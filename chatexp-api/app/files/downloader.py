"""Server-side SeaTable attachment download.

Reproduces the plugin's resolveSeaTableDownloadLink + direct-fetch resolution
order, but runs in the celery worker so file bytes stay on the host. Reuses the
same SeaTable app-download-link API the browser plugin uses.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from . import config

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised for any unrecoverable download/content problem."""


def download(payload: dict[str, Any]) -> tuple[bytes, str]:
    """Download a SeaTable attachment and return (bytes, content_type).

    Resolution is explicit, not a chain of fallbacks:
      1. ``path`` (a SeaTable asset path like /files/...) REQUIRES a base-level
         API-Token (from the plugin payload or CHATEXP_SEATABLE_API_TOKEN) and
         resolves a pre-authenticated download_link via app-download-link, the
         only server-side-valid way to fetch SeaTable attachments.
      2. ``url`` alone is fetched directly (public-URL case); credentials are
         attached only when the host matches the declared server.

    Missing credentials raise a clear DownloadError instead of silently trying
    weaker paths, so misconfiguration is visible in the job error.
    """
    max_bytes = int(payload.get("maxBytes") or config.file_max_bytes())
    name = str(payload.get("name") or "")
    path = str(payload.get("path") or "").strip()
    server = str(payload.get("server") or "").strip().rstrip("/")
    api_token = str(payload.get("apiToken") or "").strip()
    url = str(payload.get("url") or "").strip()
    timeout = config.seatable_timeout()

    # app-download-link ONLY accepts a base-level API-Token (verified against
    # the official seatable-api-python source: get_file_download_link uses
    # parse_headers(self.token), where self.token is the API-Token, not the JWT
    # accessToken). The dtableStore accessToken (JWT) is rejected 403 by that
    # endpoint, and the worker has no browser session cookie.
    #
    # Server-side configuration (CHATEXP_SEATABLE_API_TOKEN / _SERVER) is an
    # explicit deployment choice, not a silent fallback: when the plugin omits
    # credentials we REQUIRE the server config, and we say so in the log.
    if not api_token:
        api_token = config.seatable_api_token()
        token_source = "server-config" if api_token else "missing"
    else:
        token_source = "plugin"
    if not server:
        server = config.seatable_server()

    logger.info(
        "SeaTable download: name=%r path=%r server=%r apiToken=%s(%s) url=%r",
        name,
        path,
        server,
        f"<{len(api_token)} chars>" if api_token else "<empty>",
        token_source,
        url,
    )

    download_url: str | None = None
    auth_headers: dict[str, str] = {}

    if path:
        if not server:
            raise DownloadError(
                "SeaTable server is missing (plugin payload has no server and CHATEXP_SEATABLE_SERVER is not set)."
            )
        if not api_token:
            raise DownloadError(
                "SeaTable API-Token is missing (plugin payload has no apiToken and CHATEXP_SEATABLE_API_TOKEN is not set). "
                "app-download-link requires a base-level API-Token; the dtableStore JWT and session cookies do not work server-side."
            )
        # Preferred and ONLY valid path for SeaTable attachments: resolve a
        # pre-authenticated download_link via the API, then fetch it.
        _guard_server_declaration(server)
        download_url = _resolve_download_link(server, api_token, path, timeout)
        # SSRF discipline does not trust the API-resolved link either: its
        # host must pass the same guard as a direct url (private/loopback/
        # link-local targets are refused unless they ARE the declared server,
        # whose intranet address is legitimate).
        _guard_private_network_host(download_url, server)
        logger.info("Resolved app-download-link for path=%r", path)
    elif url:
        if server and not _url_host_matches(url, server):
            raise DownloadError(
                "Refused to fetch a url whose host does not match the declared SeaTable server (SSRF guard)."
            )
        _guard_private_network_host(url, server)
        download_url = url
        if api_token and server and _url_host_matches(url, server):
            auth_headers["Authorization"] = f"Token {api_token}"
    else:
        raise DownloadError("Either (server + apiToken + path) or a direct url must be provided.")

    # trust_env=False: never route SeaTable downloads through the host HTTP_PROXY.
    # SeaTable attachment URLs are intranet/same-network and the proxy breaks
    # them; the proxy is only wanted for LLM outbound calls (handled elsewhere).
    # httpx's no_proxy matching is also unreliable for IP CIDR ranges, so we
    # disable env-based proxying entirely for file downloads.
    data, content_type = _fetch(download_url, auth_headers, max_bytes, timeout, declared_server=server)
    _validate_content(name, content_type, data)
    return data, content_type


# Maximum redirects followed on a download. SeaTable asset links legitimately
# hop once or twice (auth → storage node); five is a generous ceiling and a
# redirect chain longer than that is a loop or an attack, refused loudly.
_MAX_REDIRECTS = 5

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _fetch(
    download_url: str,
    auth_headers: dict[str, str],
    max_bytes: int,
    timeout: float,
    declared_server: str = "",
) -> tuple[bytes, str]:
    """Stream-download a url with size and content-type guards. Returns
    (bytes, content_type). Raises DownloadError on transport/size/HTTP errors.

    Redirects are followed MANUALLY (follow_redirects=False): every Location
    hop is validated against the SSRF guard BEFORE it is followed, so a public
    asset host that 302s to 169.254.169.254 (or any private/loopback target)
    is refused with an error naming the blocked hop instead of silently
    fetched. Authorization headers are dropped on cross-host hops so the
    SeaTable token is never leaked to a third-party redirect target."""
    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        url = download_url
        headers = dict(auth_headers)
        redirects = 0
        while True:
            try:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise DownloadError(
                                f"SeaTable download returned HTTP {response.status_code} with no Location header."
                            )
                        if redirects >= _MAX_REDIRECTS:
                            raise DownloadError(
                                f"SeaTable download exceeded {_MAX_REDIRECTS} redirects (last hop: {url})."
                            )
                        next_url = str(httpx.URL(url).join(location))
                        _guard_redirect_hop(next_url, from_url=url, declared_server=declared_server)
                        redirects += 1
                        if (urlparse(next_url).hostname or "").lower() != (urlparse(url).hostname or "").lower():
                            headers = {}
                        url = next_url
                        continue
                    if response.status_code >= 400:
                        raise DownloadError(f"SeaTable download returned HTTP {response.status_code}.")
                    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
                    declared_length = response.headers.get("content-length")
                    if declared_length:
                        try:
                            if int(declared_length) > max_bytes:
                                raise DownloadError(f"File exceeds the {max_bytes}-byte limit (declared {declared_length}).")
                        except ValueError:
                            pass  # malformed content-length; rely on streaming guard
                    chunks = bytearray()
                    for block in response.iter_bytes(chunk_size=64 * 1024):
                        chunks.extend(block)
                        if len(chunks) > max_bytes:
                            raise DownloadError(f"File exceeds the {max_bytes}-byte limit while streaming.")
                    return bytes(chunks), content_type
            except httpx.RequestError as exc:
                raise DownloadError(f"SeaTable download failed: {exc}") from exc


def _url_host_matches(url: str, server: str) -> bool:
    """True when url and server share the same scheme://host[:port]. Used by the
    SSRF guard so a caller cannot point the downloader at an arbitrary host
    (cloud metadata, internal services, attacker token-capture endpoints)."""
    try:
        u = urlparse(url)
        s = urlparse(server)
    except ValueError:
        return False
    if not u.hostname or not s.hostname:
        return False
    if u.hostname.lower() != s.hostname.lower():
        return False
    if (u.port or 443 if u.scheme == "https" else u.port or 80) != (s.port or 443 if s.scheme == "https" else s.port or 80):
        return False
    return True


def _guard_private_network_host(url: str, declared_server: str) -> None:
    """Refuse a direct-url fetch whose host resolves into private/loopback/
    link-local space unless it is the declared SeaTable server itself (whose
    intranet address is legitimate). Blocks the classic SSRF targets — cloud
    metadata (169.254.169.254), localhost services, RFC1918 internals — while
    leaving genuinely public URLs untouched. Raises DownloadError naming the
    blocked host."""
    _assert_public_host(url, declared_server, refuse_prefix="Refused to fetch")


def _guard_redirect_hop(target_url: str, from_url: str, declared_server: str) -> None:
    """Refuse to FOLLOW a redirect whose target host resolves into private/
    loopback/link-local space (same policy as the initial-url guard; the
    declared SeaTable server is exempt). The error names the blocked hop —
    the Location target and where it came from — so the job log shows exactly
    which redirect was refused."""
    _assert_public_host(
        target_url,
        declared_server,
        refuse_prefix=f"Refused to follow the redirect from {from_url!r} to",
    )


def _assert_public_host(url: str, declared_server: str, refuse_prefix: str) -> None:
    """Shared host-resolution check behind both SSRF guards: empty-host urls
    are refused, the declared server's host is exempt, and any resolved
    private/loopback/link-local address raises DownloadError with
    ``refuse_prefix`` naming the operation."""
    hostname = (urlparse(url).hostname or "").lower()
    if not hostname:
        raise DownloadError(f"{refuse_prefix} a url with no host (SSRF guard).")
    declared_host = (urlparse(declared_server).hostname or "").lower() if declared_server else ""
    if declared_host and hostname == declared_host:
        return
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise DownloadError(f"Could not resolve host {hostname!r} for the url fetch: {exc}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_private or address.is_loopback or address.is_link_local:
            raise DownloadError(
                f"{refuse_prefix} {hostname!r} ({address}) — it resolves to a private, "
                f"loopback, or link-local address and is not the declared SeaTable server (SSRF guard)."
            )


def _guard_server_declaration(server: str) -> None:
    """The app-download-link request carries the API-Token, so its target host
    must be trustworthy. A client-declared server is trusted only when it
    matches the deploy-time CHATEXP_SEATABLE_SERVER or is a public host; a
    private/loopback/link-local declaration that the deployment did not make
    itself is an SSRF probe and is refused before any request is sent."""
    configured = config.seatable_server()
    if configured and _url_host_matches(server, configured):
        return
    _assert_public_host(server, configured, refuse_prefix="Refused declared SeaTable server")


def _resolve_download_link(server: str, api_token: str, path: str, timeout: float) -> str:
    """GET {server}/api/v2.1/dtable/app-download-link/?path=<path> with a
    base-level API-Token, returning the pre-authenticated download_link.

    This endpoint ONLY accepts the API-Token (verified against the official
    seatable-api-python implementation of get_file_download_link). The JWT
    accessToken from dtableStore is rejected with 403, and session cookies are
    not available server-side.
    """
    endpoint = f"{server.rstrip('/')}/api/v2.1/dtable/app-download-link/?path={quote(path, safe='')}"
    headers = {"Authorization": f"Token {api_token}"}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, trust_env=False) as client:
            response = client.get(endpoint, headers=headers)
    except httpx.RequestError as exc:
        raise DownloadError(f"SeaTable download-link API unreachable: {exc}") from exc
    if response.status_code >= 400:
        raise DownloadError(f"SeaTable download-link API returned HTTP {response.status_code}.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise DownloadError("SeaTable download-link API returned non-JSON.") from exc
    link = payload.get("download_link") or payload.get("downloadLink") or payload.get("url")
    if not isinstance(link, str) or not link:
        raise DownloadError("SeaTable download-link API did not return a download_link.")
    return link


def _validate_content(name: str, content_type: str, data: bytes) -> None:
    """Guard against HTML login-page redirects and basic size sanity (mirrors
    the plugin's validateDownloadedMediaContent essentials).

    An HTML response where binary was expected has many possible causes
    (403 on the URL, missing/expired session, wrong URL shape, CSP, or the
    server really served an HTML file). We report the raw symptom instead of
    guessing a single cause, so the real problem stays visible."""
    if not data:
        raise DownloadError("Downloaded file is empty.")
    # HTML login-page redirect detection: SeaTable returns an HTML page instead
    # of the binary when the request is not authenticated. Check the magic bytes.
    head = data[:512].lstrip()[:5].lower()
    if head.startswith(b"<html") or head.startswith(b"<!doc") or content_type == "text/html":
        preview = data[:300].decode("utf-8", "replace").strip()
        raise DownloadError(
            f"SeaTable returned an HTML page instead of file bytes for {name!r} "
            f"(content-type {content_type!r}). The URL likely requires authentication "
            f"that this request did not carry. Response preview: {preview[:200]}"
        )


def guess_mime(name: str) -> str:
    """Best-effort MIME guess from extension, used when SeaTable omits one."""
    lower = (name or "").lower()
    mapping = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".html": "text/html",
        ".htm": "text/html",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".json": "application/json",
        ".txt": "text/plain",
    }
    for suffix, mime in mapping.items():
        if lower.endswith(suffix):
            return mime
    return "application/octet-stream"


