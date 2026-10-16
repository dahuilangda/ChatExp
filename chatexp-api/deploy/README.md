# HTTPS deployment

When SeaTable uses HTTPS, the plugin also needs an HTTPS API endpoint. The model endpoint can remain HTTP because the API calls it server-side.

The optional Compose overlay runs Caddy in front of the API using your TLS certificate.

1. Obtain a certificate for your domain and place the files at:
   - `deploy/certs/fullchain.pem`
   - `deploy/certs/privkey.pem`
2. Allow inbound connections to the chosen HTTPS port on your server.
3. From `chatexp-api/`, start the stack:

```bash
export CHATEXP_HTTPS_PORT=18000
docker compose -f docker-compose.yml -f deploy/docker-compose.https.yml --profile https up -d --build
```

Configure the plugin Optimization endpoint as `https://your-domain.example:18000/v1` and enter your `CHATEXP_BO_API_KEY`. The LLM proxy endpoint is derived from that API endpoint.

Caddy forwards requests to `api:8000`; the API manages CORS. Certificate files stay outside version control. Renew them through your certificate provider, then restart Caddy:

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.https.yml restart caddy
```

If you already have an HTTPS reverse proxy, route it to the API instead of starting this overlay.
