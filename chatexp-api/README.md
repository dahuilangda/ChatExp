# chatexp-api

Backend for the ChatExp SeaTable plugin. Three jobs:

- BO endpoints (`/v1/bo/analyze`, `/v1/bo/suggest`) — Celery/Redis jobs
  running a torch-based optimizer in the worker container.
- LLM proxy (`POST /v1/llm/chat`) — server-side relay to an
  OpenAI-compatible model, so the browser never talks to the model directly.
- File parsing (`/v1/files/*`) — downloads SeaTable attachments and parses
  them with a sibling docling-serve container; results cached in redis and a
  shared volume.

## Quick start

```bash
cd chatexp-api
cp .env.example .env        # set CHATEXP_BO_API_KEY at minimum (required for /v1/llm/chat and /v1/files/edit)
docker compose up -d --build
docker compose ps           # redis, api, worker, worker-analysis, beat, docling-serve
curl http://localhost:8000/health          # {"status":"ok"}
```

The API listens on host port 8000. In the plugin Settings set the
Optimization endpoint to `http://<this-host>:8000/v1` and the API key to
your `CHATEXP_BO_API_KEY`.

To rebuild only the API container after code changes:

```bash
docker compose up -d --build api
```

## Configuration

Every variable is documented in [.env.example](./.env.example). Groups:

- Auth: `CHATEXP_BO_API_KEY` (gates `/v1/bo/*`, `/v1/jobs/*`, `/v1/files/*`;
  empty disables checking on those routes — not for internet-facing deploys).
  Two expensive routes are STRICT and refuse requests with 503 until the key
  is set: `POST /v1/llm/chat` (relays table content to the model) and
  `POST /v1/files/edit` (~200MB bodies, LLM-cost-incurring), so the empty
  compose default can never expose them unauthenticated.
  `CHATEXP_ADMIN_API_KEY` (gates the files admin routes).
- CORS: `CHATEXP_CORS_ORIGINS` (default `*`; safe because the API uses
  `allow_credentials=False` and authenticates via the `Authorization`
  header, not cookies).
- LLM proxy: `CHATEXP_LLM_ENDPOINT/_API_KEY/_MODEL`,
  `CHATEXP_LLM_PROXY_ALLOWED_BASE_URLS`, `CHATEXP_LLM_PROXY_TIMEOUT_SECONDS`.
- File parsing: `CHATEXP_DOCLING_BASE_URL` (default
  `http://docling-serve:5001`), `CHATEXP_FILE_MAX_BYTES`,
  `CHATEXP_FILE_CACHE_TTL`, `CHATEXP_PARSED_FILES_DIR/_MAX_BYTES`,
  `CHATEXP_SEATABLE_API_TOKEN/_SERVER`.
- Torch: `TORCH_INDEX_URL` (CPU wheels by default; see Dockerfile).

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | liveness |
| POST | `/v1/llm/chat` | BO key (required — 503 when unset) | relay chat to the model |
| POST | `/v1/bo/analyze` | BO key | enqueue analysis job (202) |
| POST | `/v1/bo/suggest` | BO key | enqueue design job (202) |
| GET | `/v1/jobs/{id}` | BO key | poll job status/result |
| DELETE/POST | `/v1/jobs/{id}/cancel` | BO key | cancel a job |
| POST | `/v1/files/parse` | BO key | enqueue attachment parse job (202) |
| GET | `/v1/files/jobs/{id}` | BO key | poll parse job status |
| POST | `/v1/files/jobs/{id}/fetch` | BO key | fetch parsed chunks under a char budget |
| POST | `/v1/files/docs/{docId}/fetch` | BO key | fetch chunks of a cached document |
| POST | `/v1/files/analyze` | BO key | enqueue agentic document analysis |
| GET | `/v1/files/analyze/{id}` | BO key | poll analysis job status |
| POST | `/v1/files/edit` | BO key (required — 503 when unset) | enqueue tracked-changes Word edit (202) |
| GET | `/v1/files/edit/{id}` | BO key | poll edit job status |
| GET | `/v1/files/docs` | admin key | list parsed documents |
| GET/DELETE | `/v1/files/docs/{docId}` | admin key | inspect/delete one document |
| POST | `/v1/files/docs/cleanup` | admin key | evict expired documents |
| GET/POST | `/v1/files/stats`, `/v1/files/stats/reset` | admin key | cache statistics |

## HTTPS deployment (required when SeaTable is HTTPS)

Browsers block HTTP requests from an HTTPS SeaTable page. Expose the API through your HTTPS reverse proxy, or use the optional [Caddy deployment](deploy/README.md):

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.https.yml --profile https up -d --build
```

Provide your TLS certificate as described in that guide, then configure the plugin with your HTTPS API endpoint.

## Example session

Save the payload:

```json
{
  "project_name": "demo",
  "table": "Table1",
  "view": "Default",
  "rows": [
    {"Name": "E1", "温度": 30, "反应时间": 30, "催化剂": "A", "产率": 62},
    {"Name": "E2", "温度": 40, "反应时间": 45, "催化剂": "A", "产率": 74},
    {"Name": "E3", "温度": 50, "反应时间": 60, "催化剂": "B", "产率": 81},
    {"Name": "E4", "温度": 60, "反应时间": 75, "催化剂": "B", "产率": 79},
    {"Name": "E5", "温度": 45, "反应时间": 50, "催化剂": "A", "产率": 77}
  ],
  "factors": [
    {"name": "温度", "kind": "continuous", "min_value": 25, "max_value": 70},
    {"name": "反应时间", "kind": "continuous", "min_value": 20, "max_value": 90},
    {"name": "催化剂", "kind": "categorical", "categories": ["A", "B"]}
  ],
  "objectives": [
    {"name": "产率", "goal": "maximize", "weight": 1}
  ],
  "n_points": 3,
  "runtime": {"candidate_pool_size": 128, "ensemble_size": 2, "training_steps": 50},
  "language": "zh"
}
```

```bash
curl -sS -X POST http://localhost:8000/v1/bo/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  --data-binary @/tmp/chatexp-bo-demo.json
# -> {"job_id": "...", "status": "queued", "status_url": ".../v1/jobs/..."}

curl -sS http://localhost:8000/v1/jobs/<job_id> \
  -H "Authorization: Bearer <your-api-key>"
# -> {"status": "succeeded", "result": {"analysis": {"ready": true, ...}}}

curl -sS -X POST http://localhost:8000/v1/bo/suggest ...   # same payload
# -> {"status": "succeeded", "result": {"suggestions": [{"factor": "BO", "score": 0.0047, "values": {...}}]}}

curl -sS -X DELETE http://localhost:8000/v1/jobs/<job_id>/cancel \
  -H "Authorization: Bearer <your-api-key>"
# -> poll the job: {"status":"canceled"}
```

## Payload notes

- `rows`: current experiment rows.
- `factors`: controllable variables (`continuous` with min/max, or
  `categorical` with categories).
- `objectives`: measured outcomes; `goal` is `maximize`, `minimize`, or
  `target`, weighted by `weight`.
- `n_points`: number of new candidate designs.
- `runtime.candidate_pool_size` / `ensemble_size` / `training_steps`:
  larger values search more candidates, improve uncertainty estimates, and
  train longer respectively.
