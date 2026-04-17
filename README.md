# immich-ml-router

A lightweight FastAPI proxy that sits between [Immich](https://immich.app) and its ML backends, routing requests based on task type.

## Motivation

When Immich is configured to use a remote GPU server for ML, taking that machine offline breaks **everything** — including CLIP semantic search, which should always be available. This router fixes that by splitting traffic: light search tasks fall back to a local CPU server, while heavy jobs simply queue until the GPU comes back.

## Routing Logic

```
Immich Server
    │  IMMICH_MACHINE_LEARNING_URL=http://immich-ml-router:3003
    ▼
immich-ml-router
    ├── CLIP (semantic search)
    │     → remote GPU server when online
    │     → local CPU server (fallback when GPU is offline)
    │
    └── facial-recognition / OCR
          → remote GPU server only
          → 503 when offline (Immich queues and retries automatically)
```

## Quick Start

### 1. Add services to your Immich `docker-compose.yml`

```yaml
services:
  immich-ml-local:
    container_name: immich_ml_local
    image: ghcr.io/immich-app/immich-machine-learning:release
    restart: always
    volumes:
      - ml-model-cache:/cache
    environment:
      - REDIS_HOSTNAME=redis
      - MACHINE_LEARNING_MODEL_TTL=300   # exit after 5min idle, freeing RAM

  immich-ml-router:
    container_name: immich_ml_router
    image: ghcr.io/yanghu/immich-ml-router:latest
    restart: always
    environment:
      - LOCAL_ML_URL=http://immich-ml-local:3003
      - REMOTE_ML_URL=http://<your-gpu-pc-ip>:3003

volumes:
  ml-model-cache:
```

### 2. Update `.env`

```
IMMICH_MACHINE_LEARNING_URL=http://immich-ml-router:3003
```

### 3. Start

```bash
docker compose pull immich-ml-local immich-ml-router
docker compose up -d immich-ml-local immich-ml-router
docker compose up -d immich-server  # recreate to pick up new env
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `LOCAL_ML_URL` | `http://immich-ml-local:3003` | Always-on CPU ML server (fallback) |
| `REMOTE_ML_URL` | `http://gpu-pc:3003` | GPU ML server (preferred) |

Timeouts: 5s connect (fast offline detection), 120s read (covers cold model load on CPU).

## Development

```bash
# Unit tests (fast, no Docker)
make test-unit

# Integration tests (spins up mock backends via Docker Compose on port 13003)
make test-integration

# Both
make test
```

## Build

```bash
# Push to ghcr.io (public)
make push-public

# Push to both ghcr.io and a private registry (set IMAGE in .env.make)
make release
```

Copy `.env.make.example` → `.env.make` and set `IMAGE` to your private registry if needed.
