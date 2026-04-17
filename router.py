import asyncio
import logging
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response

LOCAL_ML_URL = os.environ.get("LOCAL_ML_URL", "http://immich-ml-local:3003")
REMOTE_ML_URL = os.environ.get("REMOTE_ML_URL", "http://gpu-pc:3003")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()
client = httpx.AsyncClient(timeout=120.0)
# Short connect timeout detects offline PC quickly; read timeout stays long for actual inference
search_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))


@app.get("/")
async def root():
    return {"message": "Immich ML"}


@app.get("/ping")
async def ping():
    return PlainTextResponse("pong")


@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    content_type = request.headers["content-type"]

    if b'"facial-recognition"' not in body and b'"ocr"' not in body:
        # Search/CLIP: prefer remote (PC) when online, fall back to local only if offline
        try:
            log.info("→ remote search (%s bytes)", len(body))
            resp = await search_client.post(
                REMOTE_ML_URL + "/predict",
                content=body,
                headers={"content-type": content_type},
            )
            return Response(resp.content, resp.status_code, media_type=resp.headers.get("content-type"))
        except (httpx.ConnectError, httpx.TimeoutException):
            log.info("remote offline, routing search → local")

        log.info("→ local search (%s bytes)", len(body))
        for attempt in range(2):
            try:
                resp = await client.post(
                    LOCAL_ML_URL + "/predict",
                    content=body,
                    headers={"content-type": content_type},
                )
                return Response(resp.content, resp.status_code, media_type=resp.headers.get("content-type"))
            except httpx.ConnectError:
                if attempt == 0:
                    log.warning("local ML not ready, retrying in 3s...")
                    await asyncio.sleep(3)
                    continue
                log.error("local ML offline after retry")
                return Response(
                    status_code=503,
                    content=b'{"error":"local ML offline"}',
                    media_type="application/json",
                )
    else:
        log.info("→ remote (%s bytes)", len(body))
        try:
            resp = await client.post(
                REMOTE_ML_URL + "/predict",
                content=body,
                headers={"content-type": content_type},
            )
            return Response(
                resp.content,
                resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            log.warning("remote ML offline, returning 503")
            return Response(
                status_code=503,
                content=b'{"error":"remote ML offline"}',
                media_type="application/json",
            )
