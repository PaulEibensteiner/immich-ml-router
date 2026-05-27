import asyncio
import logging
import os
import socket
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response

LOCAL_ML_URL = os.environ.get("LOCAL_ML_URL", "http://immich-ml-local:3003")
REMOTE_ML_URL = os.environ.get("REMOTE_ML_URL", "http://gpu-pc:3003")
REMOTE_MAC = os.environ.get("REMOTE_MAC", "")
WOL_BROADCAST = os.environ.get("WOL_BROADCAST", "255.255.255.255")
WOL_PORT = int(os.environ.get("WOL_PORT", "9"))
WOL_ENABLED = os.environ.get("WOL_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WOL_MIN_INTERVAL_SECONDS = float(os.environ.get("WOL_MIN_INTERVAL_SECONDS", "1.0"))
# After WoL, block heavy (face/OCR) requests for up to this long while polling the remote.
# Immich's microservices have generous ML timeouts, so blocking ~60-120s avoids the job
# failing and being put on a long retry backoff.
REMOTE_BOOT_WAIT_SECONDS = float(os.environ.get("REMOTE_BOOT_WAIT_SECONDS", "90"))
REMOTE_POLL_INTERVAL_SECONDS = float(
    os.environ.get("REMOTE_POLL_INTERVAL_SECONDS", "3")
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
# Short connect timeout detects offline PC quickly; read timeout stays long for actual inference
search_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))

last_wol_sent = 0.0
wol_lock = asyncio.Lock()


def _build_magic_packet(mac: str) -> bytes:
    normalized_mac = mac.replace(":", "").replace("-", "")
    if len(normalized_mac) != 12:
        raise ValueError(f"Invalid MAC address: {mac}")

    mac_bytes = bytes.fromhex(normalized_mac)
    return b"\xff" * 6 + mac_bytes * 16


def _send_wol_packet() -> None:
    packet = _build_magic_packet(REMOTE_MAC)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as wol_socket:
        wol_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        wol_socket.sendto(packet, (WOL_BROADCAST, WOL_PORT))


async def maybe_send_wol() -> None:
    global last_wol_sent

    if not WOL_ENABLED:
        return

    if not REMOTE_MAC:
        log.warning("WOL is enabled but REMOTE_MAC is empty")
        return

    async with wol_lock:
        now = time.monotonic()
        if now - last_wol_sent < WOL_MIN_INTERVAL_SECONDS:
            return

        try:
            await asyncio.get_running_loop().run_in_executor(None, _send_wol_packet)
        except (OSError, ValueError):
            log.exception("failed to send WOL packet")
            return

        last_wol_sent = now
        log.info("sent WOL packet to %s via %s:%s", REMOTE_MAC, WOL_BROADCAST, WOL_PORT)


async def post_remote_with_wait(
    body: bytes, content_type: str, max_wait_seconds: float
) -> Response | None:
    """POST to the remote, blocking up to `max_wait_seconds` while it boots.

    Returns a Response on success, or None if the remote never came back in time.
    """
    deadline = time.monotonic() + max_wait_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            # apparently, this will wait for 5 seconds, as defined above in the client connect argument parameter
            # so the full time of every loop iteration is actually 5 + REMOTE_POLL_INTERVAL_SECONDS
            resp = await client.post(
                REMOTE_ML_URL + "/predict",
                content=body,
                headers={"content-type": content_type},
            )
            if attempt > 1:
                log.info("remote came back after %d attempts", attempt)
            return Response(
                resp.content,
                resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("remote did not come back within %.0fs", max_wait_seconds)
                return None
            sleep_for = min(REMOTE_POLL_INTERVAL_SECONDS, remaining)
            if attempt == 1:
                log.info(
                    "remote offline, blocking up to %.0fs while it boots",
                    max_wait_seconds,
                )
            await asyncio.sleep(sleep_for)


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
        await maybe_send_wol()
        try:
            log.info("-> remote search (%s bytes)", len(body))
            resp = await search_client.post(
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
            log.info("remote offline, routing search -> local")

        log.info("-> local search (%s bytes)", len(body))
        for attempt in range(2):
            try:
                resp = await client.post(
                    LOCAL_ML_URL + "/predict",
                    content=body,
                    headers={"content-type": content_type},
                )
                return Response(
                    resp.content,
                    resp.status_code,
                    media_type=resp.headers.get("content-type"),
                )
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
        await maybe_send_wol()
        log.info("-> remote (%s bytes)", len(body))
        resp = await post_remote_with_wait(body, content_type, REMOTE_BOOT_WAIT_SECONDS)
        if resp is not None:
            return resp
        log.warning("remote ML offline, returning 503")
        return Response(
            status_code=503,
            content=b'{"error":"remote ML offline"}',
            media_type="application/json",
        )
