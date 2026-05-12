from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from httpx import ASGITransport, AsyncClient

import router as router_module
from router import app

CT = "multipart/form-data; boundary=b"
LOCAL = "http://local-mock"
REMOTE = "http://remote-mock"

CLIP_BODY = (
    b"--b\r\nContent-Disposition: form-data;"
    b' name="entries"\r\n\r\n{"clip":{"textual":{}}}\r\n--b--'
)
FACE_BODY = (
    b"--b\r\nContent-Disposition: form-data;"
    b' name="entries"\r\n\r\n{"facial-recognition":{}}\r\n--b--'
)
OCR_BODY = (
    b'--b\r\nContent-Disposition: form-data; name="entries"\r\n\r\n{"ocr":{}}\r\n--b--'
)


@pytest.fixture(autouse=True)
def patch_urls(monkeypatch):
    monkeypatch.setattr(router_module, "LOCAL_ML_URL", LOCAL)
    monkeypatch.setattr(router_module, "REMOTE_ML_URL", REMOTE)


@pytest.fixture(autouse=True)
def reset_wol_state(monkeypatch):
    monkeypatch.setattr(router_module, "last_wol_sent", 0.0)


@pytest.fixture
def ac():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_root(ac):
    async with ac:
        resp = await ac.get("/")
    assert resp.json() == {"message": "Immich ML"}


async def test_ping(ac):
    async with ac:
        resp = await ac.get("/ping")
    assert resp.text == "pong"


def test_build_magic_packet():
    packet = router_module._build_magic_packet("70:85:C2:94:30:AE")

    expected_mac = bytes.fromhex("7085C29430AE")
    assert len(packet) == 102
    assert packet == b"\xff" * 6 + expected_mac * 16


async def test_maybe_send_wol_rate_limits(monkeypatch):
    calls = []

    class FakeLoop:
        async def run_in_executor(self, executor, func):
            calls.append(func)

    monotonic_values = iter([100.0, 100.5])

    monkeypatch.setattr(router_module, "WOL_ENABLED", True)
    monkeypatch.setattr(router_module, "REMOTE_MAC", "70:85:C2:94:30:AE")
    monkeypatch.setattr(router_module, "WOL_MIN_INTERVAL_SECONDS", 1.0)
    monkeypatch.setattr(router_module, "_send_wol_packet", lambda: None)
    monkeypatch.setattr(router_module.asyncio, "get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr(
        router_module,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )

    await router_module.maybe_send_wol()
    await router_module.maybe_send_wol()

    assert len(calls) == 1
    assert router_module.last_wol_sent == 100.0


@respx.mock
async def test_clip_routes_to_remote_when_online(ac):
    maybe_send_wol = AsyncMock()
    with patch.object(router_module, "maybe_send_wol", maybe_send_wol):
        respx.post(f"{REMOTE}/predict").mock(
            return_value=httpx.Response(200, json={"clip": [[0.1, 0.2]]})
        )
        async with ac:
            resp = await ac.post(
                "/predict", content=CLIP_BODY, headers={"content-type": CT}
            )
    maybe_send_wol.assert_awaited_once()
    assert resp.status_code == 200
    assert "clip" in resp.json()
    assert respx.calls.call_count == 1


@respx.mock
async def test_clip_falls_back_to_local_when_remote_offline(ac):
    maybe_send_wol = AsyncMock()
    with patch.object(router_module, "maybe_send_wol", maybe_send_wol):
        respx.post(f"{REMOTE}/predict").mock(side_effect=httpx.ConnectError("offline"))
        respx.post(f"{LOCAL}/predict").mock(
            return_value=httpx.Response(200, json={"clip": [[0.1, 0.2]]})
        )
        async with ac:
            resp = await ac.post(
                "/predict", content=CLIP_BODY, headers={"content-type": CT}
            )
    maybe_send_wol.assert_awaited_once()
    assert resp.status_code == 200
    assert "clip" in resp.json()
    assert respx.calls.call_count == 2


@respx.mock
async def test_face_routes_to_remote(ac):
    maybe_send_wol = AsyncMock()
    with patch.object(router_module, "maybe_send_wol", maybe_send_wol):
        respx.post(f"{REMOTE}/predict").mock(
            return_value=httpx.Response(200, json={"facial-recognition": []})
        )
        async with ac:
            resp = await ac.post(
                "/predict", content=FACE_BODY, headers={"content-type": CT}
            )
    maybe_send_wol.assert_awaited_once()
    assert resp.status_code == 200
    assert respx.calls.call_count == 1


@respx.mock
async def test_ocr_routes_to_remote(ac):
    maybe_send_wol = AsyncMock()
    with patch.object(router_module, "maybe_send_wol", maybe_send_wol):
        respx.post(f"{REMOTE}/predict").mock(
            return_value=httpx.Response(200, json={"ocr": []})
        )
        async with ac:
            resp = await ac.post(
                "/predict", content=OCR_BODY, headers={"content-type": CT}
            )
    maybe_send_wol.assert_awaited_once()
    assert resp.status_code == 200
    assert respx.calls.call_count == 1


@respx.mock
async def test_clip_and_face_routes_to_remote(ac):
    # both clip + facial-recognition present → remote handles everything
    body = (
        b"--b\r\nContent-Disposition: form-data;"
        b' name="entries"\r\n\r\n{"clip":{},"facial-recognition":{}}\r\n--b--'
    )
    maybe_send_wol = AsyncMock()
    with patch.object(router_module, "maybe_send_wol", maybe_send_wol):
        respx.post(f"{REMOTE}/predict").mock(
            return_value=httpx.Response(
                200, json={"clip": [], "facial-recognition": []}
            )
        )
        async with ac:
            resp = await ac.post("/predict", content=body, headers={"content-type": CT})
    maybe_send_wol.assert_awaited_once()
    assert resp.status_code == 200
    assert respx.calls.call_count == 1


@respx.mock
async def test_remote_offline_returns_503(ac):
    respx.post(f"{REMOTE}/predict").mock(side_effect=httpx.ConnectError("offline"))
    async with ac:
        resp = await ac.post(
            "/predict", content=FACE_BODY, headers={"content-type": CT}
        )
    assert resp.status_code == 503
    assert resp.json()["error"] == "remote ML offline"


@respx.mock
async def test_local_cold_start_retries_and_succeeds(ac):
    call_count = 0

    def flaky(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("not ready yet")
        return httpx.Response(200, json={"clip": [[0.1]]})

    # Remote offline so search falls through to local, which has a cold start
    respx.post(f"{REMOTE}/predict").mock(side_effect=httpx.ConnectError("offline"))
    respx.post(f"{LOCAL}/predict").mock(side_effect=flaky)
    with patch("asyncio.sleep", return_value=None):
        async with ac:
            resp = await ac.post(
                "/predict", content=CLIP_BODY, headers={"content-type": CT}
            )
    assert resp.status_code == 200
    assert call_count == 2


@respx.mock
async def test_local_offline_after_retry_returns_503(ac):
    respx.post(f"{REMOTE}/predict").mock(side_effect=httpx.ConnectError("offline"))
    respx.post(f"{LOCAL}/predict").mock(side_effect=httpx.ConnectError("still down"))
    with patch("asyncio.sleep", return_value=None):
        async with ac:
            resp = await ac.post(
                "/predict", content=CLIP_BODY, headers={"content-type": CT}
            )
    assert resp.status_code == 503
    assert resp.json()["error"] == "local ML offline"
