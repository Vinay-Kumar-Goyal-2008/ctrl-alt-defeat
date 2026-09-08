"""
tts_handler.py — FastAPI WebSocket handler for the /tts endpoint.

Client (LLM layer)  ←──WS──→  /tts (this file)  ←──WS──→  Rime /ws3

Flow:
  1. Client connects ws://server/tts
  2. Server opens WS to wss://users-ws.rime.ai/ws3 with auth
  3. Client sends JSON: {"text": "Hello"} or {"operation": "clear"}
  4. Server forwards JSON to Rime
  5. Rime streams back: {"audio": "<base64 PCM>", ...}
  6. Server decodes base64 → sends raw PCM bytes to client
  7. On client disconnect: close Rime WS gracefully

Audio: raw PCM 16-bit signed LE, 22050 Hz mono (Rime default with audioFormat=pcm).
"""

import asyncio
import base64
import json
import logging

import websockets
from fastapi import WebSocket, WebSocketDisconnect

import config

logger = logging.getLogger(__name__)


def _build_rime_tts_url() -> str:
    """Construct the Rime ws3 URL with query params for audio format."""
    base = config.RIME_TTS_WS_URL
    params = (
        f"speaker={config.RIME_SPEAKER}"
        f"&modelId={config.RIME_MODEL_ID}"
        f"&audioFormat={config.RIME_AUDIO_FORMAT}"
    )
    return f"{base}?{params}"


async def handle_tts(client_ws: WebSocket) -> None:
    """
    Main handler for ws://server/tts.
    Bridges client WebSocket ↔ Rime /ws3 TTS endpoint.
    """
    await client_ws.accept()
    client_addr = client_ws.client
    logger.info("[TTS] Client connected: %s", client_addr)

    rime_url = _build_rime_tts_url()
    rime_headers = {"Authorization": f"Bearer {config.RIME_API_KEY}"}

    try:
        async with websockets.connect(rime_url, additional_headers=rime_headers) as rime_ws:
            logger.info("[TTS] Connected to Rime: %s", rime_url)

            client_to_rime_task = asyncio.create_task(
                _client_to_rime(client_ws, rime_ws, client_addr)
            )
            rime_to_client_task = asyncio.create_task(
                _rime_to_client(rime_ws, client_ws, client_addr)
            )

            done, pending = await asyncio.wait(
                [client_to_rime_task, rime_to_client_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except websockets.exceptions.InvalidStatusCode as exc:
        logger.error(
            "[TTS] Rime rejected connection (HTTP %s). Check RIME_API_KEY and URL: %s",
            exc.status_code, rime_url,
        )
        await _close_client(client_ws, 1011, "Rime TTS connection failed")
    except websockets.exceptions.ConnectionClosedError as exc:
        logger.warning("[TTS] Rime WS closed unexpectedly: %s", exc)
    except WebSocketDisconnect:
        logger.info("[TTS] Client disconnected: %s", client_addr)
    except Exception as exc:
        logger.exception("[TTS] Unexpected error: %s", exc)
        await _close_client(client_ws, 1011, "Internal server error")
    finally:
        logger.info("[TTS] Session ended for %s", client_addr)


async def _client_to_rime(
    client_ws: WebSocket,
    rime_ws: websockets.WebSocketClientProtocol,
    client_addr: object,
) -> None:
    """Forward JSON messages from client → Rime. Validates text/operation keys."""
    try:
        async for raw in client_ws.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("[TTS] Client sent non-JSON (ignored): %.80s", raw)
                continue

            if "text" not in msg and "operation" not in msg:
                logger.warning("[TTS] Unknown message shape from client: %s", msg)
                continue

            await rime_ws.send(json.dumps(msg))

    except WebSocketDisconnect:
        logger.info("[TTS] Client disconnected during receive: %s", client_addr)
    except websockets.exceptions.ConnectionClosed:
        logger.info("[TTS] Rime WS closed while forwarding client messages")


async def _rime_to_client(
    rime_ws: websockets.WebSocketClientProtocol,
    client_ws: WebSocket,
    client_addr: object,
) -> None:
    """Receive audio from Rime, decode base64 → send raw PCM bytes to client."""
    try:
        async for raw in rime_ws:
            if isinstance(raw, str):
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("[TTS] Rime sent non-JSON text (ignored): %.80s", raw)
                    continue

                if "audio" in msg:
                    audio_bytes = base64.b64decode(msg["audio"])
                    await client_ws.send_bytes(audio_bytes)
                elif "error" in msg:
                    logger.error("[TTS] Rime error: %s", msg["error"])
                    await client_ws.send_text(json.dumps({"error": msg["error"]}))
                else:
                    logger.debug("[TTS] Rime non-audio message: %s", msg)

            elif isinstance(raw, bytes):
                await client_ws.send_bytes(raw)

    except websockets.exceptions.ConnectionClosed:
        logger.info("[TTS] Rime WS closed (stream complete)")
    except WebSocketDisconnect:
        logger.info("[TTS] Client disconnected while sending audio: %s", client_addr)


async def _close_client(ws: WebSocket, code: int, reason: str) -> None:
    """Best-effort close of the client WebSocket."""
    try:
        await ws.close(code=code, reason=reason)
    except Exception:
        pass

