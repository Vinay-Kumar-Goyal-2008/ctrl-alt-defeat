"""
stt_handler.py — FastAPI WebSocket handler for the /stt endpoint.

Architecture
────────────
Client (Twilio / test mic)  ←──WS──→  /stt (this file)  ←──WS──→  Rime STT

Flow
────
1. Client connects ws://server/stt
2. Server opens a WS connection to the Rime STT WebSocket endpoint
3. Client sends raw binary audio chunks (16-bit PCM, 16 kHz, mono)
4. Server forwards each chunk to Rime STT
5. Rime STT streams back JSON transcription events
6. Server normalises them and sends to client:
       {"text": "partial text", "is_final": false}
       {"text": "final sentence", "is_final": true}
7. On client disconnect: send EOS to Rime, close cleanly
8. On Rime WS drop: reconnect automatically (up to MAX_RECONNECT_ATTEMPTS)

--------------------------------------------------------------------------------
STUB NOTICE:
    The Rime STT endpoint URL and message format are not yet confirmed.
    _build_rime_stt_url() and _parse_rime_stt_message() are marked
    with TODO comments.  Once you have the real endpoint details, fill in:
      1. RIME_STT_WS_URL in your .env
      2. The query params in _build_rime_stt_url()
      3. The JSON parsing logic in _parse_rime_stt_message()
--------------------------------------------------------------------------------
"""

import asyncio
import json
import logging

import websockets
from fastapi import WebSocket, WebSocketDisconnect

import config

logger = logging.getLogger(__name__)

MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY_SECONDS = 2


def _build_rime_stt_url() -> str:
    """
    Construct the Rime STT WebSocket URL.

    TODO: Fill in the correct query parameters once the Rime STT
          endpoint is confirmed.  Common params for STT APIs:
              encoding=linear16&sample_rate=16000&language=en-US
    """
    base = config.RIME_STT_WS_URL
    if not base:
        raise RuntimeError(
            "RIME_STT_WS_URL is not set.  Update your .env with the "
            "Rime STT WebSocket endpoint URL."
        )
    # TODO: Add the correct query parameters for Rime STT
    params = "encoding=linear16&sample_rate=16000"
    return f"{base}?{params}"


def _parse_rime_stt_message(raw: str) -> dict | None:
    """
    Parse a message from Rime STT and normalise it to our contract:
        {"text": "...", "is_final": bool}

    Returns None if the message should be silently ignored.

    TODO: Update this function once the real Rime STT response format
          is confirmed.  The current implementation assumes a format
          similar to Deepgram/AssemblyAI for placeholder purposes.

    Example Rime STT response shapes to handle:
        {"transcript": "hello", "isFinal": false}   ← typical STT partial
        {"transcript": "hello world", "isFinal": true}
        {"type": "Results", "text": "hello", "final": true}
        {"error": "..."}
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[STT] Rime sent non-JSON text (ignored): %.80s", raw)
        return None

    # Handle error events
    if "error" in msg:
        logger.error("[STT] Rime STT error: %s", msg["error"])
        return {"error": msg["error"]}

    # TODO: Adapt these field names to match the real Rime STT response
    # ── Try common response shapes ─────────────────────────────────────────
    text = (
        msg.get("transcript")           # Deepgram-style
        or msg.get("text")              # Generic
        or msg.get("alternatives", [{}])[0].get("transcript", "")  # Google-style
    )

    # is_final: try multiple common key names
    is_final = (
        msg.get("isFinal")              # camelCase
        or msg.get("is_final")          # snake_case
        or msg.get("final")             # short form
        or False
    )

    if not text:
        return None  # Metadata-only frame, ignore

    return {"text": text, "is_final": bool(is_final)}


async def handle_stt(client_ws: WebSocket) -> None:
    """
    Main handler for ws://server/stt.

    Bridges the client WebSocket to Rime's STT endpoint:
    - Receives raw PCM binary from client → forwards to Rime STT
    - Receives transcript JSON from Rime → normalises → sends to client
    - Reconnects Rime WS automatically on drop
    """
    await client_ws.accept()
    client_addr = client_ws.client
    logger.info("[STT] Client connected: %s", client_addr)

    if not config.RIME_STT_WS_URL:
        logger.error("[STT] RIME_STT_WS_URL not configured — rejecting client")
        await client_ws.send_text(
            json.dumps({"error": "STT endpoint not configured on server"})
        )
        await client_ws.close(code=1011, reason="STT not configured")
        return

    rime_headers = {"Authorization": f"Bearer {config.RIME_API_KEY}"}

    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
        try:
            rime_url = _build_rime_stt_url()
            logger.info("[STT] Connecting to Rime STT (attempt %d): %s", attempt, rime_url)

            async with websockets.connect(
                rime_url, additional_headers=rime_headers
            ) as rime_ws:
                logger.info("[STT] Connected to Rime STT")

                client_to_rime_task = asyncio.create_task(
                    _audio_to_rime(client_ws, rime_ws, client_addr)
                )
                rime_to_client_task = asyncio.create_task(
                    _transcripts_to_client(rime_ws, client_ws, client_addr)
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

                # If the client disconnected cleanly, don't reconnect
                if _client_disconnected(done):
                    logger.info("[STT] Client disconnected — no reconnect needed")
                    return

                # Otherwise Rime dropped — fall through to reconnect
                logger.warning(
                    "[STT] Rime STT connection lost (attempt %d/%d)",
                    attempt,
                    MAX_RECONNECT_ATTEMPTS,
                )

        except websockets.exceptions.InvalidStatusCode as exc:
            logger.error(
                "[STT] Rime rejected connection HTTP %s (attempt %d). "
                "Check RIME_API_KEY and RIME_STT_WS_URL.",
                exc.status_code,
                attempt,
            )
        except WebSocketDisconnect:
            logger.info("[STT] Client disconnected: %s", client_addr)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("[STT] Unexpected error (attempt %d): %s", attempt, exc)

        if attempt < MAX_RECONNECT_ATTEMPTS:
            logger.info("[STT] Reconnecting in %ds…", RECONNECT_DELAY_SECONDS)
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)

    logger.error("[STT] Max reconnect attempts reached, closing client connection")
    await _close_client(client_ws, 1011, "STT upstream connection failed")


async def _audio_to_rime(
    client_ws: WebSocket,
    rime_ws: websockets.WebSocketClientProtocol,
    client_addr: object,
) -> None:
    """Forward raw PCM audio bytes from the client to Rime STT."""
    try:
        async for chunk in client_ws.iter_bytes():
            logger.debug("[STT] Client → Rime: %d bytes of audio", len(chunk))
            await rime_ws.send(chunk)
    except WebSocketDisconnect:
        logger.info("[STT] Client disconnected (audio stream ended): %s", client_addr)
        # Signal end-of-stream to Rime STT
        # TODO: Update this EOS signal to match Rime STT's documented format
        try:
            await rime_ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:  # noqa: BLE001
            pass
    except websockets.exceptions.ConnectionClosed:
        logger.info("[STT] Rime WS closed while forwarding audio")


async def _transcripts_to_client(
    rime_ws: websockets.WebSocketClientProtocol,
    client_ws: WebSocket,
    client_addr: object,
) -> None:
    """Receive transcript events from Rime STT and send normalised JSON to client."""
    try:
        async for raw in rime_ws:
            if isinstance(raw, bytes):
                # STT should return JSON text, not bytes — log and skip
                logger.warning("[STT] Rime sent binary data unexpectedly (%d bytes)", len(raw))
                continue

            result = _parse_rime_stt_message(raw)
            if result is None:
                continue

            logger.debug("[STT] Rime → Client: %s", result)
            await client_ws.send_text(json.dumps(result))

    except websockets.exceptions.ConnectionClosed:
        logger.info("[STT] Rime STT connection closed")
    except WebSocketDisconnect:
        logger.info("[STT] Client disconnected while sending transcripts: %s", client_addr)


def _client_disconnected(done_tasks: set) -> bool:
    """
    Return True if any of the completed tasks ended due to a client disconnect.
    Used to decide whether to reconnect to Rime STT.
    """
    for task in done_tasks:
        exc = task.exception() if not task.cancelled() else None
        if isinstance(exc, WebSocketDisconnect):
            return True
    return False


async def _close_client(ws: WebSocket, code: int, reason: str) -> None:
    """Best-effort close of the client WebSocket."""
    try:
        await ws.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001
        pass
