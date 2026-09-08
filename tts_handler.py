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
import os
import wave
from typing import AsyncIterator

import websockets
from fastapi import WebSocket, WebSocketDisconnect

import config

logger = logging.getLogger(__name__)


def _build_rime_tts_url(
    speaker: str | None = None,
    model_id: str | None = None,
    audio_format: str | None = None,
) -> str:
    """Construct the Rime ws3 URL with query params for audio format."""
    base = config.RIME_TTS_WS_URL
    spk = speaker or config.RIME_SPEAKER
    mid = model_id or config.RIME_MODEL_ID
    fmt = audio_format or config.RIME_AUDIO_FORMAT
    params = f"speaker={spk}&modelId={mid}&audioFormat={fmt}"
    return f"{base}?{params}"


async def stream_tts(
    text: str,
    speaker: str | None = None,
    model_id: str | None = None,
    audio_format: str | None = None,
) -> AsyncIterator[bytes]:
    """
    Stream audio chunks directly from Rime TTS /ws3 endpoint.
    Yields raw PCM (or requested format) bytes as each chunk arrives.
    """
    if not config.RIME_API_KEY:
        raise ValueError(
            "RIME_API_KEY is not set. Please add it to your .env file."
        )

    rime_url = _build_rime_tts_url(
        speaker=speaker, model_id=model_id, audio_format=audio_format
    )
    rime_headers = {"Authorization": f"Bearer {config.RIME_API_KEY}"}

    async with websockets.connect(rime_url, additional_headers=rime_headers) as rime_ws:
        # Send text to synthesize and EOS to flush
        await rime_ws.send(json.dumps({"text": text}))
        await rime_ws.send(json.dumps({"operation": "eos"}))

        async for raw in rime_ws:
            if isinstance(raw, str):
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                b64_audio = msg.get("data") or msg.get("audio")
                if b64_audio and msg.get("type") in ("chunk", None):
                    yield base64.b64decode(b64_audio)
                elif msg.get("type") == "done":
                    break
                elif "error" in msg:
                    raise RuntimeError(f"Rime TTS error: {msg['error']}")

            elif isinstance(raw, bytes):
                yield raw


async def synthesize_speech(
    text: str,
    speaker: str | None = None,
    model_id: str | None = None,
    audio_format: str | None = None,
) -> bytes:
    """
    Synthesize complete speech for `text` using Rime TTS /ws3.
    Returns all audio bytes concatenated.
    """
    chunks: list[bytes] = []
    async for chunk in stream_tts(
        text=text,
        speaker=speaker,
        model_id=model_id,
        audio_format=audio_format,
    ):
        chunks.append(chunk)
    return b"".join(chunks)


def save_audio(
    audio_bytes: bytes,
    output_path: str = "output.wav",
    sample_rate: int = 22050,
    channels: int = 1,
    bits_per_sample: int = 16,
    audio_format: str = "pcm",
) -> str:
    """
    Save audio bytes to disk.
    If format is PCM and filename ends in .wav, creates a standard WAV header.
    Returns the resolved output path.
    """
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if audio_format.lower() == "pcm" and output_path.lower().endswith(".wav"):
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(bits_per_sample // 8)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
    else:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

    return os.path.abspath(output_path)


async def text_to_speech(
    text: str,
    output_path: str = "output.wav",
    speaker: str | None = None,
    model_id: str | None = None,
    audio_format: str | None = None,
    sample_rate: int = 22050,
) -> str:
    """
    Convenience function: synthesizes speech from text and saves to file.
    Returns the absolute path of the saved audio file.
    """
    audio_bytes = await synthesize_speech(
        text=text,
        speaker=speaker,
        model_id=model_id,
        audio_format=audio_format,
    )
    fmt = audio_format or config.RIME_AUDIO_FORMAT
    return save_audio(
        audio_bytes=audio_bytes,
        output_path=output_path,
        sample_rate=sample_rate,
        audio_format=fmt,
    )


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

                b64_audio = msg.get("data") or msg.get("audio")
                if b64_audio and msg.get("type") in ("chunk", None):
                    audio_bytes = base64.b64decode(b64_audio)
                    await client_ws.send_bytes(audio_bytes)
                elif msg.get("type") == "done":
                    logger.info("[TTS] Rime sent done signal")
                    break
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

