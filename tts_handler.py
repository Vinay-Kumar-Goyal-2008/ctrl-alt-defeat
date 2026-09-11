
import asyncio
import base64
import json
import os
import wave

import websockets
import streamlit as st

# ============================================================
# RIME CONFIGURATION
# ============================================================

RIME_API_KEY = st.secrets.get("RIME_API_KEY")

RIME_TTS_WS_URL = st.secrets.get(
    "RIME_TTS_WS_URL",
    "wss://users-ws.rime.ai/ws3"
)

RIME_SPEAKER = st.secrets.get("RIME_SPEAKER")

RIME_MODEL_ID = st.secrets.get("RIME_MODEL_ID")

RIME_AUDIO_FORMAT = st.secrets.get(
    "RIME_AUDIO_FORMAT",
    "pcm"
)


# ============================================================
# TEXT → SPEECH
# ============================================================

async def text_to_speech(
    text: str,
    language: str,
    output_path: str = "output.wav",
    speaker: str | None = None,
    model_id: str | None = None,
    sample_rate: int = 22050,
) -> str:
    """
    Convert text to speech using Rime TTS and save it as WAV.

    Args:
        text:
            Text that should be converted to speech.

        language:
            Language code detected from the user's speech.
            Examples:
                "en"
                "hi"

        output_path:
            Path where the WAV file should be saved.

        speaker:
            Optional Rime speaker.
            Uses RIME_SPEAKER from .env if not provided.

        model_id:
            Optional Rime model ID.
            Uses RIME_MODEL_ID from .env if not provided.

        sample_rate:
            Sample rate of the output WAV file.

    Returns:
        Absolute path of the generated WAV file.
    """

    # ========================================================
    # VALIDATE API KEY
    # ========================================================

    if not RIME_API_KEY:

        raise ValueError(
            "RIME_API_KEY is not set. "
            "Check your .env file."
        )


    # ========================================================
    # GET CONFIGURATION
    # ========================================================

    speaker = speaker or RIME_SPEAKER
    model_id = model_id or RIME_MODEL_ID
    audio_format = RIME_AUDIO_FORMAT


    if not speaker:

        raise ValueError(
            "RIME_SPEAKER is not set in the .env file."
        )


    if not model_id:

        raise ValueError(
            "RIME_MODEL_ID is not set in the .env file."
        )


    # ========================================================
    # BUILD RIME WEBSOCKET URL
    # ========================================================

    rime_url = (
        f"{RIME_TTS_WS_URL}"
        f"?speaker={speaker}"
        f"&modelId={model_id}"
        f"&audioFormat={audio_format}"
    )


    # ========================================================
    # AUTHORIZATION
    # ========================================================

    headers = {
        "Authorization": f"Bearer {RIME_API_KEY}"
    }


    # ========================================================
    # AUDIO BUFFER
    # ========================================================

    audio_chunks: list[bytes] = []


    # ========================================================
    # CONNECT TO RIME
    # ========================================================

    async with websockets.connect(
        rime_url,
        additional_headers=headers
    ) as ws:


        # ====================================================
        # SEND TEXT + LANGUAGE
        # ====================================================

        request = {
            "text": text,
            "language": language
        }

        await ws.send(
            json.dumps(request)
        )


        # ====================================================
        # END OF SPEECH
        # ====================================================

        await ws.send(
            json.dumps({
                "operation": "eos"
            })
        )


        # ====================================================
        # RECEIVE AUDIO STREAM
        # ====================================================

        async for raw in ws:

            # ------------------------------------------------
            # Rime JSON message
            # ------------------------------------------------

            if isinstance(raw, str):

                try:

                    message = json.loads(raw)

                except json.JSONDecodeError:

                    continue


                # ------------------------------------------------
                # Get audio data
                # ------------------------------------------------

                encoded_audio = (
                    message.get("data")
                    or message.get("audio")
                )


                if encoded_audio:

                    audio_bytes = base64.b64decode(
                        encoded_audio
                    )

                    audio_chunks.append(
                        audio_bytes
                    )


                # ------------------------------------------------
                # Generation complete
                # ------------------------------------------------

                elif message.get("type") == "done":

                    break


                # ------------------------------------------------
                # Rime error
                # ------------------------------------------------

                elif "error" in message:

                    raise RuntimeError(
                        f"Rime TTS error: "
                        f"{message['error']}"
                    )


            # ------------------------------------------------
            # Raw binary audio
            # ------------------------------------------------

            elif isinstance(raw, bytes):

                audio_chunks.append(raw)


    # ========================================================
    # COMBINE AUDIO CHUNKS
    # ========================================================

    audio_bytes = b"".join(
        audio_chunks
    )


    if not audio_bytes:

        raise RuntimeError(
            "Rime returned no audio data."
        )


    # ========================================================
    # PREPARE OUTPUT PATH
    # ========================================================

    output_path = os.path.abspath(
        output_path
    )

    output_directory = os.path.dirname(
        output_path
    )


    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )


    # ========================================================
    # SAVE PCM → WAV
    # ========================================================

    with wave.open(
        output_path,
        "wb"
    ) as wav_file:

        # Mono
        wav_file.setnchannels(1)

        # 16-bit PCM
        wav_file.setsampwidth(2)

        # Rime sample rate
        wav_file.setframerate(
            sample_rate
        )

        wav_file.writeframes(
            audio_bytes
        )


    # ========================================================
    # RETURN PATH
    # ========================================================

    return output_path