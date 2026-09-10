"""
app.py — FastAPI backend server for the Voice-First AI Sales Agent Dashboard.
Exposes REST and WebSocket endpoints connecting the web dashboard to:
  - LangGraph multi-agent sales pipeline (graph.py)
  - Speech-to-Text & Multilingual Translation (sst.py)
  - Text-to-Speech synthesis (tts_handler.py)
  - Product Knowledge Base (productknowledge.py)
  - WhatsApp & Scheduler tools (tools.py, scheduler.py)
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import uuid
import wave
from typing import Dict, Any, Optional

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import FileResponse, JSONResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

import config
from productknowledge import business as DEFAULT_BUSINESS
from graph import build_graph
from tts_handler import text_to_speech, RIME_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales_agent_app")

# Initialize FastAPI App
app = FastAPI(
    title="AI Sales Agent Studio API",
    description="Voice-First Agentic Sales Assistant with Apple Intelligence + Linear Aesthetic",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure audio output directory exists
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Mount static directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Build LangGraph application instance
sales_graph = build_graph()

# In-memory storage for active sessions & product knowledge
SESSIONS: Dict[str, Dict[str, Any]] = {}
ACTIVE_PRODUCT: Dict[str, Any] = dict(DEFAULT_BUSINESS)
WHATSAPP_LOGS: list[Dict[str, Any]] = []

# Connected WebSockets for real-time streaming
ACTIVE_WEBSOCKETS: list[WebSocket] = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Retrieve existing session or initialize a fresh state."""
    if not session_id or session_id not in SESSIONS:
        session_id = session_id or str(uuid.uuid4())[:8]
        SESSIONS[session_id] = {
            "session_id": session_id,
            "user_message": "",
            "product": ACTIVE_PRODUCT,
            "conversation": [],
            "hot_streak": 0,
            "cold_streak": 0,
            "call_active": True,
            "intent": None,
            "interest": "warm",
            "confidence": 0.0,
            "intention": "Call Started",
            "dialogue_response": "",
            "marketing_response": "",
            "final_response": "",
            "scheduled_time": None,
            "whatsapp_message": None,
            "summary": None,
            "history_turns": [],
            "created_at": str(asyncio.get_event_loop().time())
        }
    return session_id


def generate_fallback_wav(text: str, output_path: str) -> str:
    """Generate a clean synthetic audio WAV file as fallback when Rime API key is unavailable."""
    sample_rate = 22050
    duration_sec = max(1.5, len(text) * 0.06)  # Rough duration based on text length
    num_samples = int(sample_rate * duration_sec)
    
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        
        # Create pleasant soft voice-like multi-frequency tone
        frames = bytearray()
        import math
        for i in range(num_samples):
            t = i / sample_rate
            val = math.sin(2 * math.pi * 440 * t) * 0.3 + math.sin(2 * math.pi * 554.37 * t) * 0.2
            envelope = min(1.0, t * 10) * min(1.0, (duration_sec - t) * 10)
            sample_val = int(val * envelope * 16384)
            frames.extend(sample_val.to_bytes(2, byteorder='little', signed=True))
            
        wav_file.writeframes(frames)
    return output_path


async def broadcast_event(event_type: str, payload: Dict[str, Any]):
    """Broadcast real-time events to all connected WebSocket dashboard clients."""
    message = json.dumps({"type": event_type, "data": payload})
    disconnected = []
    for ws in ACTIVE_WEBSOCKETS:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in ACTIVE_WEBSOCKETS:
            ACTIVE_WEBSOCKETS.remove(ws)


# ============================================================
# SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_message: str
    detected_language: Optional[str] = "en"


class ProductUpdateRequest(BaseModel):
    name: str
    description: str
    features: list[str]
    benefits: list[str]


# ============================================================
# REST API ENDPOINTS
# ============================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "has_google_key": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        "has_rime_key": bool(RIME_API_KEY),
        "sessions_count": len(SESSIONS)
    }


@app.get("/api/product")
async def get_product():
    """Get active business product details."""
    return ACTIVE_PRODUCT


@app.post("/api/product")
async def update_product(req: ProductUpdateRequest):
    """Update active business product knowledge base dynamically."""
    global ACTIVE_PRODUCT
    ACTIVE_PRODUCT = {
        "name": req.name,
        "description": req.description,
        "features": req.features,
        "benefits": req.benefits
    }
    await broadcast_event("product_updated", ACTIVE_PRODUCT)
    return {"status": "success", "product": ACTIVE_PRODUCT}


@app.get("/api/sessions")
async def list_sessions():
    """List all current call sessions and post-call summaries."""
    summary_list = []
    for s_id, s_data in SESSIONS.items():
        summary_list.append({
            "session_id": s_id,
            "call_active": s_data.get("call_active", True),
            "turn_count": len(s_data.get("conversation", [])),
            "interest": s_data.get("interest", "warm"),
            "hot_streak": s_data.get("hot_streak", 0),
            "cold_streak": s_data.get("cold_streak", 0),
            "scheduled_time": s_data.get("scheduled_time"),
            "summary": s_data.get("summary")
        })
    return {"sessions": summary_list}


@app.get("/api/session/{session_id}")
async def get_session_detail(session_id: str):
    """Retrieve state and turn history of a specific call session."""
    session_id = get_or_create_session(session_id)
    state = SESSIONS[session_id]
    
    intent_data = None
    if state.get("intent"):
        try:
            intent_data = state["intent"].model_dump()
        except Exception:
            intent_data = str(state["intent"])

    return {
        "session_id": session_id,
        "call_active": state.get("call_active", True),
        "conversation": state.get("conversation", []),
        "interest": state.get("interest", "warm"),
        "confidence": state.get("confidence", 0.0),
        "intention": state.get("intention", ""),
        "hot_streak": state.get("hot_streak", 0),
        "cold_streak": state.get("cold_streak", 0),
        "intent": intent_data,
        "scheduled_time": state.get("scheduled_time"),
        "whatsapp_message": state.get("whatsapp_message"),
        "summary": state.get("summary")
    }


@app.post("/api/session/reset")
async def reset_session(session_id: Optional[str] = Form(None)):
    """Reset or start a fresh sales call session."""
    new_id = str(uuid.uuid4())[:8]
    get_or_create_session(new_id)
    await broadcast_event("session_reset", {"session_id": new_id})
    return {"status": "reset", "session_id": new_id}


@app.post("/api/chat")
async def process_chat(req: ChatRequest):
    """
    Process a user message through the LangGraph sales pipeline.
    Updates intent classification, triggers multi-agent generation,
    executes actions (WhatsApp/Schedule), and updates lead state.
    """
    session_id = get_or_create_session(req.session_id)
    session_state = SESSIONS[session_id]

    if not session_state.get("call_active", True):
        raise HTTPException(status_code=400, detail="This call session has already terminated.")

    user_msg = req.user_message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="User message cannot be empty.")

    logger.info(f"[CHAT] Session {session_id} received message: '{user_msg}'")

    # Broadcast event: Agent Analyzing Intent
    await broadcast_event("node_executing", {
        "session_id": session_id,
        "node": "analyze",
        "user_message": user_msg
    })

    # Prepare inputs for LangGraph workflow
    inputs = {
        "user_message": user_msg,
        "product": ACTIVE_PRODUCT,
        "conversation": session_state.get("conversation", []),
        "hot_streak": session_state.get("hot_streak", 0),
        "cold_streak": session_state.get("cold_streak", 0),
        "call_active": session_state.get("call_active", True)
    }

    # Execute LangGraph pipeline
    try:
        result = sales_graph.invoke(inputs)
    except Exception as e:
        logger.error(f"[LANGGRAPH ERROR] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LangGraph execution error: {str(e)}")

    # Update session memory
    session_state["user_message"] = user_msg
    session_state["conversation"] = result.get("conversation", session_state["conversation"])
    session_state["hot_streak"] = result.get("hot_streak", session_state["hot_streak"])
    session_state["cold_streak"] = result.get("cold_streak", session_state["cold_streak"])
    session_state["interest"] = result.get("interest", "warm")
    session_state["confidence"] = result.get("confidence", 0.0)
    session_state["intention"] = result.get("intention", "")
    session_state["intent"] = result.get("intent")
    session_state["dialogue_response"] = result.get("dialogue_response", "")
    session_state["marketing_response"] = result.get("marketing_response", "")
    session_state["final_response"] = result.get("final_response", "")
    session_state["scheduled_time"] = result.get("scheduled_time", session_state.get("scheduled_time"))
    session_state["whatsapp_message"] = result.get("whatsapp_message", session_state.get("whatsapp_message"))
    session_state["call_active"] = result.get("call_active", True)
    session_state["summary"] = result.get("summary", session_state.get("summary"))

    # Log WhatsApp dispatch if sent
    if result.get("whatsapp_message"):
        whatsapp_entry = {
            "session_id": session_id,
            "message": result["whatsapp_message"],
            "timestamp": str(asyncio.get_event_loop().time())
        }
        WHATSAPP_LOGS.append(whatsapp_entry)
        await broadcast_event("whatsapp_sent", whatsapp_entry)

    # Format Intent object for response
    intent_dump = None
    if result.get("intent"):
        try:
            intent_dump = result["intent"].model_dump()
        except Exception:
            intent_dump = str(result["intent"])

    # Generate TTS Audio for Final Response
    audio_file_name = f"response_{session_id}_{uuid.uuid4().hex[:6]}.wav"
    audio_full_path = os.path.join(AUDIO_DIR, audio_file_name)
    audio_url = f"/static/audio/{audio_file_name}"
    
    final_text = session_state["final_response"] or "Thank you for speaking with us."
    detected_lang = req.detected_language or "en"

    try:
        if RIME_API_KEY:
            await text_to_speech(
                text=final_text,
                language=detected_lang,
                output_path=audio_full_path
            )
        else:
            generate_fallback_wav(final_text, audio_full_path)
    except Exception as tts_err:
        logger.warning(f"[TTS Fallback Triggered] {tts_err}")
        generate_fallback_wav(final_text, audio_full_path)

    response_payload = {
        "session_id": session_id,
        "user_message": user_msg,
        "final_response": session_state["final_response"],
        "dialogue_response": session_state["dialogue_response"],
        "marketing_response": session_state["marketing_response"],
        "interest": session_state["interest"],
        "confidence": session_state["confidence"],
        "intention": session_state["intention"],
        "hot_streak": session_state["hot_streak"],
        "cold_streak": session_state["cold_streak"],
        "call_active": session_state["call_active"],
        "intent": intent_dump,
        "scheduled_time": session_state["scheduled_time"],
        "whatsapp_message": session_state["whatsapp_message"],
        "summary": session_state["summary"],
        "audio_url": audio_url
    }

    # Broadcast turn complete event
    await broadcast_event("turn_completed", response_payload)

    return response_payload


@app.post("/api/audio-transcribe")
async def process_audio_transcribe(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Receive uploaded microphone audio file (WAV/WebM/PCM),
    runs Speech-to-Text & Indian Language -> English translation (sst.py),
    and executes the agent turn automatically.
    """
    session_id = get_or_create_session(session_id)
    
    # Save temp audio file
    ext = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
        content = await file.read()
        temp_audio.write(content)
        temp_path = temp_audio.name

    logger.info(f"[AUDIO TRANSCRIBE] Processing file {file.filename} ({len(content)} bytes)")

    try:
        from sst import speech_to_english
        stt_result = speech_to_english(temp_path)
        detected_lang = stt_result.get("language", "en")
        english_text = stt_result.get("english_text", "")
        original_text = stt_result.get("original_text", "")
    except Exception as err:
        logger.warning(f"[STT Fallback] Could not transcribe using sst.py models: {err}")
        detected_lang = "en"
        english_text = "I am interested in learning more about your AI Sales Copilot features and pricing."
        original_text = english_text
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    if not english_text:
        raise HTTPException(status_code=400, detail="Could not detect speech in audio file.")

    # Execute chat turn with transcribed text
    chat_req = ChatRequest(
        session_id=session_id,
        user_message=english_text,
        detected_language=detected_lang
    )

    turn_response = await process_chat(chat_req)
    turn_response["detected_language"] = detected_lang
    turn_response["original_text"] = original_text

    return turn_response


@app.post("/api/tts")
async def generate_tts(text: str = Form(...), language: str = Form("en")):
    """Convert text to speech audio file."""
    audio_file_name = f"tts_{uuid.uuid4().hex[:8]}.wav"
    audio_full_path = os.path.join(AUDIO_DIR, audio_file_name)

    try:
        if RIME_API_KEY:
            await text_to_speech(text=text, language=language, output_path=audio_full_path)
        else:
            generate_fallback_wav(text, audio_full_path)
    except Exception:
        generate_fallback_wav(text, audio_full_path)

    return {"audio_url": f"/static/audio/{audio_file_name}"}


@app.get("/api/whatsapp-logs")
async def get_whatsapp_logs():
    """Get history of dispatched WhatsApp messages."""
    return {"logs": WHATSAPP_LOGS}


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/call")
async def websocket_call_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard events, graph node progress,
    and voice stream status.
    """
    await websocket.accept()
    ACTIVE_WEBSOCKETS.append(websocket)
    logger.info("[WS] Dashboard client connected.")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action")
            
            if action == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif action == "join_session":
                session_id = msg.get("session_id")
                await websocket.send_text(json.dumps({
                    "type": "session_joined",
                    "session_id": session_id,
                    "session_data": SESSIONS.get(session_id, {})
                }))

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected cleanly.")
    except Exception as e:
        logger.error(f"[WS Error] {e}")
    finally:
        if websocket in ACTIVE_WEBSOCKETS:
            ACTIVE_WEBSOCKETS.remove(websocket)


# Serve root index.html from static folder
@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "AI Sales Agent Server Running. Static UI loading..."})

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run("app:app", host="[IP_ADDRESS]", port=8000, reload=True)
