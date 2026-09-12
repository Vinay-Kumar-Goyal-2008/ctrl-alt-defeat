import asyncio
print("1")

import os
print("2")

import tempfile
print("3")

import time
print("4")

import streamlit as st
print("5")

import soundfile as sf
print("6")

from sst import  speech_to_english
print("7")

from graph import build_graph
print("8")

from tts_handler import text_to_speech
print("9")

from productknowledge import business as PRODUCT
print("10")
# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
CHUNK_DURATION = 5
TTS_OUTPUT_DIR = "audio"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Sales Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — MODERN DARK THEME WITH NEON ACCENTS
# ============================================================

st.html(
    """
    <style>

    /* ========================================================
       GLOBAL — DARK THEME BASE
       ======================================================== */

    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-tertiary: #1a2234;
        --bg-elevated: #1f2937;
        --border-subtle: rgba(148, 163, 184, 0.12);
        --border-strong: rgba(148, 163, 184, 0.25);

        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;

        --neon-cyan: #22d3ee;
        --neon-purple: #a78bfa;
        --neon-lime: #a3e635;
        --neon-pink: #f472b6;
        --neon-amber: #fbbf24;
        --neon-red: #f87171;

        --glow-cyan: 0 0 20px rgba(34, 211, 238, 0.35);
        --glow-purple: 0 0 20px rgba(167, 139, 250, 0.35);
        --glow-lime: 0 0 20px rgba(163, 230, 53, 0.35);
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(167, 139, 250, 0.10), transparent 60%),
            radial-gradient(1000px 500px at 110% 10%, rgba(34, 211, 238, 0.08), transparent 60%),
            linear-gradient(180deg, #0a0e1a 0%, #0b1120 100%);
        color: var(--text-primary);
    }

    /* Streamlit main block padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1280px;
    }

    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--text-primary);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        letter-spacing: -0.01em;
    }

    p, span, label, li, div {
        color: var(--text-primary);
    }

    /* ========================================================
       HERO HEADER
       ======================================================== */

    .hero {
        position: relative;
        padding: 28px 32px;
        border-radius: 20px;
        margin-bottom: 24px;
        background:
            linear-gradient(135deg, rgba(34, 211, 238, 0.08), rgba(167, 139, 250, 0.10)),
            #0f172a;
        border: 1px solid var(--border-strong);
        overflow: hidden;
    }

    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(600px 200px at 0% 0%, rgba(34, 211, 238, 0.18), transparent 60%),
            radial-gradient(500px 200px at 100% 100%, rgba(167, 139, 250, 0.18), transparent 60%);
        pointer-events: none;
    }

    .hero-title {
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--neon-lime);
        border: 1px solid rgba(163, 230, 53, 0.35);
        border-radius: 999px;
        background: rgba(163, 230, 53, 0.08);
    }

    .hero-subtitle {
        color: var(--text-secondary);
        margin-top: 8px;
        font-size: 0.98rem;
        position: relative;
        z-index: 1;
    }

    .hero-live {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        color: var(--neon-cyan);
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--neon-cyan);
        box-shadow: var(--glow-cyan);
        animation: livePulse 1.4s ease-in-out infinite;
    }

    @keyframes livePulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.5; transform: scale(1.25); }
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #0a0e1a 100%);
        border-right: 1px solid var(--border-subtle);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-primary);
    }

    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 16px 0 10px 0;
    }

    /* Sidebar metric cards */
    .sb-metric {
        padding: 14px;
        border-radius: 14px;
        border: 1px solid var(--border-subtle);
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
        text-align: center;
    }

    .sb-metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sb-metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 4px;
        line-height: 1;
    }

    .sb-metric.hot .sb-metric-value  { color: var(--neon-red); text-shadow: 0 0 20px rgba(248,113,113,0.35); }
    .sb-metric.cold .sb-metric-value { color: var(--neon-cyan); text-shadow: var(--glow-cyan); }

    /* Sidebar pipeline chips */
    .pipe-chip {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 12px;
        border-radius: 10px;
        border: 1px solid var(--border-subtle);
        background: rgba(255,255,255,0.02);
        margin-bottom: 6px;
        font-size: 0.86rem;
        color: var(--text-secondary);
    }

    .pipe-chip .idx {
        width: 22px;
        height: 22px;
        border-radius: 6px;
        background: rgba(167, 139, 250, 0.15);
        color: var(--neon-purple);
        font-weight: 700;
        font-size: 0.72rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* ========================================================
       STREAMLIT BUTTONS
       ======================================================== */

    .stButton > button {
        background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
        color: #0a0e1a !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.02em;
        padding: 10px 18px;
        transition: transform 0.15s ease, box-shadow 0.2s ease, filter 0.2s ease;
        box-shadow: 0 6px 22px rgba(34, 211, 238, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
        box-shadow: 0 10px 30px rgba(167, 139, 250, 0.35);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Secondary style for reset */
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-strong);
        box-shadow: none;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--neon-purple);
        color: var(--neon-purple) !important;
        box-shadow: 0 0 18px rgba(167, 139, 250, 0.2);
    }

    /* ========================================================
       AUDIO INPUT AREA
       ======================================================== */

    div[data-testid="stAudioInput"] {
        background: linear-gradient(180deg, #101827, #0d1424);
        border: 1px dashed rgba(34, 211, 238, 0.35) !important;
        border-radius: 16px;
        padding: 14px;
    }

    /* ========================================================
       ALERTS / STATUS BOXES
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid var(--border-strong) !important;
        background: rgba(255,255,255,0.02) !important;
        color: var(--text-primary) !important;
    }

    /* ========================================================
       TABS — CLEAN, COLLAPSIBLE FEEL
       ======================================================== */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid var(--border-subtle);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary) !important;
        padding: 10px 16px;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255,255,255,0.03);
    }

    .stTabs [aria-selected="true"] {
        color: var(--neon-cyan) !important;
        background: rgba(34, 211, 238, 0.06) !important;
        border-bottom: 2px solid var(--neon-cyan) !important;
    }

    /* ========================================================
       EXPANDERS
       ======================================================== */

    details, .streamlit-expanderHeader, div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
    }

    /* ========================================================
       STEP CARDS
       ======================================================== */

    .step-card {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid var(--border-subtle);
        background: rgba(255,255,255,0.02);
        margin-bottom: 10px;
        transition: all 0.25s ease;
    }

    .step-card .step-num {
        width: 34px;
        height: 34px;
        min-width: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        background: rgba(148,163,184,0.10);
        color: var(--text-secondary);
        border: 1px solid var(--border-subtle);
    }

    .step-card .step-title {
        font-weight: 700;
        font-size: 0.98rem;
        color: var(--text-primary);
    }

    .step-card .step-details {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 3px;
    }

    .step-card.active {
        border-color: rgba(34, 211, 238, 0.45);
        background: linear-gradient(180deg, rgba(34, 211, 238, 0.06), rgba(34, 211, 238, 0.02));
        box-shadow: var(--glow-cyan);
    }
    .step-card.active .step-num {
        background: rgba(34, 211, 238, 0.15);
        color: var(--neon-cyan);
        border-color: rgba(34, 211, 238, 0.35);
    }

    .step-card.complete {
        border-color: rgba(163, 230, 53, 0.35);
        background: linear-gradient(180deg, rgba(163, 230, 53, 0.05), rgba(163, 230, 53, 0.02));
    }
    .step-card.complete .step-num {
        background: rgba(163, 230, 53, 0.15);
        color: var(--neon-lime);
        border-color: rgba(163, 230, 53, 0.35);
    }

    .step-card.error {
        border-color: rgba(248, 113, 113, 0.45);
        background: linear-gradient(180deg, rgba(248, 113, 113, 0.06), rgba(248, 113, 113, 0.02));
    }
    .step-card.error .step-num {
        background: rgba(248, 113, 113, 0.15);
        color: var(--neon-red);
        border-color: rgba(248, 113, 113, 0.35);
    }

    /* ========================================================
       PROCESSING PANEL
       ======================================================== */

    .processing-container {
        border: 1px solid var(--border-strong);
        border-radius: 18px;
        padding: 22px;
        margin: 8px 0 18px 0;
        background:
            linear-gradient(135deg, rgba(34, 211, 238, 0.05), rgba(167, 139, 250, 0.06)),
            #0f172a;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
    }

    .processing-header {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .loader {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        border: 3px solid rgba(255,255,255,0.08);
        border-top: 3px solid var(--neon-cyan);
        border-right: 3px solid var(--neon-purple);
        animation: spin 0.9s linear infinite;
        flex-shrink: 0;
        box-shadow: var(--glow-cyan);
    }

    @keyframes spin {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .processing-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
        color: var(--text-primary);
    }

    .processing-subtitle {
        color: var(--text-secondary);
        font-size: 0.88rem;
        margin-top: 3px;
    }

    .processing-bar {
        width: 100%;
        height: 8px;
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 18px;
    }

    .processing-bar-inner {
        height: 100%;
        min-width: 6%;
        border-radius: 10px;
        background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple), var(--neon-cyan));
        background-size: 200% 100%;
        animation: progress-animation 1.4s linear infinite;
        box-shadow: 0 0 14px rgba(34, 211, 238, 0.5);
    }

    @keyframes progress-animation {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    .processing-stage {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 14px;
        font-size: 0.92rem;
        color: var(--text-primary);
    }

    .pulse-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--neon-lime);
        box-shadow: var(--glow-lime);
        animation: pulse 1s infinite;
    }

    @keyframes pulse {
        0%   { transform: scale(0.8); opacity: 0.5; }
        50%  { transform: scale(1.25); opacity: 1;   }
        100% { transform: scale(0.8); opacity: 0.5; }
    }

    .elapsed-time {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 8px;
        font-variant-numeric: tabular-nums;
    }

    .processing-complete {
        border: 1px solid rgba(163, 230, 53, 0.4);
        border-radius: 18px;
        padding: 18px 22px;
        margin: 8px 0 18px 0;
        background:
            linear-gradient(135deg, rgba(163, 230, 53, 0.08), rgba(34, 211, 238, 0.05)),
            #0f172a;
    }

    .complete-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--neon-lime);
    }

    .complete-subtitle {
        color: var(--text-secondary);
        font-size: 0.88rem;
        margin-top: 4px;
    }

    /* ========================================================
       CHIP / METRIC PILL
       ======================================================== */

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 8px 0 4px 0;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        border: 1px solid var(--border-subtle);
        background: rgba(255,255,255,0.03);
        font-size: 0.86rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .chip .k { color: var(--text-muted); font-weight: 500; }
    .chip.hot   { border-color: rgba(248,113,113,0.4); color: var(--neon-red); }
    .chip.warm  { border-color: rgba(251,191,36,0.4);  color: var(--neon-amber); }
    .chip.cold  { border-color: rgba(34,211,238,0.4);  color: var(--neon-cyan); }
    .chip.info  { border-color: rgba(167,139,250,0.4); color: var(--neon-purple); }
    .chip.ok    { border-color: rgba(163,230,53,0.4);  color: var(--neon-lime); }

    /* ========================================================
       CARD
       ======================================================== */

    .card {
        border-radius: 16px;
        border: 1px solid var(--border-subtle);
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0));
        padding: 18px;
    }

    .card-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .card-content {
        color: var(--text-primary);
        font-size: 0.98rem;
        line-height: 1.55;
    }

    /* ========================================================
       CHAT MESSAGE OVERRIDE
       ======================================================== */

    div[data-testid="stChatMessage"] {
        background: linear-gradient(180deg, rgba(167,139,250,0.06), rgba(34,211,238,0.04)) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 16px !important;
        padding: 14px !important;
    }

    /* ========================================================
       CODE BLOCKS
       ======================================================== */

    code, pre {
        background: #0b1220 !important;
        color: var(--neon-cyan) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
    }

    /* ========================================================
       EMPTY STATE
       ======================================================== */

    .empty-state {
        border: 1px dashed var(--border-strong);
        border-radius: 20px;
        padding: 40px 30px;
        text-align: center;
        background:
            radial-gradient(600px 200px at 50% 0%, rgba(34,211,238,0.06), transparent 60%),
            rgba(255,255,255,0.02);
    }

    .empty-title {
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .empty-sub {
        color: var(--text-secondary);
        margin-bottom: 22px;
    }

    .empty-steps {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin-top: 18px;
    }

    .empty-step {
        padding: 14px;
        border-radius: 14px;
        border: 1px solid var(--border-subtle);
        background: rgba(255,255,255,0.02);
        text-align: left;
    }

    .empty-step .n {
        display: inline-block;
        width: 26px;
        height: 26px;
        border-radius: 8px;
        background: rgba(167,139,250,0.15);
        color: var(--neon-purple);
        text-align: center;
        line-height: 26px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 8px;
    }

    .empty-step .t {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--text-primary);
    }

    .empty-step .d {
        color: var(--text-secondary);
        font-size: 0.82rem;
        margin-top: 3px;
    }

    /* ========================================================
       SECTION HEADING
       ======================================================== */

    .section-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 8px 0 14px 0;
    }

    .section-heading::before {
        content: "";
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, var(--neon-cyan), var(--neon-purple));
        border-radius: 4px;
    }

    /* Streamlit's default success/warning/info alerts colored text fix */
    .stAlert p { color: var(--text-primary) !important; }

    /* Hide streamlit branding (subtle) */
    #MainMenu, footer { visibility: hidden; }

    </style>
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "hot_streak" not in st.session_state:
    st.session_state.hot_streak = 0

if "cold_streak" not in st.session_state:
    st.session_state.cold_streak = 0

if "call_active" not in st.session_state:
    st.session_state.call_active = True

if "app" not in st.session_state:
    st.session_state.app = build_graph()


# ============================================================
# HERO HEADER
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="hero-title">
            <span>🤖 AI Agentic Sales Assistant</span>
            <span class="hero-badge">v2 · Neon</span>
        </div>
        <div class="hero-subtitle">
            Voice-first multilingual AI sales conversation system —
            Speech Recognition · Translation · LangGraph Agent · Rime TTS.
        </div>
        <div class="hero-live">
            <span class="live-dot"></span>
            <span>Realtime pipeline online</span>
        </div>
    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html('<div class="sidebar-section-title">Conversation state</div>')

    col1, col2 = st.columns(2)

    with col1:
        st.html(
            f"""
            <div class="sb-metric hot">
                <div class="sb-metric-label">🔥 Hot</div>
                <div class="sb-metric-value">{st.session_state.hot_streak}</div>
            </div>
            """
        )

    with col2:
        st.html(
            f"""
            <div class="sb-metric cold">
                <div class="sb-metric-label">❄️ Cold</div>
                <div class="sb-metric-value">{st.session_state.cold_streak}</div>
            </div>
            """
        )

    st.html('<div class="sidebar-section-title">Pipeline flow</div>')

    st.html(
        """
        <div>
            <div class="pipe-chip"><span class="idx">1</span><span>🎙️ Voice Input</span></div>
            <div class="pipe-chip"><span class="idx">2</span><span>📝 Speech Recognition</span></div>
            <div class="pipe-chip"><span class="idx">3</span><span>🌐 Language Translation</span></div>
            <div class="pipe-chip"><span class="idx">4</span><span>🧠 LangGraph Agent</span></div>
            <div class="pipe-chip"><span class="idx">5</span><span>🔊 Rime TTS</span></div>
            <div class="pipe-chip"><span class="idx">6</span><span>▶️ Audio Response</span></div>
        </div>
        """
    )

    st.html('<div class="sidebar-section-title">Session</div>')

    call_state_color = "ok" if st.session_state.call_active else "warm"
    call_state_text = "Active" if st.session_state.call_active else "Ended"
    st.html(
        f"""
        <div class="chip-row">
            <span class="chip {call_state_color}">
                <span class="k">Call:</span> {call_state_text}
            </span>
            <span class="chip info">
                <span class="k">Turns:</span> {len(st.session_state.conversation)}
            </span>
        </div>
        """
    )

    if st.button(
        "🔄 Reset Conversation",
        use_container_width=True,
    ):
        st.session_state.conversation = []
        st.session_state.hot_streak = 0
        st.session_state.cold_streak = 0
        st.session_state.call_active = True
        st.rerun()


# ============================================================
# STEP DISPLAY
# ============================================================

def show_step(container, number, title, status="waiting", details=""):

    status_class = {
        "waiting": "",
        "active": "active",
        "complete": "complete",
        "error": "error",
    }.get(status, "")

    status_icon = {
        "waiting": "•",
        "active": "▶",
        "complete": "✓",
        "error": "✕",
    }.get(status, "•")

    html = f"""
    <div class="step-card {status_class}">
        <div class="step-num">{status_icon}</div>
        <div>
            <div class="step-title">Step {number} · {title}</div>
            {f'<div class="step-details">{details}</div>' if details else ''}
        </div>
    </div>
    """
    with container.container():
        st.html(html)


# ============================================================
# DYNAMIC PROCESSING LOADER
# ============================================================

def show_processing(container, stage, description, progress, elapsed=None):

    if elapsed is None:
        elapsed_text = "Processing request..."
    else:
        elapsed_text = f"⏱ Elapsed: {elapsed:.1f}s"

    html = f"""
    <div class="processing-container">
        <div class="processing-header">
            <div class="loader"></div>
            <div>
                <div class="processing-title">AI Pipeline Processing</div>
                <div class="processing-subtitle">{description}</div>
            </div>
        </div>
        <div class="processing-bar">
            <div class="processing-bar-inner" style="width: {progress}%"></div>
        </div>
        <div class="processing-stage">
            <div class="pulse-dot"></div>
            <b>{stage}</b>
        </div>
        <div class="elapsed-time">{elapsed_text}</div>
    </div>
    """
    with container.container():
        st.html(html)


def show_processing_complete(container, elapsed):
    html = f"""
    <div class="processing-complete">
        <div class="complete-title">✅ Pipeline Completed</div>
        <div class="complete-subtitle">
            Response generated successfully · {elapsed:.1f}s
        </div>
    </div>
    """
    with container.container():
        st.html(html)


# ============================================================
# NORMALIZE AUDIO TO EXACTLY 5 SECONDS
# ============================================================

def normalize_audio_to_5_seconds(input_path, output_path):
    """
    Converts input audio into mono, 16 kHz, exactly 5 seconds.
    Longer recordings are trimmed. Shorter ones padded with silence.
    """
    audio, sample_rate = sf.read(input_path, dtype="float32")

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        import numpy as np
        old_length = len(audio)
        new_length = int(old_length * SAMPLE_RATE / sample_rate)
        old_indices = np.linspace(0, old_length - 1, old_length)
        new_indices = np.linspace(0, old_length - 1, new_length)
        audio = np.interp(new_indices, old_indices, audio).astype("float32")

    target_samples = SAMPLE_RATE * CHUNK_DURATION

    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        import numpy as np
        silence = np.zeros(target_samples - len(audio), dtype="float32")
        audio = np.concatenate([audio, silence])

    sf.write(output_path, audio, SAMPLE_RATE)


# ============================================================
# SPEECH -> ENGLISH
# ============================================================

def process_speech(audio_path):
    result = speech_to_english(audio_path)
    language = result.get("language", "en")
    original_text = result.get("original_text", "")
    english_text = result.get("english_text", "")
    return english_text, language, original_text


# ============================================================
# RUN LANGGRAPH
# ============================================================

def run_agent(user_message):
    app = st.session_state.app
    result = app.invoke(
        {
            "user_message": user_message,
            "product": PRODUCT,
            "conversation": st.session_state.conversation,
            "hot_streak": st.session_state.hot_streak,
            "cold_streak": st.session_state.cold_streak,
            "call_active": st.session_state.call_active,
        }
    )

    st.session_state.hot_streak = result.get("hot_streak", st.session_state.hot_streak)
    st.session_state.cold_streak = result.get("cold_streak", st.session_state.cold_streak)
    st.session_state.conversation = result.get("conversation", st.session_state.conversation)
    st.session_state.call_active = result.get("call_active", True)

    return result


# ============================================================
# TTS
# ============================================================

def generate_agent_audio(response, language):
    if not response:
        return None

    os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(TTS_OUTPUT_DIR, "agent_response.wav")

    audio_path = asyncio.run(
        text_to_speech(
            text=response,
            language=language,
            output_path=output_path,
        )
    )
    return audio_path


# ============================================================
# MAIN UI — VOICE CAPTURE
# ============================================================

st.html('<div class="section-heading">🎙️ Voice conversation</div>')

st.html(
    """
    <div class="card" style="margin-bottom: 14px;">
        <div class="card-content">
            Record your message. The system processes exactly
            <b style="color: var(--neon-cyan)">5 seconds</b> of audio and sends the
            English text to the AI sales agent.
        </div>
    </div>
    """
)

audio_value = st.audio_input("Record your voice")


# ============================================================
# PROCESS FLOW
# ============================================================

if audio_value is not None:

    st.html(
        """
        <div class="chip-row">
            <span class="chip ok">✓ Audio captured</span>
            <span class="chip info">Ready to process</span>
        </div>
        """
    )

    st.audio(audio_value, format="audio/wav")

    process_button = st.button(
        "🚀 Process Audio",
        type="primary",
        use_container_width=True,
    )

    if process_button:

        pipeline_start = time.perf_counter()

        # ================================================
        # LIVE STATUS + TABS
        # ================================================

        processing_loader = st.empty()
        show_processing(
            processing_loader,
            "Initializing AI pipeline...",
            "Preparing your voice input",
            5,
            0,
        )

        tab_pipeline, tab_transcript, tab_intent, tab_response, tab_actions = st.tabs(
            [
                "⚙️  Pipeline",
                "📝  Transcript",
                "🧠  Intent",
                "🤖  Response",
                "📌  Actions",
            ]
        )

        with tab_pipeline:
            st.html('<div class="section-heading">Live processing steps</div>')
            step1 = st.empty()
            step2 = st.empty()
            step3 = st.empty()
            step4 = st.empty()
            step5 = st.empty()
            step6 = st.empty()

        temp_input = None
        temp_5sec = None

        try:

            # ================================================
            # STEP 1 — AUDIO
            # ================================================

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 1/6 — Processing Audio",
                "Normalizing microphone recording to 16 kHz mono audio",
                10, elapsed,
            )
            show_step(step1, 1, "Audio Capture", "active", "Receiving microphone recording...")

            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_input = temp_file.name
            temp_file.write(audio_value.getbuffer())
            temp_file.close()

            temp_5sec = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            normalize_audio_to_5_seconds(temp_input, temp_5sec)

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 1/6 — Audio Ready",
                "Audio successfully normalized",
                16, elapsed,
            )
            show_step(step1, 1, "Audio Capture", "complete", "5-second audio prepared at 16 kHz mono.")

            # ================================================
            # STEP 2 — SPEECH RECOGNITION
            # ================================================

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 2/6 — Speech Recognition",
                "AI is converting your voice into text",
                30, elapsed,
            )
            show_step(step2, 2, "Speech Recognition", "active", "Converting audio into text...")

            english_text, language, original_text = process_speech(temp_5sec)

            if not english_text.strip():
                show_step(step2, 2, "Speech Recognition", "error", "No speech detected.")
                processing_loader.empty()
                st.warning("No speech was detected. Please try again.")
                st.stop()

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 2/6 — Speech Recognized",
                "Voice successfully converted into text",
                35, elapsed,
            )
            show_step(step2, 2, "Speech Recognition", "complete", f"Detected speech: {original_text}")

            # ================================================
            # STEP 3 — LANGUAGE
            # ================================================

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 3/6 — Language Detection",
                "Detecting language and preparing English input",
                45, elapsed,
            )
            show_step(step3, 3, "Language Detection & Translation", "active", "Converting user speech to English...")

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 3/6 — Translation Complete",
                f"Detected language: {language}",
                50, elapsed,
            )
            show_step(
                step3, 3, "Language Detection & Translation", "complete",
                f"Language: {language}",
            )

            # ================================================
            # TRANSCRIPT TAB
            # ================================================

            with tab_transcript:
                st.html('<div class="section-heading">📝 Transcription</div>')
                col1, col2 = st.columns(2)
                with col1:
                    st.html(
                        f"""
                        <div class="card">
                            <div class="card-label">Original Speech</div>
                            <div class="card-content">{original_text or '<i style="color: var(--text-muted)">—</i>'}</div>
                        </div>
                        """
                    )
                with col2:
                    st.html(
                        f"""
                        <div class="card">
                            <div class="card-label">English</div>
                            <div class="card-content">{english_text}</div>
                        </div>
                        """
                    )
                st.html(
                    f"""
                    <div class="chip-row" style="margin-top: 12px;">
                        <span class="chip info"><span class="k">Language:</span> {language}</span>
                        <span class="chip ok"><span class="k">Length:</span> 5.0s</span>
                    </div>
                    """
                )

            # ================================================
            # MANUAL EXIT
            # ================================================

            if english_text.lower().strip() in {"exit", "quit", "bye", "end the call", "end call"}:
                st.session_state.call_active = False
                processing_loader.empty()
                st.warning("Call ended by user.")
                st.stop()

            # ================================================
            # STEP 4 — LANGGRAPH
            # ================================================

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 4/6 — LangGraph Agent",
                "Analyzing intent, conversation state and sales strategy",
                60, elapsed,
            )
            show_step(step4, 4, "LangGraph Agent", "active", "Analyzing intent and generating response...")

            result = run_agent(english_text)
            response = result.get("final_response") or result.get("dialogue_response")

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 4/6 — Agent Decision Complete",
                "Intent analyzed and response generated",
                72, elapsed,
            )
            show_step(step4, 4, "LangGraph Agent", "complete", "Intent analysis and response generation completed.")

            # ================================================
            # INTENT TAB
            # ================================================

            intent = result.get("intent")

            with tab_intent:
                st.html('<div class="section-heading">🧠 Intent intelligence</div>')

                if intent:
                    interest_class = {
                        "hot": "hot",
                        "warm": "warm",
                        "cold": "cold",
                    }.get(intent.interest, "info")

                    st.html(
                        f"""
                        <div class="chip-row">
                            <span class="chip {interest_class}">
                                <span class="k">Interest:</span> {intent.interest}
                            </span>
                            <span class="chip info">
                                <span class="k">Confidence:</span> {intent.confidence:.2f}
                            </span>
                            <span class="chip info">
                                <span class="k">Intention:</span> {intent.intention}
                            </span>
                            <span class="chip {'ok' if intent.schedule_requested else ''}">
                                <span class="k">Schedule:</span> {'Yes' if intent.schedule_requested else 'No'}
                            </span>
                            <span class="chip hot">
                                <span class="k">🔥 Hot streak:</span> {result.get('hot_streak', 0)}
                            </span>
                            <span class="chip cold">
                                <span class="k">❄️ Cold streak:</span> {result.get('cold_streak', 0)}
                            </span>
                        </div>
                        """
                    )

                    with st.expander("View detailed intent"):
                        st.write(f"**Interest:** {intent.interest}")
                        st.write(f"**Confidence:** {intent.confidence:.2f}")
                        st.write(f"**Intention:** {intent.intention}")
                        st.write(f"**Schedule requested:** {intent.schedule_requested}")
                        st.write(f"**Product details:** {intent.wants_product_details}")
                        st.write(f"**End call:** {intent.wants_to_end_call}")
                        st.write(f"**Hot streak:** {result.get('hot_streak', 0)}")
                        st.write(f"**Cold streak:** {result.get('cold_streak', 0)}")
                else:
                    st.info("No intent data available.")

            # ================================================
            # STEP 5 — TTS
            # ================================================

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 5/6 — Text-to-Speech",
                f"Generating natural voice response in {language}",
                82, elapsed,
            )
            show_step(step5, 5, "Text-to-Speech", "active", f"Generating response audio in {language}...")

            audio_path = generate_agent_audio(response=response, language=language)

            if audio_path:
                elapsed = time.perf_counter() - pipeline_start
                show_processing(
                    processing_loader,
                    "Step 5/6 — Voice Generated",
                    "Agent response converted into audio",
                    90, elapsed,
                )
                show_step(step5, 5, "Text-to-Speech", "complete", f"Audio generated in {language}.")
            else:
                show_step(step5, 5, "Text-to-Speech", "error", "TTS returned no audio.")

            # ================================================
            # STEP 6 — PLAYBACK
            # ================================================

            elapsed = time.perf_counter() - pipeline_start
            show_processing(
                processing_loader,
                "Step 6/6 — Finalizing Response",
                "Preparing agent voice for playback",
                96, elapsed,
            )

            # ================================================
            # RESPONSE TAB
            # ================================================

            with tab_response:
                st.html('<div class="section-heading">🤖 Agent response</div>')
                if response:
                    st.chat_message("assistant").write(response)
                else:
                    st.info("No response generated.")

                if audio_path and os.path.exists(audio_path):
                    show_step(step6, 6, "Audio Playback", "active", "Agent audio is ready.")
                    st.html('<div class="section-heading" style="margin-top: 18px;">🔊 Agent voice</div>')
                    st.audio(audio_path, format="audio/wav")
                    show_step(step6, 6, "Audio Playback", "complete", "Agent response ready for playback.")

            # ================================================
            # PROCESSING COMPLETE
            # ================================================

            total_elapsed = time.perf_counter() - pipeline_start
            show_processing_complete(processing_loader, total_elapsed)

            # ================================================
            # ACTIONS TAB
            # ================================================

            with tab_actions:
                st.html('<div class="section-heading">📌 Next actions</div>')

                if intent:
                    if intent.schedule_requested:
                        st.success("📅 Scheduler Agent invoked.")
                        scheduled_time = result.get("scheduled_time")
                        if scheduled_time:
                            st.write(f"Scheduled time: **{scheduled_time}**")

                    elif intent.interest == "hot" and result.get("hot_streak", 0) >= 2:
                        st.success("🔥 HOT CONFIDENCE GATE PASSED — WhatsApp product follow-up triggered.")

                    elif intent.interest == "warm":
                        st.info("🟡 Warm interest detected — detailed dialogue strategy selected.")

                    elif intent.interest == "cold":
                        st.warning(f"❄️ Cold interest detected (streak: {result.get('cold_streak', 0)}/3)")

                whatsapp_message = result.get("whatsapp_message")
                if whatsapp_message:
                    st.html('<div class="section-heading" style="margin-top: 14px;">📱 WhatsApp follow-up</div>')
                    st.code(whatsapp_message)

                scheduled_time = result.get("scheduled_time")
                if scheduled_time:
                    st.html('<div class="section-heading" style="margin-top: 14px;">📅 Schedule</div>')
                    st.success(f"Call scheduled for **{scheduled_time}**")

                # Post-call summary
                if not st.session_state.call_active:
                    st.html('<div class="section-heading" style="margin-top: 14px;">📋 Post-call summary</div>')
                    summary = result.get("summary")
                    if summary:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.html(
                                f"""
                                <div class="card">
                                    <div class="card-label">Summary</div>
                                    <div class="card-content">{summary.get("summary", "—")}</div>
                                </div>
                                """
                            )
                            st.html(
                                f"""
                                <div class="card" style="margin-top: 10px;">
                                    <div class="card-label">User Intent</div>
                                    <div class="card-content">{summary.get("user_intent", "—")}</div>
                                </div>
                                """
                            )
                            st.html(
                                f"""
                                <div class="card" style="margin-top: 10px;">
                                    <div class="card-label">Interest Level</div>
                                    <div class="card-content">{summary.get("interest_level", "—")}</div>
                                </div>
                                """
                            )
                        with col2:
                            st.html(
                                f"""
                                <div class="card">
                                    <div class="card-label">Product Interest</div>
                                    <div class="card-content">{summary.get("product_interest", "—")}</div>
                                </div>
                                """
                            )
                            st.html(
                                f"""
                                <div class="card" style="margin-top: 10px;">
                                    <div class="card-label">Recommended Follow-up</div>
                                    <div class="card-content">{summary.get("recommended_follow_up", "—")}</div>
                                </div>
                                """
                            )

                        objections = summary.get("objections", [])
                        st.html('<div class="section-heading" style="margin-top: 14px;">Objections</div>')
                        if objections:
                            for obj in objections:
                                st.write(f"- {obj}")
                        else:
                            st.caption("None")

                        details = summary.get("important_details", [])
                        st.html('<div class="section-heading" style="margin-top: 6px;">Important details</div>')
                        if details:
                            for d in details:
                                st.write(f"- {d}")
                        else:
                            st.caption("None")

                    st.warning("The conversation has ended.")

        # ====================================================
        # ERROR HANDLING
        # ====================================================

        except Exception as e:
            processing_loader.empty()
            st.error(f"Pipeline error: {str(e)}")
            st.exception(e)

        # ====================================================
        # CLEANUP
        # ====================================================

        finally:
            if temp_input and os.path.exists(temp_input):
                os.remove(temp_input)
            if temp_5sec and os.path.exists(temp_5sec):
                os.remove(temp_5sec)


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.html(
        """
        <div class="empty-state">
            <div class="empty-title">🎙️ Ready for a conversation</div>
            <div class="empty-sub">
                Record a short voice message and watch the full AI sales pipeline run in realtime.
            </div>
            <div class="empty-steps">
                <div class="empty-step">
                    <div class="n">1</div>
                    <div class="t">Click Record</div>
                    <div class="d">Grant microphone access if prompted.</div>
                </div>
                <div class="empty-step">
                    <div class="n">2</div>
                    <div class="t">Speak ~5 seconds</div>
                    <div class="d">Any supported language works.</div>
                </div>
                <div class="empty-step">
                    <div class="n">3</div>
                    <div class="t">Stop recording</div>
                    <div class="d">Preview the captured audio.</div>
                </div>
                <div class="empty-step">
                    <div class="n">4</div>
                    <div class="t">Process Audio</div>
                    <div class="d">Runs the 6-step AI pipeline.</div>
                </div>
                <div class="empty-step">
                    <div class="n">5</div>
                    <div class="t">Review results</div>
                    <div class="d">Transcript · Intent · Response.</div>
                </div>
            </div>
        </div>
        """
    )
