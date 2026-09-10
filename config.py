"""
Configuration module for loading environment variables.
"""

import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

# Rime STT Configuration
RIME_STT_WS_URL = os.getenv("RIME_STT_WS_URL", "")
RIME_API_KEY = os.getenv("RIME_API_KEY", "")

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")
CUSTOMER_WHATSAPP = os.getenv("CUSTOMER_WHATSAPP", "")

# Google AI Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY or GEMINI_API_KEY not found in environment."
    )
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
