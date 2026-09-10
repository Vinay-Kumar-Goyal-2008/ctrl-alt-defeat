import asyncio
# pyrefly: ignore [missing-import]
import sounddevice as sd
# pyrefly: ignore [missing-import]
import soundfile as sf
import tempfile
import os

from sst import speech_to_english
from graph import build_graph
from tts_handler import text_to_speech


# ============================================================
# PRODUCT
# ============================================================

PRODUCT = {
    "name": "AI Sales Copilot",
    "description": (
        "An AI-powered sales assistant that analyzes conversations, "
        "identifies customer intent, automatically follows up through "
        "WhatsApp, schedules calls, and generates post-call summaries."
    ),
    "features": [
        "Real-time intent detection",
        "Customer interest scoring",
        "Automated WhatsApp follow-up",
        "Call scheduling",
        "Post-call summaries"
    ],
    "benefits": [
        "Higher sales conversion",
        "Faster follow-up",
        "Reduced manual work",
        "Better customer qualification"
    ]
}


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds

# TTS output configuration
TTS_OUTPUT_DIR = "audio"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

app = build_graph()


# ============================================================
# CONVERSATION STATE
#
# cold_streak is tracked exactly like hot_streak because
# the graph has no checkpointer.
# ============================================================

conversation = []
hot_streak = 0
cold_streak = 0
call_active = True


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header():

    print("\n" + "=" * 70)
    print("             AI AGENTIC SALES ASSISTANT")
    print("=" * 70)

    print("Speak naturally into your microphone.")

    print("The system will:")
    print("  1. Record your voice")
    print("  2. Detect the spoken language")
    print("  3. Convert speech to English")
    print("  4. Send the English text to the AI agent")
    print("  5. Generate the agent response")
    print("  6. Convert the agent response back to speech")

    print("\nSpeak 'exit', 'quit', or 'bye' to end the call.")

    print("=" * 70 + "\n")


def print_intent(result):

    intent = result.get("intent")

    if not intent:
        return

    print("\n[INTENT ANALYSIS]")
    print("-" * 50)

    print(f"Interest       : {intent.interest}")
    print(f"Confidence     : {intent.confidence:.2f}")
    print(f"Intention      : {intent.intention}")
    print(f"Schedule       : {intent.schedule_requested}")

    if intent.schedule_time:
        print(f"Schedule time  : {intent.schedule_time}")

    if intent.schedule_preference:
        print(f"Schedule pref. : {intent.schedule_preference}")

    print(f"Product info   : {intent.wants_product_details}")
    print(f"End call       : {intent.wants_to_end_call}")

    print(
        f"Hot streak     : "
        f"{result.get('hot_streak', 0)}"
    )

    print(
        f"Cold streak    : "
        f"{result.get('cold_streak', 0)}"
    )


def print_action(result):

    intent = result.get("intent")

    if not intent:
        return

    hot_streak_value = result.get(
        "hot_streak",
        0
    )

    cold_streak_value = result.get(
        "cold_streak",
        0
    )

    call_active_value = result.get(
        "call_active",
        True
    )

    if intent.schedule_requested:

        print("\n[ACTION]")
        print("Scheduler Agent invoked.")

        scheduled_time = result.get(
            "scheduled_time"
        )

        if scheduled_time:

            print(
                f"Call scheduled for: "
                f"{scheduled_time}"
            )

    elif (
        intent.interest == "hot"
        and hot_streak_value >= 2
    ):

        print("\n[ACTION]")

        print(
            "HOT CONFIDENCE GATE PASSED "
            "(2 consecutive HOT classifications)."
        )

        print(
            "WhatsApp product follow-up triggered."
        )

    elif intent.interest == "warm":

        print("\n[ACTION]")
        print("Warm interest detected.")
        print("Using detailed dialogue prompt.")

    elif intent.interest == "cold":

        print("\n[ACTION]")

        print(
            f"Cold interest detected "
            f"(cold streak: {cold_streak_value}/3)."
        )

        if not call_active_value:
            print("Call termination triggered.")

        else:
            print(
                "Continuing conversation "
                "(streak below threshold)."
            )


def print_whatsapp(result):

    message = result.get(
        "whatsapp_message"
    )

    if not message:
        return

    print("\n[WHATSAPP]")
    print("-" * 50)

    print(message)

    print("-" * 50)


def print_summary(result):

    summary = result.get(
        "summary"
    )

    if not summary:
        return

    print("\n[POST-CALL SUMMARY]")
    print("-" * 50)

    print(
        f"Summary:\n"
        f"{summary.get('summary', '')}\n"
    )

    print(
        f"User intent:\n"
        f"{summary.get('user_intent', '')}\n"
    )

    print(
        f"Interest:\n"
        f"{summary.get('interest_level', '')}\n"
    )

    print(
        f"Product interest:\n"
        f"{summary.get('product_interest', '')}\n"
    )

    objections = summary.get(
        "objections",
        []
    )

    print(
        f"Objections:\n"
        f"{', '.join(objections) if objections else 'None'}\n"
    )

    details = summary.get(
        "important_details",
        []
    )

    print(
        f"Important details:\n"
        f"{', '.join(details) if details else 'None'}\n"
    )

    print(
        f"Recommended follow-up:\n"
        f"{summary.get('recommended_follow_up', '')}"
    )

    print("-" * 50)


# ============================================================
# RECORD AUDIO
# ============================================================

def record_audio():

    print("\nListening...")
    print("Speak now.")

    audio = sd.rec(
        int(CHUNK_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    return audio


# ============================================================
# SPEECH → ENGLISH
#
# IMPORTANT:
# This now returns BOTH:
#   1. English text
#   2. Detected language
# ============================================================

def speech_to_english_from_microphone():

    audio = record_audio()

    print("Processing speech...")

    temp_path = None

    try:

        # ----------------------------------------------------
        # Create temporary WAV file
        # ----------------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        temp_path = temp_file.name

        temp_file.close()

        # ----------------------------------------------------
        # Save microphone recording
        # ----------------------------------------------------

        sf.write(
            temp_path,
            audio,
            SAMPLE_RATE
        )

        # ----------------------------------------------------
        # Existing speech translation pipeline
        # ----------------------------------------------------

        result = speech_to_english(temp_path)

        language = result.get(
            "language",
            "en"
        )

        original_text = result.get(
            "original_text",
            ""
        )

        english_text = result.get(
            "english_text",
            ""
        )

        print("\n[SPEECH]")
        print("-" * 50)

        print(
            f"Detected language : {language}"
        )

        print(
            f"Original speech   : {original_text}"
        )

        print(
            f"English           : {english_text}"
        )

        print("-" * 50)

        # Return BOTH values
        return english_text, language

    except Exception as e:

        print("\n[SPEECH ERROR]")
        print(e)

        return "", "en"

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# TEXT → SPEECH
# ============================================================
#
# This function takes:
#
#   final agent response
#             +
#   detected language
#
# and sends them to your Rime TTS function.
#
# ============================================================

def generate_agent_audio(
    response: str,
    language: str
):

    if not response:
        print("\n[TTS] No response to synthesize.")
        return None

    try:

        # ----------------------------------------------------
        # Create audio directory
        # ----------------------------------------------------

        os.makedirs(
            TTS_OUTPUT_DIR,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Generate a unique filename
        # ----------------------------------------------------

        output_path = os.path.join(
            TTS_OUTPUT_DIR,
            "agent_response.wav"
        )

        print("\n[TTS]")
        print("-" * 50)

        print(
            f"Language : {language}"
        )

        print(
            f"Text     : {response}"
        )

        print("Generating audio...")

        # ----------------------------------------------------
        # Call your async Rime TTS function
        # ----------------------------------------------------

        audio_path = asyncio.run(
            text_to_speech(
                text=response,
                language=language,
                output_path=output_path
            )
        )

        print(
            f"Audio saved: {audio_path}"
        )

        print("-" * 50)

        return audio_path

    except Exception as e:

        print("\n[TTS ERROR]")
        print(e)

        return None


# ============================================================
# RUN LANGGRAPH AGENT
# ============================================================

def run_agent(
    user_message,
    detected_language
):

    global conversation
    global hot_streak
    global cold_streak
    global call_active

    print("\nThinking...")

    # --------------------------------------------------------
    # Invoke LangGraph
    # --------------------------------------------------------

    try:

        result = app.invoke({

            "user_message": user_message,

            "product": PRODUCT,

            "conversation": conversation,

            "hot_streak": hot_streak,

            "cold_streak": cold_streak,

            "call_active": True
        })

    except Exception as e:

        print("\n[AGENT ERROR]")
        print(e)

        return

    # --------------------------------------------------------
    # Update streak state
    # --------------------------------------------------------

    hot_streak = result.get(
        "hot_streak",
        hot_streak
    )

    cold_streak = result.get(
        "cold_streak",
        cold_streak
    )

    # --------------------------------------------------------
    # Sync conversation
    # --------------------------------------------------------

    conversation = result.get(
        "conversation",
        conversation
    )

    # --------------------------------------------------------
    # Display intent
    # --------------------------------------------------------

    print_intent(result)

    # --------------------------------------------------------
    # Display actions
    # --------------------------------------------------------

    print_action(result)

    # --------------------------------------------------------
    # Get final agent response
    #
    # Prefer final_response.
    # Fall back to dialogue_response.
    # --------------------------------------------------------

    response = (
        result.get("final_response")
        or result.get("dialogue_response")
    )

    if response:

        print("\nAgent:")
        print(response)

        # ====================================================
        # NEW:
        # Convert the agent's response to speech
        # using the ORIGINAL detected language.
        # ====================================================

        generate_agent_audio(
            response=response,
            language=detected_language
        )

    # --------------------------------------------------------
    # Display WhatsApp
    # --------------------------------------------------------

    print_whatsapp(result)

    # --------------------------------------------------------
    # Display schedule
    # --------------------------------------------------------

    scheduled_time = result.get(
        "scheduled_time"
    )

    if scheduled_time:

        print("\n[SCHEDULE]")

        print(
            f"Scheduled time: "
            f"{scheduled_time}"
        )

    # --------------------------------------------------------
    # Update call status
    # --------------------------------------------------------

    call_active = result.get(
        "call_active",
        True
    )

    # --------------------------------------------------------
    # Post-call summary
    # --------------------------------------------------------

    if not call_active:

        print("\nCall has ended.")

        print_summary(result)

        print("\nThank you.")


# ============================================================
# MAIN VOICE LOOP
# ============================================================

def main():

    global call_active

    print_header()

    while call_active:

        try:

            # =================================================
            # STEP 1
            # Record + translate speech
            #
            # Now receives:
            #
            # english_text
            # detected_language
            # =================================================

            english_text, detected_language = (
                speech_to_english_from_microphone()
            )

            # -------------------------------------------------
            # No speech detected
            # -------------------------------------------------

            if not english_text.strip():

                print(
                    "\nNo speech detected. "
                    "Try again."
                )

                continue

            # =================================================
            # STEP 2
            # Check manual exit
            # =================================================

            if english_text.lower().strip() in {

                "exit",
                "quit",
                "bye",
                "end the call",
                "end call"

            }:

                print("\nEnding call...")

                call_active = False

                break

            # =================================================
            # STEP 3
            # Send English text to LangGraph
            #
            # Also pass detected language so that the
            # agent response can be converted back to the
            # user's original language.
            # =================================================

            run_agent(
                user_message=english_text,
                detected_language=detected_language
            )

            # =================================================
            # STEP 4
            # Check whether LangGraph ended the call
            # =================================================

            if not call_active:
                break

        except KeyboardInterrupt:

            print("\n\nCall interrupted.")

            break

        except Exception as e:

            print("\n[ERROR]")
            print(e)

            continue


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
