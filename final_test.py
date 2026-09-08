import sounddevice as sd
import soundfile as sf
import tempfile
import os

from sst import speech_to_english
from graph import build_graph


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


# ============================================================
# BUILD LANGGRAPH
# ============================================================

app = build_graph()


# ============================================================
# CONVERSATION STATE
#
# FIX: cold_streak is now tracked exactly like hot_streak,
# since the graph has no checkpointer -- all cross-turn state
# must be persisted here in the orchestrator and passed back
# into every app.invoke(...) call.
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
    print("  2. Convert speech to English")
    print("  3. Send the English text to the AI agent")
    print("  4. Generate the agent response")

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

        # FIX: this used to unconditionally print
        # "Call termination triggered." on EVERY cold turn,
        # regardless of the actual streak/call_active state.
        # Now it accurately reflects what the graph decided.
        print("\n[ACTION]")
        print(
            f"Cold interest detected "
            f"(cold streak: {cold_streak_value}/3)."
        )

        if not call_active_value:
            print("Call termination triggered.")
        else:
            print("Continuing conversation (streak below threshold).")


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
            "unknown"
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
        print(f"Detected language : {language}")
        print(f"Original speech   : {original_text}")
        print(f"English           : {english_text}")
        print("-" * 50)

        return english_text

    except Exception as e:

        print("\n[SPEECH ERROR]")
        print(e)

        return ""

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# RUN LANGGRAPH AGENT
# ============================================================

def run_agent(user_message):

    global conversation
    global hot_streak
    global cold_streak
    global call_active

    print("\nThinking...")

    # --------------------------------------------------------
    # Invoke LangGraph
    #
    # FIX: do NOT manually append the user message to
    # `conversation` here. `analyze_intent` in graph.py already
    # appends it via the `conversation` reducer -- appending it
    # here too caused every user turn to be duplicated in
    # history, and it also fed the (not-yet-recorded) current
    # message into the "Previous conversation" text.
    #
    # FIX: cold_streak is now passed in just like hot_streak so
    # the 3-strike cold logic can actually accumulate turn to
    # turn.
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
    # FIX: sync the local conversation list from the graph's
    # returned state instead of hand-appending to it. The graph
    # is the single source of truth for what got added (user
    # turn + whichever assistant turn the active branch
    # produced), so this is the only place messages should be
    # added.
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
    # Display agent response
    #
    # FIX: `result["dialogue_response"]`/`final_response` are
    # plain strings (from `response.content` in graph.py), not
    # dicts -- `response['text']` was indexing a string with a
    # string key, which is exactly what raised
    # "string indices must be integers, not 'str'".
    #
    # FIX: prefer `final_response` (the judged/authoritative
    # answer for the normal dialogue+marketing path) and fall
    # back to `dialogue_response` for branches (cold) that only
    # set that field.
    # --------------------------------------------------------

    response = result.get("final_response") or result.get("dialogue_response")

    if response:
        print("\nAgent:")
        print(response)

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
            # =================================================

            english_text = speech_to_english_from_microphone()

            # -------------------------------------------------
            # No speech detected
            # -------------------------------------------------

            if not english_text.strip():

                print("\nNo speech detected. Try again.")

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
            # =================================================

            run_agent(english_text)

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