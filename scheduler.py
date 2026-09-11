from datetime import datetime, timedelta
import random

from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import ScheduleResult

import streamlit as st
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
class SchedulerAgent:

    def __init__(self):

        # Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.4,
        google_api_key=GOOGLE_API_KEY
    )

        # Structured output using your Pydantic schema
        self.structured_llm = self.llm.with_structured_output(
            ScheduleResult
        )

    def schedule(
        self,
        requested_time: str | None,
        preference: str | None,
        user_intention: str
    ):

        # ------------------------------------------------
        # If the user explicitly gave a time, preserve it.
        # ------------------------------------------------

        if requested_time:

            return ScheduleResult(
                scheduled_time=requested_time,
                reason="User explicitly provided the requested schedule."
            )

        # ------------------------------------------------
        # If they provided a preference, ask Gemini to
        # interpret the preference.
        # ------------------------------------------------

        if preference:

            now = datetime.now()

            prompt = f"""
You are a scheduling assistant.

Current time:
{now}

User scheduling preference:
{preference}

User intention:
{user_intention}

Choose a reasonable future call time that satisfies the preference.

Rules:
- Never choose a time in the past.
- If the user says morning, choose 09:00-12:00.
- If the user says afternoon, choose 12:00-17:00.
- If the user says evening, choose 17:00-21:00.
- Prefer business hours unless the user explicitly asks otherwise.
- Return the scheduled time in YYYY-MM-DD HH:MM format.
"""

            result = self.structured_llm.invoke(prompt)

            return result

        # ------------------------------------------------
        # No explicit time.
        #
        # Select a reasonable default rather than asking
        # the user again.
        # ------------------------------------------------

        now = datetime.now()

        candidate_days = [
            1,
            2,
            3
        ]

        day_offset = random.choice(candidate_days)

        scheduled = now + timedelta(days=day_offset)

        scheduled = scheduled.replace(
            hour=11,
            minute=0,
            second=0,
            microsecond=0
        )

        return ScheduleResult(
            scheduled_time=scheduled.strftime(
                "%Y-%m-%d %H:%M"
            ),
            reason=(
                "No specific time was provided, so a "
                "reasonable future business-hour slot was selected."
            )
        )