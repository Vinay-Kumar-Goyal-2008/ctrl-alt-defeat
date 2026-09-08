from typing import Literal, Optional
from pydantic import BaseModel, Field


class IntentAnalysis(BaseModel):
    interest: Literal["hot", "warm", "cold"] = Field(
        description="User's current level of purchase/product interest."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the interest classification."
    )

    intention: str = Field(
        description="Exact intention of the user."
    )

    schedule_requested: bool = Field(
        description="Whether the user wants to schedule a future call."
    )

    schedule_time: Optional[str] = Field(
        default=None,
        description="Explicit requested date/time if provided."
    )

    schedule_preference: Optional[str] = Field(
        default=None,
        description="Relative preference such as tomorrow morning, next week, evening, etc."
    )

    wants_product_details: bool = Field(
        default=False
    )

    wants_to_end_call: bool = Field(
        default=False
    )


class ScheduleResult(BaseModel):
    scheduled_time: str = Field(
        description="Final selected schedule time in ISO-like human readable format."
    )

    reason: str = Field(
        description="Why this time was selected."
    )


class PostCallSummary(BaseModel):
    summary: str

    user_intent: str

    interest_level: Literal["hot", "warm", "cold"]

    product_interest: str

    objections: list[str]

    important_details: list[str]

    recommended_follow_up: str