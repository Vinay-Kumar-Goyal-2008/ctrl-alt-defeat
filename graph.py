from typing import TypedDict, Annotated
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from schemas import (
    IntentAnalysis,
    PostCallSummary
)
import streamlit as st
from prompts import (
    INTENT_PROMPT,
    BASE_DIALOGUE_PROMPT,
    WARM_DIALOGUE_PROMPT,
    NORMAL_DIALOGUE_PROMPT,
    HOT_WHATSAPP_PROMPT,
    SCHEDULE_WHATSAPP_PROMPT,
    SUMMARY_PROMPT
)

from tools import send_whatsapp_message
from scheduler import SchedulerAgent

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from llm_conveyer import marketing_agent


# ============================================================
# STATE
# ============================================================

class ConversationState(TypedDict, total=False):

    user_message: str
    product: dict

    conversation: Annotated[
        list[dict],
        operator.add
    ]

    intent: IntentAnalysis

    interest: str
    confidence: float
    intention: str

    hot_streak: int
    cold_streak: int

    dialogue_response: str
    marketing_response: str
    final_response: str

    whatsapp_message: str
    scheduled_time: str

    call_active: bool
    summary: dict

# ============================================================
# MODELS
# ============================================================

llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.4,
        google_api_key=GOOGLE_API_KEY
    )


# ============================================================
# STRUCTURED LLMs
# ============================================================

intent_llm = llm.with_structured_output(
    IntentAnalysis
)

summary_llm = llm.with_structured_output(
    PostCallSummary
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = SchedulerAgent()


# ============================================================
# FINAL RESPONSE JUDGE SCHEMA
# ============================================================

class FinalResponse(BaseModel):

    response: str = Field(
        description=(
            "The final customer-facing response. "
            "It must directly answer the customer's latest message "
            "and sound natural and conversational."
        )
    )


judge_llm = llm.with_structured_output(
    FinalResponse
)


# ============================================================
# NODE 1
# INTENT ANALYSIS
# ============================================================

def analyze_intent(state: ConversationState):

    conversation = state.get("conversation", [])
    user_message = state["user_message"]

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation
    )

    prompt = f"""
{INTENT_PROMPT}

Previous conversation:

{history_text}

Latest user message:

{user_message}
"""

    result = intent_llm.invoke(prompt)

    # --------------------------------------------------------
    # HOT STREAK
    # --------------------------------------------------------

    previous_hot_streak = state.get("hot_streak", 0)

    if result.interest == "hot":
        new_hot_streak = previous_hot_streak + 1
    else:
        new_hot_streak = 0

    # --------------------------------------------------------
    # COLD STREAK
    # --------------------------------------------------------

    previous_cold_streak = state.get("cold_streak", 0)

    if result.interest == "cold":
        new_cold_streak = previous_cold_streak + 1
    else:
        # Any non-cold response breaks the consecutive cold streak
        new_cold_streak = 0

    return {
        "intent": result,
        "interest": result.interest,
        "confidence": result.confidence,
        "intention": result.intention,
        "hot_streak": new_hot_streak,
        "cold_streak": new_cold_streak,
        # FIX: record the user's turn into history so subsequent
        # nodes (and the *next* invocation) see it. Without this,
        # `conversation` never grows on its own.
        "conversation": [
            {"role": "user", "content": user_message}
        ]
    }

# ============================================================
# NODE 2
# EXISTING DIALOGUE GENERATION
# ============================================================

def generate_dialogue(state: ConversationState):

    conversation = state.get(
        "conversation",
        []
    )

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation
    )

    interest = state["interest"]

    if interest == "warm":
        prompt_template = WARM_DIALOGUE_PROMPT
    else:
        prompt_template = NORMAL_DIALOGUE_PROMPT

    prompt = prompt_template.format(
        product=state["product"],
        conversation=history_text,
        user_message=state["user_message"],
        interest=interest
    )

    response = llm.invoke(prompt)

    return {
        "dialogue_response": response.content
    }


# ============================================================
# NODE 3
# MARKETING AGENT
#
# ONLY USED FOR NORMAL/WARM DIALOGUE
# ============================================================

def generate_marketing_response(
    state: ConversationState
):

    conversation = state.get(
        "conversation",
        []
    )

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation
    )

    result = marketing_agent(
        query=state["user_message"],
        history_text=history_text
    )

    return {
        "marketing_response": result
    }


# ============================================================
# NODE 4
# RESPONSE JUDGE
#
# Compares:
#   A = Existing dialogue LLM
#   B = Marketing agent
#
# Then returns the best final response.
# ============================================================

def compare_responses(
    state: ConversationState
):

    conversation = state.get(
        "conversation",
        []
    )

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are the final conversation-response judge.

You have two candidate responses generated for the
same customer conversation.

BUSINESS / PRODUCT KNOWLEDGE:

{product}

CONVERSATION HISTORY:

{conversation}

CUSTOMER'S LATEST MESSAGE:

{user_message}

CANDIDATE A — EXISTING DIALOGUE AGENT:

{dialogue_response}

CANDIDATE B — MARKETING AGENT:

{marketing_response}


Evaluate both candidates carefully.

Choose or synthesize the best customer-facing response based on:

1. Relevance to the customer's latest message
2. Natural conversational tone
3. Understanding of the customer's intent
4. Business/product accuracy
5. Marketing effectiveness
6. Ability to move the conversation forward
7. Lack of unnecessary sales pressure
8. Conciseness
9. No unsupported claims
10. No repetition of information already known


IMPORTANT:

- Return ONLY the response that should be sent to the customer.
- Do not explain your decision.
- Do not mention Candidate A or Candidate B.
- Do not mention the judging process.
- Do not mention AI.
- Do not mention agents.
- Do not expose internal reasoning.
- Do not add unnecessary sales language.
- Preserve factual accuracy from the business knowledge.
"""
        ),
        (
            "human",
            "Produce the final customer-facing response."
        )
    ])

    chain = prompt | judge_llm

    result = chain.invoke({
        "product": state["product"],
        "conversation": history_text,
        "user_message": state["user_message"],
        "dialogue_response": state["dialogue_response"],
        "marketing_response": state["marketing_response"]
    })

    return {
        "final_response": result.response,
        # FIX: record the assistant's turn into history
        "conversation": [
            {"role": "assistant", "content": result.response}
        ]
    }


# ============================================================
# NODE 5
# SCHEDULER
# ============================================================

def schedule_call(state: ConversationState):

    intent = state["intent"]

    result = scheduler.schedule(
        requested_time=intent.schedule_time,
        preference=intent.schedule_preference,
        user_intention=intent.intention
    )

    return {
        "scheduled_time": result.scheduled_time
    }


# ============================================================
# NODE 6
# WHATSAPP PRODUCT MESSAGE
# ============================================================

def send_hot_whatsapp(state: ConversationState):

    conversation = state.get(
        "conversation",
        []
    )

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation
    )

    prompt = HOT_WHATSAPP_PROMPT.format(
        product=state["product"],
        intention=state["intention"],
        conversation=history_text
    )

    response = llm.invoke(prompt)

    send_whatsapp_message(
        response.content
    )

    return {
        "whatsapp_message": response.content,
        # FIX: this path used to leave final_response unset entirely.
        "final_response": response.content,
        "conversation": [
            {"role": "assistant", "content": response.content}
        ]
    }


# ============================================================
# NODE 7
# WHATSAPP SCHEDULE CONFIRMATION
# ============================================================

def send_schedule_whatsapp(
    state: ConversationState
):

    prompt = SCHEDULE_WHATSAPP_PROMPT.format(
        scheduled_time=state["scheduled_time"],
        reason="Based on the user's scheduling preference.",
        product=state["product"]
    )

    response = llm.invoke(prompt)

    send_whatsapp_message(
        response.content
    )

    return {
        "whatsapp_message": response.content,
        # FIX: this path used to leave final_response unset entirely.
        "final_response": response.content,
        "conversation": [
            {"role": "assistant", "content": response.content}
        ]
    }


# ============================================================
# NODE 8
# COLD RESPONSE
#
# FIX (main bug):
# Previously, `route_after_analysis` checked
# `intent.wants_to_end_call` BEFORE checking
# `intent.interest == "cold"`. Since the intent LLM very
# often flags `wants_to_end_call = True` on the SAME turn
# it classifies a lead as "cold" (the two are naturally
# correlated), the graph was routing straight to the "end"
# node and skipping this cold-streak handler entirely -
# terminating on the very first cold message instead of
# waiting for 3 consecutive cold turns.
#
# Fix: `cold` interest ALWAYS routes here first. This node
# is now the single source of truth for when to actually
# say goodbye. An explicit `wants_to_end_call` flag on a
# cold turn is honored as an immediate override (a user who
# is both cold AND explicitly asking to stop shouldn't need
# to wait for 3 strikes), but it no longer silently bypasses
# the streak counter for ordinary cold replies.
# ============================================================

def handle_cold(state: ConversationState):

    cold_streak = state.get("cold_streak", 0)
    intent = state.get("intent")

    explicit_end_request = bool(
        intent and getattr(intent, "wants_to_end_call", False)
    )

    # ========================================================
    # COLD #1 OR COLD #2 (and no explicit "end call" request)
    # ========================================================

    if cold_streak < 3 and not explicit_end_request:

        conversation = state.get(
            "conversation",
            []
        )

        history_text = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in conversation
        )

        marketing_response = marketing_agent(
            query=state["user_message"],
            history_text=history_text
        )

        return {
            "marketing_response": marketing_response,
            "dialogue_response": marketing_response,
            "final_response": marketing_response,
            "call_active": True,
            "conversation": [
                {"role": "assistant", "content": marketing_response}
            ]
        }

    # ========================================================
    # COLD #3 (3 consecutive cold turns) OR explicit end request
    # ========================================================

    goodbye_response = (
        "I completely understand. Thank you for your time. "
        "Have a great day."
    )

    return {
        "dialogue_response": goodbye_response,
        "final_response": goodbye_response,
        "call_active": False,
        "conversation": [
            {"role": "assistant", "content": goodbye_response}
        ]
    }


# ============================================================
# NODE 9
# END CALL
# ============================================================

def end_call(state: ConversationState):

    return {
        "call_active": False
    }


# ============================================================
# NODE 10
# POST CALL SUMMARY
# ============================================================

def generate_postcall_summary(
    state: ConversationState
):

    conversation = state.get(
        "conversation",
        []
    )

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation
    )

    prompt = SUMMARY_PROMPT.format(
        product=state["product"],
        conversation=history_text
    )

    result = summary_llm.invoke(prompt)

    summary_text = f"""
POST-CALL SUMMARY

Summary:
{result.summary}

User Intent:
{result.user_intent}

Interest:
{result.interest_level}

Product Interest:
{result.product_interest}

Objections:
{", ".join(result.objections) if result.objections else "None"}

Important Details:
{", ".join(result.important_details) if result.important_details else "None"}

Recommended Follow-up:
{result.recommended_follow_up}
"""

    send_whatsapp_message(
        summary_text
    )

    return {
        "summary": result.model_dump()
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_analysis(state: ConversationState):

    intent = state["intent"]

    # Schedule takes priority
    if intent.schedule_requested:
        return "schedule"

    # FIX: cold intent is now checked BEFORE the generic
    # wants_to_end_call check, so cold turns always go through
    # the 3-strike streak handler (`handle_cold`), which is
    # now also responsible for honoring an explicit end request.
    if intent.interest == "cold":
        return "cold"

    # Explicit "end call" request on a non-cold turn
    # (e.g. warm/hot lead who still asks to stop)
    if intent.wants_to_end_call:
        return "end"

    # Two consecutive HOT classifications
    if (
        intent.interest == "hot"
        and state.get("hot_streak", 0) >= 2
    ):
        return "hot"

    # Warm or first HOT
    return "dialogue"

# ============================================================
# BUILD GRAPH
# ============================================================

def conversation_start(state: ConversationState):
    """
    Fan-out node.

    It does not generate anything.
    It simply allows the graph to branch into:
        1. Existing dialogue agent
        2. Marketing agent
    """
    return {}

def route_after_cold(state: ConversationState):

    if state.get("call_active", True):
        return "continue"

    return "terminate"

def build_graph():

    graph = StateGraph(
        ConversationState
    )

    # ========================================================
    # NODES
    # ========================================================

    graph.add_node(
        "analyze",
        analyze_intent
    )

    # Fan-out node
    graph.add_node(
        "conversation",
        conversation_start
    )

    # Existing dialogue LLM
    graph.add_node(
        "dialogue",
        generate_dialogue
    )

    # Marketing LLM
    graph.add_node(
        "marketing",
        generate_marketing_response
    )

    # Final judge
    graph.add_node(
        "judge",
        compare_responses
    )

    # Other existing nodes
    graph.add_node(
        "schedule",
        schedule_call
    )

    graph.add_node(
        "schedule_whatsapp",
        send_schedule_whatsapp
    )

    graph.add_node(
        "hot_whatsapp",
        send_hot_whatsapp
    )

    graph.add_node(
        "cold",
        handle_cold
    )

    graph.add_node(
        "end_call",
        end_call
    )

    graph.add_node(
        "postcall_summary",
        generate_postcall_summary
    )

    # ========================================================
    # START
    # ========================================================

    graph.add_edge(
        START,
        "analyze"
    )

    # ========================================================
    # INTENT ROUTING
    # ========================================================

    graph.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            # IMPORTANT:
            # Normal/warm conversation goes to fan-out
            "dialogue": "conversation",

            # Existing paths remain unchanged
            "hot": "hot_whatsapp",
            "cold": "cold",
            "schedule": "schedule",
            "end": "end_call"
        }
    )

    # ========================================================
    # FAN-OUT
    #
    # Both nodes receive the SAME state.
    #
    # They execute independently.
    # ========================================================

    graph.add_edge(
        "conversation",
        "dialogue"
    )

    graph.add_edge(
        "conversation",
        "marketing"
    )

    # ========================================================
    # FAN-IN
    #
    # Both branches converge into judge.
    #
    # The judge should run only after both branches complete.
    # ========================================================

    graph.add_edge(
        "dialogue",
        "judge"
    )

    graph.add_edge(
        "marketing",
        "judge"
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    graph.add_edge(
        "judge",
        END
    )

    # ========================================================
    # HOT
    #
    # analyze -> hot_whatsapp -> END
    # ========================================================

    graph.add_edge(
        "hot_whatsapp",
        END
    )

    # ========================================================
    # COLD
    #
    # analyze -> cold -> summary -> END
    # ========================================================

    graph.add_conditional_edges(
        "cold",
        route_after_cold,
        {
            "continue": END,
            "terminate": "postcall_summary"
        }
    )

    # ========================================================
    # SCHEDULE
    #
    # analyze -> schedule -> whatsapp -> end -> summary -> END
    # ========================================================

    graph.add_edge(
        "schedule",
        "schedule_whatsapp"
    )

    graph.add_edge(
        "schedule_whatsapp",
        "end_call"
    )

    graph.add_edge(
        "end_call",
        "postcall_summary"
    )

    graph.add_edge(
        "postcall_summary",
        END
    )

    return graph.compile()