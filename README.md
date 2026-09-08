# 🤖 AI Sales Conversation Agent

> **An agentic, stateful customer-conversation system built with LangGraph, LangChain, and Google Gemini — capable of understanding customer intent, adapting conversation strategy, triggering business actions, and generating post-call intelligence.**

<p align="center">

**🧠 Intent Analysis** • **💬 Multi-Agent Dialogue** • **🎯 Lead Qualification** • **📅 Scheduling** • **📲 WhatsApp Automation** • **📊 Post-Call Intelligence**

</p>

---

## 🚀 Overview

This project implements an **agentic customer conversation workflow** using **LangGraph**.

Instead of relying on a single LLM to handle everything, the system uses multiple specialized agents and decision nodes:

```text
                         👤 CUSTOMER
                             │
                             ▼
                  ┌─────────────────────┐
                  │   🧠 INTENT AGENT   │
                  │      Gemini LLM      │
                  └──────────┬──────────┘
                             │
                    ┌────────┼────────┐
                    │        │        │
                    ▼        ▼        ▼
                  🔥 HOT   🌡️ WARM   ❄️ COLD
                    │        │        │
                    │        │        ▼
                    │        │   Cold Streak
                    │        │        │
                    │        │        ▼
                    │        │    👋 Goodbye
                    │        │
                    │        ▼
                    │   ┌─────────────┐
                    │   │   FAN-OUT   │
                    │   └──────┬──────┘
                    │          │
                    │    ┌─────┴─────┐
                    │    ▼           ▼
                    │  💬 Dialogue  📣 Marketing
                    │    │           │
                    │    └─────┬─────┘
                    │          ▼
                    │     ⚖️ RESPONSE
                    │        JUDGE
                    │          │
                    │          ▼
                    │     💬 FINAL RESPONSE
                    │
                    ▼
              📲 WhatsApp
              
                    📅 SCHEDULE
                         │
                         ▼
                    📲 WhatsApp
                         │
                         ▼
                    📝 SUMMARY
```

---

# ✨ Key Features

### 🧠 Intelligent Intent Analysis

Every customer message is analyzed using **Gemini structured output** to extract:

* 🎯 Customer intention
* 🔥 Interest level
* 📊 Confidence
* 📅 Scheduling request
* ⏰ Preferred scheduling time
* 🛑 End-call intent

---

### 🤝 Multi-Agent Response Generation

For normal and warm conversations, two independent agents generate responses:

```text
                 Customer Message
                        │
                ┌───────┴───────┐
                ▼               ▼
        💬 Dialogue Agent   📣 Marketing Agent
                │               │
                └───────┬───────┘
                        ▼
                 ⚖️ Response Judge
                        │
                        ▼
                 🏆 Best Response
```

The system does **not blindly use the first generated response**.

Instead, a dedicated judge evaluates both candidates based on:

* Relevance
* Intent understanding
* Product accuracy
* Naturalness
* Marketing effectiveness
* Conciseness
* Lack of sales pressure
* Unsupported claims
* Repetition

---

# 🔥 Lead Intelligence

The system maintains **temporal lead signals** instead of making important decisions from a single classification.

## 🔥 Hot Lead Detection

Two consecutive hot classifications trigger the hot-lead workflow.

```text
Turn 1        Turn 2
  🔥            🔥
  │             │
  └─────────────┘
         │
         ▼
   🔥 HOT STREAK ≥ 2
         │
         ▼
    📲 WhatsApp
```

This reduces false positives caused by a single enthusiastic message.

---

## ❄️ Cold Lead Detection

Cold leads are tracked using a consecutive cold streak.

```text
❄️ Cold #1
    │
    ▼
❄️ Cold #2
    │
    ▼
❄️ Cold #3
    │
    ▼
👋 End Conversation
    │
    ▼
📝 Post-Call Summary
```

Any non-cold response resets the cold streak.

An explicit request to end the conversation can also terminate the interaction immediately.

---

# 📅 Intelligent Scheduling

If the customer requests a callback, scheduling receives priority over normal dialogue.

```text
👤 Customer
     │
     ▼
🧠 Intent Analysis
     │
     ▼
📅 Schedule Requested
     │
     ▼
🤖 Scheduler Agent
     │
     ▼
⏰ Resolve Schedule
     │
     ▼
📲 WhatsApp Confirmation
     │
     ▼
🛑 End Call
     │
     ▼
📝 Post-Call Summary
```

The scheduler receives:

```python
scheduler.schedule(
    requested_time=intent.schedule_time,
    preference=intent.schedule_preference,
    user_intention=intent.intention
)
```

---

# 📲 WhatsApp Automation

The system can automatically trigger WhatsApp messages for important business events.

### 🔥 Hot Lead

```text
Hot Streak ≥ 2
      │
      ▼
Generate Personalized Message
      │
      ▼
📲 Send WhatsApp
```

### 📅 Scheduled Callback

```text
Schedule Detected
      │
      ▼
Scheduler Agent
      │
      ▼
Generate Confirmation
      │
      ▼
📲 Send WhatsApp
```

### 📝 Post-Call Summary

After a terminated interaction, the system generates a structured summary and sends it through WhatsApp.

---

# 📝 Post-Call Intelligence

The system generates structured post-call information using Gemini.

The summary contains:

| Field                    | Description                    |
| ------------------------ | ------------------------------ |
| 📝 Summary               | Overall conversation           |
| 🎯 User Intent           | Customer's objective           |
| 🌡️ Interest Level       | Hot / Warm / Cold              |
| 🛍️ Product Interest     | Product-specific interest      |
| ⚠️ Objections            | Customer concerns              |
| 📌 Important Details     | Relevant extracted information |
| 🔄 Recommended Follow-up | Suggested next action          |

Example:

```text
POST-CALL SUMMARY

Summary:
Customer is interested but needs additional information.

User Intent:
Evaluate the product

Interest:
Warm

Product Interest:
High

Objections:
Pricing uncertainty

Important Details:
Requested a callback

Recommended Follow-up:
Discuss pricing and implementation timeline.
```

---

# 🏗️ Architecture

The project follows a **state-driven agentic architecture**.

```text
                         ┌───────────────┐
                         │     START     │
                         └───────┬───────┘
                                 │
                                 ▼
                      ┌────────────────────┐
                      │  🧠 INTENT ANALYZER │
                      └─────────┬──────────┘
                                │
                 ┌──────────────┼───────────────┐
                 │              │               │
                 ▼              ▼               ▼
             📅 SCHEDULE      ❄️ COLD         🛑 END
                 │              │               │
                 ▼              ▼               ▼
             SCHEDULER      COLD HANDLER    END CALL
                 │              │               │
                 ▼              │               │
           WHATSAPP             │               │
                 │              │               │
                 └──────────────┼───────────────┘
                                │
                                ▼
                         📝 POST-CALL
                           SUMMARY
                                │
                                ▼
                               END


                      NORMAL / WARM PATH
                                │
                                ▼
                       ┌─────────────────┐
                       │   🔀 FAN-OUT    │
                       └────────┬────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
              💬 Dialogue Agent      📣 Marketing Agent
                     │                     │
                     └──────────┬──────────┘
                                ▼
                       ⚖️ RESPONSE JUDGE
                                │
                                ▼
                       💬 FINAL RESPONSE
                                │
                                ▼
                               END
```

---

# 🧩 LangGraph Workflow

The workflow is implemented using `StateGraph`.

```python
graph = StateGraph(ConversationState)
```

The main nodes are:

| Node                   | Responsibility                     |
| ---------------------- | ---------------------------------- |
| 🧠 `analyze`           | Intent and interest analysis       |
| 🔀 `conversation`      | Fan-out entry point                |
| 💬 `dialogue`          | Normal conversational response     |
| 📣 `marketing`         | Marketing-oriented response        |
| ⚖️ `judge`             | Selects/synthesizes final response |
| 📅 `schedule`          | Handles callback scheduling        |
| 📲 `schedule_whatsapp` | Sends schedule confirmation        |
| 🔥 `hot_whatsapp`      | Handles high-intent leads          |
| ❄️ `cold`              | Handles cold-lead streak           |
| 🛑 `end_call`          | Terminates interaction             |
| 📝 `postcall_summary`  | Generates post-call intelligence   |

---

# 🧠 Conversation State

All nodes operate over a shared `ConversationState`.

```python
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
```

This provides the graph with shared state across agents and decision nodes.

---

# 🔄 Conversation State Flow

```text
              ┌─────────────────────┐
              │   USER MESSAGE      │
              └──────────┬──────────┘
                         │
                         ▼
                  🧠 INTENT STATE
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
            🔥 HOT     🌡️ WARM    ❄️ COLD
              │          │          │
              │          │          ▼
              │          │     cold_streak
              │          │
              │          ▼
              │     Agent Responses
              │          │
              │          ▼
              │        Judge
              │
              ▼
          hot_streak
              │
              ▼
           WhatsApp
```

---

# 🛠️ Technology Stack

| Technology           | Role                         |
| -------------------- | ---------------------------- |
| 🐍 Python            | Core language                |
| 🕸️ LangGraph        | Agent workflow orchestration |
| 🔗 LangChain         | LLM framework                |
| 💎 Google Gemini     | LLM reasoning and generation |
| 📦 Pydantic          | Structured LLM outputs       |
| 🧠 TypedDict         | Graph state                  |
| 📲 WhatsApp API/Tool | Customer communication       |
| 📅 Scheduler Agent   | Callback scheduling          |

---

# 📂 Project Structure

```text
project/
│
├── graph.py              # 🕸️ Main LangGraph workflow
├── schemas.py            # 📦 Pydantic schemas
├── prompts.py            # 📝 LLM prompts
├── tools.py              # 🛠️ External tools
├── scheduler.py          # 📅 Scheduling agent
├── llm_conveyer.py       # 📣 Marketing agent
│
└── README.md             # 📖 Documentation
```

---

# 🔬 Structured LLM Architecture

The project uses Gemini with structured outputs for tasks where deterministic fields are required.

### Intent

```python
intent_llm = llm.with_structured_output(
    IntentAnalysis
)
```

### Summary

```python
summary_llm = llm.with_structured_output(
    PostCallSummary
)
```

### Final Response

```python
judge_llm = llm.with_structured_output(
    FinalResponse
)
```

This reduces the need for fragile manual parsing of LLM responses.

---

# 🎯 Routing Logic

The routing system follows a deliberate priority order:

```text
                 INTENT
                   │
                   ▼
        ┌─────────────────────┐
        │ Schedule Requested? │
        └──────────┬──────────┘
                   │ Yes
                   ▼
               📅 SCHEDULE

                   │ No
                   ▼
             ┌─────────────┐
             │    COLD?    │
             └──────┬──────┘
                    │ Yes
                    ▼
                ❄️ COLD

                    │ No
                    ▼
           ┌─────────────────┐
           │ Explicit END?   │
           └───────┬─────────┘
                   │ Yes
                   ▼
                🛑 END

                   │ No
                   ▼
          ┌──────────────────┐
          │ HOT STREAK ≥ 2?  │
          └────────┬─────────┘
                   │ Yes
                   ▼
                🔥 HOT

                   │ No
                   ▼
             💬 DIALOGUE
```

This prevents competing conditions from accidentally bypassing important stateful logic.

---

# 🧪 Example Interaction

### Customer

```text
"I'm interested. Can someone explain the pricing?"
```

### Intent Agent

```text
Interest: HOT
Confidence: 0.91
Intention: Understand pricing
```

If the customer was already classified as hot on the previous turn:

```text
hot_streak = 2
```

The system triggers:

```text
🔥 HOT LEAD
      │
      ▼
📲 Personalized WhatsApp
      │
      ▼
END
```

---

# 🏆 Why Multi-Agent?

A single LLM could generate a response, but this architecture separates responsibilities:

```text
             ONE MONOLITHIC AGENT
                     ❌
                      │
                      ▼
              Does everything


                     VS


              MULTI-AGENT SYSTEM
                     ✅
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
    Intent         Dialogue       Marketing
       │              │              │
       └──────────────┼──────────────┘
                      ▼
                    Judge
                      │
                      ▼
                   Response
```

Advantages:

* 🧩 Modular responsibilities
* 🔍 Easier debugging
* 🎯 Specialized prompts
* 📈 Better response selection
* 🔄 Easier future expansion
* 🛠️ Independent agent replacement

---

# 🔐 Design Principles

The system is built around several principles:

### 1. 🧠 Analyze Before Acting

Customer intent is evaluated before triggering business actions.

### 2. 🎯 State Over Single-Turn Decisions

Hot and cold decisions use conversation history and streaks.

### 3. 🤝 Specialized Agents

Different agents handle different responsibilities.

### 4. ⚖️ Generate → Evaluate → Select

Multiple responses can be generated before choosing the final response.

### 5. 🛡️ Structured Outputs

Important LLM decisions use validated schemas.

### 6. 🔌 Action-Oriented AI

The system does more than generate text. It can trigger real business operations such as WhatsApp communication and scheduling.

---

# 📊 High-Level Architecture

```text
                         🤖 AI SALES AGENT
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
       🧠 INTELLIGENCE                       ⚙️ ACTIONS
              │                                   │
     ┌────────┼────────┐                  ┌───────┼───────┐
     │        │        │                  │       │       │
   Intent   Lead    Context           WhatsApp Scheduler Summary
   Agent    State
     │        │
     └────┬───┘
          │
          ▼
      🔀 ROUTER
          │
    ┌─────┼─────┐
    │     │     │
    ▼     ▼     ▼
   🔥    💬    ❄️
  HOT   DIALOGUE COLD
          │
     ┌────┴────┐
     ▼         ▼
 Dialogue   Marketing
     │         │
     └────┬────┘
          ▼
       ⚖️ JUDGE
          │
          ▼
     💬 RESPONSE
```

---

# 💡 Project Objective

The objective is to build a **stateful, action-oriented AI sales agent** capable of:

> **Understanding the customer → determining their intent → adapting the conversation → evaluating response quality → triggering business actions → and producing actionable post-call intelligence.**

Rather than treating an LLM as a simple chatbot, the system uses it as part of a **controlled agentic workflow**.

---

## 👨‍💻 Built With

**Python · LangGraph · LangChain · Google Gemini · Pydantic · WhatsApp · Agentic AI**

---

## ⭐ If You Find This Interesting

Feel free to explore the architecture, experiment with the prompts, and extend the workflow with additional agents and business actions.
