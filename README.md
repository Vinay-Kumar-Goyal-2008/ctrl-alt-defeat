# 🤖 AI Sales Conversation Agent

> **An agentic, stateful voice-based customer conversation system built with LangGraph, LangChain, and Google Gemini — capable of understanding customer intent, adapting conversation strategy, qualifying leads, triggering business actions, and generating post-call intelligence.**

<p align="center">

**🎙️ Voice Input** • **🌐 Multilingual Understanding** • **🧠 Intent Analysis** • **💬 Multi-Agent Dialogue** • **🎯 Lead Qualification** • **📅 Scheduling** • **📲 WhatsApp Automation** • **🔊 Multilingual TTS** • **📊 Post-Call Intelligence**

</p>

---

# 🚀 Overview

The **AI Sales Conversation Agent** is an agentic customer-conversation system designed to automate and intelligently manage sales conversations.

Instead of using a single LLM to handle every responsibility, the system decomposes the conversation into specialized stages and agents coordinated through **LangGraph**.

The system can:

* 🎙️ Capture customer speech from a microphone
* 🌐 Detect the customer's language
* 🔄 Translate speech into English for downstream reasoning
* 🧠 Analyze customer intent and interest
* 📊 Track lead state across multiple conversation turns
* 💬 Generate conversational responses
* 📣 Generate marketing-oriented responses
* ⚖️ Evaluate and select the best response
* 🔥 Detect high-intent leads
* ❄️ Detect consistently cold leads
* 📅 Handle callback scheduling
* 📲 Trigger WhatsApp communication
* 🔊 Convert the final response back into speech
* 🌐 Generate TTS in the customer's detected language
* 📝 Generate structured post-call intelligence

The overall architecture follows:

```text
🎙️ CUSTOMER SPEECH
        │
        ▼
┌─────────────────────┐
│ Speech Recognition  │
│ + Language Detect   │
└──────────┬──────────┘
           │
           ▼
     English Text
           │
           ▼
┌─────────────────────┐
│   🧠 INTENT AGENT   │
│      Gemini LLM     │
└──────────┬──────────┘
           │
     ┌─────┼─────┐
     │     │     │
     ▼     ▼     ▼
    🔥    🌡️    ❄️
   HOT   WARM   COLD
     │     │     │
     │     │     └──────► Cold Streak
     │     │
     │     └────────────► Multi-Agent Dialogue
     │
     └──────────────────► Hot Lead Workflow
                           │
                           ▼
                       📲 WhatsApp

Multi-Agent Path
       │
 ┌─────┴─────┐
 ▼           ▼
💬 Dialogue  📣 Marketing
 Agent        Agent
 │           │
 └─────┬─────┘
       ▼
 ⚖️ Response Judge
       │
       ▼
 💬 Final Response
       │
       ▼
 🔊 Multilingual TTS
       │
       ▼
🎙️ Customer Response

Termination
       │
       ▼
📝 Post-Call Summary
       │
       ▼
📲 WhatsApp
```

---

# ✨ Key Features

## 🎙️ Voice-Based Conversation

The system supports a voice-driven interaction loop.

```text
🎙️ Microphone
     │
     ▼
Audio Recording
     │
     ▼
Speech-to-Text
     │
     ▼
Language Detection
     │
     ▼
English Translation
     │
     ▼
LangGraph Agent
     │
     ▼
Final Response
     │
     ▼
Multilingual TTS
     │
     ▼
🔊 Audio Response
```

The reasoning pipeline operates on English text while preserving the customer's detected language for the response-generation stage.

---

# 🌐 Multilingual Conversation Pipeline

The system separates **reasoning language** from **response language**.

For example:

```text
Customer speaks Hindi
        │
        ▼
Speech Recognition
        │
        ▼
Detected Language = Hindi
        │
        ▼
Translate to English
        │
        ▼
LangGraph + Gemini
        │
        ▼
English Final Response
        │
        ▼
TTS using detected language
        │
        ▼
🔊 Hindi Audio Response
```

This allows the core agent workflow to remain language-independent while preserving a natural customer experience.

---

# 🧠 Intelligent Intent Analysis

Every customer message is analyzed using **Gemini structured output**.

The intent analysis extracts signals such as:

* 🎯 Customer intention
* 🔥 Interest level
* 📊 Confidence
* 📅 Scheduling request
* ⏰ Preferred scheduling time
* 🛑 End-call intent

Example:

```text
INTENT ANALYSIS

Interest: HOT
Confidence: 0.91
Intention: Understand pricing
Schedule Requested: False
End Call: False
```

Structured outputs allow downstream routing logic to operate on validated fields instead of parsing arbitrary LLM text.

---

# 🤝 Multi-Agent Response Generation

For normal and warm conversations, the system uses multiple specialized response generators.

```text
                Customer Message
                       │
              ┌────────┴────────┐
              ▼                 ▼
       💬 Dialogue Agent   📣 Marketing Agent
              │                 │
              │                 │
              └────────┬────────┘
                       ▼
                ⚖️ Response Judge
                       │
                       ▼
                 🏆 Best Response
```

### 💬 Dialogue Agent

Focuses on:

* Natural conversation
* Customer questions
* Intent alignment
* Context preservation
* Helpful responses

### 📣 Marketing Agent

Focuses on:

* Product positioning
* Value proposition
* Lead conversion
* Relevant product benefits

### ⚖️ Response Judge

The judge evaluates candidate responses based on:

* Relevance
* Intent understanding
* Product accuracy
* Naturalness
* Marketing effectiveness
* Conciseness
* Lack of excessive sales pressure
* Unsupported claims
* Repetition

The system therefore follows:

```text
GENERATE
   │
   ▼
EVALUATE
   │
   ▼
SELECT
```

rather than blindly returning the first generated response.

---

# 🔥 Stateful Lead Intelligence

A key component of the system is **temporal lead-state tracking**.

The system does not make important lead decisions solely from one classification.

It maintains conversation-level state such as:

```python
hot_streak
cold_streak
conversation
call_active
```

---

## 🔥 Hot Lead Detection

Two consecutive hot classifications trigger the hot-lead workflow.

```text
Turn 1
 🔥 HOT
   │
   ▼
hot_streak = 1
   │
   ▼
Turn 2
 🔥 HOT
   │
   ▼
hot_streak = 2
   │
   ▼
🔥 HOT LEAD
   │
   ▼
📲 WhatsApp
```

This reduces false positives caused by a single enthusiastic message.

---

## ❄️ Cold Lead Detection

Cold leads are tracked through consecutive classifications.

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

An explicit customer request to end the conversation can also terminate the interaction immediately.

---

# 📅 Intelligent Scheduling

Scheduling takes priority over normal conversational generation when a customer requests a callback.

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
🤖 Scheduler
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

The scheduling workflow can use information such as:

```python
scheduler.schedule(
    requested_time=intent.schedule_time,
    preference=intent.schedule_preference,
    user_intention=intent.intention
)
```

---

# 📲 WhatsApp Automation

The system can trigger WhatsApp communication based on business events.

## 🔥 Hot Lead

```text
Hot Streak ≥ 2
      │
      ▼
Generate Personalized Message
      │
      ▼
📲 Send WhatsApp
```

## 📅 Scheduled Callback

```text
Schedule Detected
      │
      ▼
Scheduler
      │
      ▼
Generate Confirmation
      │
      ▼
📲 Send WhatsApp
```

## 📝 Post-Call Summary

When the conversation terminates:

```text
Conversation Ends
      │
      ▼
Generate Structured Summary
      │
      ▼
Generate Follow-up Message
      │
      ▼
📲 WhatsApp
```

---

# 🔊 Multilingual Text-to-Speech

The final agent response can be converted into speech using the customer's detected language.

```text
Final Agent Response
        │
        ▼
Detected Language
        │
        ▼
Rime TTS
        │
        ▼
PCM Audio
        │
        ▼
WAV File
```

The TTS pipeline accepts:

```python
text_to_speech(
    text=response,
    language=detected_language,
    output_path="audio/agent_response.wav"
)
```

The generated audio is saved as a WAV file and can subsequently be connected to a playback or streaming layer.

---

# 📝 Post-Call Intelligence

After a conversation terminates, Gemini generates a structured post-call summary.

The summary contains:

| Field                    | Description                    |
| ------------------------ | ------------------------------ |
| 📝 Summary               | Overall conversation           |
| 🎯 User Intent           | Customer objective             |
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

The project follows a **state-driven agentic architecture** implemented with LangGraph.

```text
                         ┌───────────────┐
                         │     START     │
                         └───────┬───────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ 🧠 INTENT ANALYZER│
                       └────────┬─────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
          📅 SCHEDULE         ❄️ COLD         🛑 END
                │               │               │
                ▼               ▼               ▼
           SCHEDULER       COLD HANDLER      END CALL
                │               │               │
                ▼               │               │
           WHATSAPP             │               │
                │               │               │
                └───────────────┼───────────────┘
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
                       🔀 FAN-OUT
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
           💬 Dialogue Agent       📣 Marketing Agent
                  │                       │
                  └───────────┬───────────┘
                              ▼
                       ⚖️ RESPONSE JUDGE
                              │
                              ▼
                       💬 FINAL RESPONSE
                              │
                              ▼
                       🔊 MULTILINGUAL TTS
                              │
                              ▼
                         AUDIO OUTPUT
```

---

# 🧩 LangGraph Workflow

The workflow is implemented using `StateGraph`.

```python
graph = StateGraph(ConversationState)
```

The major nodes include:

| Node                   | Responsibility                   |
| ---------------------- | -------------------------------- |
| 🧠 `analyze`           | Intent and interest analysis     |
| 🔀 `conversation`      | Fan-out entry point              |
| 💬 `dialogue`          | Conversational response          |
| 📣 `marketing`         | Marketing-oriented response      |
| ⚖️ `judge`             | Evaluates response candidates    |
| 📅 `schedule`          | Callback scheduling              |
| 📲 `schedule_whatsapp` | Schedule confirmation            |
| 🔥 `hot_whatsapp`      | High-intent lead workflow        |
| ❄️ `cold`              | Cold-lead streak handling        |
| 🛑 `end_call`          | Terminates interaction           |
| 📝 `postcall_summary`  | Generates post-call intelligence |

---

# 🧠 Conversation State

The graph operates over a shared state object.

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

This state allows different nodes to access and update the same conversation context.

---

# 🔄 Conversation State Flow

```text
                USER MESSAGE
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
          │          │          │
          │          ▼          │
          │     Agent Responses │
          │          │          │
          │          ▼          │
          │        Judge        │
          │          │          │
          ▼          │          │
      hot_streak     │          │
          │          │          │
          ▼          │          │
       WhatsApp      │          │
                     │          │
                     └────┬─────┘
                          │
                          ▼
                    FINAL ACTION
```

---

# 🎯 Routing Logic

The routing system follows a deliberate priority order.

```text
                    INTENT
                      │
                      ▼
          ┌─────────────────────┐
          │ Schedule Requested? │
          └──────────┬──────────┘
                     │ YES
                     ▼
                  📅 SCHEDULE
                     │
                     │ NO
                     ▼
              ┌─────────────┐
              │    COLD?    │
              └──────┬──────┘
                     │ YES
                     ▼
                   ❄️ COLD
                     │
                     │ NO
                     ▼
            ┌──────────────────┐
            │ Explicit END?    │
            └────────┬─────────┘
                     │ YES
                     ▼
                   🛑 END
                     │
                     │ NO
                     ▼
           ┌──────────────────┐
           │ HOT STREAK ≥ 2?  │
           └────────┬─────────┘
                    │ YES
                    ▼
                  🔥 HOT
                    │
                    │ NO
                    ▼
              💬 DIALOGUE
```

This priority ordering prevents competing conditions from accidentally bypassing higher-priority business actions.

---

# 🔬 Structured LLM Architecture

Gemini is used with structured outputs for tasks where deterministic fields are required.

### Intent Analysis

```python
intent_llm = llm.with_structured_output(
    IntentAnalysis
)
```

### Post-Call Summary

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

This reduces dependence on fragile manual parsing of LLM-generated text.

---

# 🧪 Example Interaction

### 🎙️ Customer

```text
"I'm interested. Can someone explain the pricing?"
```

### 🧠 Intent Agent

```text
Interest: HOT
Confidence: 0.91
Intention: Understand pricing
```

Suppose the customer was already classified as hot during the previous turn:

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
🛑 End Call
      │
      ▼
📝 Post-Call Summary
```

---

# 🏆 Why Multi-Agent?

A single LLM could theoretically generate a response and perform all reasoning.

However, separating responsibilities provides a more controllable architecture.

```text
              ONE MONOLITHIC AGENT

                       ❌

                       │

                       ▼

                 Does Everything



                       VS



                MULTI-AGENT SYSTEM

                       ✅

                       │

        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Intent        Dialogue       Marketing
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                     Judge
                       │
                       ▼
                    Response
```

### Advantages

* 🧩 Modular responsibilities
* 🔍 Easier debugging
* 🎯 Specialized prompts
* 📈 Better response evaluation
* 🔄 Easier future expansion
* 🛠️ Independent agent replacement
* ⚙️ Explicit business routing
* 📊 Better observability of individual stages

---

# 🛡️ Design Principles

## 1. 🧠 Analyze Before Acting

Customer intent is evaluated before triggering business actions.

## 2. 🎯 State Over Single-Turn Decisions

Important lead decisions use conversation history and temporal signals.

## 3. 🤝 Specialized Agents

Different agents have different responsibilities instead of one monolithic prompt.

## 4. ⚖️ Generate → Evaluate → Select

Multiple responses can be generated and evaluated before selecting the final response.

## 5. 🛡️ Structured Outputs

Important LLM decisions use validated Pydantic schemas.

## 6. 🔌 Action-Oriented AI

The system does more than generate text.

It can trigger:

* WhatsApp communication
* Scheduling
* Lead workflows
* Post-call reporting

## 7. 🌐 Separate Reasoning and Presentation

The internal reasoning pipeline can operate in English while the customer-facing voice response can use the customer's detected language.

---

# 📊 High-Level Architecture

```text
                         🤖 AI SALES AGENT
                                │
                ┌───────────────┴────────────────┐
                │                                │
         🎙️ INPUT PIPELINE                  ⚙️ ACTIONS
                │                                │
       ┌────────┼────────┐              ┌────────┼────────┐
       │        │        │              │        │        │
   Speech   Language  Translation   WhatsApp Scheduler Summary
    STT     Detect
       │        │        │
       └────────┼────────┘
                │
                ▼
          🧠 INTELLIGENCE
                │
       ┌────────┼────────┐
       │        │        │
     Intent   Lead    Context
     Agent    State
       │        │
       └────┬───┘
            │
            ▼
         🔀 ROUTER
            │
       ┌────┼────┐
       │    │    │
       ▼    ▼    ▼
      🔥   💬   ❄️
     HOT DIALOGUE COLD
            │
       ┌────┴────┐
       ▼         ▼
    Dialogue  Marketing
       │         │
       └────┬────┘
            ▼
          ⚖️ JUDGE
            │
            ▼
       💬 RESPONSE
            │
            ▼
       🔊 MULTILINGUAL
            TTS
            │
            ▼
       🎙️ AUDIO OUTPUT
```

---

# 🛠️ Technology Stack

| Technology            | Role                             |
| --------------------- | -------------------------------- |
| 🐍 Python             | Core language                    |
| 🕸️ LangGraph         | Agent workflow orchestration     |
| 🔗 LangChain          | LLM integration                  |
| 💎 Google Gemini      | Reasoning and generation         |
| 📦 Pydantic           | Structured LLM outputs           |
| 🧠 TypedDict          | Graph state management           |
| 🎙️ Speech-to-Text    | Voice input processing           |
| 🌐 Language Detection | Customer language identification |
| 🔊 Rime TTS           | Multilingual speech generation   |
| 📲 WhatsApp API/Tool  | Customer communication           |
| 📅 Scheduler          | Callback scheduling              |
| 🔐 `.env`             | API key configuration            |

---

# 📂 Project Structure

```text
project/
│
├── graph.py              # 🕸️ Main LangGraph workflow
├── schemas.py            # 📦 Pydantic schemas
├── prompts.py            # 📝 LLM prompts
├── tools.py              # 🛠️ External tools
├── scheduler.py          # 📅 Scheduling logic
├── llm_conveyer.py       # 📣 Marketing agent
├── sst.py                # 🎙️ Speech-to-text + language detection
├── tts_handler.py        # 🔊 Text-to-speech pipeline
├── orchestrator.py       # 🎛️ Voice conversation orchestrator
│
├── audio/                # 🔊 Generated audio
│
├── .env                  # 🔐 API credentials
├── .gitignore
└── README.md             # 📖 Documentation
```

> Adjust the filenames above if your actual repository structure differs.

---

# 🔐 Environment Configuration

API credentials should be stored in `.env` rather than hardcoded in source files.

Example:

```env
RIME_API_KEY=your_api_key

RIME_TTS_WS_URL=wss://users-ws.rime.ai/ws3

RIME_SPEAKER=your_speaker

RIME_MODEL_ID=your_model_id

RIME_AUDIO_FORMAT=pcm
```

Load the environment using:

```python
from dotenv import load_dotenv
import os

load_dotenv()

RIME_API_KEY = os.getenv("RIME_API_KEY")
```

Never commit `.env` to GitHub.

Your `.gitignore` should contain:

```text
.env
__pycache__/
audio/
*.pyc
```

---

# 🔄 End-to-End Execution

The complete system can be viewed as the following pipeline:

```text
                 👤 CUSTOMER
                      │
                      ▼
                🎙️ MICROPHONE
                      │
                      ▼
              🎙️ SPEECH-TO-TEXT
                      │
                      ▼
              🌐 LANGUAGE DETECTION
                      │
                      ▼
             🔄 ENGLISH TRANSLATION
                      │
                      ▼
             🕸️ LANGGRAPH WORKFLOW
                      │
                      ▼
                🧠 INTENT AGENT
                      │
             ┌────────┼────────┐
             │        │        │
             ▼        ▼        ▼
            🔥       🌡️       ❄️
           HOT       WARM     COLD
             │        │        │
             │        ▼        │
             │   MULTI-AGENT    │
             │    RESPONSE      │
             │        │         │
             │        ▼         │
             │      ⚖️ JUDGE    │
             │        │         │
             └────────┼─────────┘
                      │
                      ▼
                💬 FINAL RESPONSE
                      │
                      ▼
                🔊 RIME TTS
                      │
                      ▼
              🌐 CUSTOMER LANGUAGE
                      │
                      ▼
                 🔊 WAV AUDIO
                      │
                      ▼
                 👤 CUSTOMER

              TERMINATION PATH
                      │
                      ▼
               📝 POST-CALL
                  SUMMARY
                      │
                      ▼
                 📲 WHATSAPP
```

---

# 🎯 Project Objective

The objective is to build a **stateful, multilingual, action-oriented AI sales agent** capable of:

> **Understanding the customer → detecting language → determining intent → tracking lead state → adapting the conversation → generating and evaluating responses → triggering business actions → responding through voice → and producing actionable post-call intelligence.**

Rather than treating an LLM as a simple chatbot, this system uses the LLM as one component inside a **controlled, state-driven agentic workflow**.

---

# 🚧 Future Improvements

Potential extensions include:

* 🔊 Real-time streaming TTS instead of WAV generation
* 🎙️ Full-duplex voice conversations
* 🧠 Long-term customer memory
* 📊 CRM integration
* 📈 Lead scoring dashboard
* 📞 Automatic outbound calling
* 🗓️ Google Calendar integration
* 📲 Production WhatsApp Business integration
* 🔍 Conversation analytics
* 📊 Sales funnel analytics
* 🧪 Automated agent evaluation
* 🗂️ Persistent LangGraph checkpointing
* ⚡ Streaming LangGraph responses
* 🧠 Retrieval-augmented product knowledge
* 🌐 Expanded multilingual support

---

# 🧠 Core Architecture in One Diagram

```text
                         🤖 AI SALES CONVERSATION AGENT
                                      │
                                      ▼
                              🎙️ VOICE INPUT
                                      │
                                      ▼
                           🌐 LANGUAGE DETECTION
                                      │
                                      ▼
                             🔄 TRANSLATION
                                      │
                                      ▼
                              🧠 INTENT AGENT
                                      │
                        ┌─────────────┼─────────────┐
                        │             │             │
                        ▼             ▼             ▼
                       🔥            🌡️            ❄️
                      HOT           WARM          COLD
                        │             │             │
                        │             │             ▼
                        │             │       Cold Streak
                        │             │             │
                        │             ▼             │
                        │       ┌───────────┐       │
                        │       │ FAN-OUT   │       │
                        │       └─────┬─────┘       │
                        │             │             │
                        │      ┌──────┴──────┐      │
                        │      ▼             ▼      │
                        │   Dialogue     Marketing   │
                        │      │             │       │
                        │      └──────┬──────┘       │
                        │             ▼              │
                        │           ⚖️ JUDGE          │
                        │             │              │
                        └─────────────┼──────────────┘
                                      │
                                      ▼
                              💬 FINAL RESPONSE
                                      │
                                      ▼
                               🔊 MULTILINGUAL TTS
                                      │
                                      ▼
                                🎙️ AUDIO OUTPUT
                                      │
                                      ▼
                                   CUSTOMER

                              BUSINESS ACTIONS
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
                 📲 WhatsApp       📅 Schedule       📝 Summary
```

---

# 👨‍💻 Built With

**Python · LangGraph · LangChain · Google Gemini · Pydantic · Speech-to-Text · Rime TTS · WhatsApp · Agentic AI**

---

## ⭐ Project Focus

The central idea behind this project is simple:

> **Don't build an LLM that only talks. Build an agentic system that understands, decides, acts, and remembers.**
