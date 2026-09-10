from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from productknowledge import business


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class MarketingResponse(BaseModel):

    response: str = Field(
        description="The marketer's final response to the user's query"
    )


# ============================================================
# CREATE MARKETING AGENT
# ============================================================

def create_marketing_agent():
    import os
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "dummy_api_key_for_startup"

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.4,
    )

    structured_llm = llm.with_structured_output(
        MarketingResponse
    )

    system_prompt = """
You are an expert marketing strategist working for a specific business.

You have deep knowledge of the business provided below.

BUSINESS INFORMATION:
{business}

CONVERSATION HISTORY:
{history_text}

Your responsibilities:

1. Understand the business, its product, customers,
   positioning and goals.

2. Understand the conversation history before responding.

3. Answer the user's latest query from a practical
   marketing and customer-conversation perspective.

4. Give a response that is useful for the customer,
   not an internal marketing analysis.

5. Consider customer acquisition, positioning,
   messaging, conversion, retention, competition
   and growth when relevant.

6. Do not invent business facts, products, pricing,
   customers or features.

7. If information is missing, do not fabricate it.

8. Challenge incorrect assumptions instead of
   blindly agreeing.

9. Avoid generic marketing language when
   business-specific information is available.

10. Answer the actual latest query directly.

11. Keep the response concise and natural.

12. Do not mention that you are an AI, marketer,
    agent, or that you are analyzing the conversation.

You are generating a customer-facing response
for the business.
"""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            system_prompt
        ),
        (
            "human",
            "{query}"
        )
    ])

    return prompt | structured_llm


# ============================================================
# MARKETING AGENT
# ============================================================

def marketing_agent(
    query: str,
    history_text: str
) -> str:

    agent = create_marketing_agent()

    result = agent.invoke({
        "business": business,
        "query": query,
        "history_text": history_text
    })

    return result.response