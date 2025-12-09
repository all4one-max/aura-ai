"""
Intent classification tool for determining if user input is shopping-related.
"""

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.services.llm_service import get_llm_service


class InitialIntent(BaseModel):
    """
    Classification of user intent.
    """

    is_shopping_related: bool = Field(
        description="True if the message is related to fashion, shopping, clothing, or product recommendations. False for general questions."
    )
    response_if_not_related: str = Field(
        description="A polite response if the query is not shopping-related, redirecting to fashion assistance."
    )


async def check_initial_intent(messages: List[BaseMessage]) -> InitialIntent:
    """
    Analyzes the last 5 conversation turns plus the current message to determine
    if the current user message is shopping-related or general Q&A.

    This is context-aware: if the user is in the middle of a shopping conversation
    and asks "what is the capital of Belgium?", it should recognize this as
    NOT shopping-related even though previous messages were about shopping.

    Args:
        messages: List of conversation messages (last 5 + current)

    Returns:
        InitialIntent with classification and optional response
    """
    # Take last 5 messages (including current)
    recent_messages = messages[-5:] if len(messages) > 5 else messages

    # Format conversation for context
    conversation_context = []
    for i, msg in enumerate(recent_messages):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        is_current = i == len(recent_messages) - 1
        marker = " [CURRENT MESSAGE]" if is_current else ""
        conversation_context.append(f"{role}: {msg.content}{marker}")

    conversation_text = "\n".join(conversation_context)

    prompt = f"""You are an intent classifier for a fashion shopping assistant. Your task is to determine if the CURRENT user message (marked with [CURRENT MESSAGE]) is related to fashion/shopping or is a general question.

**IMPORTANT CONTEXT RULES:**
- Look at the conversation history for context
- Focus on the CURRENT MESSAGE to determine intent
- Even if previous messages were about shopping, if the CURRENT message is a general question (e.g., "what is the capital of Belgium?"), classify it as NOT shopping-related
- A message is shopping-related if it's about: clothing, fashion, products, recommendations, style, outfits, shopping, prices, colors, occasions, destinations for fashion purposes

**Conversation History:**
{conversation_text}

**Examples:**

Example 1 (General Q&A in middle of shopping):
user: I need a shirt
assistant: What destination?
user: Thailand
user: What is the capital of Belgium? [CURRENT MESSAGE]
**Classification:** is_shopping_related = False
**Response:** "I'm Aura AI, your fashion assistant. I can help you with clothing and style recommendations. How can I assist with your fashion needs?"

Example 2 (Continuing shopping conversation):
user: I need a shirt
assistant: What destination?
user: Thailand [CURRENT MESSAGE]
**Classification:** is_shopping_related = True

Example 3 (Direct general question):
user: What is the weather today? [CURRENT MESSAGE]
**Classification:** is_shopping_related = False
**Response:** "I'm a fashion shopping assistant. I can help you find the perfect outfit! What are you looking for?"

Example 4 (Shopping query):
user: I need clothes for a wedding [CURRENT MESSAGE]
**Classification:** is_shopping_related = True

Classify the CURRENT MESSAGE now."""

    llm_service = get_llm_service()
    return await llm_service.generate_structured_output(prompt, InitialIntent)
