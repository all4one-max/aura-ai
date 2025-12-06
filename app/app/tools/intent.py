from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class InitialIntent(BaseModel):
    """Classify if the user input is related to shopping, fashion, or the aura-ai platform capabilities."""

    is_shopping_related: bool = Field(
        description="True if the user is asking for recommendations, styling, clothes, buying, or fashion advice. False if it's general chit-chat (e.g. 'hello', 'how are you', 'weather')."
    )
    response_if_not_related: Optional[str] = Field(
        description="A polite conversational response if the input is NOT shopping related.",
        default=None,
    )


def check_initial_intent(user_input: str) -> InitialIntent:
    """
    Uses an LLM to quickly classify the user's intent before expensive processing.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(InitialIntent)
    return structured_llm.invoke(user_input)
