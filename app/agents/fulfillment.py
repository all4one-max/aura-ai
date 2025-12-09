from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore

from app.state import AgentState


def fulfillment_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    """
    Handles purchase.
    """
    selected_item = state.get("selected_item")
    if selected_item:
        return {
            "messages": [
                AIMessage(
                    content=f"Ordering {selected_item.title} for {selected_item.price}. Order placed!"
                )
            ],
            "current_agent": "fulfillment_agent",
            "next_step": None,
        }

    return {
        "messages": [AIMessage(content="Nothing to purchase.")],
        "current_agent": "fulfillment_agent",
        "next_step": None,
    }
