from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore

from app.state import AgentState


def styling_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    """
    Visualizes the product on the user.
    """
    print("Styling Agent")
    # Mock selection of the first item
    search_results = state.get("search_results", [])
    if search_results:
        selected_item = search_results[0]
        return {
            "messages": [
                AIMessage(
                    content=f"Here is how the {selected_item['name']} looks on you."
                )
            ],
            "selected_item": selected_item,
            "current_agent": "styling_agent",
            "next_step": None,
        }

    return {
        "messages": [AIMessage(content="I couldn't find any items to style.")],
        "current_agent": "styling_agent",
        "next_step": None,
    }
