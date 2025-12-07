from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore

from app.state import AgentState


def research_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    """
    Searches for products.
    """
    print("Research Agent")
    # Mock search results
    search_results = [
        {
            "id": 1,
            "name": "Blue Silk Dress",
            "price": 150,
            "url": "http://example.com/dress1",
        },
        {
            "id": 2,
            "name": "Navy Cocktail Dress",
            "price": 180,
            "url": "http://example.com/dress2",
        },
    ]

    return {
        "messages": [
            AIMessage(content="I found a few dresses that match your criteria.")
        ],
        "search_results": search_results,
        "current_agent": "research_agent",
    }
