from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore

from app.state import AgentState
from app.tools.google_shopping import (
    chat_query_to_query_filters,
    search_google_shopping,
)


def research_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    """
    Searches for products using Google Shopping API via SerpApi.
    """
    print("Research Agent")

    # Get ChatQuery from state
    chat_query = state.get("chat_query_json")

    if not chat_query:
        return {
            "messages": [
                AIMessage(
                    content="I couldn't find the search query. Please try again with a product search."
                )
            ],
            "search_results": [],
            "current_agent": "research_agent",
        }

    # Convert ChatQuery to query_filters format
    query_filters = chat_query_to_query_filters(chat_query)

    # Search Google Shopping
    products = search_google_shopping(query_filters)[:5]

    if not products:
        return {
            "messages": [
                AIMessage(
                    content="I couldn't find any products matching your criteria. Please try adjusting your search."
                )
            ],
            "search_results": [],
            "current_agent": "research_agent",
        }

    # Format message with product count
    product_count = len(products)
    message_content = f"I found {product_count} product(s) that match your criteria."
    if product_count > 0:
        message_content += f" Here are the top results."

    return {
        "messages": [AIMessage(content=message_content)],
        "search_results": products,
        "current_agent": "research_agent",
    }
