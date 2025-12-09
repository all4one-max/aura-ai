from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore

# from app.dao.chat_message_dao import get_messages
from app.dao.chat_query_dao import upsert_chat_query
from app.schema import ChatQuery
from app.state import AgentState
from app.tools.extraction import ChatQueryExtraction, extract_chat_query_tool
from app.tools.intent import check_initial_intent


def context_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    print("Context Agent")
    """
    Analyzes user intent and profile to determine next steps using an LLM.
    """
    # Extract user_id and thread_id from config
    user_id = config["metadata"]["user_id"]
    thread_id = config["metadata"]["thread_id"]

    user_input: str = state.get("messages")[-1].content  # type: ignore

    # Check if the input is related to shopping (context-aware)
    initial_intent = check_initial_intent(state.get("messages"))
    if not initial_intent.is_shopping_related:
        return {
            "user_intent": "general_chat",
            "next_step": "END",
            "current_agent": "context_agent",
            "messages": [AIMessage(content=initial_intent.response_if_not_related)],
        }

    # Fetch existing ChatQuery from DB

    # 1. Extract intents from current user message
    extracted_data: ChatQueryExtraction = extract_chat_query_tool(user_input)

    # 2. Update DB (Strong Typing)
    final_query: ChatQuery = upsert_chat_query(user_id, thread_id, extracted_data)

    # Deterministic Routing based on Missing Fields
    # Only 'query' is required - if it's missing or empty, ask for clarification
    if not final_query.query or not final_query.query.strip():
        response_msg = "To help you find products, I need to know what you're looking for. What product or item are you searching for?"
        return {
            "user_intent": "clarification",
            "next_step": "clarification_agent",
            "current_agent": "context_agent",
            "messages": [AIMessage(content=response_msg)],
        }
    else:
        return {
            "user_intent": "recommendation",
            "next_step": "research_agent",
            "current_agent": "context_agent",
            "messages": [
                AIMessage(
                    content="Great! I have all the details needed to find some recommendations for you. Let's get started."
                )
            ],
            "chat_query_json": final_query,
        }
