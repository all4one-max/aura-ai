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

    # 1. Extract intents from conversation messages (last 5 user messages)
    extracted_data: ChatQueryExtraction = extract_chat_query_tool(state.get("messages"))

    # 2. Update DB (Strong Typing)
    final_query: ChatQuery = upsert_chat_query(user_id, thread_id, extracted_data)

    # Deterministic Routing based on Missing Fields
    missing_fields = []
    if not final_query.destination:
        missing_fields.append("destination")
    if not final_query.category:
        missing_fields.append("product type")
    if not final_query.occasion:
        missing_fields.append("occasion")

    if missing_fields:
        response_msg = f"To generate the best recommendations, I need to know the {', '.join(missing_fields)}."
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
