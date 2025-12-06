from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from sqlmodel import select

from app.database import get_session
from app.schema import ChatQuery
from app.state import AgentState
from app.tools.extraction import extract_chat_query_tool
from app.tools.intent import check_initial_intent


def context_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    """
    Analyzes user intent and profile to determine next steps using an LLM.
    """

    last_message: str = state.get("messages")[-1].content  # type: ignore

    # Check if the input is related to shopping
    initial_intent = check_initial_intent(last_message)
    if not initial_intent.is_shopping_related:
        return {
            "user_intent": "general_chat",
            "next_step": "END",
            "current_agent": "context_agent",
            "messages": [AIMessage(content=initial_intent.response_if_not_related)],
        }

    # Extract user_id and thread_id from config
    # Fallback to defaults if not present
    user_id = config["metadata"]["user_id"]
    thread_id = config["metadata"]["thread_id"]

    # Fetch existing ChatQuery from DB
    session_generator = get_session()
    session = next(session_generator)
    result = None
    try:
        statement = select(ChatQuery).where(
            ChatQuery.user_id == user_id, ChatQuery.thread_id == thread_id
        )
        result = session.exec(statement).first()
    except Exception as e:
        print(f"Error fetching context: {e}")
    finally:
        session.close()

    extracted_data = extract_chat_query_tool(last_message)
    new_chat_query = ChatQuery(
        user_id=user_id,
        thread_id=thread_id,
        destination=extracted_data.destination
        if extracted_data.destination
        else (result.destination if result else None),
        occasion=extracted_data.occasion
        if extracted_data.occasion
        else (result.occasion if result else None),
        budget_range=extracted_data.budget_range
        if extracted_data.budget_range
        else (result.budget_range if result else None),
        product_type=extracted_data.product_type
        if extracted_data.product_type
        else (result.product_type if result else None),
        month_of_visit=extracted_data.month_of_visit
        if extracted_data.month_of_visit
        else (result.month_of_visit if result else None),
        color=extracted_data.color
        if extracted_data.color
        else (result.color if result else None),
    )

    # Save to DB & Deterministic Routing
    session_generator = get_session()
    session = next(session_generator)
    missing_fields = []
    try:
        statement = select(ChatQuery).where(
            ChatQuery.user_id == user_id, ChatQuery.thread_id == thread_id
        )
        existing_record = session.exec(statement).first()
        if existing_record:
            existing_record.destination = new_chat_query.destination
            existing_record.occasion = new_chat_query.occasion
            existing_record.budget_range = new_chat_query.budget_range
            existing_record.product_type = new_chat_query.product_type
            existing_record.month_of_visit = new_chat_query.month_of_visit
            existing_record.color = new_chat_query.color
            session.add(existing_record)
            query_to_check = existing_record
        else:
            session.add(new_chat_query)
            query_to_check = new_chat_query
        session.commit()

        # Check logic inside the session to avoid DetachedInstanceError
        if not existing_record:
            session.refresh(new_chat_query)

        if not query_to_check.destination:
            missing_fields.append("destination")
        if not query_to_check.product_type:
            missing_fields.append("product type")
        if not query_to_check.occasion:
            missing_fields.append("occasion")

    except Exception as e:
        print(f"Error saving context: {e}")
    finally:
        session.close()

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
        }
