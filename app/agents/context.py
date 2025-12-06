from typing import Literal, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from sqlmodel import select

from app.database import get_session
from app.schema import ChatQuery
from app.state import AgentState
from app.tools.extraction import extract_chat_query_tool


# Define the Tool Schema
class Router(BaseModel):
    """Schema for routing the conversation."""

    user_intent: str = Field(
        description="The categorization of the user's intent: 'clarification','recommendation', 'refinement', 'price_check', 'purchase', or 'general_chat'"
    )
    next_step: Literal[
        "research_agent", "styling_agent", "fulfillment_agent", "END"
    ] = Field(description="The next agent to call.")
    response_message: str = Field(
        description="The response message to show to the user."
    )


SYSTEM_PROMPT_2 = """You are the 'Context Agent', the central Orchestrator of the Aura AI fashion platform.
Your SOLE goal is to perform **Initial Data Validation** and route the user to the correct next step.

**CRITICAL STATE INFORMATION:**
{last_agent_info}

**MINIMUM DATA REQUIREMENT FOR SEARCH:**
To successfully call the `research_agent`, the system must have clear intent on the **product type** (e.g., shirt, dress, pants) and the **occasion/context** (e.g., beach vacation, business event).

**Agents available for routing:**
1. `research_agent`: Use ONLY when the **MINIMUM DATA REQUIREMENT** is fully satisfied (product type and occasion are clear).
2. `clarification_agent`: Use when a search request is made, but the **MINIMUM DATA REQUIREMENT** is missing or incomplete. This agent will PAUSE the graph to ask the user for the missing data.

**Routing Logic:**
- **HIGH PRIORITY RULE 1 (Off-Topic/Direct Answer):** If the user's input is not related to fashion, styling, or shopping (e.g., "Hello," "How is the weather?"), route to `END` and provide a helpful, conversational response.
- **RULE 2 (Data Validation and Loop):**
    - If the user's input implies a recommendation or search (e.g., "I need an outfit") AND the **MINIMUM DATA REQUIREMENT** is missing or incomplete (you do not know the occasion or specific product type), route to `clarification_agent`.
    - Otherwise (the user input and history provide the MINIMUM DATA REQUIREMENT for a successful search), route to `research_agent`.

**Important:** Always provide a brief, professional `response_message` to the user explaining the transition.
"""


def context_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    """
    Analyzes user intent and profile to determine next steps using an LLM.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Bind the tool
    llm_with_tools = llm.bind_tools([Router])

    # Extract user_id and thread_id from config
    # Fallback to defaults if not present
    user_id = config["metadata"]["user_id"]
    thread_id = config["metadata"]["thread_id"]

    # Fetch existing ChatQuery from DB
    existing_context = ""
    session_generator = get_session()
    session = next(session_generator)
    result = None
    try:
        statement = select(ChatQuery).where(
            ChatQuery.user_id == user_id, ChatQuery.thread_id == thread_id
        )
        result = session.exec(statement).first()
        if result:
            existing_context = f"""
            **PREVIOUS CONTEXT FOUND:**
            - Destination: {result.destination}
            - Occasion: {result.occasion}
            - Product Type: {result.product_type}
            - Budget: {result.budget_range}
            - Color: {result.color}
            """
    except Exception as e:
        print(f"Error fetching context: {e}")
    finally:
        session.close()

    extracted_data = extract_chat_query_tool(state.get("messages")[-1].content)
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

        missing_fields = []
        if not query_to_check.destination:
            missing_fields.append("destination")
        if not query_to_check.product_type:
            missing_fields.append("product type")
        if not query_to_check.occasion:
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
            }

    except Exception as e:
        print(f"Error saving context: {e}")
    finally:
        session.close()

    # Extract the last agent's execution status (assuming these fields exist in state)
    if state.get("current_agent"):
        last_agent_info = f"""
        NOTE: THIS MESSAGE IS FROM THE LAST AGENT
        Last Agent: {state.get("current_agent")}
        Last Agent Response: {state.get("messages")[-1].content}
        {existing_context}
        """
    else:
        # this is user input
        last_agent_info = f"""
        NOTE: THIS MESSAGE IS USER INPUT
        User Input: {state.get("messages")[-1].content}
        {existing_context}
        """

    # Format the SYSTEM_PROMPT with the current state info
    formatted_system_prompt = SYSTEM_PROMPT_2.format(last_agent_info=last_agent_info)

    # Invoke
    # logic to handle potential cyclic history where previous agent outputs might confuse the LLM
    # For now, passing full history is fine as it includes "I found X" messages from agents.

    ai_msg = llm_with_tools.invoke(formatted_system_prompt)

    # Parse Tool Call
    if ai_msg.tool_calls:
        tool_call = ai_msg.tool_calls[0]
        args = tool_call["args"]

        user_intent = args.get("user_intent")
        next_step = args.get("next_step")
        response_msg = args.get("response_message")

        return {
            "user_intent": user_intent,
            "next_step": next_step,
            "current_agent": "context_agent",
            "messages": [AIMessage(content=response_msg)],
        }

    # Fallback if no tool call (should shouldn't happen with forced tool calling, but for safety)
    return {
        "user_intent": "general_chat",
        "next_step": "END",
        "current_agent": "context_agent",
        "messages": [ai_msg],
    }
