from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore

# from app.dao.chat_message_dao import get_messages
from app.schema import ChatQuery
from app.state import AgentState
from app.tools.extraction import ChatQueryExtraction, extract_chat_query_tool
from app.tools.intent import check_initial_intent


async def context_agent(
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
    initial_intent = await check_initial_intent(state.get("messages"))
    if not initial_intent.is_shopping_related:
        return {
            "user_intent": "general_chat",
            "next_step": "END",
            "current_agent": "context_agent",
            "messages": [AIMessage(content=initial_intent.response_if_not_related)],
        }

    # 1. Extract intents from conversation messages (last 5 user messages)
    # Context agent uses the whole messages history (emphasizes last messages)
    all_messages = state.get("messages", [])
    print(f"📋 Context agent using {len(all_messages)} messages from conversation history")
    extracted_data: ChatQueryExtraction = await extract_chat_query_tool(all_messages)

    # 2. Convert extraction to ChatQuery (stored in AgentState, not separate table)
    # Get existing chat_query_json from state if available
    existing_query = state.get("chat_query_json")
    
    if existing_query:
        # Handle both dict (from database) and ChatQuery object cases
        if isinstance(existing_query, dict):
            # Convert dict to ChatQuery object
            existing_query = ChatQuery(**existing_query)
        
        # Update existing ChatQuery with new extracted data
        update_data = extracted_data.model_dump(exclude_none=True)
        for key, value in update_data.items():
            if hasattr(existing_query, key):
                setattr(existing_query, key, value)
        final_query = existing_query
    else:
        # Create new ChatQuery from extraction (no user_id/thread_id needed - stored in AgentState)
        final_query = ChatQuery(**extracted_data.model_dump(exclude_none=True))
    
    # 3. Merge user profile metadata into ChatQuery (if not already set from extraction)
    user_profile = state.get("user_profile")
    if user_profile:
        # Map user profile fields to ChatQuery fields
        # Only set if not already extracted from user messages
        
        # Gender: user_profile.gender -> chat_query.gender
        if not final_query.gender and user_profile.get("gender"):
            final_query.gender = user_profile["gender"]
        
        # Age group: user_profile.age_group -> chat_query.age_group
        if not final_query.age_group and user_profile.get("age_group"):
            final_query.age_group = user_profile["age_group"]
        
        # Size: user_profile.upper_body_size or lower_body_size -> chat_query.size
        # Prefer upper_body_size, fallback to lower_body_size
        if not final_query.size:
            if user_profile.get("upper_body_size"):
                final_query.size = user_profile["upper_body_size"]
            elif user_profile.get("lower_body_size"):
                final_query.size = user_profile["lower_body_size"]
        
        # Region: user_profile.region -> chat_query.gl and chat_query.location
        if user_profile.get("region"):
            region = user_profile["region"].lower()
            
            # Set gl (country code) if not already set
            if not final_query.gl or final_query.gl == "in":  # Only override default
                final_query.gl = region
            
            # Set location if not already set (convert region code to location string)
            if not final_query.location or final_query.location == "India":  # Only override default
                region_to_location = {
                    "in": "India",
                    "us": "United States",
                    "uk": "United Kingdom",
                    "ca": "Canada",
                    "au": "Australia",
                    "de": "Germany",
                    "fr": "France",
                    "it": "Italy",
                    "es": "Spain",
                    "jp": "Japan",
                    "cn": "China",
                    "br": "Brazil",
                    "mx": "Mexico",
                    "sg": "Singapore",
                    "ae": "UAE",
                }
                final_query.location = region_to_location.get(region, "India")
        
        print(f"✅ Merged user profile metadata into ChatQuery: gender={final_query.gender}, age_group={final_query.age_group}, size={final_query.size}, gl={final_query.gl}, location={final_query.location}")

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
            "next_step": "END",
            "current_agent": "context_agent",
            "messages": [AIMessage(content=response_msg)],
        }
    else:
        # All fields present - proceed to research
        # Send initial message to user that we're processing
        return {
            "user_intent": "recommendation",
            "next_step": "research_agent",
            "current_agent": "context_agent",
            "messages": [
                AIMessage(
                    content="Great! I have all the details. Searching for products and generating styling visualizations... This may take a moment."
                )
            ],
            "chat_query_json": final_query,
        }
