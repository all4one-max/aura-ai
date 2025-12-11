"""
DAO for AgentStateTable - manages agent state in our own table.
Replaces LangGraph's checkpoint system - we sync state here directly.
"""
import json
from datetime import datetime
from typing import Optional, Any
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.schema import AgentStateTable
from app.state import AgentState
from app.database import engine


def _serialize_for_json(obj: Any) -> Any:
    """Helper to serialize objects for JSON storage."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    # Handle Pydantic models
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    # Handle LangChain messages
    if hasattr(obj, "content") and hasattr(obj, "type"):
        msg_dict = {
            "type": obj.type if hasattr(obj, "type") else type(obj).__name__,
            "content": obj.content if hasattr(obj, "content") else str(obj),
        }
        # Include additional_kwargs if present (e.g., ranked_products)
        if hasattr(obj, "additional_kwargs") and obj.additional_kwargs:
            msg_dict["additional_kwargs"] = _serialize_for_json(obj.additional_kwargs)
        return msg_dict
    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        return {k: _serialize_for_json(v) for k, v in obj.__dict__.items()}
    return str(obj)


async def sync_agent_state_from_checkpoint(
    thread_id: str,
    user_id: str,
    state: AgentState,
    request_id: Optional[str] = None,
) -> AgentStateTable:
    """
    Sync agent state to AgentStateTable - stores individual fields in separate columns.
    
    Updates the same row incrementally as each agent completes:
    - context_agent → updates chat_query_json
    - research_agent → updates search_results
    - styling_agent → updates styled_products and merged_images
    - ranking_agent → updates ranked_products
    
    Args:
        thread_id: Chat/conversation identifier
        user_id: User identifier
        state: Current AgentState from LangGraph
        request_id: Optional request identifier
        
    Returns:
        AgentStateTable instance
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        # Check if state already exists
        statement = select(AgentStateTable).where(AgentStateTable.thread_id == thread_id)
        result = await session.execute(statement)
        existing_state = result.scalar_one_or_none()
        
        # Extract fields from state
        current_agent = state.get("current_agent")
        user_intent = state.get("user_intent")
        next_step = state.get("next_step")
        
        # Serialize complex objects for JSON storage
        messages = state.get("messages")
        user_profile = state.get("user_profile")
        search_results = state.get("search_results")
        selected_item = state.get("selected_item")
        chat_query_json = state.get("chat_query_json")
        styled_products = state.get("styled_products")
        ranked_products = state.get("ranked_products")
        merged_images = state.get("merged_images")
        
        now = datetime.utcnow()
        
        if existing_state:
            # Update existing state - only update fields that are present
            existing_state.user_id = user_id
            existing_state.request_id = request_id
            existing_state.current_agent = current_agent
            existing_state.user_intent = user_intent
            existing_state.next_step = next_step
            
            # Append messages if present (don't replace, append to existing)
            # Messages flow incrementally - append new ones to existing array
            if messages is not None:
                # Get existing messages from database
                existing_messages_raw = existing_state.messages or []
                
                # If existing_messages is a dict (JSONB), convert to list
                if isinstance(existing_messages_raw, dict):
                    if "messages" in existing_messages_raw:
                        existing_messages = existing_messages_raw["messages"]
                    else:
                        existing_messages = []
                elif isinstance(existing_messages_raw, list):
                    existing_messages = existing_messages_raw.copy()  # Make a copy to avoid modifying original
                else:
                    existing_messages = []
                
                print(f"🔍 Existing messages in DB: {len(existing_messages)}")
                
                # Serialize new messages from LangGraph state
                # LangGraph's messages already contain ALL accumulated messages (via add_messages reducer)
                new_messages_serialized = _serialize_for_json(messages)
                
                # Ensure new_messages_serialized is a list
                if isinstance(new_messages_serialized, dict):
                    if "messages" in new_messages_serialized:
                        new_messages_serialized = new_messages_serialized["messages"]
                    else:
                        new_messages_serialized = []
                elif not isinstance(new_messages_serialized, list):
                    new_messages_serialized = []
                
                print(f"🔍 New messages from LangGraph: {len(new_messages_serialized)}")
                
                # Normalize message type names for comparison
                def normalize_type(msg_type):
                    """Normalize message type for comparison."""
                    if not msg_type:
                        return ""
                    msg_type_lower = str(msg_type).lower()
                    if "human" in msg_type_lower or "user" in msg_type_lower:
                        return "human"
                    elif "ai" in msg_type_lower or "assistant" in msg_type_lower:
                        return "ai"
                    return msg_type_lower
                
                # Create a set of existing message signatures for quick lookup
                # Use content + normalized type as the signature
                existing_signatures = set()
                for msg in existing_messages:
                    if isinstance(msg, dict):
                        content = str(msg.get("content", "")).strip()
                        msg_type = normalize_type(msg.get("type", ""))
                        signature = f"{msg_type}:{content}"
                        existing_signatures.add(signature)
                
                # Count how many new messages we're adding
                new_count = 0
                
                # Add only new messages that aren't already present
                for new_msg in new_messages_serialized:
                    if isinstance(new_msg, dict):
                        new_content = str(new_msg.get("content", "")).strip()
                        new_type = normalize_type(new_msg.get("type", ""))
                        signature = f"{new_type}:{new_content}"
                        
                        if signature not in existing_signatures:
                            existing_messages.append(new_msg)
                            existing_signatures.add(signature)
                            new_count += 1
                            print(f"   ➕ Appending: [{new_type}] {new_content[:60]}...")
                    else:
                        # If it's not a dict, check by string representation
                        msg_str = str(new_msg).strip()
                        existing_strs = [str(m).strip() for m in existing_messages if not isinstance(m, dict)]
                        if msg_str not in existing_strs:
                            existing_messages.append(new_msg)
                            new_count += 1
                            print(f"   ➕ Appending (non-dict): {msg_str[:60]}...")
                
                # Log what we're doing
                print(f"📝 Final messages: {len(existing_messages)} total ({new_count} new appended, {len(existing_messages) - new_count} existing)")
                
                # Store the combined messages array
                existing_state.messages = existing_messages
            
            # Update user_profile if present
            if user_profile is not None:
                existing_state.user_profile = _serialize_for_json(user_profile)
            
            # Update search_results if present (research_agent completes)
            if search_results is not None:
                existing_state.search_results = _serialize_for_json(search_results)
            
            # Update selected_item if present
            if selected_item is not None:
                existing_state.selected_item = _serialize_for_json(selected_item)
            
            # Update chat_query_json if present (context_agent completes)
            if chat_query_json is not None:
                existing_state.chat_query_json = _serialize_for_json(chat_query_json)
            
            # Update styled_products if present (styling_agent completes)
            if styled_products is not None:
                existing_state.styled_products = _serialize_for_json(styled_products)
            
            # Update ranked_products if present
            if ranked_products is not None:
                existing_state.ranked_products = _serialize_for_json(ranked_products)
            
            # Update merged_images if present (styling_agent completes)
            if merged_images is not None:
                existing_state.merged_images = _serialize_for_json(merged_images)
            
            existing_state.updated_at = now
            await session.commit()
            await session.refresh(existing_state)
            return existing_state
        else:
            # Create new state
            new_state = AgentStateTable(
                thread_id=thread_id,
                user_id=user_id,
                request_id=request_id,
                messages=_serialize_for_json(messages),
                user_profile=_serialize_for_json(user_profile),
                search_results=_serialize_for_json(search_results),
                selected_item=_serialize_for_json(selected_item),
                chat_query_json=_serialize_for_json(chat_query_json),
                styled_products=_serialize_for_json(styled_products),
                ranked_products=_serialize_for_json(ranked_products),
                merged_images=_serialize_for_json(merged_images),
                current_agent=current_agent,
                user_intent=user_intent,
                next_step=next_step,
                created_at=now,
                updated_at=now,
            )
            session.add(new_state)
            await session.commit()
            await session.refresh(new_state)
            return new_state


async def get_agent_state(thread_id: str) -> Optional[AgentStateTable]:
    """
    Get agent state for a thread.
    
    Args:
        thread_id: Chat/conversation identifier
        
    Returns:
        AgentStateTable instance or None
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        statement = select(AgentStateTable).where(AgentStateTable.thread_id == thread_id)
        result = await session.execute(statement)
        
        # Handle case where multiple rows exist (shouldn't happen, but handle gracefully)
        rows = result.scalars().all()
        if len(rows) == 0:
            return None
        elif len(rows) == 1:
            return rows[0]
        else:
            # Multiple rows found - this shouldn't happen if thread_id is primary key
            # Return the most recently updated one
            print(f"⚠️  Warning: Multiple agent_state rows found for thread_id {thread_id}, using most recent")
            # Sort by updated_at descending and return the first (most recent)
            # Use a fallback datetime if both are None
            def get_sort_key(x):
                if x.updated_at:
                    return x.updated_at
                elif x.created_at:
                    return x.created_at
                else:
                    return datetime.min
            sorted_rows = sorted(rows, key=get_sort_key, reverse=True)
            return sorted_rows[0]


def agent_state_table_to_agent_state(db_state: AgentStateTable) -> AgentState:
    """
    Convert AgentStateTable database record back to AgentState TypedDict.
    Reconstructs AgentState from individual columns.
    
    Args:
        db_state: AgentStateTable instance from database
        
    Returns:
        AgentState TypedDict
    """
    from app.schema import ChatQuery, Product, ProductWithEmbedding
    
    # Reconstruct messages
    messages = db_state.messages if db_state.messages else []
    
    # Reconstruct user_profile
    user_profile = db_state.user_profile if db_state.user_profile else None
    
    # Reconstruct search_results (List[Product])
    search_results = []
    if db_state.search_results:
        if isinstance(db_state.search_results, list):
            search_results = [Product(**item) if isinstance(item, dict) else item for item in db_state.search_results]
        elif isinstance(db_state.search_results, dict):
            # Handle case where it's stored as dict
            search_results = [Product(**db_state.search_results)] if db_state.search_results else []
    
    # Reconstruct selected_item
    selected_item = None
    if db_state.selected_item:
        selected_item = Product(**db_state.selected_item) if isinstance(db_state.selected_item, dict) else db_state.selected_item
    
    # Reconstruct chat_query_json
    chat_query_json = None
    if db_state.chat_query_json:
        chat_query_json = ChatQuery(**db_state.chat_query_json) if isinstance(db_state.chat_query_json, dict) else db_state.chat_query_json
    
    # Reconstruct styled_products (List[ProductWithEmbedding])
    styled_products = []
    if db_state.styled_products:
        if isinstance(db_state.styled_products, list):
            styled_products = [ProductWithEmbedding(**item) if isinstance(item, dict) else item for item in db_state.styled_products]
        elif isinstance(db_state.styled_products, dict):
            styled_products = [ProductWithEmbedding(**db_state.styled_products)]
    
    # Reconstruct ranked_products
    ranked_products = []
    if db_state.ranked_products:
        if isinstance(db_state.ranked_products, list):
            ranked_products = [ProductWithEmbedding(**item) if isinstance(item, dict) else item for item in db_state.ranked_products]
        elif isinstance(db_state.ranked_products, dict):
            ranked_products = [ProductWithEmbedding(**db_state.ranked_products)]
    
    # Reconstruct merged_images
    merged_images = []
    if db_state.merged_images:
        if isinstance(db_state.merged_images, list):
            merged_images = db_state.merged_images
        elif isinstance(db_state.merged_images, dict):
            merged_images = list(db_state.merged_images.values()) if isinstance(db_state.merged_images, dict) else []
    
    return {
        "messages": messages,
        "user_profile": user_profile,
        "search_results": search_results,
        "selected_item": selected_item,
        "next_step": db_state.next_step,
        "user_intent": db_state.user_intent,
        "current_agent": db_state.current_agent,
        "thread_id": db_state.thread_id,
        "request_id": db_state.request_id,
        "chat_query_json": chat_query_json,
        "styled_products": styled_products,
        "ranked_products": ranked_products,
        "merged_images": merged_images,
    }


async def delete_agent_state(thread_id: str) -> bool:
    """
    Delete agent state for a thread.
    
    Args:
        thread_id: Chat/conversation identifier
        
    Returns:
        True if deleted, False if not found
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        statement = select(AgentStateTable).where(AgentStateTable.thread_id == thread_id)
        result = await session.execute(statement)
        state = result.scalar_one_or_none()
        
        if state:
            await session.delete(state)
            await session.commit()
            return True
        return False

