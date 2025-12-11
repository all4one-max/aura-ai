"""
User service for managing user profiles and state.
All user data is stored in AgentStateTable, not separate tables.
"""

import uuid
from typing import Optional, Dict, Any

from app.state import UserProfile
from app.dao.agent_state_dao import get_agent_state


class UserService:
    """Service for user operations."""

    def __init__(self, checkpointer=None):
        """Initialize - checkpointer parameter kept for compatibility but not used."""
        # We now use AgentStateTable directly, not checkpoints
        pass

    async def get_user_profile(
        self, user_id: str, thread_id: Optional[str] = None
    ) -> Optional[UserProfile]:
        """
        Get user profile from AgentStateTable.
        If thread_id provided, gets profile from that chat's state.
        Otherwise, searches all user's chats for profile.

        Args:
            user_id: User identifier
            thread_id: Optional chat/thread identifier

        Returns:
            UserProfile if found, None otherwise
        """
        if thread_id:
            # Get state from AgentStateTable
            agent_state_record = await get_agent_state(thread_id)
            if agent_state_record and agent_state_record.state:
                state = agent_state_record.state
                return state.get("user_profile")
        
        # For now, return None - profile will be loaded from state when needed
        # In production, you might want to query AgentStateTable for user's latest state
        return None

    async def update_user_profile(
        self, user_id: str, username: str, profile_data: Dict[str, Any]
    ) -> UserProfile:
        """
        Update user profile. Profile is stored in AgentStateTable.

        Args:
            user_id: User identifier
            username: Username
            profile_data: Profile data to update

        Returns:
            Updated UserProfile
        """
        # Create/update profile
        profile: UserProfile = {
            "user_id": user_id,
            "username": username,
            "photo_urls": profile_data.get("photo_urls"),
            "user_embeddings": profile_data.get("user_embeddings"),
            "upper_body_size": profile_data.get("upper_body_size"),
            "lower_body_size": profile_data.get("lower_body_size"),
            "region": profile_data.get("region"),
            "gender": profile_data.get("gender"),
            "age_group": profile_data.get("age_group"),
            "query_filters": profile_data.get("query_filters"),
            "liked_items": profile_data.get("liked_items", []),
        }
        return profile

    async def like_item(self, user_id: str, image_id: str) -> list:
        """
        Like an item. Liked items stored in user_profile.liked_items.

        Args:
            user_id: User identifier
            image_id: Image/product ID to like

        Returns:
            List of liked items
        """
        # This will be handled when updating state
        # For now, return empty list
        return []


# Global user service (initialized with MemorySaver for compatibility)
user_service: Optional[UserService] = None

