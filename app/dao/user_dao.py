"""
Data Access Object for User operations.
"""

import json
from typing import Optional, Dict, Any
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.database import engine
from app.schema import User
from app.state import UserProfile


async def create_user(username: str, user_id: Optional[str] = None) -> User:
    """
    Create a new user in PostgreSQL with username.
    User ID is auto-generated if not provided.
    All other fields are empty initially.
    
    If user with same username already exists, returns existing user instead.

    Args:
        username: Username (required)
        user_id: Optional user identifier. If not provided, generates one automatically.

    Returns:
        Created User object (or existing user if username already exists)
    """
    async with AsyncSession(engine) as session:
        # Check if user with this username already exists
        existing_user = await get_user(username=username)
        if existing_user:
            return existing_user
        
        # Generate user_id if not provided
        if not user_id:
            import uuid
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        # Create new user
        user = User(user_id=user_id, username=username)
        session.add(user)
        try:
            await session.commit()
            await session.refresh(user)
            return user
        except Exception as e:
            await session.rollback()
            # If duplicate error (shouldn't happen due to check above, but handle gracefully)
            existing_user = await get_user(username=username)
            if existing_user:
                return existing_user
            raise


async def get_user(username: Optional[str] = None, user_id: Optional[str] = None) -> Optional[User]:
    """
    Get user by username or user_id.

    Args:
        username: Username to search for
        user_id: User ID to search for

    Returns:
        User object if found, None otherwise
    """
    async with AsyncSession(engine) as session:
        if username:
            statement = select(User).where(User.username == username)
        elif user_id:
            statement = select(User).where(User.user_id == user_id)
        else:
            return None

        result = await session.execute(statement)
        return result.scalar_one_or_none()


async def update_user_profile(user_id: str, profile_updates: Dict[str, Any]) -> User:
    """
    Update user profile fields.
    Only provided fields will be updated.

    Args:
        user_id: User identifier
        profile_updates: Dictionary of fields to update (matching UserProfile)

    Returns:
        Updated User object
    """
    async with AsyncSession(engine) as session:
        statement = select(User).where(User.user_id == user_id)
        result = await session.execute(statement)
        user = result.scalar_one_or_none()

        if not user:
            raise ValueError(f"User {user_id} not found")

        # Update fields
        if "photo_urls" in profile_updates:
            photo_urls = profile_updates["photo_urls"]
            user.photo_urls = json.dumps(photo_urls) if photo_urls else None

        if "user_embeddings" in profile_updates:
            user_embeddings = profile_updates["user_embeddings"]
            # UserEmbedding is a BaseModel, serialize it properly
            if user_embeddings:
                from app.schema import UserEmbedding
                if isinstance(user_embeddings, dict):
                    # Validate and serialize UserEmbedding BaseModel
                    try:
                        embedding_obj = UserEmbedding(**user_embeddings)
                        user.user_embeddings = embedding_obj.model_dump_json()
                    except Exception as e:
                        print(f"Warning: Could not serialize user_embeddings: {e}")
                        user.user_embeddings = json.dumps(user_embeddings)
                else:
                    user.user_embeddings = json.dumps(user_embeddings)
            else:
                user.user_embeddings = None

        if "upper_body_size" in profile_updates:
            user.upper_body_size = profile_updates["upper_body_size"]

        if "lower_body_size" in profile_updates:
            user.lower_body_size = profile_updates["lower_body_size"]

        if "region" in profile_updates:
            user.region = profile_updates["region"]

        if "gender" in profile_updates:
            user.gender = profile_updates["gender"]

        if "age_group" in profile_updates:
            user.age_group = profile_updates["age_group"]

        if "query_filters" in profile_updates:
            query_filters = profile_updates["query_filters"]
            user.query_filters = json.dumps(query_filters) if query_filters else None

        if "liked_items" in profile_updates:
            liked_items = profile_updates["liked_items"]
            user.liked_items = json.dumps(liked_items) if liked_items else None

        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


def user_to_profile(user: User) -> UserProfile:
    """
    Convert User SQLModel to UserProfile TypedDict.
    
    UserEmbedding (BaseModel) is stored as JSON string in database and
    deserialized back to UserEmbedding dict structure here.

    Args:
        user: User object from database

    Returns:
        UserProfile dictionary (with UserEmbedding properly deserialized)
    """
    from app.schema import UserEmbedding
    
    # Deserialize UserEmbedding from JSON if present
    user_embeddings = None
    if user.user_embeddings:
        try:
            embeddings_dict = json.loads(user.user_embeddings)
            # UserEmbedding is a BaseModel, so we validate it
            user_embeddings = UserEmbedding(**embeddings_dict).model_dump()
        except Exception as e:
            print(f"Warning: Could not deserialize user_embeddings: {e}")
            user_embeddings = None
    
    return {
        "user_id": user.user_id,
        "username": user.username,
        "photo_urls": json.loads(user.photo_urls) if user.photo_urls else [],
        "user_embeddings": user_embeddings,  # UserEmbedding BaseModel structure
        "upper_body_size": user.upper_body_size,
        "lower_body_size": user.lower_body_size,
        "region": user.region,
        "gender": user.gender,
        "age_group": user.age_group,
        "query_filters": json.loads(user.query_filters) if user.query_filters else None,
        "liked_items": json.loads(user.liked_items) if user.liked_items else [],
    }

