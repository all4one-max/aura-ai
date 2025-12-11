"""
DAO for UserChat operations.
Manages user-chat relationships for fetching chats and showing chat history.
"""
from typing import List, Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from datetime import datetime

from app.database import engine
from app.schema import UserChat


async def create_user_chat(username: str, chat_room_id: str, user_id: str) -> UserChat:
    """
    Create or update a UserChat record.
    
    Since chat_room_id is the primary key, we query by chat_room_id only.
    If it exists, we update username and user_id (in case they changed).
    If it doesn't exist, we create a new record.
    
    Args:
        username: Username
        chat_room_id: Chat room/thread identifier (primary key)
        user_id: User identifier
        
    Returns:
        Created or updated UserChat object
    """
    async with AsyncSession(engine) as session:
        # Query by primary key (chat_room_id) only
        statement = select(UserChat).where(
            UserChat.chat_room_id == chat_room_id
        )
        result = await session.execute(statement)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing record (username and user_id might have changed)
            existing.username = username
            existing.user_id = user_id
            existing.updated_at = datetime.utcnow()
            await session.commit()
            await session.refresh(existing)
            return existing
        
        # Create new
        user_chat = UserChat(
            username=username,
            chat_room_id=chat_room_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(user_chat)
        await session.commit()
        await session.refresh(user_chat)
        return user_chat


async def get_user_chats(username: str) -> List[UserChat]:
    """
    Get all chats for a user by username (deprecated - use get_user_chats_by_user_id instead).
    
    Args:
        username: Username
        
    Returns:
        List of UserChat objects ordered by updated_at (newest first)
    """
    async with AsyncSession(engine) as session:
        statement = select(UserChat).where(
            UserChat.username == username
        ).order_by(UserChat.updated_at.desc())
        result = await session.execute(statement)
        # Use scalars() to get UserChat objects, not Row objects
        return list(result.scalars().all())


async def get_user_chats_by_user_id(user_id: str) -> List[UserChat]:
    """
    Get all chats for a user by user_id.
    
    Args:
        user_id: User identifier
        
    Returns:
        List of UserChat objects ordered by updated_at (newest first)
    """
    async with AsyncSession(engine) as session:
        statement = select(UserChat).where(
            UserChat.user_id == user_id
        ).order_by(UserChat.updated_at.desc())
        result = await session.execute(statement)
        # Use scalars() to get UserChat objects, not Row objects
        return list(result.scalars().all())


async def get_user_chat_by_room_id(username: str, chat_room_id: str) -> Optional[UserChat]:
    """
    Get a specific chat for a user by chat_room_id.
    
    Args:
        username: Username
        chat_room_id: Chat room identifier
        
    Returns:
        UserChat if found, None otherwise
    """
    async with AsyncSession(engine) as session:
        statement = select(UserChat).where(
            UserChat.username == username,
            UserChat.chat_room_id == chat_room_id
        )
        result = await session.execute(statement)
        return result.scalar_one_or_none()


async def delete_user_chat(username: str, chat_room_id: str) -> bool:
    """
    Delete a user chat.
    
    Args:
        username: Username
        chat_room_id: Chat room identifier
        
    Returns:
        True if deleted, False if not found
    """
    async with AsyncSession(engine) as session:
        statement = select(UserChat).where(
            UserChat.username == username,
            UserChat.chat_room_id == chat_room_id
        )
        result = await session.execute(statement)
        user_chat = result.scalar_one_or_none()
        
        if user_chat:
            await session.delete(user_chat)
            await session.commit()
            return True
        return False

