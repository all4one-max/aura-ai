from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.database import engine
from app.schema import ChatQuery
from app.tools.extraction import ChatQueryExtraction


async def upsert_chat_query(
    user_id: str, thread_id: str, extraction: ChatQueryExtraction
) -> ChatQuery:
    """
    Data Access Object for ChatQuery.
    Creates or updates a ChatQuery record based on extracted data.
    Only fields that are not None in the extraction will update the DB record.
    """
    async with AsyncSession(engine) as session:
        statement = select(ChatQuery).where(
            ChatQuery.user_id == user_id, ChatQuery.thread_id == thread_id
        )
        result = await session.exec(statement)
        existing = result.first()

        update_data = extraction.model_dump(exclude_none=True)

        if existing:
            # Update provided fields
            for key, value in update_data.items():
                setattr(existing, key, value)
            session.add(existing)
            await session.commit()
            await session.refresh(existing)
            return existing
        else:
            # Create new record
            new_record = ChatQuery(user_id=user_id, thread_id=thread_id, **update_data)
            session.add(new_record)
            await session.commit()
            await session.refresh(new_record)
            return new_record
