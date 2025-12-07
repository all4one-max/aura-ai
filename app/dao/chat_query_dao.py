from sqlmodel import Session, select

from app.database import engine
from app.schema import ChatQuery
from app.tools.extraction import ChatQueryExtraction


def upsert_chat_query(
    user_id: str, thread_id: str, extraction: ChatQueryExtraction
) -> ChatQuery:
    """
    Data Access Object for ChatQuery.
    Creates or updates a ChatQuery record based on extracted data.
    Only fields that are not None in the extraction will update the DB record.
    """
    with Session(engine) as session:
        statement = select(ChatQuery).where(
            ChatQuery.user_id == user_id, ChatQuery.thread_id == thread_id
        )
        existing = session.exec(statement).first()

        update_data = extraction.model_dump(exclude_none=True)

        if existing:
            # Update provided fields
            for key, value in update_data.items():
                setattr(existing, key, value)
            session.add(existing)
            session.commit()
            session.refresh(existing)
            return existing
        else:
            # Create new record
            new_record = ChatQuery(user_id=user_id, thread_id=thread_id, **update_data)
            session.add(new_record)
            session.commit()
            session.refresh(new_record)
            return new_record
