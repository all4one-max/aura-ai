from sqlmodel import Session, select

from app.database import create_db_and_tables
from app.schema import ChatQuery


def verify_database():
    print("--- Verifying Database Setup ---")

    # 1. Initialize DB (create tables)
    create_db_and_tables()
    # print("✅ Tables Created")

    # # 2. Create a dummy record
    # query_data = ChatQuery(
    #     user_id="user_123",
    #     thread_id="thread_abc",
    #     destination="Hawaii",
    #     occasion="Summer Vacation",
    #     budget_range="$100-$300",
    #     product_type=ProductType.SHIRT,
    #     month_of_visit="July",
    #     color="Blue",
    # )

    # # 3. Save to DB
    # with Session(engine) as session:
    #     session.add(query_data)
    #     session.commit()
    #     session.refresh(query_data)
    #     saved_id = query_data.id
    #     print(f"✅ Saved Record with ID: {saved_id}")

    # # 4. Read from DB
    # with Session(engine) as session:
    #     statement = select(ChatQuery).where(ChatQuery.id == saved_id)
    #     retrieved_record = session.exec(statement).first()

    #     if retrieved_record:
    #         print(f"✅ Retrieved Record: {retrieved_record}")
    #         # Verify fields
    #         assert retrieved_record.destination == "Hawaii"
    #         assert retrieved_record.product_type == ProductType.SHIRT
    #         assert retrieved_record.color == "Blue"
    #         print("✅ Field Verification Passed")
    #     else:
    #         print("❌ Failed to retrieve record")


if __name__ == "__main__":
    verify_database()
