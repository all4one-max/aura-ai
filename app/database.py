from sqlmodel import Session, SQLModel, create_engine, delete
from app.config import DATABASE_URL

# Create the engine with connection pooling for PostgreSQL
engine = create_engine(
    DATABASE_URL,
    echo=False,
    # PostgreSQL-specific settings (ignored by SQLite)
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,  # Connection pool size
    max_overflow=10,  # Max overflow connections
)


def create_db_and_tables():
    """
    Create all database tables.
    """
    print("here it is", DATABASE_URL)
    SQLModel.metadata.create_all(engine)


def get_session():
    """
    Dependency that provides a new database session.
    """
    with Session(engine) as session:
        yield session


def clear_table(table: type[SQLModel]):
    """
    Clears all rows from the given table.
    """
    with Session(engine) as session:
        statement = delete(table)
        session.exec(statement)
        session.commit()
