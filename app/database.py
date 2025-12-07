import os

from sqlmodel import Session, SQLModel, create_engine, delete

# Define the SQLite database file name

sqlite_url = os.getenv("SQLLITE_URL", "")

# Create the engine
engine = create_engine(sqlite_url, echo=False)


def create_db_and_tables():
    """
    Creates the database tables based on the SQLModel metadata.
    """
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
