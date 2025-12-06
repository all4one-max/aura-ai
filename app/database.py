from sqlmodel import Session, SQLModel, create_engine

# Define the SQLite database file name
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

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
