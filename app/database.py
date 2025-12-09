from sqlmodel import SQLModel, delete
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL

# Adjust DATABASE_URL for async drivers if necessary
async_database_url = DATABASE_URL
if DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+asyncpg://"):
    # Using asyncpg as it is generally recommended for async SQLAlchemy
    async_database_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create the async engine
engine = create_async_engine(
    async_database_url,
    echo=False,
    # PostgreSQL-specific settings
    # pool_pre_ping=True,  # Verify connections before using
    # pool_size=5,  # Connection pool size
    # max_overflow=10,  # Max overflow connections
)

async def create_db_and_tables():
    """
    Create all database tables.
    """
    print("Database URL:", async_database_url)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_session():
    """
    Dependency that provides a new database session.
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session

async def clear_table(table: type[SQLModel]):
    """
    Clears all rows from the given table.
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        statement = delete(table)
        await session.exec(statement)
        await session.commit()
