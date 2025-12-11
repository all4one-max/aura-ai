#!/usr/bin/env python3
"""
Script to fix LangGraph checkpoints table schema issue.
This script drops and recreates the checkpoints table with proper constraints.

WARNING: This will delete all checkpoint history!
"""

import asyncio
import os
import sys
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def fix_checkpoints_table():
    """Drop and recreate the checkpoints table."""
    database_url = os.getenv("DATABASE_URL", "")
    
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Clean the URL
    if "postgresql+asyncpg://" in database_url:
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    
    if "sqlite://" in database_url:
        print("⚠️  SQLite detected. This script is for PostgreSQL only.")
        print("   For SQLite, just delete the database file and restart the server.")
        sys.exit(1)
    
    if "postgresql://" not in database_url:
        print("❌ Invalid DATABASE_URL. Expected PostgreSQL connection string.")
        sys.exit(1)
    
    print(f"🔧 Connecting to database...")
    pool = None
    try:
        pool = AsyncConnectionPool(database_url)
        await pool.open()
        
        # Get a connection to execute raw SQL
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                print("🗑️  Dropping existing checkpoints table...")
                await cur.execute("DROP TABLE IF EXISTS checkpoints CASCADE;")
                await conn.commit()
                print("✅ Checkpoints table dropped")
        
        print("🔧 Recreating checkpoints table with proper schema...")
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        print("✅ Checkpoints table recreated successfully!")
        print("\n📝 Next steps:")
        print("   1. Restart your server")
        print("   2. Test the chat functionality")
        print("   3. The checkpoints table should now work correctly")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if pool:
            await pool.close()

if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph Checkpoints Table Fix Script")
    print("=" * 60)
    print("\n⚠️  WARNING: This will delete all checkpoint history!")
    print("   Press Ctrl+C to cancel, or wait 5 seconds to continue...")
    print()
    
    try:
        import signal
        
        def timeout_handler(signum, frame):
            print("\n⏱️  Timeout - proceeding with fix...")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        # Wait for user input or timeout
        try:
            input()
            signal.alarm(0)  # Cancel timeout
        except (EOFError, KeyboardInterrupt):
            signal.alarm(0)
            print("\n❌ Cancelled by user")
            sys.exit(0)
    except (ImportError, AttributeError):
        # Windows doesn't support SIGALRM
        import time
        time.sleep(2)
        print("   Proceeding...")
    
    asyncio.run(fix_checkpoints_table())

