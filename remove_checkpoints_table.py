#!/usr/bin/env python3
"""
Script to remove LangGraph checkpoints table from PostgreSQL.
We now use only AgentStateTable for state management.
"""
import asyncio
import os
import sys
from psycopg_pool import AsyncConnectionPool

async def remove_checkpoints_table():
    """Drop the checkpoints table from PostgreSQL."""
    database_url = os.getenv("DATABASE_URL", "")
    
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Clean the URL
    if "postgresql+asyncpg://" in database_url:
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    
    if "sqlite://" in database_url:
        print("⚠️  SQLite detected. This script is for PostgreSQL only.")
        print("   For SQLite, just delete the database file if needed.")
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
                # Check if table exists
                await cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'checkpoints'
                    );
                """)
                exists = await cur.fetchone()
                
                if exists and exists[0]:
                    print("🗑️  Dropping checkpoints table...")
                    await cur.execute("DROP TABLE IF EXISTS checkpoints CASCADE;")
                    await conn.commit()
                    print("✅ Checkpoints table dropped successfully!")
                else:
                    print("ℹ️  Checkpoints table does not exist (already removed)")
                
                # Verify removal
                await cur.execute("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public' AND tablename = 'checkpoints';
                """)
                result = await cur.fetchone()
                
                if result:
                    print("⚠️  Warning: Table still exists after drop command")
                else:
                    print("✅ Verified: checkpoints table has been removed")
                
                # List remaining tables for reference
                await cur.execute("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public' 
                    ORDER BY tablename;
                """)
                tables = await cur.fetchall()
                print("\n📋 Remaining tables in database:")
                for table in tables:
                    print(f"   - {table[0]}")
        
    except Exception as e:
        print(f"❌ Error removing checkpoints table: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if pool:
            await pool.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Remove LangGraph Checkpoints Table")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Drop the 'checkpoints' table from PostgreSQL")
    print("  2. Verify removal")
    print("  3. Show remaining tables")
    print("\n⚠️  WARNING: This will delete all checkpoint history!")
    print("   (But we're not using it anymore - we use AgentStateTable)")
    print()
    
    try:
        response = input("Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Cancelled by user")
            sys.exit(0)
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Cancelled by user")
        sys.exit(0)
    
    asyncio.run(remove_checkpoints_table())

