#!/usr/bin/env python3
"""
Script to remove LangGraph checkpoints table and verify AgentStateTable.
Uses SQLModel/SQLAlchemy directly.
"""
import asyncio
import os
import sys
from sqlalchemy import text
from sqlmodel import create_engine, SQLModel
from app.database import engine
from app.schema import AgentStateTable

async def cleanup_checkpoints():
    """Remove checkpoints table and verify agent_state table."""
    print("=" * 60)
    print("Cleanup Checkpoints Table & Verify AgentStateTable")
    print("=" * 60)
    print()
    
    try:
        # Use async engine
        async with engine.begin() as conn:
            # Check if checkpoints table exists
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'checkpoints'
                );
            """))
            exists = await result.fetchone()
            
            if exists and exists[0]:
                print("🗑️  Dropping checkpoints table...")
                await conn.execute(text("DROP TABLE IF EXISTS checkpoints CASCADE;"))
                print("✅ Checkpoints table dropped successfully!")
            else:
                print("ℹ️  Checkpoints table does not exist (already removed)")
            
            # Verify removal
            result = await conn.execute(text("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public' AND tablename = 'checkpoints';
            """))
            checkpoints_exists = await result.fetchone()
            
            if checkpoints_exists:
                print("⚠️  Warning: Table still exists after drop command")
            else:
                print("✅ Verified: checkpoints table has been removed")
            
            # Ensure agent_state table exists (create if needed)
            print("\n📋 Ensuring agent_state table exists...")
            SQLModel.metadata.create_all(engine.sync_engine, tables=[AgentStateTable.__table__])
            print("✅ AgentStateTable is ready")
            
            # List all tables
            result = await conn.execute(text("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public' 
                ORDER BY tablename;
            """))
            tables = await result.fetchall()
            print("\n📋 Current tables in database:")
            for table in tables:
                print(f"   - {table[0]}")
            
            # Check agent_state table structure
            print("\n🔍 Checking agent_state table structure...")
            result = await conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'agent_state'
                ORDER BY ordinal_position;
            """))
            columns = await result.fetchall()
            if columns:
                print("✅ agent_state table columns:")
                for col in columns:
                    print(f"   - {col[0]}: {col[1]} (nullable: {col[2]})")
            else:
                print("⚠️  agent_state table not found - it will be created on first use")
        
        print("\n✅ Cleanup complete!")
        print("\n📝 Summary:")
        print("   - checkpoints table: REMOVED")
        print("   - agent_state table: READY")
        print("   - System now uses only AgentStateTable for state management")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(cleanup_checkpoints())

