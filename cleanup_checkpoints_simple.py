#!/usr/bin/env python3
"""
Script to remove LangGraph checkpoints table and verify AgentStateTable.
Uses psycopg2 directly (synchronous).
"""
import os
import sys
import psycopg2
from urllib.parse import urlparse, unquote

# Load config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.config import DATABASE_URL

def cleanup_checkpoints():
    """Remove checkpoints table and verify agent_state table."""
    print("=" * 60)
    print("Cleanup Checkpoints Table & Verify AgentStateTable")
    print("=" * 60)
    print()
    
    database_url = DATABASE_URL
    if not database_url:
        print("❌ DATABASE_URL not found in config")
        sys.exit(1)
    
    # Parse database URL
    if "postgresql+asyncpg://" in database_url:
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    
    if "sqlite://" in database_url:
        print("⚠️  SQLite detected. This script is for PostgreSQL only.")
        print("   For SQLite, checkpoints table is not used.")
        sys.exit(0)
    
    if "postgresql://" not in database_url:
        print("❌ Invalid DATABASE_URL. Expected PostgreSQL connection string.")
        sys.exit(1)
    
    try:
        # Parse and connect
        parsed = urlparse(database_url)
        # Decode URL-encoded password
        password = unquote(parsed.password) if parsed.password else None
        
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path[1:] if parsed.path.startswith('/') else parsed.path,  # Remove leading /
            user=parsed.username,
            password=password,
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if checkpoints table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'checkpoints'
            );
        """)
        exists = cur.fetchone()[0]
        
        # Remove all checkpoint-related tables
        checkpoint_tables = ['checkpoints', 'checkpoint_blobs', 'checkpoint_migrations', 'checkpoint_writes']
        removed_tables = []
        
        for table_name in checkpoint_tables:
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table_name}'
                );
            """)
            exists = cur.fetchone()[0]
            
            if exists:
                print(f"🗑️  Dropping {table_name} table...")
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                removed_tables.append(table_name)
                print(f"✅ {table_name} table dropped successfully!")
            else:
                print(f"ℹ️  {table_name} table does not exist (already removed)")
        
        if removed_tables:
            print(f"\n✅ Removed {len(removed_tables)} checkpoint-related table(s)")
        else:
            print("\n✅ All checkpoint tables already removed")
        
        # Check if agent_state table exists
        print("\n📋 Checking agent_state table...")
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'agent_state'
            );
        """)
        agent_state_exists = cur.fetchone()[0]
        
        if agent_state_exists:
            print("✅ agent_state table exists")
            
            # Show table structure
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'agent_state'
                ORDER BY ordinal_position;
            """)
            columns = cur.fetchall()
            print("\n📊 agent_state table structure:")
            for col in columns:
                nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                print(f"   - {col[0]}: {col[1]} ({nullable})")
        else:
            print("⚠️  agent_state table does not exist")
            print("   It will be created automatically when the server starts")
        
        # List all tables
        print("\n📋 Current tables in database:")
        cur.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            ORDER BY tablename;
        """)
        tables = cur.fetchall()
        for table in tables:
            print(f"   - {table[0]}")
        
        cur.close()
        conn.close()
        
        print("\n✅ Cleanup complete!")
        print("\n📝 Summary:")
        print("   - checkpoints table: REMOVED")
        print("   - agent_state table: READY")
        print("   - System now uses only AgentStateTable for state management")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    cleanup_checkpoints()

