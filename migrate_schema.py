#!/usr/bin/env python3
"""
Migration script to:
1. Drop ChatQuery table
2. Create UserChat table
3. Update AgentStateTable schema (change timestamps from VARCHAR to TIMESTAMP)
"""
import os
import sys
import psycopg2
from urllib.parse import urlparse, unquote

# Load config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.config import DATABASE_URL

def migrate_schema():
    """Migrate database schema."""
    print("=" * 60)
    print("Database Schema Migration")
    print("=" * 60)
    print()
    
    database_url = DATABASE_URL
    if not database_url:
        print("❌ DATABASE_URL not found in config")
        sys.exit(1)
    
    if "sqlite://" in database_url:
        print("⚠️  SQLite detected. This script is for PostgreSQL only.")
        sys.exit(0)
    
    if "postgresql://" not in database_url:
        print("❌ Invalid DATABASE_URL. Expected PostgreSQL connection string.")
        sys.exit(1)
    
    try:
        # Parse and connect
        parsed = urlparse(database_url)
        password = unquote(parsed.password) if parsed.password else None
        
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path[1:] if parsed.path.startswith('/') else parsed.path,
            user=parsed.username,
            password=password,
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # 1. Drop ChatQuery table
        print("🗑️  Dropping ChatQuery table...")
        cur.execute("DROP TABLE IF EXISTS chatquery CASCADE;")
        print("✅ ChatQuery table dropped")
        
        # 2. Create UserChat table
        print("\n📋 Creating UserChat table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.user_chat (
                id SERIAL PRIMARY KEY,
                username VARCHAR NOT NULL,
                chat_room_id VARCHAR NOT NULL,
                user_id VARCHAR NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
            );
        """)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS ix_user_chat_username ON public.user_chat(username);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_user_chat_chat_room_id ON public.user_chat(chat_room_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_user_chat_user_id ON public.user_chat(user_id);")
        
        print("✅ UserChat table created")
        
        # 3. Update AgentStateTable timestamps (if they're VARCHAR, convert to TIMESTAMP)
        print("\n📋 Checking AgentStateTable schema...")
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns
            WHERE table_name = 'agent_state' 
            AND column_name IN ('created_at', 'updated_at');
        """)
        timestamp_cols = cur.fetchall()
        
        if timestamp_cols:
            for col_name, data_type in timestamp_cols:
                if data_type == 'character varying':
                    print(f"   Converting {col_name} from VARCHAR to TIMESTAMP...")
                    # First, set a default for existing NULL values, then alter column
                    cur.execute(f"""
                        ALTER TABLE public.agent_state 
                        ALTER COLUMN {col_name} TYPE TIMESTAMP WITH TIME ZONE 
                        USING CASE 
                            WHEN {col_name} IS NULL THEN NOW()
                            ELSE {col_name}::TIMESTAMP WITH TIME ZONE
                        END;
                    """)
                    cur.execute(f"""
                        ALTER TABLE public.agent_state 
                        ALTER COLUMN {col_name} SET DEFAULT NOW();
                    """)
                    cur.execute(f"""
                        ALTER TABLE public.agent_state 
                        ALTER COLUMN {col_name} SET NOT NULL;
                    """)
                    print(f"   ✅ {col_name} converted to TIMESTAMP")
                else:
                    print(f"   ℹ️  {col_name} is already {data_type}")
        
        # Verify tables
        print("\n📋 Current tables:")
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
        
        print("\n✅ Migration complete!")
        print("\n📝 Summary:")
        print("   - ChatQuery table: REMOVED")
        print("   - UserChat table: CREATED")
        print("   - AgentStateTable timestamps: UPDATED")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    migrate_schema()

