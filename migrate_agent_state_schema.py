#!/usr/bin/env python3
"""
Migration script to update AgentStateTable schema:
1. Add individual field columns (messages, user_profile, search_results, etc.)
2. Migrate data from 'state' JSON column to individual columns
3. Optionally drop 'state' column (or keep it for backward compatibility)
"""
import os
import sys
import psycopg2
from urllib.parse import urlparse, unquote

# Load config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.config import DATABASE_URL

def migrate_agent_state_schema():
    """Migrate AgentStateTable to have individual field columns."""
    print("=" * 60)
    print("AgentStateTable Schema Migration")
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
        
        # Check current schema
        print("📋 Checking current AgentStateTable schema...")
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns
            WHERE table_name = 'agent_state'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        print("Current columns:")
        for col in columns:
            print(f"   - {col[0]}: {col[1]}")
        
        # Add new columns if they don't exist
        new_columns = [
            ("messages", "JSON"),
            ("user_profile", "JSON"),
            ("search_results", "JSON"),
            ("selected_item", "JSON"),
            ("chat_query_json", "JSON"),
            ("styled_products", "JSON"),
            ("ranked_products", "JSON"),
            ("merged_images", "JSON"),
        ]
        
        print("\n📋 Adding new columns...")
        for col_name, col_type in new_columns:
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_name = 'agent_state' 
                    AND column_name = '{col_name}'
                );
            """)
            exists = cur.fetchone()[0]
            
            if not exists:
                print(f"   Adding {col_name}...")
                cur.execute(f"""
                    ALTER TABLE public.agent_state 
                    ADD COLUMN {col_name} {col_type};
                """)
                print(f"   ✅ {col_name} added")
            else:
                print(f"   ℹ️  {col_name} already exists")
        
        # Migrate data from 'state' JSON column to individual columns if 'state' column exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'agent_state' 
                AND column_name = 'state'
            );
        """)
        has_state_column = cur.fetchone()[0]
        
        if has_state_column:
            print("\n📋 Migrating data from 'state' column to individual columns...")
            cur.execute("SELECT thread_id, state FROM agent_state WHERE state IS NOT NULL;")
            rows = cur.fetchall()
            
            migrated_count = 0
            for thread_id, state_json in rows:
                if state_json:
                    try:
                        import json
                        state_data = json.loads(state_json) if isinstance(state_json, str) else state_json
                        
                        # Update individual columns from state JSON
                        updates = []
                        if "messages" in state_data:
                            updates.append(f"messages = '{json.dumps(state_data['messages'])}'::json")
                        if "user_profile" in state_data:
                            updates.append(f"user_profile = '{json.dumps(state_data['user_profile'])}'::json")
                        if "search_results" in state_data:
                            updates.append(f"search_results = '{json.dumps(state_data['search_results'])}'::json")
                        if "selected_item" in state_data:
                            updates.append(f"selected_item = '{json.dumps(state_data['selected_item'])}'::json")
                        if "chat_query_json" in state_data:
                            updates.append(f"chat_query_json = '{json.dumps(state_data['chat_query_json'])}'::json")
                        if "styled_products" in state_data:
                            updates.append(f"styled_products = '{json.dumps(state_data['styled_products'])}'::json")
                        if "ranked_products" in state_data:
                            updates.append(f"ranked_products = '{json.dumps(state_data['ranked_products'])}'::json")
                        if "merged_images" in state_data:
                            updates.append(f"merged_images = '{json.dumps(state_data['merged_images'])}'::json")
                        
                        if updates:
                            update_sql = f"""
                                UPDATE agent_state 
                                SET {', '.join(updates)}
                                WHERE thread_id = %s;
                            """
                            cur.execute(update_sql, (thread_id,))
                            migrated_count += 1
                    except Exception as e:
                        print(f"   ⚠️  Error migrating thread_id {thread_id}: {e}")
            
            print(f"   ✅ Migrated {migrated_count} rows")
            print("\n   Note: 'state' column kept for backward compatibility")
            print("   You can drop it later if not needed: ALTER TABLE agent_state DROP COLUMN state;")
        
        # Verify final schema
        print("\n📋 Final AgentStateTable schema:")
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'agent_state'
            ORDER BY ordinal_position;
        """)
        final_columns = cur.fetchall()
        for col in final_columns:
            nullable = "NULL" if col[2] == "YES" else "NOT NULL"
            print(f"   - {col[0]}: {col[1]} ({nullable})")
        
        cur.close()
        conn.close()
        
        print("\n✅ Migration complete!")
        print("\n📝 Summary:")
        print("   - Added individual field columns")
        print("   - Migrated data from 'state' column (if existed)")
        print("   - AgentStateTable now stores fields separately")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    migrate_agent_state_schema()

