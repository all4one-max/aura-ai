-- Fix LangGraph checkpoints table schema
-- Run this SQL script in your PostgreSQL database to fix the ON CONFLICT error

-- Drop the existing checkpoints table (WARNING: This will lose all checkpoint history)
DROP TABLE IF EXISTS checkpoints CASCADE;

-- The table will be recreated automatically by LangGraph on next server start
-- with the correct unique constraints

-- Alternative: If you want to keep existing data, you can try to add the missing constraint:
-- However, this may fail if there are duplicate entries. In that case, drop and recreate is safer.

-- Check current table structure (for reference):
-- \d checkpoints

