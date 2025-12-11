-- Remove LangGraph checkpoints table from PostgreSQL
-- This script drops the checkpoints table since we're now using only AgentStateTable

-- Drop the checkpoints table and all related objects
DROP TABLE IF EXISTS checkpoints CASCADE;

-- Verify the table is gone
SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = 'checkpoints';

-- If the above query returns no rows, the table has been successfully removed
-- You should only see the 'agent_state' table now

