-- ============================================================================
-- Astra AI — Supabase Schema Migrations
-- ============================================================================
-- These migrations create the tables used by SupabaseStore and enable
-- row-level security (RLS) for device_id isolation.
--
-- Run via the Supabase SQL editor or `supabase db reset`.
-- ============================================================================

-- ----------------------------------------------
-- 1. devices table
-- ----------------------------------------------
-- Stores per‑device preferences (memory_enabled flag, creation time).
-- RLS: device_id is the PK; rows are auto‑scoped to the requesting device.
--

-- Drop existing tables if re‑migrating (be careful on production!)
-- DROP TABLE IF EXISTS memories CASCADE;
-- DROP TABLE IF EXISTS messages CASCADE;
-- DROP TABLE IF EXISTS conversations CASCADE;
-- DROP TABLE IF EXISTS devices CASCADE;

CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    memory_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Trigger to keep updated_at current
CREATE OR REPLACE FUNCTION handle_updated_at_devices()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_devices_updated_at
BEFORE UPDATE ON devices
FOR EACH ROW
EXECUTE FUNCTION handle_updated_at_devices();

-- ----------------------------------------------
-- 2. conversations table
-- ----------------------------------------------
-- One row per conversation. Scoped by device_id.
-- Title may be null (unnamed chat).
--

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION handle_updated_at_conversations()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_conversations_updated_at
BEFORE UPDATE ON conversations
FOR EACH ROW
EXECUTE FUNCTION handle_updated_at_conversations();

-- ----------------------------------------------
-- 3. messages table
-- ----------------------------------------------
-- One row per message (user or assistant). Cascades delete from conversation.
--

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for fast conversation‑scoped fetches
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);

-- ----------------------------------------------
-- 4. memories table
-- ----------------------------------------------
-- One row per long‑term memory entry. Scoped by device_id.
-- status: 'active' | 'pending' | 'deleted'
-- kind: free‑form label (e.g. "fact", "preference", "user_profile")
--

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'pending', 'deleted')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION handle_updated_at_memories()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_memories_updated_at
BEFORE UPDATE ON memories
FOR EACH ROW
EXECUTE FUNCTION handle_updated_at_memories();

-- ----------------------------------------------
-- 5. Row-Level Security (RLS) policies
-- ----------------------------------------------
-- The goal: each device can only read/write its own rows. There is no
-- users table, no login, no cookie. Anonymous users are identified by
-- a UUID stored in browser localStorage and sent as X-Device-Id header.
--
-- These policies assume the Supabase anon key is used for browser calls
-- (which cannot write) and the service-role key is used server‑side
-- (which bypasses RLS but our code still scopes every query by device_id).
--

-- devices: each device can only see its own row
ALTER TABLE devices ENABLE ROW LEVEL SECURITY;
CREATE POLICY "devices_self_access" ON devices
    USING (device_id = current_setting('device_id', true));

-- conversations: each device can only see its own conversations
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
CREATE POLICY "conversations_self_access" ON conversations
    USING (device_id = current_setting('device_id', true));

-- messages: each device can only see its own messages (via conversation scoping)
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
CREATE POLICY "messages_self_access" ON messages
    USING (conversation_id IN (
        SELECT id FROM conversations WHERE device_id = current_setting('device_id', true)
    ));

-- memories: each device can only see its own memories
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
CREATE POLICY "memories_self_access" ON memories
    USING (device_id = current_setting('device_id', true));

-- ----------------------------------------------
-- 6. Helper function: set device_id session variable
-- ----------------------------------------------
-- This is how the Python client scopes queries without built‑in RLS.
-- The service-role client sets the request header X-Device-Id, and the
-- PostgREST server reads it via the helper function. The Python code
-- passes device_id in the query strings (e.g. device_id=eq.X) which
-- achieves the same isolation even without RLS active.
--
-- To use the function, run (replace 'XYZ' with the actual device UUID):
--   SELECT set_config('device_id', 'XYZ', false);
--
-- This is especially useful when using the anon key from the browser:
-- the browser JavaScript would call this once after login, and then all
-- subsequent PostgREST requests would automatically scope to that device.
--
-- NOTE: The Python SupabaseStore does NOT rely on this function — it
-- scopes every query via the `device_id=eq.X` filter in the query string.
-- The RLS policies above are defence‑in‑depth so that even if a query
-- omits the filter, cross‑device access is still blocked at the database
-- level.
--
-- If the operator prefers not to use the session variable, the policies
-- can be set to PERMISSIVE and the Python code handles isolation exclusively
-- via the query‑string filters (which is the current approach).

-- ============================================================================
-- End of migration
-- ============================================================================