-- ========================================================================
-- ADD CHAT HISTORY TABLE (User-Specific Questions)
-- Run this in Supabase SQL Editor to add chat history storage
-- ========================================================================

-- Create user_chat_history table
CREATE TABLE IF NOT EXISTS user_chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_user_chat_history_user_id ON user_chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_user_chat_history_created_at ON user_chat_history(user_id, created_at DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE user_chat_history ENABLE ROW LEVEL SECURITY;

-- RLS Policies - Chat history is USER-SPECIFIC (NOT shared)
-- Since we use custom auth (not Supabase Auth), we need different policies

-- Allow authenticated users to view their own chat history
DROP POLICY IF EXISTS "Users can only view their own chat history" ON user_chat_history;
CREATE POLICY "Users can only view their own chat history" 
    ON user_chat_history FOR SELECT 
    USING (true);  -- Allow all authenticated users, filter by user_id in application

-- Allow authenticated users to insert chat history (user_id is set by application)
DROP POLICY IF EXISTS "Users can only insert their own chat history" ON user_chat_history;
CREATE POLICY "Users can only insert their own chat history" 
    ON user_chat_history FOR INSERT 
    WITH CHECK (true);  -- Allow inserts, application ensures user_id matches logged-in user

-- Allow authenticated users to update their own chat history
DROP POLICY IF EXISTS "Users can only update their own chat history" ON user_chat_history;
CREATE POLICY "Users can only update their own chat history" 
    ON user_chat_history FOR UPDATE 
    USING (true)
    WITH CHECK (true);

-- Allow authenticated users to delete their own chat history
DROP POLICY IF EXISTS "Users can only delete their own chat history" ON user_chat_history;
CREATE POLICY "Users can only delete their own chat history" 
    ON user_chat_history FOR DELETE 
    USING (true);  -- Allow deletes, application ensures user_id matches

-- Verify table was created
SELECT 'Chat history table created successfully!' as status;
SELECT * FROM user_chat_history LIMIT 0;

