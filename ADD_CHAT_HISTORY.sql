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
DROP POLICY IF EXISTS "Users can only view their own chat history" ON user_chat_history;
CREATE POLICY "Users can only view their own chat history" 
    ON user_chat_history FOR SELECT 
    USING (auth.uid() = user_id);  -- Users can only see their own questions

DROP POLICY IF EXISTS "Users can only insert their own chat history" ON user_chat_history;
CREATE POLICY "Users can only insert their own chat history" 
    ON user_chat_history FOR INSERT 
    WITH CHECK (auth.uid() = user_id);  -- Users can only add their own questions

DROP POLICY IF EXISTS "Users can only update their own chat history" ON user_chat_history;
CREATE POLICY "Users can only update their own chat history" 
    ON user_chat_history FOR UPDATE 
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can only delete their own chat history" ON user_chat_history;
CREATE POLICY "Users can only delete their own chat history" 
    ON user_chat_history FOR DELETE 
    USING (auth.uid() = user_id);  -- Users can only delete their own questions

-- Verify table was created
SELECT 'Chat history table created successfully!' as status;
SELECT * FROM user_chat_history LIMIT 0;

