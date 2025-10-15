-- ========================================================================
-- ADD PDF STORAGE TABLE
-- Run this in Supabase SQL Editor to add PDF storage functionality
-- ========================================================================

-- Create user_pdfs table
CREATE TABLE IF NOT EXISTS user_pdfs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    pdf_text TEXT NOT NULL,
    ai_summary TEXT,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_user_pdfs_user_id ON user_pdfs(user_id);
CREATE INDEX IF NOT EXISTS idx_user_pdfs_uploaded_at ON user_pdfs(uploaded_at DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE user_pdfs ENABLE ROW LEVEL SECURITY;

-- RLS Policies
DROP POLICY IF EXISTS "Users can view their own PDFs" ON user_pdfs;
CREATE POLICY "Users can view their own PDFs" 
    ON user_pdfs FOR SELECT 
    USING (true);

DROP POLICY IF EXISTS "Users can insert their own PDFs" ON user_pdfs;
CREATE POLICY "Users can insert their own PDFs" 
    ON user_pdfs FOR INSERT 
    WITH CHECK (true);

DROP POLICY IF EXISTS "Users can update their own PDFs" ON user_pdfs;
CREATE POLICY "Users can update their own PDFs" 
    ON user_pdfs FOR UPDATE 
    USING (true);

DROP POLICY IF EXISTS "Users can delete their own PDFs" ON user_pdfs;
CREATE POLICY "Users can delete their own PDFs" 
    ON user_pdfs FOR DELETE 
    USING (true);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_user_pdfs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_user_pdfs_timestamp ON user_pdfs;
CREATE TRIGGER update_user_pdfs_timestamp
    BEFORE UPDATE ON user_pdfs
    FOR EACH ROW
    EXECUTE FUNCTION update_user_pdfs_updated_at();

-- Verify table was created
SELECT 'PDF storage table created successfully!' as status;
SELECT * FROM user_pdfs LIMIT 0;

