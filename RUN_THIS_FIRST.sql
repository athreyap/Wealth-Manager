-- ============================================================================
-- STEP 1 + STEP 2 COMBINED: Clean & Create New Shared Architecture
-- Copy this ENTIRE file and run in Supabase SQL Editor
-- ============================================================================

-- ============================================================================
-- STEP 1: CLEAN EVERYTHING
-- ============================================================================

-- Drop all tables
DROP TABLE IF EXISTS file_uploads CASCADE;
DROP TABLE IF EXISTS historical_prices CASCADE;
DROP TABLE IF EXISTS price_history CASCADE;
DROP TABLE IF EXISTS holdings CASCADE;
DROP TABLE IF EXISTS user_transactions CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS stock_master CASCADE;
DROP TABLE IF EXISTS portfolios CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Drop views
DROP VIEW IF EXISTS user_holdings_detailed CASCADE;
DROP VIEW IF EXISTS user_transactions_detailed CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS get_or_create_stock(TEXT, TEXT, TEXT, TEXT) CASCADE;
DROP FUNCTION IF EXISTS update_stock_live_price(UUID, DECIMAL) CASCADE;

-- Drop triggers
DROP TRIGGER IF EXISTS update_holdings_updated_at ON holdings CASCADE;

RAISE NOTICE '✅ Step 1: Cleanup complete!';

-- ============================================================================
-- STEP 2: CREATE NEW SHARED ARCHITECTURE
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USER MANAGEMENT
-- ============================================================================

CREATE TABLE users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE portfolios (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    portfolio_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- SHARED STOCK/ASSET MASTER DATA (NEW!)
-- ============================================================================

CREATE TABLE stock_master (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    sector TEXT,
    live_price DECIMAL,
    last_updated TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(ticker, stock_name)
);

CREATE INDEX idx_stock_master_ticker ON stock_master(ticker);
CREATE INDEX idx_stock_master_type ON stock_master(asset_type);

-- ============================================================================
-- SHARED HISTORICAL PRICES (NEW!)
-- ============================================================================

CREATE TABLE historical_prices (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    stock_id UUID NOT NULL REFERENCES stock_master(id) ON DELETE CASCADE,
    price_date DATE NOT NULL,
    price DECIMAL NOT NULL,
    volume BIGINT,
    source TEXT,
    iso_year INTEGER,
    iso_week INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(stock_id, price_date)
);

CREATE INDEX idx_historical_prices_stock ON historical_prices(stock_id);
CREATE INDEX idx_historical_prices_date ON historical_prices(price_date);
CREATE INDEX idx_historical_prices_week ON historical_prices(iso_year, iso_week);

-- ============================================================================
-- USER TRANSACTIONS (UPDATED!)
-- ============================================================================

CREATE TABLE user_transactions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    stock_id UUID NOT NULL REFERENCES stock_master(id) ON DELETE CASCADE,
    quantity DECIMAL NOT NULL,
    price DECIMAL NOT NULL,
    transaction_date DATE NOT NULL,
    transaction_type TEXT NOT NULL,
    channel TEXT,
    notes TEXT,
    -- Week tracking (as per your image requirements)
    iso_year INTEGER,
    iso_week INTEGER,
    week_label TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_user_transactions_user ON user_transactions(user_id);
CREATE INDEX idx_user_transactions_portfolio ON user_transactions(portfolio_id);
CREATE INDEX idx_user_transactions_stock ON user_transactions(stock_id);
CREATE INDEX idx_user_transactions_date ON user_transactions(transaction_date);
CREATE INDEX idx_user_transactions_week ON user_transactions(iso_year, iso_week);

-- ============================================================================
-- HOLDINGS (UPDATED!)
-- ============================================================================

CREATE TABLE holdings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    stock_id UUID NOT NULL REFERENCES stock_master(id) ON DELETE CASCADE,
    total_quantity DECIMAL NOT NULL,
    average_price DECIMAL NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, portfolio_id, stock_id)
);

CREATE INDEX idx_holdings_user ON holdings(user_id);
CREATE INDEX idx_holdings_portfolio ON holdings(portfolio_id);
CREATE INDEX idx_holdings_stock ON holdings(stock_id);

-- ============================================================================
-- FILE UPLOADS
-- ============================================================================

CREATE TABLE file_uploads (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER,
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_status TEXT DEFAULT 'pending',
    error_message TEXT
);

-- ============================================================================
-- VIEWS FOR EASY QUERYING
-- ============================================================================

CREATE OR REPLACE VIEW user_holdings_detailed AS
SELECT 
    h.id,
    h.user_id,
    h.portfolio_id,
    h.total_quantity,
    h.average_price,
    h.last_updated,
    sm.id as stock_id,
    sm.ticker,
    sm.stock_name,
    sm.asset_type,
    sm.sector,
    sm.live_price AS current_price
FROM holdings h
JOIN stock_master sm ON h.stock_id = sm.id;

CREATE OR REPLACE VIEW user_transactions_detailed AS
SELECT 
    ut.id,
    ut.user_id,
    ut.portfolio_id,
    ut.quantity,
    ut.price,
    ut.transaction_date,
    ut.transaction_type,
    ut.channel,
    ut.notes,
    sm.id as stock_id,
    sm.ticker,
    sm.stock_name,
    sm.asset_type,
    sm.sector
FROM user_transactions ut
JOIN stock_master sm ON ut.stock_id = sm.id;

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('users', 'portfolios', 'stock_master', 'historical_prices', 'user_transactions', 'holdings', 'file_uploads');
    
    IF table_count = 7 THEN
        RAISE NOTICE '';
        RAISE NOTICE '=================================================================';
        RAISE NOTICE '✅ SUCCESS: All 7 tables created!';
        RAISE NOTICE '=================================================================';
        RAISE NOTICE '';
        RAISE NOTICE 'Tables created:';
        RAISE NOTICE '  ✅ users';
        RAISE NOTICE '  ✅ portfolios';
        RAISE NOTICE '  ✅ stock_master (SHARED)';
        RAISE NOTICE '  ✅ historical_prices (SHARED)';
        RAISE NOTICE '  ✅ user_transactions';
        RAISE NOTICE '  ✅ holdings';
        RAISE NOTICE '  ✅ file_uploads';
        RAISE NOTICE '';
        RAISE NOTICE 'Views created:';
        RAISE NOTICE '  ✅ user_holdings_detailed';
        RAISE NOTICE '  ✅ user_transactions_detailed';
        RAISE NOTICE '';
        RAISE NOTICE '🎉 New shared architecture ready!';
        RAISE NOTICE '🚀 Now run: streamlit run app_complete.py';
        RAISE NOTICE '';
        RAISE NOTICE '=================================================================';
    ELSE
        RAISE NOTICE '⚠️  Only % of 7 tables created - something went wrong!', table_count;
    END IF;
END $$;

