-- ============================================================================
-- RESET DATABASE - Fix Constraint and Delete All Data
-- ============================================================================
-- This script will:
-- 1. Fix the stock_master constraint (UNIQUE ticker instead of UNIQUE ticker+name)
-- 2. Delete ALL data (transactions, holdings, prices, stock_master, users, portfolios)
-- 3. Keep table structure intact
-- 4. Verify everything is clean and ready
--
-- After running this:
-- - All duplicate ticker issues will be fixed
-- - Tables are ready for fresh data
-- - You can recreate users and upload files
-- - No duplicates will be created (enforced by constraint)
-- ============================================================================

BEGIN;

-- ============================================================================
-- STEP 1: FIX DATABASE CONSTRAINT
-- Change from UNIQUE(ticker, stock_name) to UNIQUE(ticker)
-- ============================================================================

-- Remove the old constraint that allows duplicate tickers with different names
ALTER TABLE stock_master 
DROP CONSTRAINT IF EXISTS stock_master_ticker_stock_name_key;

-- Add new constraint: ticker must be unique (one ticker = one stock)
ALTER TABLE stock_master 
ADD CONSTRAINT stock_master_ticker_unique UNIQUE(ticker);

-- Verify constraint was added
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 
        FROM information_schema.table_constraints 
        WHERE table_name = 'stock_master' 
        AND constraint_name = 'stock_master_ticker_unique'
    ) THEN
        RAISE NOTICE '✅ STEP 1: Constraint fixed - ticker is now UNIQUE';
    ELSE
        RAISE EXCEPTION '❌ STEP 1 FAILED: Could not add constraint';
    END IF;
END $$;

-- ============================================================================
-- STEP 2: DELETE ALL DATA
-- Delete in correct order to respect foreign key constraints
-- ============================================================================

-- 1. Delete historical prices first (references stock_master)
DELETE FROM historical_prices;

-- 2. Delete user transactions (references stock_master, users, portfolios)
DELETE FROM user_transactions;

-- 3. Delete holdings (references stock_master, users, portfolios)
DELETE FROM holdings;

-- 4. Delete stock_master (now safe, nothing references it)
DELETE FROM stock_master;

-- 5. Delete user chat history (references users)
DELETE FROM user_chat_history;

-- 6. Delete user PDFs (references users)
DELETE FROM user_pdfs;

-- 7. Delete file uploads (references users)
DELETE FROM file_uploads;

-- 8. Delete portfolios (references users)
DELETE FROM portfolios;

-- 9. Delete users (last, nothing references it)
DELETE FROM users;

-- ============================================================================
-- STEP 3: VERIFY DELETION
-- ============================================================================

DO $$
DECLARE
    stock_count INTEGER;
    holding_count INTEGER;
    tx_count INTEGER;
    price_count INTEGER;
    user_count INTEGER;
    portfolio_count INTEGER;
BEGIN
    -- Count remaining records
    SELECT COUNT(*) INTO stock_count FROM stock_master;
    SELECT COUNT(*) INTO holding_count FROM holdings;
    SELECT COUNT(*) INTO tx_count FROM user_transactions;
    SELECT COUNT(*) INTO price_count FROM historical_prices;
    SELECT COUNT(*) INTO user_count FROM users;
    SELECT COUNT(*) INTO portfolio_count FROM portfolios;
    
    -- Verify all data is deleted
    IF stock_count = 0 AND holding_count = 0 AND tx_count = 0 
       AND price_count = 0 AND user_count = 0 AND portfolio_count = 0 THEN
        RAISE NOTICE '✅ STEP 3: Verification passed - All data deleted';
        RAISE NOTICE '   - stock_master: % records', stock_count;
        RAISE NOTICE '   - holdings: % records', holding_count;
        RAISE NOTICE '   - user_transactions: % records', tx_count;
        RAISE NOTICE '   - historical_prices: % records', price_count;
        RAISE NOTICE '   - users: % records', user_count;
        RAISE NOTICE '   - portfolios: % records', portfolio_count;
    ELSE
        RAISE EXCEPTION '❌ STEP 3 FAILED: Data still exists. stocks: %, holdings: %, transactions: %, prices: %, users: %, portfolios: %', 
            stock_count, holding_count, tx_count, price_count, user_count, portfolio_count;
    END IF;
END $$;

-- ============================================================================
-- STEP 4: VERIFY CONSTRAINT
-- ============================================================================

DO $$
DECLARE
    constraint_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 
        FROM information_schema.table_constraints 
        WHERE table_name = 'stock_master' 
        AND constraint_name = 'stock_master_ticker_unique'
        AND constraint_type = 'UNIQUE'
    ) INTO constraint_exists;
    
    IF constraint_exists THEN
        RAISE NOTICE '✅ STEP 4: Constraint verified - stock_master.ticker is UNIQUE';
    ELSE
        RAISE EXCEPTION '❌ STEP 4 FAILED: Constraint verification failed';
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES (Run separately to confirm)
-- ============================================================================

-- Check all table counts (all should be 0)
SELECT 
    'stock_master' as table_name, COUNT(*) as record_count FROM stock_master
UNION ALL
SELECT 'holdings', COUNT(*) FROM holdings
UNION ALL
SELECT 'user_transactions', COUNT(*) FROM user_transactions
UNION ALL
SELECT 'historical_prices', COUNT(*) FROM historical_prices
UNION ALL
SELECT 'users', COUNT(*) FROM users
UNION ALL
SELECT 'portfolios', COUNT(*) FROM portfolios;

-- Verify constraint exists
SELECT 
    constraint_name, 
    constraint_type,
    table_name
FROM information_schema.table_constraints 
WHERE table_name = 'stock_master' 
AND constraint_type = 'UNIQUE';

