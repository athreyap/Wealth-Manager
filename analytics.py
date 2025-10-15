"""
Portfolio analytics and performance calculations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import streamlit as st

class PortfolioAnalytics:
    """Calculates portfolio performance, P&L, and analytics"""
    
    def __init__(self, holdings: List[Dict[str, Any]], transactions: List[Dict[str, Any]]):
        self.holdings = holdings
        self.transactions = transactions
    
    def calculate_portfolio_summary(self) -> Dict[str, Any]:
        """Calculate overall portfolio summary"""
        total_investment = 0
        current_value = 0
        total_realized_pnl = 0
        
        # Calculate from holdings
        for holding in self.holdings:
            try:
                quantity = float(holding.get('total_quantity', 0))
                avg_price = float(holding.get('average_price', 0))
                
                # Handle None current_price - use avg_price as fallback
                current_price_raw = holding.get('current_price')
                if current_price_raw is None or current_price_raw == 0:
                    current_price = avg_price
                else:
                    current_price = float(current_price_raw)
                
                investment = quantity * avg_price
                current = quantity * current_price
                
                total_investment += investment
                current_value += current
            except (ValueError, TypeError) as e:
                # Skip holdings with invalid data
                continue
        
        # Calculate realized P&L from transactions
        total_realized_pnl = self._calculate_realized_pnl()
        
        # Calculate unrealized P&L
        unrealized_pnl = current_value - total_investment
        
        # Calculate total P&L
        total_pnl = unrealized_pnl + total_realized_pnl
        
        # Calculate returns
        total_return_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        return {
            'total_investment': round(total_investment, 2),
            'current_value': round(current_value, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'realized_pnl': round(total_realized_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_holdings': len(self.holdings)
        }
    
    def calculate_holding_details(self) -> pd.DataFrame:
        """Calculate detailed metrics for each holding"""
        holdings_data = []
        
        for holding in self.holdings:
            try:
                quantity = float(holding.get('total_quantity', 0))
                avg_price = float(holding.get('average_price', 0))
                
                # Handle None current_price
                current_price_raw = holding.get('current_price')
                if current_price_raw is None or current_price_raw == 0:
                    current_price = avg_price
                else:
                    current_price = float(current_price_raw)
                
                investment = quantity * avg_price
                current_value = quantity * current_price
                pnl = current_value - investment
                pnl_pct = (pnl / investment * 100) if investment > 0 else 0
                
                holdings_data.append({
                    'Asset Name': holding['asset_name'],
                    'Symbol': holding['asset_symbol'],
                    'Type': holding['asset_type'].replace('_', ' ').title(),
                    'Quantity': quantity,
                    'Avg Price': round(avg_price, 2),
                    'Current Price': round(current_price, 2),
                    'Investment': round(investment, 2),
                    'Current Value': round(current_value, 2),
                    'P&L': round(pnl, 2),
                    'P&L %': round(pnl_pct, 2)
                })
            except (ValueError, TypeError) as e:
                # Skip holdings with invalid data
                continue
        
        df = pd.DataFrame(holdings_data)
        
        if not df.empty:
            # Sort by current value descending
            df = df.sort_values('Current Value', ascending=False)
        
        return df
    
    def calculate_asset_allocation(self) -> pd.DataFrame:
        """Calculate asset allocation by type"""
        allocation = {}
        
        for holding in self.holdings:
            try:
                asset_type = holding['asset_type'].replace('_', ' ').title()
                quantity = float(holding.get('total_quantity', 0))
                avg_price = float(holding.get('average_price', 0))
                
                # Handle None current_price
                current_price_raw = holding.get('current_price')
                if current_price_raw is None or current_price_raw == 0:
                    current_price = avg_price
                else:
                    current_price = float(current_price_raw)
                
                value = quantity * current_price
                
                if asset_type in allocation:
                    allocation[asset_type] += value
                else:
                    allocation[asset_type] = value
            except (ValueError, TypeError):
                # Skip holdings with invalid data
                continue
        
        total = sum(allocation.values())
        
        allocation_data = []
        for asset_type, value in allocation.items():
            percentage = (value / total * 100) if total > 0 else 0
            allocation_data.append({
                'Asset Type': asset_type,
                'Value': round(value, 2),
                'Percentage': round(percentage, 2)
            })
        
        df = pd.DataFrame(allocation_data)
        
        if not df.empty:
            df = df.sort_values('Value', ascending=False)
        
        return df
    
    def calculate_top_performers(self, top_n: int = 5) -> Dict[str, pd.DataFrame]:
        """Calculate top gainers and losers"""
        holdings_df = self.calculate_holding_details()
        
        if holdings_df.empty:
            return {
                'top_gainers': pd.DataFrame(),
                'top_losers': pd.DataFrame()
            }
        
        # Sort by P&L percentage
        sorted_by_pnl = holdings_df.sort_values('P&L %', ascending=False)
        
        top_gainers = sorted_by_pnl.head(top_n)
        top_losers = sorted_by_pnl.tail(top_n)
        
        return {
            'top_gainers': top_gainers,
            'top_losers': top_losers
        }
    
    def _calculate_realized_pnl(self) -> float:
        """Calculate realized P&L from sell transactions"""
        realized_pnl = 0
        
        # Group transactions by asset
        asset_transactions = {}
        
        for trans in self.transactions:
            symbol = trans['asset_symbol']
            if symbol not in asset_transactions:
                asset_transactions[symbol] = []
            asset_transactions[symbol].append(trans)
        
        # Calculate realized P&L for each asset
        for symbol, transactions in asset_transactions.items():
            # Sort by date
            sorted_trans = sorted(transactions, key=lambda x: x['transaction_date'])
            
            # FIFO calculation
            buy_queue = []
            
            for trans in sorted_trans:
                if trans['transaction_type'] == 'buy':
                    buy_queue.append({
                        'quantity': float(trans['quantity']),
                        'price': float(trans['price'])
                    })
                else:  # sell
                    sell_qty = float(trans['quantity'])
                    sell_price = float(trans['price'])
                    
                    while sell_qty > 0 and buy_queue:
                        buy = buy_queue[0]
                        
                        if buy['quantity'] <= sell_qty:
                            # Use entire buy lot
                            pnl = (sell_price - buy['price']) * buy['quantity']
                            realized_pnl += pnl
                            sell_qty -= buy['quantity']
                            buy_queue.pop(0)
                        else:
                            # Partial buy lot
                            pnl = (sell_price - buy['price']) * sell_qty
                            realized_pnl += pnl
                            buy['quantity'] -= sell_qty
                            sell_qty = 0
        
        return realized_pnl
    
    def calculate_monthly_performance(self) -> pd.DataFrame:
        """Calculate month-wise performance"""
        if not self.transactions:
            return pd.DataFrame()
        
        # Convert transactions to DataFrame
        trans_df = pd.DataFrame(self.transactions)
        trans_df['transaction_date'] = pd.to_datetime(trans_df['transaction_date'])
        trans_df['month'] = trans_df['transaction_date'].dt.to_period('M')
        
        # Group by month
        monthly_data = []
        
        for month, group in trans_df.groupby('month'):
            buys = group[group['transaction_type'] == 'buy']
            sells = group[group['transaction_type'] == 'sell']
            
            buy_value = (buys['quantity'].astype(float) * buys['price'].astype(float)).sum()
            sell_value = (sells['quantity'].astype(float) * sells['price'].astype(float)).sum()
            
            monthly_data.append({
                'Month': str(month),
                'Buy Value': round(buy_value, 2),
                'Sell Value': round(sell_value, 2),
                'Net Flow': round(buy_value - sell_value, 2)
            })
        
        df = pd.DataFrame(monthly_data)
        
        if not df.empty:
            df = df.sort_values('Month', ascending=False)
        
        return df
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate advanced portfolio metrics"""
        if not self.holdings:
            return {}
        
        holdings_df = self.calculate_holding_details()
        
        if holdings_df.empty:
            return {}
        
        # Calculate concentration (top 5 holdings %)
        total_value = holdings_df['Current Value'].sum()
        top_5_value = holdings_df.head(5)['Current Value'].sum()
        concentration = (top_5_value / total_value * 100) if total_value > 0 else 0
        
        # Calculate weighted average return
        holdings_df['weight'] = holdings_df['Current Value'] / total_value
        weighted_return = (holdings_df['P&L %'] * holdings_df['weight']).sum()
        
        return {
            'concentration_top5': round(concentration, 2),
            'weighted_avg_return': round(weighted_return, 2),
            'total_holdings_count': len(self.holdings)
        }

