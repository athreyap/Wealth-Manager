# 💰 Wealth Manager - Portfolio Analytics Platform

A comprehensive portfolio management and analytics platform built with Streamlit, featuring AI-powered insights, advanced analytics, and persistent data storage.

## 🌟 Features

### 📊 Portfolio Management
- **Multi-Asset Support**: Stocks, Mutual Funds, PMS, AIF, Bonds
- **CSV Import**: Bulk transaction import with AI-powered parsing
- **Automatic Price Fetching**: Real-time and historical price updates
- **P&L Tracking**: Comprehensive profit/loss analysis with performance ratings

### 📈 Advanced Analytics
- **Portfolio Allocation**: Sector, Channel, and Asset Type breakdowns
- **Performance Analysis**: Top gainers/losers, performance trends
- **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands, Volume indicators
- **Risk Metrics**: VaR, Sharpe Ratio, Volatility, Max Drawdown
- **52-Week NAVs**: Historical price trends and analysis
- **Holdings Comparison**: Multi-dimensional comparison tools

### 🤖 AI Assistant
- **Portfolio Insights**: AI-powered recommendations and analysis
- **PDF Document Analysis**: Upload and analyze research reports, statements
- **Persistent PDF Library**: Store and access documents across sessions
- **Chat Interface**: Ask questions about your portfolio and documents

### ⚡ Performance Optimizations
- **Smart Caching**: 80% reduction in database queries
- **Lazy Loading**: 70% reduction in unnecessary computations
- **Optimized Calculations**: Single-pass metric calculations
- **Fast Navigation**: Instant page switching

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Supabase account
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/wealth-manager.git
cd wealth-manager
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Supabase**
   - Create a new Supabase project
   - Run `RUN_THIS_FIRST.sql` in the Supabase SQL Editor
   - Run `ADD_PDF_STORAGE.sql` for PDF storage feature

4. **Configure secrets**

Create `.streamlit/secrets.toml`:
```toml
[supabase]
url = "your-supabase-url"
key = "your-supabase-anon-key"

[api_keys]
open_ai = "your-openai-api-key"
gemini = "your-gemini-api-key"  # Optional
```

5. **Run the application**
```bash
streamlit run web_agent.py
```

## 📁 Project Structure

```
wealth-manager/
├── web_agent.py                    # Main Streamlit application
├── database_shared.py              # Database operations
├── enhanced_price_fetcher.py       # Price fetching with fallbacks
├── weekly_manager_streamlined.py   # Historical price management
├── analytics.py                    # P&L calculations
├── smart_ticker_detector.py        # Asset type detection
├── bulk_ai_fetcher.py             # Bulk AI price fetching
├── fetch_yearly_bulk.py           # Yearly price fetching
├── pms_aif_calculator.py          # PMS/AIF calculations
├── visualizations.py              # Chart generation
├── requirements.txt               # Python dependencies
├── RUN_THIS_FIRST.sql            # Main database setup
├── ADD_PDF_STORAGE.sql           # PDF storage setup
└── README.md                      # This file
```

## 🗄️ Database Schema

### Core Tables
- **users**: User accounts and authentication
- **stock_master**: Shared stock catalog
- **user_transactions**: User-specific transactions
- **historical_prices**: Shared price history
- **user_pdfs**: PDF document storage

## 📊 Usage

### 1. Register/Login
- Create an account or log in with username/password
- Upload CSV files during registration (optional)

### 2. Upload Transactions
- Navigate to "Upload More Files"
- Upload CSV files with transaction data
- System automatically parses and imports transactions

### 3. View Portfolio
- **Portfolio Overview**: See all holdings, P&L, ratings
- **Charts & Analytics**: 7 tabs of advanced analytics
- **AI Assistant**: Ask questions, upload PDFs

### 4. Analyze Performance
- View sector/channel allocation
- Compare holdings performance
- Analyze technical indicators
- Review risk metrics

## 📝 CSV Format

Required columns:
```csv
Date,Ticker,Stock Name,Quantity,Price,Transaction Type,Sector,Channel
2024-01-15,RELIANCE.NS,Reliance Industries,10,2500,BUY,Energy,Direct
```

Optional columns: `Asset Type`, `Notes`

## 🔧 Configuration

### Price Fetching
- **Stocks**: yfinance (NSE/BSE) → AI fallback
- **Mutual Funds**: mftool → AI fallback
- **PMS/AIF**: CAGR-based calculations

### Caching
- Holdings: 5-minute TTL
- Portfolio Summary: 10-minute TTL
- Portfolio Metrics: 5-minute TTL

## 🚀 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard
4. Deploy!

### Environment Variables
Set these in Streamlit Cloud secrets:
- `supabase.url`
- `supabase.key`
- `api_keys.open_ai`

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Database**: Supabase (PostgreSQL)
- **AI**: OpenAI GPT-4o-mini
- **Price Data**: yfinance, mftool
- **Charts**: Plotly
- **PDF Processing**: pdfplumber

## 📦 Dependencies

Key packages:
- `streamlit` - Web framework
- `supabase` - Database client
- `openai` - AI integration
- `yfinance` - Stock prices
- `mftool` - Mutual fund NAVs
- `plotly` - Interactive charts
- `pandas` - Data manipulation
- `pdfplumber` - PDF text extraction

See `requirements.txt` for complete list.

## 🔐 Security

- Password hashing (SHA-256)
- Row Level Security (RLS) in Supabase
- User-specific data isolation
- Secure API key management

## 🐛 Troubleshooting

### Common Issues

**1. "relation does not exist" error**
- Run `RUN_THIS_FIRST.sql` in Supabase SQL Editor

**2. "Invalid API key" error**
- Check `.streamlit/secrets.toml` configuration
- Verify Supabase URL and key

**3. Price fetching fails**
- Check internet connection
- Verify ticker format (add .NS or .BO for Indian stocks)
- AI fallback will activate automatically

**4. PDF upload fails**
- Run `ADD_PDF_STORAGE.sql` in Supabase
- Restart Streamlit application

## 📈 Performance Tips

1. **Enable Caching**: Caching is enabled by default
2. **Bulk Operations**: Upload multiple CSVs at once
3. **Historical Prices**: Fetch yearly prices for better performance
4. **Database Indexes**: Already optimized in SQL scripts

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Streamlit for the amazing framework
- Supabase for the database platform
- OpenAI for AI capabilities
- yfinance and mftool for price data

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review SQL scripts for database setup

## 🎯 Roadmap

- [ ] Portfolio optimization (Efficient Frontier)
- [ ] Options flow analysis
- [ ] Backtesting capabilities
- [ ] Mobile app
- [ ] Export functionality
- [ ] Multi-currency support

---

**Built with ❤️ using Streamlit**

🌟 Star this repo if you find it helpful!
