"""
Streamlined Wealth Manager - Matches Your Image Requirements
- Register/Login with file upload
- Store files to DB and calculate historical from date in file based on week of year
- Fetch missing weeks till current week based on week of year
- Calculate P&L based on current week price
- Portfolio analysis based on sector/channel
- PMS CAGR when price mentioned, 52-week NAVs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
import functools
import csv
import importlib
import difflib
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from dateutil import parser as dateutil_parser
warnings.filterwarnings('ignore')

# Add current directory to Python path for AI agents import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also add the current working directory as a fallback
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# Import AI agents for insights and recommendations
AI_AGENTS_AVAILABLE = False
AI_FILE_EXTRACTION_ENABLED = True  # Unified uploader (Python-first with AI fallback)
AI_TICKER_RESOLUTION_ENABLED = True  # ENABLED - uses Gemini fallback (free/cheap)
PRICE_FETCHER_VERSION = 3

_preview_price_cache: Dict[Tuple[str, str, str], Optional[float]] = {}

AMFI_NAV_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"

try:
    _assistant_helper = importlib.import_module("assistant_helper")
    run_gpt5_completion = getattr(_assistant_helper, "run_gpt5_completion")
except Exception as exc:  # pragma: no cover - runtime dependency
    run_gpt5_completion = None  # type: ignore[assignment]
    AI_COMPLETION_IMPORT_ERROR = exc
else:
    AI_COMPLETION_IMPORT_ERROR = None

try:
    from ai_agents.agent_manager import get_agent_manager, run_ai_analysis, get_ai_recommendations, get_ai_alerts
    from ai_agents.ai_file_processor import AIFileProcessor
    AI_AGENTS_AVAILABLE = True
    import logging
    logging.info("✅ AI Agents imported successfully")
except ImportError as e:
    import logging
    logging.error(f"❌ AI Agents import failed: {e}")
    logging.error(f"   Error type: {type(e).__name__}")
    # Try to show more helpful error in Streamlit
    if 'ai_agents' in str(e):
        logging.error(f"   Current directory: {os.getcwd()}")
        logging.error(f"   ai_agents exists: {os.path.exists('ai_agents')}")
        logging.error(f"   ai_agents/__init__.py exists: {os.path.exists('ai_agents/__init__.py')}")
        logging.error(f"   Python path: {sys.path[:5]}")
        logging.error(f"   File directory: {os.path.dirname(os.path.abspath(__file__))}")
    AI_AGENTS_AVAILABLE = False
except KeyError as e:
    import logging
    logging.error(f"❌ AI Agents KeyError (likely missing secrets): {e}")
    logging.error(f"   This usually means st.secrets['api_keys']['open_ai'] is missing")
    AI_AGENTS_AVAILABLE = False
except Exception as e:
    import logging
    logging.error(f"❌ AI Agents error: {e}")
    logging.error(f"   Error type: {type(e).__name__}")
    logging.error(f"   Error args: {e.args if hasattr(e, 'args') else 'N/A'}")
    import traceback
    logging.error(f"   Traceback: {traceback.format_exc()}")
    AI_AGENTS_AVAILABLE = False

# Performance optimization decorators
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_holdings(user_id: str):
    """Cache holdings data to avoid repeated database calls"""
    from database_shared import SharedDatabaseManager
    db = SharedDatabaseManager()
    holdings = db.get_user_holdings_silent(user_id)
    # CRITICAL: Double-check filtering at cache level to ensure zero-quantity holdings are never cached
    # This is a safety net in case database filtering has any issues
    filtered = []
    for h in holdings:
        quantity_raw = h.get('total_quantity', 0)
        try:
            quantity = float(quantity_raw) if quantity_raw is not None else 0.0
        except (ValueError, TypeError):
            quantity = 0.0
        if quantity > 0.0001:  # Only cache holdings with positive quantity
            filtered.append(h)
    return filtered

@st.cache_data(ttl=21600)  # 6 hours
def get_cached_amfi_schemes() -> Dict[str, str]:
    """Cache AMFI scheme list to avoid repeated downloads."""
    try:
        from mftool import Mftool
        mf = Mftool()
        schemes = mf.get_scheme_codes()
        if not schemes:
            return {}
        return {code: name for code, name in schemes.items() if code and name}
    except Exception:
        return {}

@st.cache_data(ttl=86400)  # 1 day
def get_cached_amfi_nav_text() -> Optional[str]:
    """Cache the raw AMFI NAV file text to avoid repeated downloads."""
    try:
        response = requests.get(AMFI_NAV_URL)
        if response.status_code == 200 and response.text:
            return response.text
    except Exception:
        pass
    return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_cached_portfolio_summary(holdings: List[Dict]) -> str:
    """Cache comprehensive portfolio summary calculation"""
    if not holdings:
        return "No holdings found"
    
    # Safe conversion to handle None values
    def safe_float(value, default=0):
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    total_investment = 0
    total_current = 0
    
    # Asset type breakdown
    asset_types = {}
    channels = {}
    sectors = {}
    
    for holding in holdings:
        # CRITICAL: Skip holdings with zero or negative quantity (fully sold positions)
        # Use a small epsilon to handle floating point precision issues (e.g., 0.0001)
        quantity = safe_float(holding.get('total_quantity'), 0)
        if quantity <= 0.0001:  # Treat anything <= 0.0001 as effectively zero
            continue  # Skip fully sold positions - they shouldn't count in invested amount or P&L
        
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        
        # Use safe_float to handle None values
        current_value = safe_float(current_price, 0) * quantity
        investment = quantity * safe_float(holding.get('average_price'), 0)
        total_investment += investment
        total_current += current_value
        
        # Track asset types
        asset_type = holding.get('asset_type', 'Unknown')
        if asset_type not in asset_types:
            asset_types[asset_type] = {'investment': 0, 'current': 0, 'count': 0}
        asset_types[asset_type]['investment'] += investment
        asset_types[asset_type]['current'] += current_value
        asset_types[asset_type]['count'] += 1
        
        # Track channels
        channel = holding.get('channel', 'Unknown')
        if channel not in channels:
            channels[channel] = {'investment': 0, 'current': 0, 'count': 0}
        channels[channel]['investment'] += investment
        channels[channel]['current'] += current_value
        channels[channel]['count'] += 1
        
        # Track sectors
        sector = holding.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = {'investment': 0, 'current': 0, 'count': 0}
        sectors[sector]['investment'] += investment
        sectors[sector]['current'] += current_value
        sectors[sector]['count'] += 1
    
    total_pnl = total_current - total_investment
    total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
    
    portfolio_summary = f"""📊 COMPREHENSIVE PORTFOLIO OVERVIEW:

💰 FINANCIAL SUMMARY:
• Total Holdings: {len(holdings)} assets
• Total Investment: ₹{total_investment:,.0f}
• Current Value: ₹{total_current:,.0f}
• Total P&L: ₹{total_pnl:,.0f} ({total_pnl_pct:+.1f}%)

📈 ASSET TYPE BREAKDOWN:"""
    
    for asset_type, data in sorted(asset_types.items(), key=lambda x: x[1]['current'], reverse=True):
        pnl = data['current'] - data['investment']
        pnl_pct = (pnl / data['investment'] * 100) if data['investment'] > 0 else 0
        portfolio_summary += f"\n• {asset_type.title()}: {data['count']} holdings, ₹{data['current']:,.0f} ({pnl_pct:+.1f}%)"
    
    portfolio_summary += f"\n\n🏢 CHANNEL BREAKDOWN:"
    for channel, data in sorted(channels.items(), key=lambda x: x[1]['current'], reverse=True):
        pnl = data['current'] - data['investment']
        pnl_pct = (pnl / data['investment'] * 100) if data['investment'] > 0 else 0
        portfolio_summary += f"\n• {channel}: {data['count']} holdings, ₹{data['current']:,.0f} ({pnl_pct:+.1f}%)"
    
    portfolio_summary += f"\n\n🏭 SECTOR BREAKDOWN:"
    for sector, data in sorted(sectors.items(), key=lambda x: x[1]['current'], reverse=True):
        pnl = data['current'] - data['investment']
        pnl_pct = (pnl / data['investment'] * 100) if data['investment'] > 0 else 0
        portfolio_summary += f"\n• {sector}: {data['count']} holdings, ₹{data['current']:,.0f} ({pnl_pct:+.1f}%)"
    
    # Calculate P&L for all holdings for top gainers/losers
    holdings_with_pnl = []
    for holding in holdings:
        # CRITICAL: Skip holdings with zero or negative quantity (fully sold positions)
        # Use a small epsilon to handle floating point precision issues (e.g., 0.0001)
        quantity = safe_float(holding.get('total_quantity'), 0)
        if quantity <= 0.0001:  # Treat anything <= 0.0001 as effectively zero
            continue  # Skip fully sold positions
        
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        
        current_value = safe_float(current_price, 0) * quantity
        investment = quantity * safe_float(holding.get('average_price'), 0)
        pnl_pct = ((current_value - investment) / investment * 100) if investment > 0 else 0
        
        holdings_with_pnl.append({
            'ticker': holding.get('ticker', 'N/A'),
            'stock_name': holding.get('stock_name', 'N/A'),
            'asset_type': holding.get('asset_type', 'Unknown'),
            'channel': holding.get('channel', 'Unknown'),
            'sector': holding.get('sector', 'Unknown'),
            'current_value': current_value,
            'pnl_pct': pnl_pct
        })
    
    portfolio_summary += f"\n\n🏆 TOP 5 GAINERS (by P&L %):"
    top_gainers = sorted(holdings_with_pnl, key=lambda h: h['pnl_pct'], reverse=True)[:5]
    for holding in top_gainers:
        emoji = "🚀" if holding['pnl_pct'] > 10 else "📈"
        portfolio_summary += f"\n{emoji} {holding['ticker']} ({holding['asset_type']}) - {holding['stock_name'][:30]}"
        portfolio_summary += f"\n   Channel: {holding['channel']} | Sector: {holding['sector']} | P&L: {holding['pnl_pct']:+.1f}% | Value: ₹{holding['current_value']:,.0f}"
    
    portfolio_summary += f"\n\n📉 TOP 5 LOSERS (by P&L %):"
    top_losers = sorted(holdings_with_pnl, key=lambda h: h['pnl_pct'])[:5]
    for holding in top_losers:
        emoji = "📉" if holding['pnl_pct'] < 0 else "➡️"
        portfolio_summary += f"\n{emoji} {holding['ticker']} ({holding['asset_type']}) - {holding['stock_name'][:30]}"
        portfolio_summary += f"\n   Channel: {holding['channel']} | Sector: {holding['sector']} | P&L: {holding['pnl_pct']:+.1f}% | Value: ₹{holding['current_value']:,.0f}"
    
    portfolio_summary += f"\n\n🏆 TOP 10 HOLDINGS (by current value):"
    
    # Sort holdings by current value and take top 10
    sorted_holdings = sorted(
        holdings_with_pnl, 
        key=lambda h: h['current_value'], 
        reverse=True
    )
    
    for holding in sorted_holdings[:10]:
        emoji = "🚀" if holding['pnl_pct'] > 10 else "📈" if holding['pnl_pct'] > 0 else "📉" if holding['pnl_pct'] < -5 else "➡️"
        
        portfolio_summary += f"\n{emoji} {holding['ticker']} ({holding['asset_type']}) - {holding['stock_name'][:30]}"
        portfolio_summary += f"\n   Channel: {holding['channel']} | Sector: {holding['sector']} | P&L: {holding['pnl_pct']:+.1f}% | Value: ₹{holding['current_value']:,.0f}"
    
    return portfolio_summary


def _format_dataframe_preview(df: pd.DataFrame, max_rows: int = 15, max_cols: int = 20) -> str:
    """Return a readable string preview of a DataFrame with row/column limits."""
    if df is None or df.empty:
        return "No data available in this sheet."
    preview = df.copy()
    if max_cols and preview.shape[1] > max_cols:
        # Keep first max_cols columns
        remaining = preview.shape[1] - max_cols
        preview = preview.iloc[:, :max_cols]
        preview[f"... (+{remaining} more columns)"] = ""
    if max_rows and preview.shape[0] > max_rows:
        preview = preview.head(max_rows)
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols):
        return preview.to_string(index=False)


def _build_document_payload_from_file(uploaded_file) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Convert an uploaded file (PDF/CSV/Excel) into a document payload for AI analysis.
    Returns (payload, error_message).
    """
    if not uploaded_file:
        return None, "No file provided."

    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()
    metadata: Dict[str, Any] = {
        'extension': extension,
        'original_filename': filename,
    }

    try:
        if extension == '.pdf':
            try:
                import pdfplumber  # type: ignore
                import PyPDF2  # type: ignore
            except ImportError as exc:  # pragma: no cover - runtime dependency
                return None, f"PDF processing libraries missing: {exc}"

            uploaded_file.seek(0)
            
            # Convert uploaded_file to BytesIO for pdfplumber compatibility
            import io
            if hasattr(uploaded_file, 'read'):
                pdf_bytes = uploaded_file.read()
            elif hasattr(uploaded_file, 'getvalue'):
                pdf_bytes = uploaded_file.getvalue()
                if isinstance(pdf_bytes, str):
                    pdf_bytes = pdf_bytes.encode('utf-8')
            else:
                pdf_bytes = uploaded_file
            
            # Ensure we have bytes
            if isinstance(pdf_bytes, str):
                pdf_bytes = pdf_bytes.encode('latin-1')
            
            pdf_text = ""
            tables_found: List[Dict[str, Any]] = []
            page_count = 0

            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    page_count = len(pdf.pages)
                    print(f"[AI_ASSISTANT_PDF] PDF opened successfully with {page_count} pages")
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                pdf_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                                print(f"[AI_ASSISTANT_PDF] Page {page_num}: Extracted {len(page_text)} characters")

                            tables = page.extract_tables()
                            if tables:
                                for table_idx, table in enumerate(tables, 1):
                                    tables_found.append(
                                        {
                                            'page': page_num,
                                            'table': table_idx,
                                            'rows': len(table),
                                        }
                                    )
                        except Exception as page_error:
                            print(f"[AI_ASSISTANT_PDF] Page {page_num}: Error during extraction: {str(page_error)}")
                            continue
            except Exception as pdf_error:
                print(f"[AI_ASSISTANT_PDF] pdfplumber failed: {str(pdf_error)}")
                pdf_text = ""  # Reset to trigger fallback

            if not pdf_text.strip():
                # Fallback to PyPDF2 if pdfplumber couldn't extract text
                print(f"[AI_ASSISTANT_PDF] pdfplumber returned no text, trying PyPDF2...")
                try:
                    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                    page_count = len(reader.pages)
                    extracted_pages = []
                    for page_num, page in enumerate(reader.pages, 1):
                        try:
                            page_content = page.extract_text()
                            if page_content:
                                extracted_pages.append(page_content)
                                print(f"[AI_ASSISTANT_PDF] PyPDF2 Page {page_num}: Extracted {len(page_content)} characters")
                        except Exception as page_error:
                            print(f"[AI_ASSISTANT_PDF] PyPDF2 Page {page_num}: Error: {str(page_error)}")
                            continue
                    pdf_text = "\n\n".join(extracted_pages)
                    if pdf_text.strip():
                        print(f"[AI_ASSISTANT_PDF] PyPDF2 successfully extracted {len(pdf_text)} characters")
                except Exception as pypdf_error:
                    print(f"[AI_ASSISTANT_PDF] PyPDF2 also failed: {str(pypdf_error)}")
                    pdf_text = ""  # Will trigger OCR fallback

            if not pdf_text.strip():
                # Last resort: Try direct PDF upload to OpenAI first (fastest, no dependencies)
                print(f"[AI_ASSISTANT_PDF] No text extracted, attempting direct PDF upload to OpenAI...")
                try:
                    import streamlit as st
                    import openai
                    openai_client = openai.OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
                    uploaded_file.seek(0)
                    direct_text = _extract_pdf_with_direct_upload(uploaded_file, filename, openai_client, show_ui_errors=False)
                    if direct_text and direct_text.strip():
                        pdf_text = direct_text
                        metadata['extraction_method'] = 'Direct OpenAI Upload'
                        metadata['info_message'] = (
                            f"✅ Extracted {len(pdf_text)} characters using direct OpenAI PDF processing"
                        )
                        metadata['info_lines'].append("- Extraction: Direct OpenAI PDF Upload (fastest)")
                        print(f"[AI_ASSISTANT_PDF] ✅ Direct OpenAI upload successfully extracted {len(pdf_text)} characters")
                    else:
                        # Try Vision API (page-by-page) as fallback
                        print(f"[AI_ASSISTANT_PDF] Direct upload failed, trying Vision API...")
                        uploaded_file.seek(0)
                        vision_text = _extract_pdf_with_vision_api(uploaded_file, filename, show_ui_errors=False)
                        if vision_text and vision_text.strip():
                            pdf_text = vision_text
                            metadata['extraction_method'] = 'Vision API'
                            metadata['info_message'] = (
                                f"✅ Extracted {len(pdf_text)} characters using GPT-4 Vision API"
                            )
                            metadata['info_lines'].append("- Extraction: GPT-4 Vision API (page-by-page)")
                            print(f"[AI_ASSISTANT_PDF] ✅ Vision API successfully extracted {len(pdf_text)} characters")
                        else:
                            # Final fallback: Try OCR (requires system dependencies)
                            print(f"[AI_ASSISTANT_PDF] Vision API failed, attempting OCR...")
                            uploaded_file.seek(0)
                            ocr_text = _extract_pdf_with_ocr(uploaded_file)
                            if ocr_text and ocr_text.strip():
                                pdf_text = ocr_text
                                metadata['extraction_method'] = 'OCR'
                                metadata['info_message'] = (
                                    f"✅ Extracted {len(pdf_text)} characters using OCR from {page_count} pages"
                                )
                                metadata['info_lines'].append("- Extraction: OCR (image-based PDF)")
                                print(f"[AI_ASSISTANT_PDF] ✅ OCR successfully extracted {len(pdf_text)} characters")
                            else:
                                return None, f"Could not extract readable text from '{filename}'. PDF appears to be image-based. Tried: 1) Direct OpenAI upload, 2) GPT-4 Vision API, 3) OCR - all failed. Please check your OpenAI API key or use a text-selectable PDF."
                except Exception as ai_error:
                    print(f"[AI_ASSISTANT_PDF] AI extraction methods failed: {str(ai_error)}")
                    # Try OCR as final fallback
                    uploaded_file.seek(0)
                    ocr_text = _extract_pdf_with_ocr(uploaded_file)
                    if ocr_text and ocr_text.strip():
                        pdf_text = ocr_text
                        metadata['extraction_method'] = 'OCR'
                        metadata['info_message'] = (
                            f"✅ Extracted {len(pdf_text)} characters using OCR from {page_count} pages"
                        )
                        metadata['info_lines'].append("- Extraction: OCR (image-based PDF)")
                        print(f"[AI_ASSISTANT_PDF] ✅ OCR successfully extracted {len(pdf_text)} characters")
                    else:
                        return None, f"Could not extract readable text from '{filename}'. PDF appears to be image-based. All extraction methods failed. Error: {str(ai_error)}"

            metadata['pages'] = page_count
            metadata['tables_found'] = len(tables_found)
            metadata['info_lines'] = [
                "- Type: PDF Document",
                f"- Pages: {page_count or 'Unknown'}",
                f"- Tables detected: {len(tables_found)}",
            ]
            metadata['info_message'] = (
                f"✅ Extracted {len(pdf_text)} characters from {page_count} pages, "
                f"found {len(tables_found)} tables"
            )

            tables_summary = ""
            if tables_found:
                sample = tables_found[:3]
                tables_summary = "Sample tables detected:\n"
                for table in sample:
                    tables_summary += f"• Page {table['page']} – Table {table['table']} ({table['rows']} rows)\n"
            metadata['tables_summary'] = tables_summary.strip()

            return {
                'name': filename,
                'display_name': filename,
                'text': pdf_text.strip(),
                'type_label': 'PDF Document',
                'metadata': metadata,
            }, None

        if extension in ('.csv', '.tsv'):
            uploaded_file.seek(0)
            read_kwargs = {}
            if extension == '.tsv':
                read_kwargs['sep'] = '\t'
            try:
                df = pd.read_csv(uploaded_file, **read_kwargs)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1', **read_kwargs)

            rows, cols = df.shape
            preview_rows = min(rows, 20) if rows else 0
            preview_text = _format_dataframe_preview(df, max_rows=preview_rows)
            stats_text = ""
            try:
                stats = df.describe(include='all').transpose()
                stats_text = stats.to_string()
            except Exception:
                stats_text = "Summary statistics unavailable."

            doc_text = (
                f"CSV Datasheet: {filename}\n"
                f"Rows: {rows}, Columns: {cols}\n\n"
                f"Column Names: {', '.join(map(str, df.columns))}\n\n"
                f"Preview (first {preview_rows} rows):\n{preview_text}\n\n"
                f"Summary Statistics:\n{stats_text}\n"
            )

            metadata['rows'] = rows
            metadata['columns'] = cols
            metadata['info_lines'] = [
                "- Type: CSV Datasheet",
                f"- Rows: {rows}",
                f"- Columns: {cols}",
            ]
            metadata['info_message'] = (
                f"✅ Parsed CSV with {rows} rows × {cols} columns "
                f"(showing first {preview_rows or 0} rows)"
            )

            return {
                'name': filename,
                'display_name': filename,
                'text': doc_text.strip(),
                'type_label': 'CSV Datasheet',
                'metadata': metadata,
            }, None

        if extension in ('.xlsx', '.xls'):
            uploaded_file.seek(0)
            try:
                engine = 'openpyxl' if extension == '.xlsx' else None
                sheets = pd.read_excel(uploaded_file, sheet_name=None, engine=engine)
            except ImportError as e:
                return None, f"Missing Excel dependency for '{filename}'. Install with: pip install openpyxl xlrd"
            except ValueError as exc:
                return None, f"Failed to read Excel file '{filename}': {exc}"
            except Exception as e:
                if 'openpyxl' in str(e).lower() or 'xlrd' in str(e).lower():
                    return None, f"Missing Excel dependency for '{filename}'. Install with: pip install openpyxl xlrd"
                return None, f"Failed to read Excel file '{filename}': {e}"

            if not sheets:
                return None, f"No sheets found in Excel file '{filename}'."

            sheet_count = len(sheets)
            total_rows = 0
            sheet_texts = [
                f"Excel Datasheet: {filename}",
                f"Sheets: {sheet_count}",
            ]

            for sheet_name, sheet_df in sheets.items():
                rows, cols = sheet_df.shape
                total_rows += rows
                preview = _format_dataframe_preview(sheet_df, max_rows=15)
                sheet_texts.append(
                    f"\n--- Sheet: {sheet_name} ({rows} rows × {cols} columns) ---\n{preview}\n"
                )

            doc_text = "\n".join(sheet_texts)

            metadata['sheets'] = sheet_count
            metadata['total_rows'] = total_rows
            metadata['info_lines'] = [
                "- Type: Excel Datasheet",
                f"- Sheets: {sheet_count}",
                f"- Approx. rows: {total_rows}",
            ]
            metadata['info_message'] = (
                f"✅ Parsed Excel workbook with {sheet_count} sheet(s) "
                f"and approximately {total_rows} rows"
            )

            return {
                'name': filename,
                'display_name': filename,
                'text': doc_text.strip(),
                'type_label': 'Excel Datasheet',
                'metadata': metadata,
            }, None

        return None, f"Unsupported file type '{extension or 'unknown'}' for '{filename}'."

    except Exception as exc:
        return None, f"Error reading '{filename}': {exc}"


def _build_document_payload_from_url(url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetch a news/article URL and convert it into a document payload.
    Returns (payload, error_message).
    """
    if not url or not url.strip():
        return None, "Please provide a news article URL."

    cleaned_url = url.strip()

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:  # pragma: no cover - runtime dependency
        return None, "BeautifulSoup (bs4) is required to process article URLs."

    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }
        response = requests.get(cleaned_url, headers=headers)
        response.raise_for_status()
    except Exception as exc:
        return None, f"Failed to fetch article: {exc}"

    soup = BeautifulSoup(response.content, 'html.parser')

    for tag in soup(['script', 'style', 'noscript', 'iframe', 'header', 'footer']):
        tag.extract()

    title = soup.title.string.strip() if soup.title and soup.title.string else cleaned_url
    text_chunks = [
        line.strip()
        for line in soup.get_text(separator='\n').splitlines()
        if line and line.strip()
    ]
    article_body = "\n".join(text_chunks)

    if not article_body.strip():
        return None, "Could not extract readable content from the article."

    published_at = None
    potential_date_fields = [
        ('meta', {'property': 'article:published_time'}),
        ('meta', {'property': 'og:published_time'}),
        ('meta', {'name': 'pubdate'}),
        ('meta', {'name': 'date'}),
        ('meta', {'itemprop': 'datePublished'}),
    ]
    for tag_name, attrs in potential_date_fields:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get('content'):
            try:
                published_at = dateutil_parser.parse(tag['content'])
                break
            except Exception:
                continue

    if not published_at:
        time_tag = soup.find('time')
        if time_tag:
            candidate = time_tag.get('datetime') or time_tag.text
            if candidate:
                try:
                    published_at = dateutil_parser.parse(candidate)
                except Exception:
                    published_at = None

    domain = urlparse(cleaned_url).netloc or "unknown"
    published_display = published_at.strftime('%Y-%m-%d %H:%M') if published_at else "Unknown"

    filename_base = re.sub(r'[^A-Za-z0-9 _\-]+', '', title).strip()
    if not filename_base:
        filename_base = re.sub(r'[^A-Za-z0-9]+', '', domain) or "news-article"
    filename = f"News - {filename_base[:80]}.txt"

    doc_text = (
        f"News Article: {title}\n"
        f"Source: {domain}\n"
        f"URL: {cleaned_url}\n"
        f"Published: {published_display}\n\n"
        f"{article_body}"
    )

    metadata: Dict[str, Any] = {
        'type': 'news_article',
        'source': domain,
        'url': cleaned_url,
        'published_at': published_at.isoformat() if published_at else None,
        'info_lines': [
            "- Type: News Article",
            f"- Source: {domain}",
            f"- Published: {published_display}",
        ],
        'info_message': f"✅ Fetched news article from {domain} ({len(article_body)} characters)",
    }

    return {
        'name': filename,
        'display_name': title,
        'text': doc_text.strip(),
        'type_label': 'News Article',
        'metadata': metadata,
    }, None


def _run_document_analysis(
    document_payloads: List[Dict[str, Any]],
    user: Dict[str, Any],
    holdings: List[Dict[str, Any]],
    db,
    section_key: str,
) -> Tuple[int, int]:
    """
    Run AI analysis for provided document payloads.
    Returns (success_count, failure_count).
    """
    if not document_payloads:
        return 0, 0

    try:
        import openai  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        st.error(f"❌ OpenAI package not installed: {exc}")
        return 0, len(document_payloads)

    try:
        openai.api_key = st.secrets["api_keys"]["open_ai"]
    except Exception as exc:
        st.error(f"❌ OpenAI API key missing: {exc}")
        return 0, len(document_payloads)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    portfolio_summary = get_cached_portfolio_summary(holdings)

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_bar = None
    if len(document_payloads) > 1:
        progress_bar = progress_placeholder.progress(0)

    success_count = 0
    failure_count = 0

    for idx, doc in enumerate(document_payloads, 1):
        if progress_bar:
            progress_bar.progress(idx / len(document_payloads))
        status_placeholder.text(f"📄 Processing {idx}/{len(document_payloads)}: {doc.get('display_name', doc.get('name'))}")

        try:
            doc_text = doc.get('text', '').strip()
            if not doc_text:
                failure_count += 1
                st.warning(f"⚠️ Skipping {doc.get('name')}: No readable content extracted.")
                continue

            metadata = doc.get('metadata', {})
            info_lines = metadata.get('info_lines') or []
            info_block = "\n".join(info_lines) if info_lines else "- No additional metadata"
            tables_summary = metadata.get('tables_summary', '')
            if tables_summary:
                info_block += f"\n{tables_summary}"

            preview_limit = 10000
            doc_preview = doc_text[:preview_limit]
            if len(doc_text) > preview_limit:
                doc_preview += "\n...[truncated for AI analysis]..."

            doc_type = doc.get('type_label', 'Document')
            display_name = doc.get('display_name', doc.get('name'))

            analysis_prompt = (
                f"Analyze this {doc_type} for portfolio management insights.\n\n"
                f"📄 DOCUMENT INFO:\n{info_block}\n\n"
                f"💼 USER'S PORTFOLIO:\n{portfolio_summary}\n\n"
                f"📝 DOCUMENT CONTENT (preview):\n{doc_preview}\n\n"
                "Provide a structured analysis with these sections:\n\n"
                "📋 DOCUMENT SUMMARY\n"
                "📊 KEY METRICS & DATA\n"
                "📈 INSIGHTS FROM DATA/TABLES\n"
                "💡 MAIN FINDINGS\n"
                "🎯 PORTFOLIO RELEVANCE\n"
                "⚡ RECOMMENDED ACTIONS\n\n"
                "Be specific, reference concrete numbers where possible, and tailor the insights to the portfolio context."
            )

            response = openai.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_completion_tokens=1000,
            )

            ai_analysis = response.choices[0].message.content

            with st.expander(f"📄 {display_name} - Analysis", expanded=(idx == 1)):
                st.markdown("### 🤖 AI Analysis")
                st.markdown(ai_analysis)

            info_message = metadata.get('info_message')
            if info_message:
                st.info(info_message)

            st.session_state.chat_history.append({
                "q": f"Analyze {doc_type}: {display_name}",
                "a": ai_analysis,
            })
            if hasattr(db, 'save_chat_history'):
                try:
                    db.save_chat_history(user['id'], f"Analyze {doc_type}: {display_name}", ai_analysis)
                except Exception:
                    pass

            save_result = db.save_pdf(
                user_id=user['id'],
                filename=doc.get('name', display_name),
                pdf_text=doc_text,
                ai_summary=ai_analysis,
            )
            if not save_result.get('success'):
                failure_count += 1
                st.error(f"❌ Failed to save '{display_name}': {save_result.get('error', 'Unknown error')}")
            else:
                success_count += 1

        except Exception as exc:
            failure_count += 1
            st.error(f"❌ Error processing '{doc.get('name', 'document')}': {str(exc)[:120]}")

    if len(document_payloads) > 1:
        progress_placeholder.empty()
        status_placeholder.empty()

    if success_count > 0:
        try:
            st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
        except Exception:
            pass

    return success_count, failure_count


def _render_document_upload_section(
    section_key: str,
    user: Dict[str, Any],
    holdings: List[Dict[str, Any]],
    db,
    header_text: str = "**📤 Upload Documents for AI Analysis**",
) -> None:
    """Render shared document upload UI section for AI assistant."""
    st.markdown("---")
    st.markdown(header_text)
    st.caption("💡 Upload PDFs, CSVs, or Excel files together — the AI will extract insights for you.")

    uploaded_files = st.file_uploader(
        "Choose document file(s) to analyze",
        type=['pdf', 'csv', 'tsv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        key=f"{section_key}_file_uploader",
        help="Supported formats: PDF, CSV/TSV, Excel (XLSX/XLS).",
    )

    document_payloads: List[Dict[str, Any]] = []
    extraction_errors: List[str] = []

    if uploaded_files:
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]

        for file_obj in uploaded_files:
            payload, error = _build_document_payload_from_file(file_obj)
            if payload:
                document_payloads.append(payload)
            elif error:
                extraction_errors.append(error)

    if document_payloads:
        button_text = (
            "🔍 Analyze 1 Document" if len(document_payloads) == 1
            else f"🔍 Analyze & Upload {len(document_payloads)} Documents"
        )
        if st.button(button_text, type="primary", key=f"{section_key}_analyze_button"):
            success_count, failure_count = _run_document_analysis(
                document_payloads,
                user,
                holdings,
                db,
                section_key,
            )
            if success_count > 0 and failure_count == 0:
                st.success(f"✅ Successfully processed {success_count} document(s)!")
                st.info("📄 Documents saved to the shared library for future chats.")
                st.rerun()
            elif success_count > 0:
                st.warning(f"⚠️ Processed {success_count} document(s), {failure_count} failed.")
                st.info("📄 Successfully processed documents are now available in the shared library.")
            else:
                st.error("❌ All documents failed to process. Please check the files and try again.")

    elif uploaded_files:
        st.warning("⚠️ Could not extract any usable content from the uploaded files.")

    for error_msg in extraction_errors:
        st.error(error_msg)

    st.markdown("**🔗 Analyze Financial News Article**")
    st.caption("Paste a URL to fetch, store, and analyze the article alongside your portfolio.")
    news_url = st.text_input(
        "News/article URL",
        key=f"{section_key}_news_url",
        placeholder="https://example.com/news-article",
    )
    if st.button("🔍 Fetch & Analyze Article", key=f"{section_key}_news_button"):
        article_payload, article_error = _build_document_payload_from_url(news_url)
        if article_error:
            st.error(article_error)
        elif article_payload:
            success_count, failure_count = _run_document_analysis(
                [article_payload],
                user,
                holdings,
                db,
                f"{section_key}_news",
            )
            if success_count:
                st.success("✅ Article analyzed and stored successfully!")
                st.info("📰 Article text saved to the shared document library.")
            else:
                st.error("❌ Failed to analyze the article. Please try another link.")


# ====================================================================
# Transaction file extraction helpers (Python first, AI fallback)
# ====================================================================

_TX_COLUMN_ALIASES: Dict[str, List[str]] = {
    'date': [
        'date', 'transaction date', 'trade date', 'tx date', 'tran date',
        'order date', 'purchase date', 'nav date', 'valuation date',
        'execution', 'execution date', 'execution date and time', 'deal date'
    ],
    'ticker': [
        'ticker', 'symbol','scrip symbol', 'code', 'isin', 'scrip', 'stock code',
        'security code', 'instrument code', 'scheme code', 'amfi',
        'amfi code', 'isin code', 'investment code'
    ],
    'stock_name': [
        'stock name', 'security name', 'instrument', 'name', 'scheme name',
        'company', 'fund name', 'description', 'asset name', 'holding',
        'product'
    ],
    'scheme_name': [
        'scheme name', 'fund scheme', 'scheme', 'schemename'
    ],
    'quantity': [
        'quantity', 'qty', 'units', 'shares', 'quantity/unit',
        'no. of units', 'no of units', 'units/qty', 'units (credit)',
        'units (debit)'
    ],
    'price': [
        'price', 'rate', 'nav', 'purchase price', 'per unit price',
        'trade price', 'unit price', 'cost price', 'nav per unit',
        'deal price', 'executed price'
    ],
    'amount': [
        'amount', 'value', 'total value', 'consideration', 'txn amount',
        'gross amount', 'net amount', 'investment amount', 'order value',
        'total', 'investment', 'transaction value'
    ],
    'transaction_type': [
        'transaction type', 'type', 'action', 'side', 'buy/sell',
        'txn type', 'order type', 'direction', 'nature', 'transaction',
        'mode'
    ],
    'asset_type': [
        'asset type', 'instrument type', 'category', 'asset class',
        'type of asset', 'class', 'instrument category'
    ],
    'channel': [
        'channel', 'Channel', 'platform', 'broker', 'account', 'source', 'portfolio',
        'through', 'partner', 'advisor', 'exchange', 'Exchange', 'EXCHANGE'
    ],
    'sector': [
        'sector', 'industry', 'sector name', 'segment'
    ],
    'notes': [
        'notes', 'Notes', 'remarks', 'comment', 'description', 'order status', 'Order status', 'Order Status', 'ORDER STATUS',
        'exchange order id', 'Exchange Order Id', 'Exchange Order ID', 'order id', 'Order Id'
    ],
    'folio': [
        'folio', 'folio no', 'folio number'
    ],
}


def _tx_safe_str(value: Any) -> str:
    """Convert to clean string, removing NaN/None placeholders."""
    if value is None:
        return ''
    try:
        text = str(value).strip()
    except Exception:
        return ''
    lowered = text.lower()
    if lowered in {'nan', 'none', 'null', 'nan%', 'nat', 'n/a', ''}:
        return ''
    return text


def _tx_safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float, handling currency symbols and commas."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = _tx_safe_str(value)
    if not text:
        return default
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    cleaned = (
        text.replace('₹', '')
        .replace('INR', '')
        .replace('Rs.', '')
        .replace('Rs', '')
        .replace(',', '')
        .replace('\u20b9', '')
        .replace('%', '')
        .strip()
    )
    try:
        return float(cleaned)
    except Exception:
        match = re.search(r'-?\d+(\.\d+)?', cleaned)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                pass
    return default


def _tx_normalize_ticker(raw: Any) -> str:
    """Normalize ticker values, handling numeric codes and suffixes."""
    text = _tx_safe_str(raw)
    if not text and isinstance(raw, (int, float)):
        text = f"{raw}"
    if not text:
        return ''
    text = text.strip()
    if ':' in text:
        parts = [segment for segment in re.split(r'[:/]', text) if segment]
        if parts:
            text = parts[-1].strip()
    text = text.replace('\u00a0', ' ')
    text = ''.join(text.split())  # remove whitespace
    text = text.lstrip('$')
    text = re.sub(r'^(NSE|BSE|NSI|BOM)(EQ|BE|BZ|XT|XD|P|W|Z|T0|T1)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:[-_.]?)(EQ|BE|BZ|XT|XD|P|W|Z|T0|T1)$', '', text, flags=re.IGNORECASE)
    if text.endswith('.0') and text.replace('.0', '').isdigit():
        text = text[:-2]
    text = text.replace('-', '')
    if text.isdigit():
        return text
    return text.upper()


def _tx_fallback_ticker_from_name(name: str) -> str:
    """Generate a synthetic ticker from the stock name when none is available."""
    if not name:
        return ''
    candidate = re.sub(r'[^A-Za-z0-9]', '', name.upper())
    return candidate or ''


def _tx_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names based on aliases."""
    rename_map: Dict[str, str] = {}
    used_targets: set[str] = set()
    for col in df.columns:
        col_name = _tx_safe_str(col).lower()
        target = None
        for canonical, aliases in _TX_COLUMN_ALIASES.items():
            for alias in aliases:
                alias_lower = alias.lower()
                if col_name == alias_lower or alias_lower in col_name:
                    target = canonical
                    break
            if target:
                break
        if target and target not in used_targets:
            rename_map[col] = target
            used_targets.add(target)
        else:
            rename_map[col] = col
    return df.rename(columns=rename_map)


def _tx_normalize_transaction_type(
    raw_type: Any,
    quantity: Optional[float] = None,
    amount: Optional[float] = None,
) -> str:
    """Normalize transaction type to 'buy' or 'sell'."""
    value = _tx_safe_str(raw_type).lower()
    if value in {'sell', 's', 'redeem', 'redemption', 'withdrawal', 'withdraw', 'exit', 'debit', 'outflow'}:
        return 'sell'
    if value in {'buy', 'b', 'purchase', 'invest', 'investment', 'inflow', 'credit', 'add', 'subscription'}:
        return 'buy'
    if quantity is not None:
        if quantity < 0:
            return 'sell'
        if quantity > 0:
            return 'buy'
    if amount is not None:
        if amount < 0:
            return 'sell'
        if amount > 0:
            return 'buy'
    return 'buy'


def _tx_infer_asset_type(ticker: str, stock_name: str, asset_hint: str = '') -> str:
    """Infer asset type from ticker/name/hint.
    CRITICAL: For numeric tickers, checks actual data sources (AMFI, yfinance) before heuristics.
    """
    hint = asset_hint.lower()
    name = stock_name.lower()
    ticker_upper = ticker.upper()

    # Check if ticker is numeric (including decimals like "10.65" - AMFI codes)
    is_numeric = False
    try:
        # Remove any non-numeric characters except decimal point
        cleaned = ticker_upper.replace(',', '').replace('$', '').strip()
        numeric_value = float(cleaned)
        is_numeric = True
    except (ValueError, AttributeError):
        pass

    # For numeric tickers, check actual data sources to determine type
    if is_numeric and ticker_upper.replace('.', '').isdigit():
        ticker_clean = ticker_upper.replace('.', '').split('.')[0]  # Remove decimal part
        
        # First, check if it's in AMFI dataset (mutual fund)
        try:
            amfi_data = get_amfi_dataset()
            if amfi_data and 'code_lookup' in amfi_data:
                if ticker_clean in amfi_data['code_lookup']:
                    return 'mutual_fund'  # Found in AMFI = mutual fund
        except Exception:
            pass  # AMFI check failed, continue to stock check
        
        # Try to fetch from yfinance (for stocks) - but only if it's a 5-6 digit number
        if len(ticker_clean) >= 5:
            try:
                import yfinance as yf
                # Try NSE first
                nse_ticker = f"{ticker_clean}.NS"
                stock = yf.Ticker(nse_ticker)
                hist = stock.history(period='1d')
                if not hist.empty:
                    return 'stock'  # Found in yfinance NSE = stock
                
                # Try BSE
                bse_ticker = f"{ticker_clean}.BO"
                stock = yf.Ticker(bse_ticker)
                hist = stock.history(period='1d')
                if not hist.empty:
                    return 'stock'  # Found in yfinance BSE = stock
            except Exception:
                pass  # yfinance check failed, fall back to heuristics
        
        # Fallback to heuristics if both checks failed
        # Most 6-digit numbers starting with '1' are mutual funds
        if len(ticker_clean) == 6 and ticker_clean.startswith('1'):
            return 'mutual_fund'
        elif len(ticker_clean) >= 5:
            # Other 5-6 digit numeric codes are likely BSE stocks
            return 'stock'
        else:
            # Shorter numeric codes are likely stocks
            return 'stock'

    if hint:
        if 'mutual' in hint or 'mf' in hint or hint == 'mutual_fund':
            return 'mutual_fund'
        if 'pms' in hint:
            return 'pms'
        if 'aif' in hint:
            return 'aif'
        if 'bond' in hint or 'debenture' in hint:
            return 'bond'
        if 'stock' in hint or 'equity' in hint:
            return 'stock'

    if ticker_upper.startswith('INP') or 'pms' in name or 'portfolio management' in name:
        return 'pms'
    if ticker_upper.startswith('AIF') or 'aif' in name or 'alternative investment' in name:
        return 'aif'
    
    # Check for mutual fund keywords FIRST (even if "bond" is in name - bond funds are still MFs)
    mf_keywords = ['fund', 'scheme', 'growth', 'nav', 'mutual', 'mf', 'plan']
    has_mf_keyword = any(keyword in name for keyword in mf_keywords)
    
    if has_mf_keyword:
        return 'mutual_fund'
    
    if ticker_upper.isdigit() and len(ticker_upper) >= 5:
        return 'mutual_fund'
    
    # Only check for bonds if NOT a mutual fund and NOT numeric
    # SGB (Sovereign Gold Bonds) are actual bonds, not mutual funds
    if ticker_upper.startswith('SGB') or 'sgb' in name:
        return 'bond'
    # Debentures are bonds
    if 'debenture' in name and not has_mf_keyword:
        return 'bond'
    # Other "bond" mentions might be bond mutual funds, so only classify as bond if no MF keywords
    is_numeric = ticker_upper.replace('.', '').isdigit()
    if 'bond' in name and not has_mf_keyword and not is_numeric:
        return 'bond'
    
    if '.NS' in ticker_upper or '.BO' in ticker_upper:
        return 'stock'
    
    return 'stock'


def _extract_pdf_with_ocr(uploaded_file) -> str:
    """
    Extract text from image-based PDF using OCR.
    Tries multiple OCR methods:
    1. Tesseract (if available - works locally)
    2. EasyOCR (pure Python - works on Streamlit Cloud)
    
    Returns extracted text or empty string if OCR fails or is unavailable.
    """
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    
    # Method 1: Try Tesseract (works locally, may not work on Streamlit Cloud)
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        from PIL import Image
        
        print(f"[PDF_OCR] Attempting Tesseract OCR...")
        images = convert_from_bytes(pdf_bytes, dpi=300, fmt='RGB')
        print(f"[PDF_OCR] Converted {len(images)} pages to images")
        
        all_text = []
        for page_num, image in enumerate(images, 1):
            try:
                print(f"[PDF_OCR] Processing page {page_num}/{len(images)} with Tesseract...")
                page_text = pytesseract.image_to_string(image, lang='eng')
                
                if page_text and page_text.strip():
                    all_text.append(f"--- Page {page_num} ---\n{page_text}\n")
                    char_count = len(page_text)
                    print(f"[PDF_OCR] Page {page_num}: Extracted {char_count} characters")
            except Exception as e:
                # Tesseract not available or failed - try EasyOCR
                print(f"[PDF_OCR] Tesseract failed for page {page_num}: {str(e)}")
                break
        
        if all_text:
            combined_text = "\n".join(all_text)
            if combined_text.strip():
                print(f"[PDF_OCR] ✅ Tesseract extracted {len(combined_text)} total characters")
                return combined_text
    except Exception as e:
        print(f"[PDF_OCR] Tesseract not available: {str(e)}")
        print(f"[PDF_OCR] Falling back to EasyOCR (works on Streamlit Cloud)...")
    
    # Method 2: Try EasyOCR (pure Python - works on Streamlit Cloud)
    try:
        import easyocr
        import numpy as np
        from PIL import Image
        import io
        
        images = []
        
        # Try to convert PDF to images - may fail on Streamlit Cloud if Poppler not available
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(pdf_bytes, dpi=300, fmt='RGB')
            print(f"[PDF_OCR] pdf2image converted {len(images)} pages to images")
        except Exception as pdf_conv_error:
            print(f"[PDF_OCR] pdf2image failed (may need Poppler): {str(pdf_conv_error)}")
            print(f"[PDF_OCR] Trying alternative: Extract images directly from PDF using PyPDF2...")
            
            # Alternative: Try to extract images directly from PDF using PyPDF2
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        # Try to get page as image (if PDF has embedded images)
                        if '/XObject' in page.get('/Resources', {}):
                            xObject = page['/Resources']['/XObject'].get_object()
                            for obj in xObject:
                                if xObject[obj]['/Subtype'] == '/Image':
                                    # Extract image from PDF
                                    img_obj = xObject[obj]
                                    # This is complex - pdf2image is much easier
                                    pass
                    except:
                        pass
                
                # If we couldn't extract images, try rendering page as image using pdfplumber
                try:
                    import pdfplumber
                    pdf_plumber = pdfplumber.open(io.BytesIO(pdf_bytes))
                    for page in pdf_plumber.pages:
                        # pdfplumber doesn't directly render to image, but we can try
                        # Actually, pdfplumber also needs Poppler for image rendering
                        pass
                    pdf_plumber.close()
                except:
                    pass
                
                print(f"[PDF_OCR] ⚠️ Cannot extract images without Poppler (pdf2image dependency)")
                print(f"[PDF_OCR]   Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases/")
                print(f"[PDF_OCR]   Or use a text-selectable PDF")
                return ""
            except Exception as alt_error:
                print(f"[PDF_OCR] Alternative extraction also failed: {str(alt_error)}")
                return ""
        
        if not images:
            print(f"[PDF_OCR] ⚠️ No images extracted - cannot proceed with OCR")
            return ""
        
        print(f"[PDF_OCR] Attempting EasyOCR (Streamlit Cloud compatible)...")
        # Initialize EasyOCR reader (English only for speed)
        # This will download models on first run (~100MB, cached after first use)
        reader = easyocr.Reader(['en'], gpu=False)
        
        all_text = []
        for page_num, image in enumerate(images, 1):
            try:
                print(f"[PDF_OCR] Processing page {page_num}/{len(images)} with EasyOCR...")
                # Convert PIL image to numpy array
                img_array = np.array(image)
                # Run OCR
                results = reader.readtext(img_array)
                # Extract text from results
                page_text = '\n'.join([result[1] for result in results])
                
                if page_text and page_text.strip():
                    all_text.append(f"--- Page {page_num} ---\n{page_text}\n")
                    char_count = len(page_text)
                    print(f"[PDF_OCR] Page {page_num}: Extracted {char_count} characters")
                else:
                    print(f"[PDF_OCR] Page {page_num}: No text extracted")
            except Exception as e:
                print(f"[PDF_OCR] Page {page_num}: EasyOCR failed: {str(e)}")
                continue
        
        combined_text = "\n".join(all_text)
        if combined_text.strip():
            print(f"[PDF_OCR] ✅ EasyOCR extracted {len(combined_text)} total characters from {len(images)} pages")
            return combined_text
        else:
            print(f"[PDF_OCR] ⚠️ EasyOCR completed but no text was extracted")
            return ""
            
    except ImportError as e:
        print(f"[PDF_OCR] ⚠️ EasyOCR not available: {e}")
        print(f"[PDF_OCR]   Install with: pip install easyocr")
        return ""
    except Exception as e:
        print(f"[PDF_OCR] ❌ EasyOCR extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


def _extract_pdf_with_direct_upload(uploaded_file, filename: str, openai_client, show_ui_errors: bool = True) -> str:
    """
    Try to process PDF directly with OpenAI (if supported).
    Currently, GPT-4o Vision API requires converting PDF to images.
    This function tries Assistants API file upload as an alternative.
    
    Returns extracted text or empty string if not supported.
    """
    try:
        # Read PDF bytes
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()
        pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
        
        # Check file size (OpenAI has limits)
        if pdf_size_mb > 50:  # Reduced limit for safety
            print(f"[PDF_DIRECT] PDF too large ({pdf_size_mb:.2f} MB) for direct upload")
            return ""
        
        print(f"[PDF_DIRECT] Attempting direct PDF processing ({pdf_size_mb:.2f} MB)...")
        
        # Note: Current OpenAI Chat Completions API (gpt-4o) doesn't support direct PDF upload
        # It requires converting PDF pages to images first.
        # The File API is for Assistants API, which is a different use case.
        # 
        # So we return empty string to trigger the fallback (page-by-page image conversion)
        # which we know works.
        print(f"[PDF_DIRECT] Direct PDF upload not available in Chat Completions API")
        print(f"[PDF_DIRECT] GPT-4o requires PDF pages to be converted to images")
        print(f"[PDF_DIRECT] Falling back to page-by-page Vision API processing...")
        return ""
        
    except Exception as e:
        print(f"[PDF_DIRECT] Direct PDF upload check failed: {str(e)}")
        return ""


def _extract_pdf_with_vision_api(uploaded_file, filename: str = "uploaded_file.pdf", show_ui_errors: bool = True) -> str:
    """
    Extract text from image-based PDF using GPT-4 Vision API.
    This tries direct PDF upload first, then falls back to page-by-page processing.
    
    Args:
        uploaded_file: PDF file to process
        filename: Name of the file
        show_ui_errors: If True, display errors in Streamlit UI (default: True)
    
    Returns extracted text or empty string if Vision API fails or is unavailable.
    """
    error_messages = []  # Collect error messages for UI display
    diagnostic_info = []  # Collect diagnostic information
    
    try:
        import openai
        import streamlit as st
        import base64
        import io
        from PIL import Image
        
        # Diagnostic: Test PyMuPDF import (try multiple import methods)
        fitz_available = False
        fitz = None
        try:
            # Try standard import first
            import fitz
            fitz_version = fitz.version if hasattr(fitz, 'version') else "unknown"
            msg = f"✅ PyMuPDF (fitz) import test: SUCCESS (version: {fitz_version})"
            print(f"[PDF_VISION] {msg}")
            diagnostic_info.append(msg)
            fitz_available = True
        except ImportError as fitz_import_test:
            msg = f"❌ PyMuPDF (fitz) import test: FAILED - {str(fitz_import_test)}"
            print(f"[PDF_VISION] {msg}")
            diagnostic_info.append(msg)
            error_messages.append(f"PyMuPDF not installed: {str(fitz_import_test)}")
            print(f"[PDF_VISION] 💡 Install with: pip install pymupdf")
            print(f"[PDF_VISION] 💡 Note: After installing, restart Streamlit to load the module")
        except Exception as fitz_test_err:
            msg = f"⚠️ PyMuPDF (fitz) import test: ERROR - {str(fitz_test_err)}"
            print(f"[PDF_VISION] {msg}")
            diagnostic_info.append(msg)
            error_messages.append(f"PyMuPDF error: {str(fitz_test_err)}")
        
        # Initialize OpenAI client
        try:
            openai_client = openai.OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
            diagnostic_info.append("✅ OpenAI client initialized successfully")
        except Exception as e:
            error_msg = f"OpenAI client not available: {str(e)}"
            print(f"[PDF_VISION] ❌ {error_msg}")
            error_messages.append(error_msg)
            if show_ui_errors:
                try:
                    st.error(f"❌ **Vision API Error**: {error_msg}\n\nPlease check your OpenAI API key in Streamlit secrets.")
                except:
                    pass
            return ""
        
        # FIRST: Try direct PDF upload (fastest, simplest - processes entire PDF in one call)
        print(f"[PDF_VISION] Attempting direct PDF upload first (fastest method)...")
        uploaded_file.seek(0)
        direct_text = _extract_pdf_with_direct_upload(uploaded_file, filename, openai_client, show_ui_errors=show_ui_errors)
        if direct_text and direct_text.strip():
            print(f"[PDF_VISION] ✅ Direct PDF upload successful! Extracted {len(direct_text)} characters")
            diagnostic_info.append(f"✅ Direct PDF upload: Success ({len(direct_text)} chars)")
            return direct_text
        else:
            print(f"[PDF_VISION] Direct PDF upload not supported or failed, falling back to page-by-page Vision API...")
            diagnostic_info.append("⚠️ Direct PDF upload: Not supported, using page-by-page method")
        
        # FALLBACK: Convert PDF to images using PyMuPDF (fitz) - doesn't require Poppler
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()
        pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
        print(f"[PDF_VISION] Read {len(pdf_bytes)} bytes ({pdf_size_mb:.2f} MB) from PDF file: {filename}")
        diagnostic_info.append(f"📄 PDF size: {pdf_size_mb:.2f} MB")
        
        # File size validation for Streamlit Cloud
        MAX_FILE_SIZE_MB = 10  # Streamlit Cloud limit
        if pdf_size_mb > MAX_FILE_SIZE_MB:
            error_msg = f"File too large: {pdf_size_mb:.2f} MB. Maximum allowed: {MAX_FILE_SIZE_MB} MB for Streamlit Cloud."
            print(f"[PDF_VISION] ❌ {error_msg}")
            error_messages.append(error_msg)
            diagnostic_info.append(f"❌ {error_msg}")
            if show_ui_errors:
                try:
                    st.error(f"❌ **File Size Error**: {error_msg}\n\nPlease use a smaller PDF file or split it into multiple files.")
                except:
                    pass
            return ""
        
        images = []
        image_conversion_error = None
        try:
            # Try to use PyMuPDF (even if diagnostic check failed, retry import here)
            fitz_module = None
            if fitz_available:
                # Already imported in diagnostic check
                fitz_module = fitz
            else:
                # Retry import - sometimes modules load differently at runtime
                try:
                    import fitz
                    fitz_module = fitz
                    print(f"[PDF_VISION] PyMuPDF import succeeded on retry!")
                    fitz_available = True
                except ImportError:
                    pass
            
            if fitz_module:
                print(f"[PDF_VISION] Using PyMuPDF to convert PDF to images...")
                try:
                    pdf_doc = fitz_module.open(stream=pdf_bytes, filetype="pdf")
                    total_pages = len(pdf_doc)
                    
                    # Page count validation for Streamlit Cloud
                    MAX_PDF_PAGES = 50  # Reasonable limit for Streamlit Cloud
                    if total_pages > MAX_PDF_PAGES:
                        error_msg = f"PDF too large: {total_pages} pages. Maximum allowed: {MAX_PDF_PAGES} pages for Streamlit Cloud."
                        print(f"[PDF_VISION] ❌ {error_msg}")
                        error_messages.append(error_msg)
                        diagnostic_info.append(f"❌ {error_msg}")
                        pdf_doc.close()
                        if show_ui_errors:
                            try:
                                st.error(f"❌ **PDF Size Error**: {error_msg}\n\nPlease split the PDF into smaller files (max {MAX_PDF_PAGES} pages each).")
                            except:
                                pass
                        return ""
                    
                    # Check if PDF is encrypted
                    is_encrypted = pdf_doc.is_encrypted
                    needs_password = pdf_doc.needs_pass
                    print(f"[PDF_VISION] PDF info - Pages: {total_pages}, Encrypted: {is_encrypted}, Needs Password: {needs_password}")
                    diagnostic_info.append(f"📄 PDF pages: {total_pages}")
                    if is_encrypted:
                        diagnostic_info.append(f"🔒 PDF is encrypted: {is_encrypted}")
                    if needs_password:
                        error_msg = "PDF is password-protected. Please provide a password or use an unprotected PDF."
                        print(f"[PDF_VISION] ❌ {error_msg}")
                        error_messages.append(error_msg)
                        diagnostic_info.append(f"❌ {error_msg}")
                        if show_ui_errors:
                            try:
                                st.error(f"❌ **PDF Error**: {error_msg}\n\nPlease use an unprotected PDF or provide the password.")
                            except:
                                pass
                        pdf_doc.close()
                        return ""
                    
                    # Check PDF metadata
                    metadata = pdf_doc.metadata
                    if metadata:
                        print(f"[PDF_VISION] PDF metadata: {metadata}")
                    
                    print(f"[PDF_VISION] ✅ PDF opened successfully with {total_pages} pages")
                    diagnostic_info.append(f"✅ PDF opened successfully")
                    
                    failed_pages = []
                    # Process pages incrementally to save memory (don't store all images)
                    # Images will be processed immediately with Vision API
                    for page_num in range(total_pages):
                        try:
                            page = pdf_doc[page_num]
                            
                            # Try to render page to image (150 DPI optimized for Streamlit Cloud - 4x less memory)
                            try:
                                mat = fitz_module.Matrix(150/72, 150/72)  # 150 DPI (reduced from 300 for memory optimization)
                                pix = page.get_pixmap(matrix=mat)
                                
                                # Check if pixmap is valid
                                if pix.width == 0 or pix.height == 0:
                                    error_msg = f"Page {page_num + 1}: Invalid page dimensions (width: {pix.width}, height: {pix.height})"
                                    print(f"[PDF_VISION] ⚠️ {error_msg}")
                                    error_messages.append(error_msg)
                                    failed_pages.append(page_num + 1)
                                    continue
                                
                                # Convert to PIL Image
                                img_data = pix.tobytes("png")
                                if not img_data or len(img_data) == 0:
                                    error_msg = f"Page {page_num + 1}: Failed to generate image data (empty)"
                                    print(f"[PDF_VISION] ⚠️ {error_msg}")
                                    error_messages.append(error_msg)
                                    failed_pages.append(page_num + 1)
                                    continue
                                
                                img = Image.open(io.BytesIO(img_data))
                                if img.size[0] == 0 or img.size[1] == 0:
                                    error_msg = f"Page {page_num + 1}: Invalid image size {img.size}"
                                    print(f"[PDF_VISION] ⚠️ {error_msg}")
                                    error_messages.append(error_msg)
                                    failed_pages.append(page_num + 1)
                                    continue
                                
                                # Compress image if too large (max 2048px dimension)
                                max_dimension = 2048
                                if img.width > max_dimension or img.height > max_dimension:
                                    ratio = min(max_dimension / img.width, max_dimension / img.height)
                                    new_size = (int(img.width * ratio), int(img.height * ratio))
                                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                                    print(f"[PDF_VISION] ⚠️ Page {page_num + 1}: Resized from {img.width}x{img.height} to {new_size[0]}x{new_size[1]}")
                                
                                # Store image for processing (will be processed immediately after conversion)
                                images.append(img)
                                
                                if (page_num + 1) % 3 == 0 or page_num == 0 or page_num == total_pages - 1:
                                    img_size_kb = len(img_data) / 1024
                                    print(f"[PDF_VISION] ✅ Page {page_num + 1}/{total_pages}: Converted to image ({img.width}x{img.height}, {img_size_kb:.1f} KB)")
                                
                                # Free pixmap memory immediately
                                pix = None
                                
                            except Exception as render_error:
                                error_msg = f"Page {page_num + 1}: Render error - {str(render_error)}"
                                print(f"[PDF_VISION] ⚠️ {error_msg}")
                                import traceback
                                print(f"[PDF_VISION] Render error traceback: {traceback.format_exc()}")
                                error_messages.append(error_msg)
                                failed_pages.append(page_num + 1)
                                continue
                        except Exception as page_error:
                            error_msg = f"Page {page_num + 1}: Failed to process - {str(page_error)}"
                            print(f"[PDF_VISION] ⚠️ {error_msg}")
                            import traceback
                            print(f"[PDF_VISION] Page error traceback: {traceback.format_exc()}")
                            error_messages.append(error_msg)
                            failed_pages.append(page_num + 1)
                            continue
                    
                    pdf_doc.close()
                    
                    if failed_pages:
                        warning_msg = f"⚠️ Failed to convert {len(failed_pages)} pages: {failed_pages}"
                        print(f"[PDF_VISION] {warning_msg}")
                        diagnostic_info.append(warning_msg)
                    
                    success_msg = f"✅ Successfully converted {len(images)}/{total_pages} pages to images"
                    print(f"[PDF_VISION] {success_msg}")
                    diagnostic_info.append(success_msg)
                    
                    if len(images) == 0:
                        error_msg = f"Failed to convert any pages to images. Failed pages: {failed_pages}"
                        print(f"[PDF_VISION] ❌ {error_msg}")
                        error_messages.append(error_msg)
                        diagnostic_info.append(f"❌ {error_msg}")
                        if show_ui_errors:
                            try:
                                st.error(f"❌ **PDF Conversion Error**: {error_msg}\n\n**Diagnostic Info**:\n" + "\n".join(f"- {info}" for info in diagnostic_info))
                            except:
                                pass
                        return ""
                except Exception as fitz_error:
                    error_msg = f"PyMuPDF error opening/converting PDF: {str(fitz_error)}"
                    print(f"[PDF_VISION] ⚠️ {error_msg}")
                    import traceback
                    traceback_str = traceback.format_exc()
                    print(f"[PDF_VISION] Error traceback: {traceback_str}")
                    error_messages.append(error_msg)
                    diagnostic_info.append(f"❌ {error_msg}")
                    image_conversion_error = fitz_error
                    fitz_available = False  # Mark as failed to try fallback
            
            # Fallback to pdf2image if PyMuPDF failed or not available
            if not images and not fitz_available:
                print(f"[PDF_VISION] Attempting pdf2image fallback...")
                try:
                    from pdf2image import convert_from_bytes
                    print(f"[PDF_VISION] Using pdf2image to convert PDF to images...")
                    images = convert_from_bytes(pdf_bytes, dpi=150, fmt='RGB')  # Reduced from 300 for memory optimization
                    success_msg = f"✅ pdf2image converted {len(images)} pages"
                    print(f"[PDF_VISION] {success_msg}")
                    diagnostic_info.append(success_msg)
                except ImportError:
                    error_msg = "Neither PyMuPDF nor pdf2image available for PDF to image conversion"
                    print(f"[PDF_VISION] ❌ {error_msg}")
                    error_messages.append(error_msg)
                    diagnostic_info.append(f"❌ {error_msg}")
                    diagnostic_info.append("💡 Install PyMuPDF with: pip install pymupdf (recommended, no system dependencies)")
                    diagnostic_info.append("💡 Or install pdf2image with: pip install pdf2image (requires Poppler)")
                    if show_ui_errors:
                        try:
                            st.error(f"❌ **PDF Vision API Error**: {error_msg}\n\n**Solution**: Install PyMuPDF: `pip install pymupdf` (recommended, no system dependencies)\n\n**Diagnostic Info**:\n" + "\n".join(f"- {info}" for info in diagnostic_info))
                        except:
                            pass
                    return ""
                except Exception as pdf2img_error:
                    error_msg = f"pdf2image failed: {str(pdf2img_error)}"
                    print(f"[PDF_VISION] ❌ {error_msg}")
                    import traceback
                    traceback_str = traceback.format_exc()
                    print(f"[PDF_VISION] pdf2image error traceback: {traceback_str}")
                    error_messages.append(error_msg)
                    diagnostic_info.append(f"❌ {error_msg}")
                    if show_ui_errors:
                        try:
                            st.error(f"❌ **PDF Vision API Error**: {error_msg}\n\n**Diagnostic Info**:\n" + "\n".join(f"- {info}" for info in diagnostic_info))
                        except:
                            pass
                    return ""
        
        except Exception as conv_error:
            error_msg = f"Unexpected error during PDF to image conversion: {str(conv_error)}"
            print(f"[PDF_VISION] ❌ {error_msg}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"[PDF_VISION] Unexpected error traceback: {traceback_str}")
            error_messages.append(error_msg)
            diagnostic_info.append(f"❌ {error_msg}")
            if show_ui_errors:
                try:
                    st.error(f"❌ **PDF Vision API Error**: {error_msg}\n\n**Full Error**:\n```\n{traceback_str}\n```")
                except:
                    pass
            return ""
        
        if not images:
            error_msg = "No images extracted from PDF - cannot process with Vision API"
            print(f"[PDF_VISION] ❌ {error_msg}")
            error_messages.append(error_msg)
            diagnostic_info.append(f"❌ {error_msg}")
            if show_ui_errors:
                try:
                    st.warning(f"⚠️ **PDF Vision API Warning**: {error_msg}\n\n**Diagnostic Info**:\n" + "\n".join(f"- {info}" for info in diagnostic_info))
                except:
                    pass
            return ""
        
        print(f"[PDF_VISION] Converted {len(images)} PDF pages to images")
        diagnostic_info.append(f"🖼️ Converted {len(images)} pages to images")
        
        # Process images with GPT-4 Vision (process first 10 pages to avoid token limits)
        max_pages = min(10, len(images))
        all_text = []
        failed_vision_pages = []
        
        print(f"[PDF_VISION] Processing {max_pages} pages with GPT-4 Vision API...")
        if show_ui_errors and len(images) > max_pages:
            try:
                st.info(f"ℹ️ Processing first {max_pages} of {len(images)} pages (to manage API costs)")
            except:
                pass
        
        for page_num, image in enumerate(images[:max_pages], 1):
            try:
                # Convert PIL Image to base64 with compression
                buffered = io.BytesIO()
                # Use optimize=True and compress_level for smaller file size
                image.save(buffered, format="PNG", optimize=True, compress_level=6)
                img_size_bytes = len(buffered.getvalue())
                img_size_mb = img_size_bytes / (1024 * 1024)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                base64_size_kb = len(img_base64) / 1024
                
                # Check image size (Vision API has limits)
                if img_size_mb > 20:
                    error_msg = f"Page {page_num}: Image too large ({img_size_mb:.2f} MB, max ~20 MB)"
                    print(f"[PDF_VISION] ⚠️ {error_msg}")
                    error_messages.append(error_msg)
                    failed_vision_pages.append(page_num)
                    continue
                
                print(f"[PDF_VISION] Processing page {page_num}/{max_pages} with GPT-4 Vision (image: {image.width}x{image.height}, {img_size_mb:.2f} MB, base64: {base64_size_kb:.1f} KB)...")
                
                # Call GPT-4 Vision API
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",  # GPT-4 Vision model
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert at extracting financial transaction data from documents. Extract all transaction information including dates, tickers, quantities, prices, amounts, and transaction types. Return the data as structured text that can be parsed."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"""Extract all financial transactions from this PDF page (Page {page_num} of {len(images)} from file: {filename}).

Look for transaction data including:
- Transaction dates (in any format - will be normalized)
- Stock/scheme names and tickers/symbols
- Quantities (shares/units)
- Prices (per unit)
- Amounts (total value)
- Transaction types (buy, sell, purchase, redemption, etc.)
- Asset types (stocks, mutual funds, PMS, AIF, bonds)
- Channels/brokers/platforms

Format the output as plain text with one transaction per line, like:
Date: YYYY-MM-DD | Ticker: SYMBOL | Name: Stock Name | Quantity: 100 | Price: 50.00 | Amount: 5000 | Type: buy | Asset: stock | Channel: Broker Name

Or if it's a table, extract all rows with transaction data.

If this page doesn't contain any transaction data, return exactly: "No transactions on this page"

Return ALL transactions found on this page, even if the format is slightly different."""
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=4000,
                    )
                    
                    if not response or not response.choices:
                        error_msg = f"Page {page_num}: Empty response from Vision API"
                        print(f"[PDF_VISION] ⚠️ {error_msg}")
                        error_messages.append(error_msg)
                        failed_vision_pages.append(page_num)
                        continue
                    
                    page_text = response.choices[0].message.content
                    if page_text and page_text.strip():
                        if "No transactions" not in page_text and "no transaction" not in page_text.lower():
                            all_text.append(f"--- Page {page_num} ---\n{page_text}\n")
                            print(f"[PDF_VISION] ✅ Page {page_num}: Extracted {len(page_text)} characters of transaction data")
                        else:
                            print(f"[PDF_VISION] ℹ️ Page {page_num}: Vision API found no transactions on this page")
                    else:
                        print(f"[PDF_VISION] ⚠️ Page {page_num}: Vision API returned empty response")
                        failed_vision_pages.append(page_num)
                        
                except openai.RateLimitError as rate_error:
                    error_msg = f"Page {page_num}: Rate limit exceeded - {str(rate_error)}"
                    print(f"[PDF_VISION] ❌ {error_msg}")
                    error_messages.append(error_msg)
                    failed_vision_pages.append(page_num)
                    if show_ui_errors:
                        try:
                            st.warning(f"⚠️ **API Rate Limit**: {error_msg}\n\nPlease wait a moment and try again.")
                        except:
                            pass
                    continue
                except openai.APIError as api_error:
                    error_msg = f"Page {page_num}: OpenAI API error - {str(api_error)}"
                    print(f"[PDF_VISION] ❌ {error_msg}")
                    import traceback
                    print(f"[PDF_VISION] API error traceback: {traceback.format_exc()}")
                    error_messages.append(error_msg)
                    failed_vision_pages.append(page_num)
                    continue
                except Exception as api_error:
                    error_msg = f"Page {page_num}: Vision API call failed - {str(api_error)}"
                    print(f"[PDF_VISION] ❌ {error_msg}")
                    import traceback
                    print(f"[PDF_VISION] API error traceback: {traceback.format_exc()}")
                    error_messages.append(error_msg)
                    failed_vision_pages.append(page_num)
                    continue
                    
            except Exception as page_error:
                error_msg = f"Page {page_num}: Error processing image - {str(page_error)}"
                print(f"[PDF_VISION] ❌ {error_msg}")
                import traceback
                print(f"[PDF_VISION] Page processing error traceback: {traceback.format_exc()}")
                error_messages.append(error_msg)
                failed_vision_pages.append(page_num)
                continue
        
        if failed_vision_pages:
            warning_msg = f"⚠️ Failed to process {len(failed_vision_pages)} pages with Vision API: {failed_vision_pages}"
            print(f"[PDF_VISION] {warning_msg}")
            diagnostic_info.append(warning_msg)
        
        if all_text:
            combined_text = "\n".join(all_text)
            success_msg = f"✅ Extracted {len(combined_text)} characters from {len(all_text)} pages"
            print(f"[PDF_VISION] {success_msg}")
            diagnostic_info.append(success_msg)
            if show_ui_errors and failed_vision_pages:
                try:
                    st.warning(f"⚠️ **Partial Success**: Extracted data from {len(all_text)} pages, but {len(failed_vision_pages)} pages failed. Failed pages: {failed_vision_pages}")
                except:
                    pass
            return combined_text
        else:
            error_msg = f"No transaction data extracted from any pages. Failed pages: {failed_vision_pages}"
            print(f"[PDF_VISION] ❌ {error_msg}")
            error_messages.append(error_msg)
            diagnostic_info.append(f"❌ {error_msg}")
            if show_ui_errors:
                try:
                    error_display = f"❌ **Vision API Error**: {error_msg}\n\n"
                    if error_messages:
                        error_display += "**Errors encountered**:\n" + "\n".join(f"- {msg}" for msg in error_messages[:5]) + "\n"
                    error_display += "\n**Diagnostic Info**:\n" + "\n".join(f"- {info}" for info in diagnostic_info)
                    st.error(error_display)
                except:
                    pass
            return ""
            
    except ImportError as import_error:
        error_msg = f"Required library not available: {import_error}"
        print(f"[PDF_VISION] ⚠️ {error_msg}")
        import traceback
        traceback_str = traceback.format_exc()
        print(f"[PDF_VISION] Import error traceback: {traceback_str}")
        try:
            import streamlit as st
            st.error(f"❌ **PDF Vision API Error**: {error_msg}\n\n**Solution**: Install required libraries:\n- `pip install pymupdf` (recommended)\n- `pip install openai`\n- `pip install pillow`")
        except:
            pass
        return ""
    except Exception as e:
        error_msg = f"GPT-4 Vision extraction failed: {str(e)}"
        print(f"[PDF_VISION] ❌ {error_msg}")
        import traceback
        traceback_str = traceback.format_exc()
        print(f"[PDF_VISION] Full error traceback:")
        print(traceback_str)
        try:
            import streamlit as st
            st.error(f"❌ **PDF Vision API Error**: {error_msg}\n\n**Full Error**:\n```\n{traceback_str[:500]}...\n```\n\nCheck terminal logs for complete error details.")
        except:
            pass
        return ""


def _tx_extract_tables_from_pdf(uploaded_file) -> List[pd.DataFrame]:
    """Extract tables from PDF as DataFrames."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        print("[PDF_EXTRACT] ⚠️ pdfplumber not installed - cannot extract PDF tables")
        return []

    tables: List[pd.DataFrame] = []
    try:
        uploaded_file.seek(0)
        
        # Convert uploaded_file to BytesIO for pdfplumber compatibility
        import io
        if hasattr(uploaded_file, 'read'):
            pdf_bytes = uploaded_file.read()
        elif hasattr(uploaded_file, 'getvalue'):
            pdf_bytes = uploaded_file.getvalue()
            if isinstance(pdf_bytes, str):
                pdf_bytes = pdf_bytes.encode('utf-8')
        else:
            pdf_bytes = uploaded_file
        
        # Ensure we have bytes
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin-1')
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            print(f"[PDF_EXTRACT] Processing PDF with {total_pages} pages")
            
            for page_index, page in enumerate(pdf.pages, start=1):
                try:
                    page_tables = page.extract_tables()
                    print(f"[PDF_EXTRACT] Page {page_index}: Found {len(page_tables or [])} tables")
                except Exception as e:
                    print(f"[PDF_EXTRACT] Page {page_index}: Error extracting tables: {str(e)}")
                    continue
                    
                for table_index, table in enumerate(page_tables or [], start=1):
                    if not table or len(table) < 2:
                        print(f"[PDF_EXTRACT] Page {page_index}, Table {table_index}: Skipped (too small: {len(table) if table else 0} rows)")
                        continue
                    
                    # Debug: Check raw cell content before cleaning
                    if table_index == 1 and page_index <= 2:  # Debug first few tables
                        print(f"[PDF_EXTRACT] Page {page_index}, Table {table_index}: Raw table sample")
                        print(f"[PDF_EXTRACT]   First row raw: {table[0] if table else 'None'}")
                        if len(table) > 1:
                            print(f"[PDF_EXTRACT]   Second row raw: {table[1] if table else 'None'}")
                    
                    # Clean cells but be more lenient - allow rows with some empty cells
                    # PDF tables often have sparse data where some columns are empty
                    cleaned_rows = []
                    for row_idx, row in enumerate(table):
                        if not row:
                            continue
                        # Clean each cell - preserve content even if it looks empty
                        # PDF extraction might return None or empty strings, but try to get actual content
                        cleaned_cells = []
                        for cell in row:
                            if cell is None:
                                cleaned_cells.append('')
                            elif isinstance(cell, str):
                                # Preserve the string as-is first, then clean
                                cell_str = cell if cell else ''
                                # Only strip if there's actual content
                                if cell_str.strip():
                                    cleaned_cells.append(cell_str.strip())
                                else:
                                    cleaned_cells.append(cell_str)  # Keep original even if empty
                            else:
                                # Try to convert to string - might be a number or other type
                                try:
                                    cell_str = str(cell) if cell is not None else ''
                                    cleaned_cells.append(cell_str.strip() if cell_str.strip() else cell_str)
                                except:
                                    cleaned_cells.append('')
                        
                        # Keep row if it has at least 1 non-empty cell (very lenient for PDFs)
                        # PDFs can have rows with just one important value
                        non_empty_count = sum(1 for cell in cleaned_cells if cell and str(cell).strip())
                        if non_empty_count >= 1:  # At least 1 cell with data (very lenient)
                            cleaned_rows.append(cleaned_cells)
                        elif row_idx == 0:
                            # Always keep first row as potential header, even if mostly empty
                            cleaned_rows.append(cleaned_cells)
                    
                    # More lenient: accept tables with at least 1 data row (header + 1 row minimum)
                    if len(cleaned_rows) < 2:
                        print(f"[PDF_EXTRACT] Page {page_index}, Table {table_index}: Skipped (only {len(cleaned_rows)} valid rows after cleaning)")
                        print(f"[PDF_EXTRACT]   Original table had {len(table)} rows")
                        if table and len(table) > 0:
                            print(f"[PDF_EXTRACT]   First row sample: {table[0][:5] if len(table[0]) > 5 else table[0]}")
                            if len(table) > 1:
                                print(f"[PDF_EXTRACT]   Second row sample: {table[1][:5] if len(table[1]) > 5 else table[1]}")
                        continue
                    
                    # Log table info for debugging - show actual cell content
                    if len(cleaned_rows) >= 2:
                        print(f"[PDF_EXTRACT] Page {page_index}, Table {table_index}: Keeping table with {len(cleaned_rows)} rows")
                        # Show first few cells of header and first data row
                        header_preview = [str(c)[:20] for c in cleaned_rows[0][:5]]  # First 5 columns, first 20 chars
                        print(f"[PDF_EXTRACT]   Header (first 5 cols): {header_preview}")
                        if len(cleaned_rows) > 1:
                            data_preview = [str(c)[:20] for c in cleaned_rows[1][:5]]
                            print(f"[PDF_EXTRACT]   First data row (first 5 cols): {data_preview}")
                        
                        # Check if cells actually have content (not just empty strings)
                        header_has_content = any(c and str(c).strip() for c in cleaned_rows[0])
                        data_has_content = any(c and str(c).strip() for c in cleaned_rows[1]) if len(cleaned_rows) > 1 else False
                        if not header_has_content and not data_has_content:
                            print(f"[PDF_EXTRACT]   ⚠️ WARNING: Table cells appear to be empty - may need OCR or different extraction method")
                    
                    header = cleaned_rows[0]
                    body = cleaned_rows[1:]
                    
                    # Normalize header: clean up column names for better mapping
                    # PDF extraction can produce headers with extra spaces, None values, etc.
                    normalized_header = []
                    for i, col in enumerate(header):
                        # Use _tx_safe_str to normalize (handles None, empty, whitespace)
                        clean_col = _tx_safe_str(col)
                        # If column is empty/None, create a default name
                        if not clean_col:
                            clean_col = f"Column_{i+1}"
                        normalized_header.append(clean_col)
                    
                    try:
                        df = pd.DataFrame(body, columns=normalized_header)
                        df.attrs['__tx_source'] = f"page_{page_index}_table_{table_index}"
                        print(f"[PDF_EXTRACT] Page {page_index}, Table {table_index}: Created DataFrame {df.shape}, columns: {list(df.columns)}")
                        tables.append(df)
                    except Exception as e:
                        print(f"[PDF_EXTRACT] Page {page_index}, Table {table_index}: Error creating DataFrame: {str(e)}")
                        continue
                        
            print(f"[PDF_EXTRACT] Total tables extracted: {len(tables)}")
            
            # If no tables found, try alternative extraction methods
            if len(tables) == 0:
                print(f"[PDF_EXTRACT] No tables found with standard extraction, trying alternative methods...")
                uploaded_file.seek(0)
                
                # Convert uploaded_file to BytesIO for pdfplumber compatibility
                import io
                if hasattr(uploaded_file, 'read'):
                    pdf_bytes = uploaded_file.read()
                elif hasattr(uploaded_file, 'getvalue'):
                    pdf_bytes = uploaded_file.getvalue()
                    if isinstance(pdf_bytes, str):
                        pdf_bytes = pdf_bytes.encode('utf-8')
                else:
                    pdf_bytes = uploaded_file
                
                # Ensure we have bytes
                if isinstance(pdf_bytes, str):
                    pdf_bytes = pdf_bytes.encode('latin-1')
                
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    # Try extracting all text and looking for tabular patterns
                    for page_index, page in enumerate(pdf.pages, start=1):
                        try:
                            # Try with different table settings (removed invalid edge_tolerance)
                            page_tables = page.extract_tables(table_settings={
                                "vertical_strategy": "lines_strict",
                                "horizontal_strategy": "lines_strict",
                                "explicit_vertical_lines": [],
                                "explicit_horizontal_lines": [],
                                "snap_tolerance": 3,
                                "join_tolerance": 3,
                                "min_words_vertical": 1,
                                "min_words_horizontal": 1,
                            })
                            if page_tables:
                                print(f"[PDF_EXTRACT] Page {page_index}: Found {len(page_tables)} tables with alternative settings")
                                # Process these tables with the same logic above
                                for table_index, table in enumerate(page_tables, start=1):
                                    if not table or len(table) < 2:
                                        continue
                                    cleaned_rows = []
                                    for row in table:
                                        if not row:
                                            continue
                                        cleaned_cells = [cell.strip() if isinstance(cell, str) else (str(cell).strip() if cell is not None else '') for cell in row]
                                        non_empty_count = sum(1 for cell in cleaned_cells if cell and str(cell).strip())
                                        if non_empty_count >= 2:
                                            cleaned_rows.append(cleaned_cells)
                                    if len(cleaned_rows) >= 2:
                                        header = cleaned_rows[0]
                                        body = cleaned_rows[1:]
                                        normalized_header = []
                                        for i, col in enumerate(header):
                                            clean_col = _tx_safe_str(col)
                                            if not clean_col:
                                                clean_col = f"Column_{i+1}"
                                            normalized_header.append(clean_col)
                                        try:
                                            df = pd.DataFrame(body, columns=normalized_header)
                                            df.attrs['__tx_source'] = f"page_{page_index}_table_{table_index}_alt"
                                            print(f"[PDF_EXTRACT] Page {page_index}, Table {table_index} (alt): Created DataFrame {df.shape}")
                                            tables.append(df)
                                        except Exception as e:
                                            print(f"[PDF_EXTRACT] Page {page_index}, Table {table_index} (alt): Error creating DataFrame: {str(e)}")
                                            continue
                        except Exception as e:
                            print(f"[PDF_EXTRACT] Page {page_index}: Alternative extraction failed: {str(e)}")
                            continue
                            
                print(f"[PDF_EXTRACT] After alternative extraction: {len(tables)} total tables")
            
            # If still no tables, try text-based extraction (for image-based PDFs or when table detection fails)
            if len(tables) == 0:
                print(f"[PDF_EXTRACT] Still no tables found, trying text extraction as fallback...")
                uploaded_file.seek(0)
                
                # Convert uploaded_file to BytesIO for pdfplumber compatibility
                import io
                if hasattr(uploaded_file, 'read'):
                    pdf_bytes = uploaded_file.read()
                elif hasattr(uploaded_file, 'getvalue'):
                    pdf_bytes = uploaded_file.getvalue()
                    if isinstance(pdf_bytes, str):
                        pdf_bytes = pdf_bytes.encode('utf-8')
                else:
                    pdf_bytes = uploaded_file
                
                # Ensure we have bytes
                if isinstance(pdf_bytes, str):
                    pdf_bytes = pdf_bytes.encode('latin-1')
                
                try:
                    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                        all_text_content = []
                        total_chars = 0
                        for page_index, page in enumerate(pdf.pages, start=1):
                            try:
                                # Try multiple extraction methods
                                page_text = page.extract_text()
                                if not page_text or not page_text.strip():
                                    # Try extracting from words/chars directly
                                    words = page.extract_words()
                                    if words:
                                        page_text = ' '.join([w.get('text', '') for w in words if w.get('text')])
                                
                                if page_text and page_text.strip():
                                    all_text_content.append(f"--- Page {page_index} ---\n{page_text}")
                                    char_count = len(page_text)
                                    total_chars += char_count
                                    print(f"[PDF_EXTRACT] Page {page_index}: Extracted {char_count} characters of text")
                                    # Show sample of extracted text
                                    sample = page_text[:200].replace('\n', ' ')
                                    print(f"[PDF_EXTRACT]   Sample: {sample}...")
                                else:
                                    print(f"[PDF_EXTRACT] Page {page_index}: No text found - may be image-based/scanned PDF")
                                    # Check if page has any words/chars at all
                                    words = page.extract_words()
                                    chars = page.chars
                                    print(f"[PDF_EXTRACT]   Words found: {len(words) if words else 0}, Chars found: {len(chars) if chars else 0}")
                            except Exception as e:
                                print(f"[PDF_EXTRACT] Page {page_index}: Text extraction failed: {str(e)}")
                                continue
                        
                        if all_text_content:
                            print(f"[PDF_EXTRACT] ✅ Extracted text from {len(all_text_content)}/{len(pdf.pages)} pages ({total_chars} total characters)")
                            print(f"[PDF_EXTRACT]   AI fallback should be able to process this text content")
                        else:
                            print(f"[PDF_EXTRACT] ⚠️ No extractable text found - PDF appears to be image-based/scanned")
                            print(f"[PDF_EXTRACT]   Attempting OCR extraction (same method as AI Assistant uses)...")
                            # Try OCR as last resort - same approach as AI Assistant
                            ocr_text = _extract_pdf_with_ocr(uploaded_file)
                            if ocr_text and ocr_text.strip():
                                print(f"[PDF_EXTRACT] ✅ OCR extracted {len(ocr_text)} characters")
                                # Store OCR text for AI processing
                                # The AI fallback will process this OCR text
                                all_text_content = [ocr_text]
                            else:
                                print(f"[PDF_EXTRACT]   OCR also failed or not available")
                                print(f"[PDF_EXTRACT]   Attempting GPT-4 Vision API as final fallback...")
                                # Try GPT-4 Vision API as final fallback
                                vision_text = _extract_pdf_with_vision_api(uploaded_file, "uploaded_file.pdf")
                                if vision_text and vision_text.strip():
                                    print(f"[PDF_EXTRACT] ✅ GPT-4 Vision extracted {len(vision_text)} characters")
                                    all_text_content = [vision_text]
                                    # Store the extracted text for AI processing
                                    # This will be used by the AI extraction function
                                    uploaded_file._vision_api_text = vision_text
                                    print(f"[PDF_EXTRACT] ✅ Stored Vision API text on file object (attribute set: {hasattr(uploaded_file, '_vision_api_text')})")
                                else:
                                    print(f"[PDF_EXTRACT]   GPT-4 Vision also failed or not available")
                                    print(f"[PDF_EXTRACT]   Recommendation: Install OCR dependencies (pdf2image, pytesseract, Tesseract) or convert PDF to text-selectable format")
                except Exception as e:
                    print(f"[PDF_EXTRACT] Text extraction fallback failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
    except Exception as e:
        print(f"[PDF_EXTRACT] Error processing PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    return tables


_TX_COLUMN_ALIASES: Dict[str, List[str]] = {
    'date': [
        'date', 'Date', 'transaction date', 'trade date', 'tx date', 'tran date',
        'order date', 'purchase date', 'nav date', 'valuation date',
        'execution', 'execution date', 'Execution date', 'execution date and time', 'Execution date and time',
        'Execution Date and Time', 'deal date', 'order datetime', 'order date / time'
    ],
    'ticker': [
        'ticker', 'Ticker', 'symbol', 'Symbol', 'SYMBOL', 'scrip symbol', 'scrip', 'scrip sym',
        'trading symbol', 'script', 'code', 'bse code', 'nse code',
        'security code', 'instrument code', 'scheme code',
        'isin code', 'isin', 'ISIN', 'amfi code', 'amfi',
        'investment code', 'contract symbol', 'security id'
    ],
    'stock_name': [
        'stock name', 'Stock name', 'Stock Name', 'STOCK NAME', 'scrip name', 'security name', 'instrument', 'name', 'scheme name',
        'company', 'fund name', 'description', 'asset name', 'holding',
        'product', 'contract description', 'company name'
    ],
    'scheme_name': [
        'scheme name', 'fund scheme', 'scheme', 'schemename', 'plan name'
    ],
    'quantity': [
        'quantity', 'Quantity', 'QTY', 'qty', 'units', 'Units', 'UNITS', 'shares', 'quantity/unit',
        'no. of units', 'no of units', 'units/qty', 'units (credit)',
        'units (debit)', 'lot size', 'filled quantity', 'executed quantity',
        'number of units', 'no. of shares', 'no of shares', 'unit balance',
        'units held', 'units balance', 'total units', 'current units',
        'holding quantity', 'holding units', 'balance units', 'balance quantity'
    ],
    'price': [
        'price', 'rate', 'nav', 'purchase price', 'per unit price',
        'trade price', 'unit price', 'cost price', 'nav per unit',
        'deal price', 'executed price', 'execution price', 'order price', 'strike price'
    ],
    'amount': [
        'amount', 'Amount', 'value', 'Value', 'VALUE', 'total value', 'Total Value', 'consideration', 'txn amount',
        'gross amount', 'net amount', 'investment amount', 'order value',
        'total', 'investment', 'transaction value', 'contract value', 'turnover'
    ],
    'transaction_type': [
        'transaction type', 'Transaction Type', 'type', 'Type', 'TYPE', 'action', 'side', 'buy/sell',
        'txn type', 'order type', 'direction', 'nature', 'transaction',
        'mode', 'buy/sell/other', 'transaction mode', 'trade type'
    ],
    'asset_type': [
        'asset type', 'instrument type', 'category', 'asset class',
        'type of asset', 'class', 'instrument category'
    ],
    'channel': [
        'channel', 'Channel', 'platform', 'broker', 'account', 'source', 'portfolio',
        'through', 'partner', 'advisor', 'exchange', 'Exchange', 'EXCHANGE'
    ],
    'sector': [
        'sector', 'industry', 'sector name', 'segment'
    ],
    'notes': [
        'notes', 'Notes', 'remarks', 'comment', 'description', 'order status', 'Order status', 'Order Status', 'ORDER STATUS',
        'exchange order id', 'Exchange Order Id', 'Exchange Order ID', 'order id', 'Order Id'
    ],
    'folio': [
        'folio', 'folio no', 'folio number'
    ],
}

_TX_CANONICAL_COLUMNS = set(_TX_COLUMN_ALIASES.keys())
_TX_AUTO_MAP_TARGETS = (
    'date',
    'ticker',
    'stock_name',
    'quantity',
    'price',
    'amount',
    'transaction_type',
)

_TX_AUTO_MAP_THRESHOLD: Dict[str, float] = {
    'date': 0.7,
    'ticker': 0.65,
    'stock_name': 0.6,
    'quantity': 0.6,
    'price': 0.6,
    'amount': 0.6,
    'transaction_type': 0.55,
}


def _tx_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names based on aliases."""
    rename_map: Dict[str, str] = {}
    used_targets: set[str] = set()
    mapping_entries: List[Dict[str, Any]] = []

    # Hard rules for common broker exports (e.g. "Scrip Symbol" → ticker, "Scrip Name" → stock_name)
    HARD_MAP = {
        'ticker': {'scrip symbol', 'scrip-symbol', 'scrip_symbol', 'trading symbol'},
        'stock_name': {'scrip name', 'scrip-name', 'scrip_name'},
    }

    # First, mark columns that already have canonical names as used
    all_canonical_names = set(_TX_COLUMN_ALIASES.keys())
    all_canonical_names.update(HARD_MAP.keys())
    for col in df.columns:
        col_lower = _tx_safe_str(col).lower()
        if col_lower in all_canonical_names:
            used_targets.add(col_lower)

    # Process columns in two passes:
    # 1. First pass: Exact matches (prioritize these)
    # 2. Second pass: Partial matches
    
    exact_matches = {}
    partial_matches = {}
    
    for col in df.columns:
        col_name = _tx_safe_str(col).lower()
        target = None
        is_exact = False

        # If column already has canonical name, keep it (exact match)
        if col_name in all_canonical_names:
            target = col_name
            is_exact = True
        else:
            # Apply hard mappings first
            for hard_target, aliases in HARD_MAP.items():
                if col_name in aliases:
                    target = hard_target
                    is_exact = (col_name == hard_target.lower())
                    break

            if target is None:
                for canonical, aliases in _TX_COLUMN_ALIASES.items():
                    for alias in aliases:
                        alias_lower = alias.lower()
                        # Exact match (preferred)
                        if col_name == alias_lower:
                            target = canonical
                            is_exact = True
                            break
                        # Partial match (only if no exact match found)
                        elif alias_lower in col_name and target is None:
                            target = canonical
                            is_exact = False
                    if target:
                        break

        if target:
            if is_exact:
                exact_matches[col] = target
            else:
                partial_matches[col] = target
        else:
            rename_map[col] = col
    
    # Process exact matches first (they take priority)
    for col, target in exact_matches.items():
        if target not in used_targets:
            rename_map[col] = target
            used_targets.add(target)
            if col != target:
                mapping_entries.append({
                    'source': col,
                    'target': target,
                    'reason': 'exact_match',
                })
        else:
            rename_map[col] = col
    
    # Then process partial matches (only if target not already used)
    for col, target in partial_matches.items():
        if target not in used_targets:
            rename_map[col] = target
            used_targets.add(target)
            mapping_entries.append({
                'source': col,
                'target': target,
                'reason': 'partial_match',
            })
        else:
            # If target already used by exact match, keep original name
            rename_map[col] = col

    df = df.rename(columns=rename_map)
    
    # CRITICAL: Normalize all column names to lowercase after mapping
    # This ensures "Quantity" becomes "quantity" so checks like 'quantity' in df.columns work
    column_normalization = {}
    for col in df.columns:
        col_lower = col.lower()
        if col != col_lower:
            column_normalization[col] = col_lower
    
    if column_normalization:
        df = df.rename(columns=column_normalization)
        print(f"[COLUMN_MAP] Normalized column names to lowercase: {column_normalization}")
    else:
        # Double-check: if we still have "Quantity" or other capitalized canonical names, force normalize
        canonical_lowercase = {'quantity', 'amount', 'price', 'date', 'ticker', 'stock_name', 'transaction_type'}
        force_normalize = {}
        for col in df.columns:
            if col.lower() in canonical_lowercase and col != col.lower():
                force_normalize[col] = col.lower()
        if force_normalize:
            df = df.rename(columns=force_normalize)
            print(f"[COLUMN_MAP] Force normalized canonical columns: {force_normalize}")
    
    if mapping_entries:
        df.attrs.setdefault('__tx_column_mapping', [])
        df.attrs['__tx_column_mapping'].extend(mapping_entries)
        # Log the mappings for debugging
        print(f"[COLUMN_MAP] Column mappings applied:")
        for entry in mapping_entries:
            print(f"[COLUMN_MAP]   '{entry['source']}' → '{entry['target']}' ({entry['reason']})")
    
    # Log final column names after mapping
    print(f"[COLUMN_MAP] Final columns after mapping: {list(df.columns)}")
    
    return df


def _tx_auto_map_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Use statistical heuristics to map remaining columns to canonical names."""
    if df is None or df.empty:
        return df

    existing_lower = {col.lower() for col in df.columns}
    missing_targets = [target for target in _TX_AUTO_MAP_TARGETS if target not in existing_lower]
    if not missing_targets:
        return df

    def _sample(series: pd.Series, sample_size: int = 60) -> pd.Series:
        return series.dropna().head(sample_size)

    def _score_date(series: pd.Series) -> float:
        sample = _sample(series)
        if sample.empty:
            return 0.0
        sample_str = sample.astype(str).str.strip()
        parsed_dayfirst = pd.to_datetime(sample_str, errors='coerce', dayfirst=True)
        parsed_monthfirst = pd.to_datetime(sample_str, errors='coerce', dayfirst=False)
        return float(max(parsed_dayfirst.notna().mean(), parsed_monthfirst.notna().mean()))

    def _score_transaction_type(series: pd.Series) -> float:
        sample = _sample(series)
        if sample.empty:
            return 0.0
        keywords = {
            'buy', 'sell', 'redeem', 'redemption', 'sip', 'switch in', 'switch out',
            'subscription', 'allotment', 'investment', 'withdrawal', 'debit', 'credit',
        }
        values = [str(val).strip().lower() for val in sample]
        matches = sum(1 for value in values if any(token in value for token in keywords))
        return float(matches / len(values))

    def _score_numeric(series: pd.Series) -> float:
        sample = _sample(series)
        if sample.empty:
            return 0.0
        numeric = pd.to_numeric(sample, errors='coerce')
        if numeric.empty:
            return 0.0
        return float(numeric.notna().mean())

    def _score_quantity(series: pd.Series) -> float:
        sample = _sample(series)
        if sample.empty:
            return 0.0
        numeric = pd.to_numeric(sample, errors='coerce')
        numeric = numeric.dropna()
        if numeric.empty:
            return 0.0
        integerish = (numeric.round(4) - numeric.round()).abs() <= 0.01
        positive_ratio = (numeric > 0).mean() if not numeric.empty else 0.0
        return float(min(1.0, 0.6 * integerish.mean() + 0.4 * positive_ratio))

    def _score_price(series: pd.Series) -> float:
        sample = _sample(series)
        if sample.empty:
            return 0.0
        numeric = pd.to_numeric(sample, errors='coerce').dropna()
        if numeric.empty:
            return 0.0
        mean_val = numeric.abs().mean()
        if mean_val <= 0:
            return 0.0
        decimal_ratio = ((numeric - numeric.round(2)).abs() > 0.005).mean()
        return float(min(1.0, 0.5 * decimal_ratio + 0.5 * min(mean_val / 1000, 1.0)))

    def _score_amount(series: pd.Series) -> float:
        sample = _sample(series)
        if sample.empty:
            return 0.0
        numeric = pd.to_numeric(sample, errors='coerce').dropna()
        if numeric.empty:
            return 0.0
        mean_val = numeric.abs().mean()
        if mean_val <= 0:
            return 0.0
        return float(min(1.0, 0.4 + min(mean_val / 50000, 0.6)))

    def _score_ticker(series: pd.Series) -> float:
        sample = _sample(series)
        if sample.empty:
            return 0.0
        values = [str(val).strip() for val in sample]

        def looks_like_ticker(value: str) -> bool:
            if not value or len(value) > 25:
                return False
            stripped = value.replace('.', '').replace('-', '').replace(' ', '')
            if not stripped:
                return False
            if stripped.isdigit() and 3 <= len(stripped) <= 12:
                return True
            if not all(ch.isalnum() or ch in {'.', '-', '/', '&'} for ch in value):
                return False
            alpha_chars = sum(ch.isalpha() for ch in value)
            if alpha_chars == 0:
                return False
            upper_ratio = sum(ch.isupper() for ch in value if ch.isalpha()) / alpha_chars
            return upper_ratio >= 0.6

        matches = sum(1 for value in values if looks_like_ticker(value))
        uniqueness = len(set(values)) / len(values)
        base_score = matches / len(values)
        return float(min(1.0, 0.7 * base_score + 0.3 * uniqueness))

    def _score_stock_name(series: pd.Series) -> float:
        sample = _sample(series)
        if sample.empty:
            return 0.0
        values = [str(val).strip() for val in sample]
        meaningful = sum(1 for value in values if len(value) >= 3 and any(ch.isalpha() for ch in value))
        spaced = sum(1 for value in values if ' ' in value or value.title() == value)
        return float(min(1.0, 0.5 * (meaningful / len(values)) + 0.5 * (spaced / len(values))))

    scoring_functions = {
        'date': _score_date,
        'ticker': _score_ticker,
        'stock_name': _score_stock_name,
        'quantity': _score_quantity,
        'price': _score_price,
        'amount': _score_amount,
        'transaction_type': _score_transaction_type,
    }

    rename_map: Dict[str, str] = {}
    mapping_entries: List[Dict[str, Any]] = []
    used_sources: set[str] = set()

    for target in missing_targets:
        best_col = None
        best_score = 0.0
        scorer = scoring_functions.get(target)
        if not scorer:
            continue

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in _TX_CANONICAL_COLUMNS or col_lower == target:
                continue
            if col in rename_map or col in used_sources:
                continue
            score = scorer(df[col])
            if score > best_score:
                best_score = score
                best_col = col

        threshold = _TX_AUTO_MAP_THRESHOLD.get(target, 0.7)
        if best_col and best_score >= threshold:
            rename_map[best_col] = target
            used_sources.add(best_col)
            mapping_entries.append({
                'source': best_col,
                'target': target,
                'reason': 'heuristic_match',
                'score': round(best_score, 3),
            })

    if not rename_map:
        return df

    df = df.rename(columns=rename_map)
    if mapping_entries:
        df.attrs.setdefault('__tx_column_mapping', [])
        df.attrs['__tx_column_mapping'].extend(mapping_entries)
        # Log the auto-mappings for debugging
        print(f"[COLUMN_MAP] Auto-mapped columns (heuristic):")
        for entry in mapping_entries:
            print(f"[COLUMN_MAP]   '{entry['source']}' → '{entry['target']}' (score: {entry.get('score', 'N/A')})")
    
    return df


def _tx_ai_map_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Use AI to intelligently map columns to canonical names.
    This is used when standard mapping fails to identify columns.
    """
    if df is None or df.empty:
        return df
    
    try:
        import openai
        openai_client = openai.OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
    except Exception as e:
        print(f"[COLUMN_MAP] ⚠️ Cannot use AI mapping: {e}")
        return df
    
    # Get sample data to help AI understand the columns
    sample_rows = min(5, len(df))
    sample_data = df.head(sample_rows).to_dict('records')
    
    # Prepare column information for AI
    column_info = []
    for col in df.columns:
        sample_values = [str(row.get(col, ''))[:50] for row in sample_data[:3] if row.get(col) is not None]
        column_info.append({
            'name': col,
            'sample_values': sample_values[:3]  # First 3 non-null values
        })
    
    prompt = f"""You are a financial data expert. Analyze the columns in this transaction file and map them to standard column names.

File: {filename}
Columns found: {[col['name'] for col in column_info]}

Column details:
{chr(10).join([f"- '{col['name']}': {col['sample_values']}" for col in column_info])}

Standard column names we need:
- 'date': Transaction date (any date format)
- 'ticker': Stock/symbol identifier (e.g., RELIANCE.NS, ISIN codes, mutual fund codes)
- 'stock_name': Full name of the security/stock
- 'quantity': Number of shares/units (numeric)
- 'price': Price per unit (numeric)
- 'amount': Total transaction value/amount (numeric)
- 'transaction_type': Buy/Sell indicator
- 'scheme_name': Mutual fund scheme name (if applicable)

Your task:
1. Analyze each column name and its sample values
2. Map each column to the most appropriate standard name
3. Return ONLY a JSON object with mappings: {{"original_column_name": "standard_name"}}

Example output:
{{"Execution date and time": "date", "Symbol": "ticker", "Stock name": "stock_name", "Quantity": "quantity", "Value": "amount"}}

IMPORTANT:
- Map based on meaning, not just name similarity
- If a column clearly represents quantity (has numeric values that look like share counts), map to "quantity"
- If a column represents total value/amount, map to "amount"
- If price column is missing but we have quantity and amount, that's OK (we'll calculate it)
- Return ONLY the JSON mapping, no explanation
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for column mapping
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing financial transaction files and mapping columns to standard names. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=500
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        import json
        import re
        
        # Find JSON object in response (handle nested objects)
        # Try to find the JSON object by looking for { ... }
        json_start = ai_response.find('{')
        json_end = ai_response.rfind('}')
        if json_start >= 0 and json_end > json_start:
            json_str = ai_response[json_start:json_end + 1]
            try:
                mapping_json = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to extract just the mapping part
                mapping_json = None
        else:
            mapping_json = None
        
        if mapping_json:
            
            # Apply mappings
            rename_map = {}
            mapping_entries = []
            for original_col, standard_col in mapping_json.items():
                if original_col in df.columns and standard_col in _TX_COLUMN_ALIASES:
                    rename_map[original_col] = standard_col
                    mapping_entries.append({
                        'source': original_col,
                        'target': standard_col,
                        'reason': 'ai_mapping'
                    })
            
            if rename_map:
                df = df.rename(columns=rename_map)
                df.attrs.setdefault('__tx_column_mapping', [])
                df.attrs['__tx_column_mapping'].extend(mapping_entries)
                print(f"[COLUMN_MAP] ✅ AI mapped columns:")
                for entry in mapping_entries:
                    print(f"[COLUMN_MAP]   '{entry['source']}' → '{entry['target']}' (AI)")
                return df
            else:
                print(f"[COLUMN_MAP] ⚠️ AI returned mappings but none were valid")
        else:
            print(f"[COLUMN_MAP] ⚠️ Could not extract JSON from AI response: {ai_response[:200]}")
    
    except Exception as e:
        print(f"[COLUMN_MAP] ⚠️ AI column mapping failed: {e}")
    
    return df


def _tx_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up raw dataframes by stripping metadata rows and fixing headers."""
    if df is None or df.empty:
        return df

    df = df.copy()
    df.attrs['__tx_original_columns'] = list(df.columns)
    df.attrs.setdefault('__tx_column_mapping', [])

    # Normalize column labels
    normalized_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            col = ' '.join([_tx_safe_str(part) for part in col if part is not None])
        normalized_columns.append(_tx_safe_str(col))
    df.columns = normalized_columns

    # Drop empty rows/cols
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    if df.empty:
        return df

    # Check if current columns already match known aliases - if so, don't replace them
    current_cols_lower = [col.lower() for col in df.columns]
    alias_tokens = set()
    canonical_names = set()
    for canonical, aliases in _TX_COLUMN_ALIASES.items():
        canonical_names.add(canonical.lower())
        for alias in aliases:
            alias_tokens.add(alias.lower())
    
    # Count how many current columns match known aliases
    matching_cols = sum(1 for col in current_cols_lower if col in alias_tokens or col in canonical_names)
    has_good_headers = matching_cols >= 3  # If 3+ columns match, assume headers are correct
    
    # Only detect header row if current headers don't look correct
    if not has_good_headers:
        def score_row(row: pd.Series) -> float:
            score = 0.0
            for value in row.values:
                text = _tx_safe_str(value).lower()
                if not text:
                    continue
                if text in alias_tokens:
                    score += 3.0
                elif any(token in text for token in alias_tokens):
                    score += 1.5
                elif len(text) <= 25:
                    score += 0.5
            return score

        header_candidates = df.head(8)
        best_idx = None
        best_score = 0.0
        for idx, row in header_candidates.iterrows():
            score = score_row(row)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Only replace headers if the found row scores significantly higher than current columns
        if best_idx is not None and best_score > matching_cols * 2.0:
            header_row = df.loc[best_idx]
            new_columns = [ _tx_safe_str(val) for val in header_row.values ]
            df = df.loc[best_idx + 1 :].reset_index(drop=True)
            df.columns = new_columns

    return df


def _tx_dataframe_to_transactions(
    df: pd.DataFrame,
    filename: str,
    sheet_label: Optional[str] = None,
    debug_log: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Convert a standardized DataFrame into transaction dicts."""
    transactions: List[Dict[str, Any]] = []

    if df is None or df.empty:
        return transactions

    # Debug: Check original DataFrame
    print(f"[FILE_PARSE] Original DataFrame columns: {list(df.columns)}")
    print(f"[FILE_PARSE] Original DataFrame shape: {df.shape}")
    
    prepared_df = _tx_prepare_dataframe(df)
    print(f"[FILE_PARSE] After _tx_prepare_dataframe columns: {list(prepared_df.columns)}")
    
    prepared_df = prepared_df.dropna(how='all')
    if prepared_df.empty:
        return transactions

    prepared_df = _tx_standardize_columns(prepared_df)
    print(f"[FILE_PARSE] After _tx_standardize_columns columns: {list(prepared_df.columns)}")
    
    prepared_df = _tx_auto_map_missing_columns(prepared_df)
    print(f"[FILE_PARSE] After _tx_auto_map_missing_columns columns: {list(prepared_df.columns)}")
    
    # If critical columns are still missing, use AI to intelligently map them
    critical_columns = ['quantity', 'amount', 'date', 'ticker']
    missing_critical = [col for col in critical_columns if col not in [c.lower() for c in prepared_df.columns]]
    if missing_critical and not prepared_df.empty:
        print(f"[COLUMN_MAP] ⚠️ Missing critical columns after standard mapping: {missing_critical}")
        print(f"[COLUMN_MAP] Attempting AI-powered column mapping...")
        prepared_df = _tx_ai_map_columns(prepared_df, filename)
        print(f"[FILE_PARSE] After _tx_ai_map_columns columns: {list(prepared_df.columns)}")

    if debug_log is not None:
        # Extract just the filename from path (handles full paths and hyphens correctly)
        sheet_name = sheet_label or Path(filename).stem
        debug_log.append({
            'sheet': sheet_name,
            'rows': int(prepared_df.shape[0]),
            'columns': list(prepared_df.columns),
            'original_columns': prepared_df.attrs.get('__tx_original_columns', []),
            'mapping': prepared_df.attrs.get('__tx_column_mapping', []),
        })

    rows_processed = 0
    rows_skipped_date = 0
    rows_skipped_ticker = 0
    rows_added = 0

    # Debug: Print actual column names
    if rows_processed == 0:
        print(f"[FILE_PARSE] DataFrame columns: {list(prepared_df.columns)}")
        print(f"[FILE_PARSE] DataFrame shape: {prepared_df.shape}")
        if not prepared_df.empty:
            print(f"[FILE_PARSE] First row sample: {prepared_df.iloc[0].to_dict()}")

    for idx, row in prepared_df.iterrows():
        rows_processed += 1
        
        # Get date - try multiple column name variations
        # Accept ANY date format - normalization will handle parsing later
        raw_date = None
        date_cols = ['date', 'transaction_date', 'transaction date', 'Date', 'DATE', 'Transaction Date']
        
        # First try exact column name match using prepared_df.columns (not row.index)
        for col in date_cols:
            if col in prepared_df.columns:
                try:
                    val = row[col]
                    # Check if value exists and is not NaN/None/empty
                    if val is not None and not pd.isna(val):
                        val_str = str(val).strip()
                        if val_str and val_str.lower() not in ['nan', 'none', 'null', 'n/a', '']:
                            raw_date = val
                            break
                except (KeyError, IndexError):
                    continue
        
        # If not found, try case-insensitive search on actual DataFrame columns
        if raw_date is None:
            for col in prepared_df.columns:
                if isinstance(col, str) and col.lower() in [d.lower() for d in date_cols]:
                    try:
                        val = row[col]
                        if val is not None and not pd.isna(val):
                            val_str = str(val).strip()
                            if val_str and val_str.lower() not in ['nan', 'none', 'null', 'n/a', '']:
                                raw_date = val
                                break
                    except (KeyError, IndexError):
                        continue
        
        # Check if date is missing - be lenient, just check if there's any value
        if raw_date is None:
            rows_skipped_date += 1
            if rows_skipped_date <= 3:  # Log first 3 skipped rows for debugging
                print(f"[FILE_PARSE] Row {idx} skipped - no date found")
                print(f"[FILE_PARSE]   Looking for columns: {date_cols}")
                print(f"[FILE_PARSE]   Available columns: {list(prepared_df.columns)}")
                print(f"[FILE_PARSE]   Row data: {dict(row)}")
            continue
        
        # Convert to string for processing (normalization happens later)
        date_str = str(raw_date).strip()
        
        # Get values from mapped columns (after standardization)
        # These should be the canonical names: 'quantity', 'amount', 'price'
        # Check for both lowercase and any case variations
        raw_quantity = None
        raw_amount = None
        raw_price = None
        
        # Try lowercase first (canonical name)
        if 'quantity' in prepared_df.columns:
            raw_quantity = row.get('quantity')
        else:
            # Try case variations
            for col in prepared_df.columns:
                if col.lower() == 'quantity':
                    raw_quantity = row.get(col)
                    break
        
        if 'amount' in prepared_df.columns:
            raw_amount = row.get('amount')
        else:
            for col in prepared_df.columns:
                if col.lower() == 'amount':
                    raw_amount = row.get(col)
                    break
        
        if 'price' in prepared_df.columns:
            raw_price = row.get('price')
        else:
            for col in prepared_df.columns:
                if col.lower() == 'price':
                    raw_price = row.get(col)
                    break

        # Debug: Log what we found
        if idx == 0:  # Only log for first row to avoid spam
            print(f"[FILE_PARSE] Row {idx}: Checking for columns in prepared_df")
            print(f"[FILE_PARSE]   Available columns: {list(prepared_df.columns)}")
            print(f"[FILE_PARSE]   'quantity' in columns: {'quantity' in prepared_df.columns}, value: {raw_quantity}")
            print(f"[FILE_PARSE]   'amount' in columns: {'amount' in prepared_df.columns}, value: {raw_amount}")
            print(f"[FILE_PARSE]   'price' in columns: {'price' in prepared_df.columns}, value: {raw_price}")

        # Check if values are actually present in the row (not just None)
        # For amount and quantity, also check if the value is 0 (0 is a valid value, not missing)
        quantity_present = raw_quantity is not None and not pd.isna(raw_quantity) and str(raw_quantity).strip() != ''
        amount_present = raw_amount is not None and not pd.isna(raw_amount) and str(raw_amount).strip() != ''
        price_present = raw_price is not None and not pd.isna(raw_price) and str(raw_price).strip() != ''
        
        # Debug: Log presence flags for first row
        if idx == 0:
            print(f"[FILE_PARSE] Row {idx} presence flags:")
            print(f"[FILE_PARSE]   quantity_present={quantity_present}, raw_quantity={raw_quantity}")
            print(f"[FILE_PARSE]   amount_present={amount_present}, raw_amount={raw_amount}")
            print(f"[FILE_PARSE]   price_present={price_present}, raw_price={raw_price}")

        quantity_value = _tx_safe_float(raw_quantity) if quantity_present else 0.0
        amount_value = _tx_safe_float(raw_amount) if amount_present else 0.0
        price_value = _tx_safe_float(raw_price) if price_present else 0.0
        
        # Debug logging for first row
        if idx == 0:
            print(f"[FILE_PARSE] Row {idx} values:")
            print(f"[FILE_PARSE]   quantity_present={quantity_present}, quantity_value={quantity_value}")
            print(f"[FILE_PARSE]   amount_present={amount_present}, amount_value={amount_value}")
            print(f"[FILE_PARSE]   price_present={price_present}, price_value={price_value}")

        # CRITICAL: If we have properly mapped columns, NEVER swap or recalculate
        # Trust the file values when columns are correctly mapped
        has_proper_columns = ('quantity' in prepared_df.columns and 'amount' in prepared_df.columns)
        
        should_swap = False
        
        # Only do swap detection if column names are NOT clear AND we don't have proper mappings
        # If we have proper column names (after mapping), trust the file values completely
        if not has_proper_columns and quantity_present and amount_present:
            # Smart detection: Check if quantity and amount are swapped
            # Only swap when it's VERY CLEAR they're wrong - be conservative!
            if price_value > 0 and quantity_value > 0 and amount_value > 0:
                # Validate using price: quantity * price should approximately equal amount
                expected_amount = quantity_value * price_value
                expected_quantity_from_swapped = amount_value / price_value
                
                # Calculate how well each combination matches
                current_match = abs(amount_value - expected_amount) / max(1.0, expected_amount)
                swapped_match = abs(quantity_value - expected_quantity_from_swapped) / max(1.0, expected_quantity_from_swapped) if expected_quantity_from_swapped > 0 else 999
                
                # Only swap if:
                # 1. Current values are WAY off (more than 10% difference)
                # 2. Swapped values match MUCH better (at least 80% better match)
                # 3. Swapped quantity is reasonable (positive and not too large)
                if (current_match > 0.10 and  # Current values are clearly wrong
                    swapped_match < current_match * 0.2 and  # Swapped is at least 80% better
                    expected_quantity_from_swapped > 0 and
                    expected_quantity_from_swapped < 1000000):  # Reasonable quantity
                    should_swap = True
                    print(f"[FILE_PARSE] Row {idx}: Swapping quantity/amount (price validation: current_diff={current_match:.2%}, swapped_diff={swapped_match:.2%})")
            
            # Fallback: Heuristic check if no price available
            # Only swap if it's VERY obvious (quantity is huge, amount is tiny)
            elif quantity_value > 0 and amount_value > 0:
                # Quantities are typically small (1-10000), amounts are typically larger (100+)
                # Only swap if quantity is VERY large (like an amount) and amount is VERY small (like quantity)
                if quantity_value > 10000 and amount_value < 100 and quantity_value > amount_value * 100:
                    should_swap = True
                    print(f"[FILE_PARSE] Row {idx}: Swapping quantity/amount (heuristic: qty={quantity_value} looks like amount, amt={amount_value} looks like quantity)")
        
        if should_swap:
            print(f"[FILE_PARSE] Row {idx}: ⚠️ Swapping quantity and amount (qty={quantity_value} ↔ amt={amount_value})")
            quantity_value, amount_value = amount_value, quantity_value
            # After swap, update presence flags
            quantity_present = True  # After swap, we have quantity
            amount_present = True    # After swap, we have amount

        transaction_type = _tx_normalize_transaction_type(
            row.get('transaction_type'),
            quantity=quantity_value,
            amount=amount_value,
        )

        if quantity_value < 0:
            quantity_value = abs(quantity_value)
        if amount_value < 0:
            amount_value = abs(amount_value)
        if price_value < 0:
            price_value = abs(price_value)

        stock_name = _tx_safe_str(row.get('stock_name') or row.get('name'))
        scheme_name = _tx_safe_str(row.get('scheme_name'))
        ticker = _tx_normalize_ticker(row.get('ticker') or row.get('symbol'))
        
        # For mutual funds, try to resolve ticker using scheme_name first
        if not ticker and scheme_name:
            # Try to resolve AMFI code from scheme name
            # This will be done later during import, but we can set a flag here
            ticker = _tx_normalize_ticker(scheme_name)
        
        if not ticker:
            ticker = _tx_normalize_ticker(stock_name)

        if not ticker and not stock_name and not scheme_name:
            rows_skipped_ticker += 1
            continue

        asset_hint = _tx_safe_str(row.get('asset_type'))
        asset_type = _tx_infer_asset_type(ticker, stock_name or scheme_name, asset_hint)
        
        # For mutual funds, properly set scheme_name and stock_name
        if asset_type == 'mutual_fund':
            # If scheme_name is missing, use stock_name as scheme_name
            if not scheme_name:
                scheme_name = stock_name
            # For mutual funds, stock_name should be the fund name (same as scheme_name if not provided separately)
            # But if we have a separate scheme_name, keep stock_name as is (it might be the display name)
            # If stock_name is just the ticker, try to resolve from AMFI
            if stock_name == ticker or not stock_name or stock_name == 'Unknown':
                # Try to resolve fund name from AMFI
                try:
                    amfi_data = get_amfi_dataset()
                    if amfi_data and 'code_lookup' in amfi_data:
                        scheme = amfi_data['code_lookup'].get(ticker)
                        if scheme and scheme.get('name'):
                            stock_name = scheme.get('name')
                            scheme_name = scheme.get('name')  # Use same for both
                except Exception:
                    pass
        else:
            # For non-mutual funds, scheme_name should be None
            scheme_name = None

        # CRITICAL: Never recalculate quantity if it's present in the file
        # Quantity is the source of truth - if it's in the file, use it as-is
        original_quantity = quantity_value  # Store original before any calculations
        
        # Only recalculate missing values, don't override existing values from file
        # Priority 1: Calculate price if missing (when we have quantity and amount)
        # CRITICAL: Always calculate price from amount/quantity if price is missing, even if it's 0
        if not price_present and amount_present and quantity_present:
            if amount_value > 0 and quantity_value > 0:
                price_value = amount_value / quantity_value
                print(f"[FILE_PARSE] Row {idx}: ✅ Calculated price from amount/quantity: {price_value} (qty={quantity_value}, amt={amount_value})")
            else:
                # Even if amount or quantity is 0, set price to 0 (don't leave it unset)
                price_value = 0.0
                print(f"[FILE_PARSE] Row {idx}: ⚠️ Cannot calculate price (qty={quantity_value}, amt={amount_value}), setting to 0")
        
        # Priority 2: Only calculate quantity if it's TRULY missing (not present in file)
        # AND we have amount+price, AND quantity was never present
        if not quantity_present and amount_present and price_present and amount_value > 0 and price_value > 0:
            quantity_value = amount_value / price_value
            print(f"[FILE_PARSE] Row {idx}: ✅ Calculated quantity from amount/price: {quantity_value} (amt={amount_value}, price={price_value})")
        
        # Safety check: If quantity was present, never override it
        if quantity_present and quantity_value != original_quantity:
            print(f"[FILE_PARSE] Row {idx}: ⚠️ WARNING: Quantity was present ({original_quantity}) but got changed to {quantity_value}. Restoring original.")
            quantity_value = original_quantity
        
        # Calculate amount if missing or significantly wrong (but only if we have quantity and price)
        if quantity_value > 0 and price_value > 0:
            computed_amount = quantity_value * price_value
            # Only override amount if it's missing or way off (more than 1% difference)
            # This allows for rounding differences but fixes major errors
            if amount_value <= 0:
                amount_value = computed_amount
            elif abs(amount_value - computed_amount) > 0.01 * max(1.0, computed_amount):
                # Amount exists but doesn't match - only fix if difference is significant
                amount_value = computed_amount
        
        # Don't recalculate quantity - use what's in the file
        # Quantity is the source of truth from the transaction record

        channel = _tx_safe_str(row.get('channel'))
        if not channel:
            # Extract just the filename from path (handles full paths like E:\kalyan\Files_checked\deepak.pdf)
            # Path().stem correctly handles hyphens in filenames (e.g., deepak-2.pdf -> deepak-2)
            base = Path(filename).stem
            if sheet_label:
                channel = f"{base} - {sheet_label}"
            else:
                channel = base
        sector = _tx_safe_str(row.get('sector'))
        if not sector:
            sector = 'Mutual Fund' if asset_type == 'mutual_fund' else 'Unknown'

        scheme_name = _tx_safe_str(row.get('scheme_name')) if asset_type == 'mutual_fund' else None
        notes = _tx_safe_str(row.get('notes'))

        transaction = {
            'date': raw_date,
            'transaction_type': transaction_type,
            'ticker': ticker,
            'stock_name': stock_name or ticker if asset_type != 'mutual_fund' else (scheme_name or stock_name or ticker),
            'scheme_name': scheme_name if asset_type == 'mutual_fund' else None,
            'quantity': quantity_value,
            'price': price_value,
            'amount': amount_value,
            'asset_type': asset_type,
            'channel': channel,
            'sector': sector,
            'notes': notes,
            # Store metadata about which fields were originally present in the file
            '__original_quantity_present': quantity_present,
            '__original_amount_present': amount_present,
            '__original_price_present': price_present,
            '__original_quantity': original_quantity if quantity_present else None,
            '__original_amount': raw_amount if amount_present else None,
            '__original_price': raw_price if price_present else None,
        }
        transactions.append(transaction)
        rows_added += 1

    # Debug logging
    if rows_processed > 0:
        print(f"[FILE_PARSE] Processed {rows_processed} rows: {rows_added} added, {rows_skipped_date} skipped (date), {rows_skipped_ticker} skipped (ticker/name)")

    return transactions


def extract_transactions_python(uploaded_file, filename: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Try to extract transactions using deterministic Python parsing.

    Returns:
        Tuple of (transactions, debug_log) where debug_log contains per-sheet mapping details.
    """
    suffix = Path(filename).suffix.lower()
    transactions: List[Dict[str, Any]] = []
    debug_log: List[Dict[str, Any]] = []

    try:
        if suffix in {'.csv', '.tsv'}:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep='\t' if suffix == '.tsv' else ',', encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep='\t' if suffix == '.tsv' else ',', encoding='latin-1')
            if df.empty:
                print(f"[FILE_PARSE] CSV {filename} is empty after reading")
            else:
                print(f"[FILE_PARSE] CSV {filename} has {len(df)} rows, columns: {list(df.columns)}")
            transactions = _tx_dataframe_to_transactions(df, filename, debug_log=debug_log)
            if not transactions:
                print(f"[FILE_PARSE] No transactions extracted from {filename} after processing {len(df)} rows")
        elif suffix in {'.xlsx', '.xls'}:
            uploaded_file.seek(0)
            try:
                # Use appropriate engine for Excel files
                engine = 'openpyxl' if suffix == '.xlsx' else None
                workbook = pd.read_excel(uploaded_file, sheet_name=None, engine=engine)
                rows: List[Dict[str, Any]] = []
                for sheet_name, sheet_df in (workbook or {}).items():
                    rows.extend(_tx_dataframe_to_transactions(sheet_df, filename, sheet_label=sheet_name, debug_log=debug_log))
                transactions = rows
            except ImportError as e:
                print(f"[FILE_PARSE] ❌ Missing Excel dependency for {filename}: {e}")
                print(f"[FILE_PARSE] 💡 Install with: pip install openpyxl xlrd")
                transactions = []
            except Exception as e:
                print(f"[FILE_PARSE] ❌ Error reading Excel file {filename}: {e}")
                if 'openpyxl' in str(e).lower() or 'xlrd' in str(e).lower():
                    print(f"[FILE_PARSE] 💡 Install missing dependency: pip install openpyxl xlrd")
                transactions = []
        elif suffix == '.pdf':
            tables = _tx_extract_tables_from_pdf(uploaded_file)
            print(f"[FILE_PARSE] PDF {filename}: Extracted {len(tables)} tables")
            rows: List[Dict[str, Any]] = []
            for table_idx, table in enumerate(tables):
                label = table.attrs.get('__tx_source')
                print(f"[FILE_PARSE] Processing table {table_idx + 1}/{len(tables)}: {label}")
                print(f"[FILE_PARSE]   Table shape: {table.shape}, columns: {list(table.columns)}")
                if not table.empty:
                    print(f"[FILE_PARSE]   First row sample: {table.iloc[0].to_dict() if len(table) > 0 else 'Empty'}")
                table_transactions = _tx_dataframe_to_transactions(table, filename, sheet_label=label, debug_log=debug_log)
                print(f"[FILE_PARSE]   Extracted {len(table_transactions)} transactions from this table")
                rows.extend(table_transactions)
            transactions = rows
            print(f"[FILE_PARSE] PDF {filename}: Total {len(transactions)} transactions from all tables")
        else:
            transactions = []
    except Exception as exc:
        print(f"[FILE_PARSE] Python extraction failed for {filename}: {exc}")
        transactions = []
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    filtered = [tx for tx in transactions if tx.get('quantity') or tx.get('amount')]
    print(f"[FILE_PARSE] {filename}: {len(transactions)} total transactions, {len(filtered)} after filtering (must have quantity or amount)")
    if len(transactions) > 0 and len(filtered) == 0:
        print(f"[FILE_PARSE] ⚠️ All transactions filtered out - showing sample:")
        for i, tx in enumerate(transactions[:3]):
            print(f"[FILE_PARSE]   Transaction {i+1}: quantity={tx.get('quantity')}, amount={tx.get('amount')}, ticker={tx.get('ticker')}, date={tx.get('date')}")
    return filtered, debug_log


def _tx_build_db_transaction(
    tx: Dict[str, Any],
    fallback_channel: str,
    filename: str,
    user_id: str,
    portfolio_id: str,
    db,
    price_fetcher,
    price_cache: Dict[Tuple[str, str, str], Optional[float]],
) -> Optional[Dict[str, Any]]:
    """Normalize extracted transaction into DB-ready payload."""
    raw_date = tx.get('transaction_date') or tx.get('date') or tx.get('Date')
    if not raw_date:
        return None

    normalized_date = db._normalize_transaction_date(raw_date) if hasattr(db, '_normalize_transaction_date') else None
    if not normalized_date:
        normalized_date = db._normalize_transaction_date(_tx_safe_str(raw_date)) if hasattr(db, '_normalize_transaction_date') else None
    if not normalized_date:
        return None

    ticker = _tx_normalize_ticker(tx.get('ticker'))
    stock_name = _tx_safe_str(tx.get('stock_name') or tx.get('scheme_name') or '')
    if not ticker and stock_name:
        ticker = _tx_normalize_ticker(stock_name)
    if not ticker:
        return None
    if re.fullmatch(r'\d+(\.\d+)?', ticker):
        fallback_ticker = _tx_fallback_ticker_from_name(stock_name)
        ticker = fallback_ticker or ticker
    if not stock_name:
        stock_name = ticker

    # Check if values are actually present (not just zero)
    # First check metadata (from Python extraction), then fall back to transaction dict
    quantity_present = tx.get('__original_quantity_present', False)
    amount_present = tx.get('__original_amount_present', False)
    price_present = tx.get('__original_price_present', False)
    
    # Get original raw values if stored, otherwise get from transaction dict
    quantity_raw = tx.get('__original_quantity')
    if quantity_raw is None:
        quantity_raw = tx.get('quantity')
    units_raw = tx.get('units')
    
    amount_raw = tx.get('__original_amount')
    if amount_raw is None:
        amount_raw = tx.get('amount') or tx.get('value')
    
    price_raw = tx.get('__original_price')
    if price_raw is None:
        price_raw = tx.get('price')
    
    # If metadata not available, infer presence from raw values
    if not quantity_present and quantity_raw is not None and str(quantity_raw).strip() != '':
        quantity_present = True
    if not amount_present and amount_raw is not None and str(amount_raw).strip() != '':
        amount_present = True
    if not price_present and price_raw is not None and str(price_raw).strip() != '':
        price_present = True
    
    quantity_value = _tx_safe_float(quantity_raw)
    # Only use units if quantity is truly missing (None), not if it's zero
    if (quantity_raw is None or quantity_raw == '') and (units_raw is None or units_raw == ''):
        quantity_value = _tx_safe_float(units_raw)
    amount_value = _tx_safe_float(amount_raw)
    price_value = _tx_safe_float(price_raw)

    # CRITICAL: Never recalculate quantity if it was present in the file
    original_quantity = quantity_value  # Store before any calculations
    
    # Calculate missing values from available data
    # Only calculate if value is truly missing (None/empty), not if it's zero
    # Priority 1: Calculate price from amount/quantity if price is missing
    if not price_present and amount_present and quantity_present and amount_value > 0 and quantity_value > 0:
        price_value = amount_value / quantity_value
        print(f"[TX_BUILD] ✅ Calculated price from amount/quantity: {price_value} (qty={quantity_value}, amt={amount_value})")
    
    # Priority 2: Only calculate quantity if it's TRULY missing (not present in file)
    if not quantity_present and amount_present and price_present and amount_value > 0 and price_value > 0:
        quantity_value = amount_value / price_value
        print(f"[TX_BUILD] ✅ Calculated quantity from amount/price: {quantity_value} (amt={amount_value}, price={price_value})")
    
    # Priority 3: Calculate amount from quantity/price if amount is missing
    if not amount_present and quantity_present and price_present and quantity_value > 0 and price_value > 0:
        amount_value = quantity_value * price_value
        print(f"[TX_BUILD] ✅ Calculated amount from quantity/price: {amount_value}")
    
    # Safety check: If quantity was present, never override it
    if quantity_present and quantity_value != original_quantity:
        print(f"[TX_BUILD] ⚠️ WARNING: Quantity was present ({original_quantity}) but got changed to {quantity_value}. Restoring original.")
        quantity_value = original_quantity
    
    transaction_type = _tx_normalize_transaction_type(
        tx.get('transaction_type'),
        quantity=quantity_value,
        amount=amount_value,
    )
    if quantity_value < 0:
        quantity_value = abs(quantity_value)
        transaction_type = 'sell'

    if quantity_value <= 0:
        return None
    if quantity_value > 0 and price_value > 0:
        computed_amount = quantity_value * price_value
        if amount_value <= 0 or abs(amount_value - computed_amount) > 0.01 * max(1.0, computed_amount):
            amount_value = computed_amount

    asset_hint = _tx_safe_str(tx.get('asset_type'))
    asset_type = _tx_infer_asset_type(ticker, stock_name, asset_hint)
    
    # For PMS/AIF, calculate current value using CAGR and store that instead of original values
    if asset_type in ['pms', 'aif']:
        try:
            from pms_aif_calculator import PMS_AIF_Calculator
            
            # Get the original investment amount (from file)
            original_investment = amount_value if amount_value > 0 else (quantity_value * price_value if quantity_value > 0 and price_value > 0 else 0)
            
            if original_investment > 0 and normalized_date:
                # Calculate current value using CAGR
                calculator = PMS_AIF_Calculator()
                result = calculator.calculate_pms_aif_value(
                    ticker,
                    normalized_date,
                    original_investment,
                    is_aif=(asset_type == 'aif')
                )
                
                current_value = result.get('current_value', 0)
                if current_value > 0:
                    # Store current value as amount
                    amount_value = current_value
                    # For PMS/AIF, quantity is typically 1 (single investment)
                    quantity_value = 1.0
                    # Price is the current value (since quantity = 1)
                    price_value = current_value
                    print(f"[TX_BUILD] ✅ PMS/AIF: Calculated current value ₹{current_value:,.2f} from original ₹{original_investment:,.2f} (CAGR: {result.get('cagr_used', 0)*100:.1f}%)")
        except Exception as e:
            print(f"[TX_BUILD] ⚠️ Failed to calculate PMS/AIF current value: {e}")
            # Continue with original values if CAGR calculation fails

    # For mutual funds, prefer scheme_name for ticker resolution (AMFI code lookup)
    scheme_name = _tx_safe_str(tx.get('scheme_name'))
    fund_name_for_lookup = scheme_name if (asset_type == 'mutual_fund' and scheme_name) else stock_name

    cache_key = (ticker, normalized_date, asset_type)
    if price_value <= 0 and price_cache is not None and cache_key in price_cache:
        cached = price_cache[cache_key]
        if cached:
            price_value = cached

    if price_value <= 0 and price_fetcher:
        fetched_price = None
        try:
            fetched_price = price_fetcher.get_historical_price(ticker, asset_type, normalized_date, fund_name=fund_name_for_lookup)
        except Exception:
            fetched_price = None
        if not fetched_price:
            try:
                fetched_price, _ = price_fetcher.get_current_price(ticker, asset_type, fund_name=fund_name_for_lookup)
            except Exception:
                fetched_price = None
        if fetched_price and fetched_price > 0:
            price_value = float(fetched_price)
            if price_cache is not None:
                price_cache[cache_key] = price_value
        elif price_cache is not None and cache_key not in price_cache:
            price_cache[cache_key] = None

    if price_value < 0:
        price_value = abs(price_value)

    if price_value:
        price_value = round(price_value, 4)
    else:
        price_value = 0.0

    channel = _tx_safe_str(tx.get('channel')) or fallback_channel or 'Direct'
    sector = _tx_safe_str(tx.get('sector'))
    if not sector:
        sector = 'Mutual Fund' if asset_type == 'mutual_fund' else 'Unknown'
    scheme_name = _tx_safe_str(tx.get('scheme_name')) if asset_type == 'mutual_fund' else None
    notes = _tx_safe_str(tx.get('notes'))

    transaction_type = transaction_type if transaction_type in {'buy', 'sell'} else 'buy'

    return {
        'user_id': user_id,
        'portfolio_id': portfolio_id,
        'ticker': ticker,
        'stock_name': stock_name,
        'scheme_name': scheme_name,
        'quantity': quantity_value,
        'price': price_value,
        'transaction_date': normalized_date,
        'transaction_type': transaction_type,
        'asset_type': asset_type,
        'channel': channel,
        'sector': sector,
        'filename': filename,
        'notes': notes,
    }

# Import modules
from database_shared import SharedDatabaseManager
from enhanced_price_fetcher import EnhancedPriceFetcher
from bulk_ai_fetcher import BulkAIFetcher

# AI Ticker Resolver not needed - using ai_resolve_tickers_from_names() function instead
TICKER_RESOLVER_AVAILABLE = False
AITickerResolver = None

# from weekly_manager_streamlined import StreamlinedWeeklyManager  # Removed - file deleted

# Page configuration
st.set_page_config(
    page_title="Wealth Manager - Streamlined",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None
if 'db' not in st.session_state:
    st.session_state.db = SharedDatabaseManager()
if (
    'price_fetcher' not in st.session_state
    or st.session_state.get('price_fetcher_version') != PRICE_FETCHER_VERSION
):
    st.session_state.price_fetcher = EnhancedPriceFetcher()
    st.session_state.price_fetcher_version = PRICE_FETCHER_VERSION
if 'missing_weeks_fetched' not in st.session_state:
    st.session_state.missing_weeks_fetched = False
if 'needs_initial_refresh' not in st.session_state:
    st.session_state.needs_initial_refresh = True
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None

# Ensure SharedDatabaseManager has latest methods after code updates
if not hasattr(st.session_state.db, "get_channel_weekly_history") or not hasattr(st.session_state.db, "get_sector_weekly_history"):
    st.session_state.db = SharedDatabaseManager()

# Function to detect corporate actions (splits/bonus)
def detect_corporate_actions(user_id, db, holdings=None):
    """
    Detect stock splits and bonus shares by comparing CSV prices with current prices
    Returns list of stocks with likely corporate actions
    """
    try:
        import yfinance as yf

        # Use provided holdings or fetch if not provided
        if holdings is None:
            holdings = db.get_user_holdings(user_id)
        corporate_actions: List[Dict[str, Any]] = []

        @functools.lru_cache(maxsize=128)
        def _latest_split_info(symbol: str) -> Optional[Tuple[str, float]]:
            try:
                split_series = yf.Ticker(symbol).splits
                if split_series is not None and not split_series.empty:
                    last_ratio = float(split_series.iloc[-1])
                    if last_ratio > 0:
                        # yfinance reports 0.25 for 4:1 split etc.
                        resolved_ratio = round(1.0 / last_ratio)
                        return split_series.index[-1], resolved_ratio
            except Exception:
                pass
            return None

        def _find_split_confirmation(ticker_code: str) -> Optional[Tuple[str, float]]:
            candidates = []
            if ticker_code.endswith(('.NS', '.BO')):
                candidates.append(ticker_code)
            elif ticker_code.isdigit():
                candidates.append(f"{ticker_code}.BO")
            else:
                candidates.extend([f"{ticker_code}.NS", f"{ticker_code}.BO", ticker_code])

            for candidate in candidates:
                info = _latest_split_info(candidate)
                if info:
                    return info
            return None

        for holding in holdings:
            if holding.get('asset_type') != 'stock':
                continue

            ticker = str(holding.get('ticker') or '').strip()
            avg_price = float(holding.get('average_price') or 0)
            current_price = holding.get('current_price')
            quantity = holding.get('total_quantity', 0)

            if avg_price == 0 or not current_price:
                continue

            current_price = float(current_price)
            price_ratio = avg_price / current_price if current_price else 0

            if price_ratio < 1.5:
                continue

            confirmation = _find_split_confirmation(ticker)
            if not confirmation:
                continue

            split_date, confirmed_ratio = confirmation
            if confirmed_ratio <= 1:
                continue

            corporate_actions.append({
                'ticker': ticker,
                'stock_name': holding.get('stock_name'),
                'stock_id': holding.get('stock_id'),
                'avg_price': avg_price,
                'current_price': current_price,
                'quantity': quantity,
                'ratio': price_ratio,
                'split_ratio': confirmed_ratio,
                'split_date': str(split_date),
                'action_type': 'split',
            })
        
        if corporate_actions:
            print(f"[CORPORATE_ACTIONS] Detected {len(corporate_actions)} stocks with splits/bonus")
            for action in corporate_actions:
                print(f"  - {action['ticker']}: 1:{action['split_ratio']} {action['action_type']} on {action.get('split_date')}")
        
        return corporate_actions

    except Exception as e:
        print(f"[CORPORATE_ACTIONS] Error detecting: {e}")
        return []

def adjust_for_corporate_action(user_id, stock_id, split_ratio, db):
    """
    Adjust transaction quantities and prices for a stock split/bonus
    
    Args:
        user_id: User ID
        stock_id: Stock master ID
        split_ratio: Split ratio (e.g., 20 for 1:20 split)
        db: Database manager
    """
    try:
        # Get all transactions for this stock and user
        transactions = db.supabase.table('user_transactions').select('*').eq(
            'user_id', user_id
        ).eq('stock_id', stock_id).execute()
        
        if not transactions.data:
            return 0
        
        updated_count = 0
        for txn in transactions.data:
            old_quantity = float(txn['quantity'])
            old_price = float(txn['price'])
            
            # Adjust for split
            new_quantity = old_quantity * split_ratio
            new_price = old_price / split_ratio
            
            # Update transaction
            db.supabase.table('user_transactions').update({
                'quantity': new_quantity,
                'price': new_price,
                'notes': f"Auto-adjusted for 1:{split_ratio} stock split"
            }).eq('id', txn['id']).execute()
            
            updated_count += 1
        
        print(f"[CORPORATE_ACTIONS] Adjusted {updated_count} transactions for 1:{split_ratio} split")
        return updated_count
        
    except Exception as e:
        print(f"[CORPORATE_ACTIONS] Error adjusting: {e}")
        return 0

# Function to update bond prices using AI
def update_bond_prices_with_ai(user_id, db, bonds=None):
    """Update bond prices using AI (called automatically on login)"""
    try:
        # Use provided bonds or fetch if not provided
        if bonds is None:
            all_holdings = db.get_user_holdings(user_id)
            bonds = [h for h in all_holdings if h.get('asset_type') == 'bond']
        
        if not bonds:
            return  # No bonds to update
        
        print(f"[BOND_UPDATE] Found {len(bonds)} bonds, fetching current prices...")
        
        from enhanced_price_fetcher import EnhancedPriceFetcher
        price_fetcher = EnhancedPriceFetcher()
        
        updated_count = 0
        for bond in bonds:
            ticker = bond.get('ticker')
            stock_name = bond.get('stock_name')
            
            if ticker and stock_name:
                print(f"[BOND_UPDATE] Fetching price for {stock_name} ({ticker})...")
                
                # Try to get bond price from AI
                price, source = price_fetcher._get_bond_price(ticker, stock_name)
                
                if price and price > 0:
                    # Update the price in database
                    db._store_current_price(ticker, price, 'bond')
                    print(f"[BOND_UPDATE] ✅ Updated {ticker}: ₹{price:.2f} (from {source})")
                    updated_count += 1
                else:
                    print(f"[BOND_UPDATE] ❌ Failed to get price for {ticker}")
        
        if updated_count > 0:
            print(f"[BOND_UPDATE] Successfully updated {updated_count}/{len(bonds)} bond prices")
        else:
            print(f"[BOND_UPDATE] No bond prices updated (all failed)")
    except Exception as e:
        print(f"[BOND_UPDATE] Error updating bond prices: {e}")
        pass  # Silent failure - don't break login


def run_portfolio_refresh(user_id: str, *, auto: bool = False) -> None:
    """Refresh current prices and weekly history on-demand or automatically."""
    spinner_message = "🔄 Refreshing portfolio data..." if not auto else "🔄 Initializing portfolio data..."
    with st.spinner(spinner_message):
        try:
            update_bond_prices_with_ai(user_id, db)
        except Exception as exc:
            if not auto:
                st.warning(f"Bond price refresh skipped: {str(exc)[:80]}")

        holdings = db.get_user_holdings(user_id)

        if holdings:
            try:
                missing_weeks = db.get_missing_weeks_for_user(user_id)
                if missing_weeks:
                    db.fetch_and_store_missing_weekly_prices(user_id, missing_weeks)
            except Exception as exc:
                if not auto:
                    st.warning(f"Weekly history refresh failed: {str(exc)[:120]}")

            try:
                holdings_needing_update = should_update_prices_today(holdings, db)
                if holdings_needing_update and bulk_ai_fetcher.available:
                    unique_tickers = list({h['ticker'] for h in holdings_needing_update if h.get('ticker')})
                    if unique_tickers:
                        asset_types = {h['ticker']: h.get('asset_type', 'stock') for h in holdings_needing_update if h.get('ticker')}
                        db.bulk_process_new_stocks_with_comprehensive_data(
                            tickers=unique_tickers,
                            asset_types=asset_types
                        )
                    else:
                        st.session_state.price_fetcher.update_live_prices_for_holdings(holdings_needing_update, db)
                elif holdings_needing_update:
                    st.session_state.price_fetcher.update_live_prices_for_holdings(holdings_needing_update, db)
            except Exception as exc:
                if not auto:
                    st.warning(f"Live price update failed: {str(exc)[:120]}")

        st.session_state.missing_weeks_fetched = True
        st.session_state.needs_initial_refresh = False
        st.session_state.last_fetch_time = datetime.now()

    if auto:
        st.success("✅ Portfolio refreshed automatically.")
    else:
        st.success("✅ Portfolio data refreshed.")
        st.rerun()

# Function to update all live prices
def update_all_live_prices():
    """Update live prices for all holdings"""
    if 'user_id' in st.session_state:
        holdings = db.get_user_holdings(st.session_state.user_id)
        if holdings:
            st.session_state.price_fetcher.update_live_prices_for_holdings(holdings, db)
            st.success("✅ Live prices updated successfully!")
            st.rerun()
        else:
            st.warning("No holdings found to update.")

def should_update_prices_today(holdings, db_manager):
    """Check if prices need to be updated today"""
    from datetime import datetime, date
    
    today = date.today()
    needs_update = []
    
    if not db_manager:
        # No database manager, assume all need update
        return holdings
    
    for holding in holdings:
        stock_id = holding.get('stock_id')
        if stock_id:
            # Get last updated date from database
            try:
                if hasattr(db_manager, 'get_stock_last_updated'):
                    last_updated = db_manager.get_stock_last_updated(stock_id)
                    if last_updated:
                        try:
                            last_updated_date = datetime.fromisoformat(last_updated).date()
                            if last_updated_date < today:
                                needs_update.append(holding)
                        except (ValueError, TypeError):
                            # Invalid date format, needs update
                            needs_update.append(holding)
                    else:
                        # No last_updated record, needs update
                        needs_update.append(holding)
                else:
                    # Method not available, assume needs update
                    needs_update.append(holding)
            except Exception as e:
                # Any error, assume needs update (conservative approach)
                print(f"[PRICE_CHECK] Error checking last_updated for stock_id {stock_id}: {str(e)}")
                needs_update.append(holding)
        else:
            # No stock_id, needs update
            needs_update.append(holding)
    
    return needs_update

if 'bulk_ai_fetcher' not in st.session_state:
    st.session_state.bulk_ai_fetcher = BulkAIFetcher()
if 'weekly_manager' not in st.session_state:
    st.session_state.weekly_manager = None  # Simplified - removed StreamlinedWeeklyManager

db = st.session_state.db
price_fetcher = st.session_state.price_fetcher
bulk_ai_fetcher = st.session_state.bulk_ai_fetcher
weekly_manager = st.session_state.weekly_manager

# ============================================================================
# AUTHENTICATION
# ============================================================================

def login_page():
    """Login page"""
    st.title("💰 Wealth Manager - Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            # Convert username to lowercase for case-insensitive login
            user = db.login_user(username.lower(), password)
            if user:
                st.session_state.user = user
                
                # OPTIMIZATION: Recalculate first, then fetch holdings ONCE and reuse
                # This avoids multiple separate database calls (was 3+ calls, now 2 max)
                try:
                    db.recalculate_holdings(user['id'])
                except:
                    pass  # Silent failure
                
                # Fetch holdings ONCE after recalculation and reuse for all operations
                try:
                    holdings = db.get_user_holdings(user['id'])
                except:
                    holdings = []
                
                # Update bond prices automatically on login (AI-powered)
                # Pass holdings to avoid another fetch
                try:
                    if holdings:
                        bonds = [h for h in holdings if h.get('asset_type') == 'bond']
                        if bonds:
                            update_bond_prices_with_ai(user['id'], db, bonds=bonds)
                except:
                    pass  # Silent failure - don't break login
                
                st.session_state.needs_initial_refresh = True
                st.session_state.missing_weeks_fetched = False
                st.session_state.last_fetch_time = None
                
                # Detect corporate actions (splits/bonus) on login
                # Pass holdings to avoid another fetch
                try:
                    if holdings:
                        corporate_actions = detect_corporate_actions(user['id'], db, holdings=holdings)
                        if corporate_actions:
                            st.session_state.corporate_actions_detected = corporate_actions
                        else:
                            st.session_state.corporate_actions_detected = None
                    else:
                        st.session_state.corporate_actions_detected = None
                except:
                    st.session_state.corporate_actions_detected = None
                
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with tab2:
        st.subheader("Register")
        full_name = st.text_input("Full Name", key="register_name")
        username = st.text_input("Username", key="register_username")
        email = st.text_input("Email (Optional)", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm")
        
        # File upload during registration
        st.subheader("Upload Transaction Files (Optional)")
        uploaded_files = st.file_uploader(
            "Upload transaction files (CSV, Excel, PDF)",
            type=['csv', 'tsv', 'xlsx', 'xls', 'pdf'],
            accept_multiple_files=True,
            help="Python will auto-map standard columns; unsupported layouts fall back to AI extraction automatically."
        )

        if uploaded_files:
            if st.button("📝 Convert to CSV (Preview)", key="register_convert_csv"):
                with st.spinner("🔄 Converting files..."):
                    csv_rows: List[Dict[str, Any]] = []
                    summary_messages: List[str] = []

                    for uploaded_file in uploaded_files:
                        rows, method_used = extract_transactions_for_csv(uploaded_file, uploaded_file.name, None)
                        if rows:
                            csv_rows.extend(rows)
                            summary_messages.append(f"📄 {uploaded_file.name}: {len(rows)} rows ({method_used.upper()})")
                        else:
                            # Provide more helpful error message
                            if uploaded_file.name.lower().endswith('.pdf'):
                                error_msg = f"⚠️ {uploaded_file.name}: Image-based PDF detected. Use OCR or provide text-selectable PDF/CSV/Excel."
                            else:
                                error_msg = f"⚠️ {uploaded_file.name}: No transactions detected."
                                if method_used == 'python':
                                    error_msg += " Check terminal logs for extraction details."
                                if not AI_AGENTS_AVAILABLE:
                                    error_msg += " AI fallback unavailable - check AI agents configuration."
                            summary_messages.append(error_msg)

                if csv_rows:
                    df_preview = pd.DataFrame(csv_rows)
                    st.success(f"✅ Extracted {len(csv_rows)} transactions. Review below or download as CSV.")
                    for msg in summary_messages:
                        st.caption(msg)
                    st.dataframe(df_preview, use_container_width=True)

                    csv_data = df_preview.to_csv(index=False).encode('utf-8')
                    timestamp = int(time.time())
                    st.download_button(
                        "⬇️ Download Converted CSV",
                        csv_data,
                        file_name=f"registration_transactions_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"register_csv_download_{timestamp}"
                    )
                else:
                    for msg in summary_messages:
                        st.caption(msg)
                    st.warning("No transactions extracted from the selected files. Please review the input.")
        
        if st.button("Register"):
            if password != confirm_password:
                st.error("Passwords don't match")
            elif not username:
                st.error("Username is required")
            else:
                # Convert username to lowercase for case-insensitive registration
                result = db.register_user(username.lower(), password, full_name, email)
                if result['success']:
                    user = result['user']
                    st.session_state.user = user
                    
                    # Create default portfolio
                    portfolio_result = db.create_portfolio(user['id'], "Main Portfolio")
                    if portfolio_result['success']:
                        portfolio_id = portfolio_result['portfolio']['id']
                        
                        # Process uploaded files (as per your image)
                        if uploaded_files:
                            st.info("📁 Processing uploaded files...")
                            imported_count = process_uploaded_files(uploaded_files, user['id'], portfolio_id)
                            
                            if imported_count > 0:
                                # Auto-fetch comprehensive data (info + prices + weekly) in bulk
                                st.info("🔍 Auto-fetching comprehensive data (info + prices + historical)...")
                                
                                try:
                                    holdings = db.get_user_holdings(user['id'])
                                    if holdings:
                                        # Get unique tickers and asset types
                                        unique_tickers = list(set([h['ticker'] for h in holdings if h.get('ticker')]))
                                        asset_types = {h['ticker']: h.get('asset_type', 'stock') for h in holdings if h.get('ticker')}
                                        
                                        if unique_tickers and st.session_state.bulk_ai_fetcher.available:
                                            st.caption("📊 Bulk fetching all data in one AI call...")
                                            # Fetch everything (stock info + current price + 52-week data) in ONE AI call
                                            stock_ids = db.bulk_process_new_stocks_with_comprehensive_data(
                                                tickers=unique_tickers,
                                                asset_types=asset_types
                                            )
                                            st.caption(f"✅ Fetched comprehensive data for {len(stock_ids)} tickers")
                                        else:
                                            # Fallback to individual updates
                                            st.caption("📊 Fetching prices individually...")
                                            st.session_state.price_fetcher.update_live_prices_for_holdings(holdings, db)
                                            st.caption(f"✅ Updated {len(holdings)} holdings")
                                    
                                    st.success("✅ Registration, file processing, and comprehensive data fetching complete!")
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ Registration successful, but data fetching had issues: {str(e)[:100]}")
                                    st.success("✅ Registration and file processing complete!")
                            else:
                                st.success("✅ Registration and file processing complete!")
                        else:
                            st.success("✅ Registration successful!")
                        
                        st.info("🔄 Redirecting to dashboard...")
                        time.sleep(2)  # Brief pause to show success message
                        st.rerun()
                else:
                    st.error(f"Registration failed: {result['error']}")


def normalize_scheme_name(name: str) -> str:
    """Normalize mutual fund scheme names for comparison."""
    cleaned = name.lower().strip()
    replacements = [
        ("- regular plan - growth", ""),
        ("- regular plan growth", ""),
        ("regular plan - growth", ""),
        ("regular plan growth", ""),
        ("- growth", ""),
        ("growth", ""),
        (" plan", ""),
        (" (regular)", ""),
        (" direct plan", ""),
        (" direct growth", ""),
    ]
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    return "".join(ch for ch in cleaned if ch.isalnum())


@st.cache_data(ttl=3600)  # Cache AMFI download for 1 hour
def get_amfi_dataset() -> Dict[str, Any]:
    """Download AMFI NAV dataset and build lookup tables."""
    try:
        response = requests.get(AMFI_NAV_URL)
        response.raise_for_status()

        data = response.text.splitlines()
        reader = csv.DictReader(data, delimiter=';')

        schemes: List[Dict[str, str]] = []
        code_lookup: Dict[str, Dict[str, str]] = {}
        name_lookup: Dict[str, List[Dict[str, str]]] = {}

        for row in reader:
            scheme = {
                "code": (row.get("Scheme Code") or "").strip(),
                "name": (row.get("Scheme Name") or "").strip(),
                "nav": (row.get("Net Asset Value") or "").strip(),
                "date": (row.get("Date") or "").strip(),
            }
            if not scheme["code"] or not scheme["name"]:
                continue

            schemes.append(scheme)
            code_lookup[scheme["code"]] = scheme

            normalized = normalize_scheme_name(scheme["name"])
            if normalized:
                name_lookup.setdefault(normalized, []).append(scheme)

        return {
            "schemes": schemes,
            "code_lookup": code_lookup,
            "name_lookup": name_lookup,
        }
    except Exception as exc:  # pragma: no cover - network dependent
        st.caption(f"   ⚠️ AMFI dataset unavailable: {str(exc)[:80]}")
        return {"schemes": [], "code_lookup": {}, "name_lookup": {}}


def match_scheme_by_name(
    scheme_name: str,
    name_lookup: Dict[str, List[Dict[str, str]]],
    *,
    max_matches: int = 5,
) -> List[Dict[str, Any]]:
    """Return candidate AMFI schemes ranked by similarity."""
    normalized = normalize_scheme_name(scheme_name)
    if not normalized or not name_lookup:
        return []

    matches: List[Dict[str, Any]] = []
    if normalized in name_lookup:
        matches = [
            {
                "code": scheme["code"],
                "name": scheme["name"],
                "nav": scheme["nav"],
                "date": scheme["date"],
                "score": 1.0,
            }
            for scheme in name_lookup[normalized]
        ]
    else:
        candidates = difflib.get_close_matches(
            normalized,
            name_lookup.keys(),
            n=max_matches * 5,
            cutoff=0.6,
        )

        for candidate in candidates:
            base_score = difflib.SequenceMatcher(a=normalized, b=candidate).ratio()
            for scheme in name_lookup.get(candidate, []):
                matches.append(
                    {
                        "code": scheme["code"],
                        "name": scheme["name"],
                        "nav": scheme["nav"],
                        "date": scheme["date"],
                        "score": base_score,
                    }
                )

    if not matches:
        return []

    target_upper = (scheme_name or "").upper()

    def adjusted_score(entry: Dict[str, Any]) -> float:
        weight = entry.get("score", 0.0)
        name_upper = entry.get("name", "").upper()

        def tweak(keyword: str, bonus: float) -> None:
            nonlocal weight
            if keyword in target_upper:
                if keyword in name_upper:
                    weight += bonus
                else:
                    weight -= bonus

        tweak("DIRECT", 0.08)
        tweak("REGULAR", 0.05)
        tweak("GROWTH", 0.04)
        tweak("DIVIDEND", 0.04)
        tweak("IDCW", 0.04)

        # Keep score within a sensible band
        return max(0.0, min(weight, 1.2))

    for entry in matches:
        entry["adjusted_score"] = adjusted_score(entry)

    matches.sort(
        key=lambda item: (item.get("adjusted_score", 0.0), item.get("score", 0.0)),
        reverse=True,
    )

    return matches[:max_matches]


def resolve_mutual_fund_with_amfi(
    scheme_name: str,
    current_code: str,
    dataset: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve mutual fund against AMFI dataset using code and name heuristics."""
    code_lookup = dataset.get("code_lookup", {})
    name_lookup = dataset.get("name_lookup", {})

    result: Dict[str, Any] = {
        "status": "unresolved",
        "direct_scheme": None,
        "matches": [],
    }

    cleaned_code = str(current_code).strip()
    direct_scheme = code_lookup.get(cleaned_code)
    matches = match_scheme_by_name(scheme_name, name_lookup)

    result["matches"] = matches

    if direct_scheme:
        normalized_target = normalize_scheme_name(scheme_name) if scheme_name else ""
        normalized_direct = normalize_scheme_name(direct_scheme["name"]) if direct_scheme["name"] else ""
        similarity = (
            difflib.SequenceMatcher(a=normalized_target, b=normalized_direct).ratio()
            if normalized_target and normalized_direct
            else 0.0
        )
        result["direct_scheme"] = {**direct_scheme, "similarity": similarity}
        result["status"] = "direct" if similarity >= 0.9 else "direct_mismatch"
        return result

    if matches:
        result["status"] = "name_matches"

    return result


def _format_amfi_matches_for_prompt(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return "None"

    lines = []
    for candidate in matches[:5]:
        lines.append(
            f"- code: {candidate['code']} | name: {candidate['name']} | NAV: {candidate['nav']} ({candidate['date']}) | score: {candidate['score']:.3f}"
        )
    return "\n".join(lines)


def _extract_json_block(text: str) -> Optional[str]:
    """Extract first JSON object or array from text."""
    if not text:
        return None
    pattern = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else None


def _parse_ai_amfi_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse AI response for AMFI code suggestions."""
    if not raw:
        return None

    candidates = []
    try:
        candidates.append(json.loads(raw))
    except json.JSONDecodeError:
        block = _extract_json_block(raw)
        if block:
            try:
                candidates.append(json.loads(block))
            except json.JSONDecodeError:
                pass

    for payload in candidates:
        if isinstance(payload, dict):
            if "code" in payload:
                return payload
            suggestions = payload.get("suggestions")
            if isinstance(suggestions, list) and suggestions:
                first = suggestions[0]
                if isinstance(first, dict) and "code" in first:
                    return first
        elif isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict) and "code" in first:
                return first

    return None


def ai_select_amfi_code(
    scheme_name: str,
    user_code: str,
    matches: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ask GPT to choose the best AMFI code from candidates."""
    if run_gpt5_completion is None:
        return None

    matches_prompt = _format_amfi_matches_for_prompt(matches)
    user_content = (
        "Determine the most likely AMFI mutual fund scheme code based on the user-provided name.\n"
        "Respond with JSON containing at minimum 'code' and 'confidence' (0-1). Include 'reason' if helpful.\n"
        "If none of the candidates are suitable, return an empty JSON object.\n\n"
        f"User scheme name: {scheme_name}\n"
        f"User provided code: {user_code}\n"
        f"Candidate matches:\n{matches_prompt}"
    )

    try:
        response = run_gpt5_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in Indian mutual funds. Choose the correct AMFI scheme code "
                        "given candidate matches. Only output JSON."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=200,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        st.caption(f"   ⚠️ AI AMFI suggestion failed: {str(exc)[:80]}")
        return None

    suggestion = _parse_ai_amfi_response(response)
    return suggestion


def ai_suggest_market_identifiers(
    name: str,
    user_ticker: str,
    *,
    asset_hint: Optional[str] = None,
    max_suggestions: int = 3,
) -> List[Dict[str, Any]]:
    """Use GPT to propose market identifiers (ticker/AMFI/PMS/AIF) with type classification."""
    if run_gpt5_completion is None:
        return []

    hint_text = asset_hint or "unknown"
    system_message = {
        "role": "system",
        "content": (
            "You map Indian financial instruments to tickers/codes that work with finance APIs. "
            "Return ONLY JSON array. Each item must include ticker, instrument_type "
            "(stock, mutual_fund, bond, pms, aif), confidence (0-1), and source "
            "indicating which API/database to use (yfinance_nse, yfinance_bse, mftool, manual, isin). "
            "Ensure tickers are directly usable with the stated API."
        ),
    }
    user_message = {
        "role": "user",
        "content": (
            f"Instrument name: {name}\n"
            f"User ticker/code: {user_ticker}\n"
            f"Instrument type hint: {hint_text}\n"
            "Provide up to {max_suggestions} high-confidence identifiers.\n"
            "Rules:\n"
            "- Stocks/BSE listings: return NSE (.NS) or BSE (.BO) symbols that resolve on Yahoo Finance.\n"
            "- Mutual funds: return the 6-digit AMFI scheme code that works with the mftool library.\n"
            "- Bonds (incl. SGB): return the exchange symbol or ISIN that works with yfinance (e.g., SGBFEB32IV).\n"
            "- PMS: return SEBI registration code (INP...).\n"
            "- AIF: return the AIF registration code.\n"
            "- Exclude guesses that fail these rules.\n"
            "Respond with a JSON array, e.g.:\n"
            '[{\"ticker\": \"TIMEX.BO\", \"instrument_type\": \"stock\", \"source\": \"yfinance_bse\", \"confidence\": 0.82, \"notes\": \"BSE symbol\"}]'
        ).format(max_suggestions=max_suggestions),
    }

    try:
        response = run_gpt5_completion(
            messages=[system_message, user_message],
            temperature=0,
            max_tokens=400,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        st.caption(f"   ⚠️ AI identifier suggestion failed: {str(exc)[:80]}")
        return []

    suggestions: List[Dict[str, Any]] = []
    candidates: List[Any] = []
    try:
        candidates.append(json.loads(response))
    except json.JSONDecodeError:
        block = _extract_json_block(response)
        if block:
            try:
                candidates.append(json.loads(block))
            except json.JSONDecodeError:
                pass

    for payload in candidates:
        if isinstance(payload, dict):
            raw = payload.get("suggestions")
            if isinstance(raw, list):
                suggestions = [s for s in raw if isinstance(s, dict)]
                break
        elif isinstance(payload, list):
            suggestions = [item for item in payload if isinstance(item, dict)]
            break

    if suggestions:
        return suggestions[:max_suggestions]

    # Deterministic fallbacks when AI cannot help
    fallback_suggestions: List[Dict[str, Any]] = []
    ticker_clean = (user_ticker or "").strip()
    name_lower = (name or "").lower()
    hint_lower = hint_text.lower()

    if ticker_clean.isdigit() and len(ticker_clean) <= 6:
        fallback_suggestions.append({
            "ticker": f"{ticker_clean}.BO" if not ticker_clean.endswith(".BO") else ticker_clean,
            "instrument_type": "stock",
            "confidence": 0.6,
            "source": "yfinance_bse",
            "notes": "Numeric ticker mapped to BSE code",
        })

    if ("mutual_fund" in hint_lower) or ("fund" in name_lower):
        try:
            dataset = get_amfi_dataset()
            amfi_resolution = resolve_mutual_fund_with_amfi(name or ticker_clean, ticker_clean, dataset)

            candidate_code = None
            confidence = 0.0

            if amfi_resolution.get("direct_scheme"):
                candidate_code = amfi_resolution["direct_scheme"]["code"]
                confidence = 0.9 if amfi_resolution["status"] == "direct" else 0.75
            elif amfi_resolution.get("matches"):
                candidate_code = amfi_resolution["matches"][0]["code"]
                confidence = amfi_resolution["matches"][0].get("score", 0.7)

            if candidate_code:
                fallback_suggestions.append({
                    "ticker": candidate_code,
                    "instrument_type": "mutual_fund",
                    "confidence": min(1.0, confidence),
                    "source": "amfi_lookup",
                    "notes": "Resolved using AMFI dataset",
                })
        except Exception:
            pass

    if fallback_suggestions:
        return fallback_suggestions[:max_suggestions]

    return []


def _should_refine_identifier(
    original_ticker: str,
    resolved_data: Dict[str, Any],
) -> bool:
    verified_ticker = (resolved_data.get('ticker') or "").strip()
    asset_type = (resolved_data.get('asset_type') or "").lower()
    if not verified_ticker:
        return True
    if verified_ticker == original_ticker:
        return True
    if asset_type == 'stock' and not (verified_ticker.endswith('.NS') or verified_ticker.endswith('.BO')):
        return True
    if asset_type == 'mutual_fund' and not verified_ticker.isdigit():
        return True
    if asset_type == 'bond' and verified_ticker.isdigit():
        return True
    if asset_type in {'pms', 'aif'} and verified_ticker == original_ticker:
        return True
    return False


def refine_resolved_identifier_with_ai(
    original_ticker: str,
    resolved_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Refine ticker/type using GPT suggestions when deterministic mapping is weak."""
    if run_gpt5_completion is None:
        return resolved_data

    if not _should_refine_identifier(original_ticker, resolved_data):
        return resolved_data

    name = resolved_data.get('name') or original_ticker
    asset_hint = resolved_data.get('asset_type')
    suggestions = ai_suggest_market_identifiers(name, original_ticker, asset_hint=asset_hint)
    if not suggestions:
        return resolved_data

    for candidate in suggestions:
        ticker_candidate = (candidate.get('ticker') or "").strip()
        instrument_type = (candidate.get('instrument_type') or "").strip().lower()
        if not ticker_candidate or not instrument_type:
            continue

        resolved_data['ticker'] = ticker_candidate
        resolved_data['asset_type'] = instrument_type
        resolved_data['source'] = candidate.get('source', resolved_data.get('source', 'ai'))
        resolved_data['confidence'] = candidate.get('confidence')
        resolved_data['notes'] = candidate.get('notes')
        break

    return resolved_data


def search_mftool_for_amfi_code(scheme_name):
    """
    Search mftool database for AMFI code by scheme name
    More reliable than AI! Uses intelligent keyword matching.
    """
    try:
        from mftool import Mftool
        mf = Mftool()
        
        # Get all schemes
        schemes = mf.get_scheme_codes()
        
        # Extract important keywords (filter out common words)
        name_lower = scheme_name.lower()
        skip_words = {'fund', 'plan', 'option', 'scheme', 'the', 'and', 'of', '-'}
        keywords = [word for word in name_lower.split() if len(word) > 2 and word not in skip_words]
        
        # Search for best match
        best_matches = []
        for code, name in schemes.items():
            scheme_lower = name.lower()
            
            # Count matching keywords
            match_count = sum(1 for kw in keywords if kw in scheme_lower)
            
            # Calculate match percentage
            match_pct = match_count / len(keywords) if keywords else 0
            
            # Bonus points for exact company/fund house match
            company_match = 0
            if 'sbi' in name_lower and 'sbi' in scheme_lower:
                company_match = 1
            elif 'hdfc' in name_lower and 'hdfc' in scheme_lower:
                company_match = 1
            elif 'tata' in name_lower and 'tata' in scheme_lower:
                company_match = 1
            elif 'quant' in name_lower and 'quant' in scheme_lower:
                company_match = 1
            elif 'iifl' in name_lower and 'iifl' in scheme_lower:
                company_match = 1
            elif '360' in name_lower and '360' in scheme_lower:
                company_match = 1
            
            # Only consider if at least 70% keywords match OR company matches with 50%+ keywords
            if match_pct >= 0.7 or (company_match and match_pct >= 0.5):
                best_matches.append((code, name, match_count, match_pct, company_match))
        
        # Sort by: company match, then keyword count, then percentage
        best_matches.sort(key=lambda x: (x[4], x[2], x[3]), reverse=True)
        
        if best_matches:
            # Return best match
            code, name, count, pct, company = best_matches[0]
            return {
                'ticker': code,
                'name': name,
                'sector': 'Mutual Fund',
                'source': 'mftool',
                'match_confidence': pct
            }
        
        return None
        
    except Exception as e:
        return None

def ai_resolve_tickers_from_names(ticker_name_pairs):
    """
    Use AI to resolve tickers from stock/fund names
    Returns verified tickers that work with yfinance/mftool
    """
    try:
        import openai
        
        # Try OpenAI first
        api_key = st.secrets["api_keys"].get("openai") or st.secrets["api_keys"].get("open_ai")
        use_gemini = False
        
        if api_key:
            client = openai.OpenAI(api_key=api_key)
        else:
            use_gemini = True
        
        # Build prompt
        ticker_list = []
        for ticker, info in ticker_name_pairs.items():
            asset_type = info.get('asset_type', 'stock')
            name = info.get('name', ticker)
            ticker_list.append(f"- Ticker: {ticker}, Name: {name}, Type: {asset_type}")
        
        prompt = f"""For each holding below, provide the VERIFIED ticker/code that works with yfinance (for stocks) or mftool (for mutual funds).

HOLDINGS:
{chr(10).join(ticker_list)}

YOUR TASK:
1. For STOCKS: 
   - Use the STOCK NAME (not ticker) as primary identifier
   - Search for the correct NSE/BSE ticker based on the company NAME
   - Try NSE first (.NS suffix): Verify it works with yfinance
   - If NSE fails, try BSE (.BO suffix): Verify it works with yfinance
   - Return whichever exchange ticker actually works
   - Example: "ITC LTD" → Search online → Find "ITC.NS" → Verify with yfinance → Return ITC.NS
   - IMPORTANT: Some stocks are ONLY on BSE, not NSE!
   - CRITICAL: Use the NAME to find correct ticker, don't just add .NS to existing ticker
   
2. For MUTUAL_FUND:
   - **EACH FUND MUST HAVE A UNIQUE AMFI CODE** - DO NOT reuse codes!
   - Search AMFI website or Value Research for the EXACT scheme code
   - Find the exact AMFI scheme code (6-digit number like 120760, 101305, etc.)
   - Different funds ALWAYS have different codes
   - Verify it works with mftool in Python
   - CRITICAL: Return numeric AMFI code, not scheme name
   - DOUBLE CHECK: Make sure you're not returning the same code for different funds!

3. For PMS (Portfolio Management Service):
   - Use the PMS registration code (format: INP000001234)
   - If not found, create a unique identifier based on PMS name
   - Return as-is (no API to verify)

4. For AIF (Alternative Investment Fund):
   - Use the AIF registration code (format: AIF-CAT1-12345)
   - If not found, create identifier from AIF name
   - Return as-is (no API to verify)

5. For BONDS:
   - For Sovereign Gold Bonds (SGB): Use NSE ticker (e.g., SGBFEB32IV)
   - For other bonds: Use ISIN code or exchange ticker
   - Try yfinance verification if possible
   
6. Provide sector information based on the stock/fund/bond name

7. Indicate source: yfinance_nse, yfinance_bse, mftool, manual, or isin

Return ONLY this JSON format:
{{
  "ORIGINAL_TICKER_OR_NAME": {{
    "ticker": "VERIFIED_TICKER_OR_AMFI_CODE",
    "name": "Full Name",
    "sector": "Sector Name",
    "source": "yfinance_nse|yfinance_bse|mftool",
    "verified": true
  }}
}}

EXAMPLES:
{{
  "RELIANCE": {{
    "ticker": "RELIANCE.NS",
    "name": "Reliance Industries Limited",
    "sector": "Oil & Gas",
    "source": "yfinance_nse",
    "verified": true
  }},
  "IDEA": {{
    "ticker": "IDEA.NS",
    "name": "Vodafone Idea Limited",
    "sector": "Telecom",
    "source": "yfinance_nse",
    "verified": true
  }},
  "BEDMUTHA": {{
    "ticker": "BEDMUTHA.BO",
    "name": "Bedmutha Industries Limited",
    "sector": "Chemicals",
    "source": "yfinance_bse",
    "verified": true
  }},
  "SBI Gold Direct Plan Growth": {{
    "ticker": "101305",
    "name": "SBI Gold Direct Plan Growth",
    "sector": "Gold",
    "source": "mftool",
    "verified": true
  }},
  "HDFC ELSS Tax Saver Direct Plan Growth": {{
    "ticker": "100104",
    "name": "HDFC ELSS Tax Saver Direct Plan Growth", 
    "sector": "ELSS",
    "source": "mftool",
    "verified": true
  }},
  "Quant Tax Plan Direct Growth": {{
    "ticker": "120760",
    "name": "Quant Tax Plan Direct Growth", 
    "sector": "ELSS",
    "source": "mftool",
    "verified": true
  }},
  "IDFC Nifty 50 Index Direct Plan Growth": {{
    "ticker": "134997",
    "name": "IDFC Nifty 50 Index Direct Plan Growth", 
    "sector": "Index Fund",
    "source": "mftool",
    "verified": true
  }},
  "Tata Small Cap Fund Direct Growth": {{
    "ticker": "125497",
    "name": "Tata Small Cap Fund Direct Growth", 
    "sector": "Small Cap",
    "source": "mftool",
    "verified": true
  }},
  "2.50% Gold Bonds 2032 SR-IV": {{
    "ticker": "SGBFEB32IV",
    "name": "Sovereign Gold Bond 2032 Series IV",
    "sector": "Government Securities",
    "source": "yfinance_nse",
    "verified": true
  }},
  "Buoyant Opportunities Fund": {{
    "ticker": "INP000012345",
    "name": "Buoyant Opportunities Fund",
    "sector": "PMS",
    "source": "manual",
    "verified": true
  }},
  "Private Equity Fund XYZ": {{
    "ticker": "AIF-CAT1-12345",
    "name": "Private Equity Fund XYZ",
    "sector": "AIF",
    "source": "manual",
    "verified": true
  }}
}}

CRITICAL RULES:
- **USE THE STOCK NAME** as primary identifier, not the ticker! Search for the correct ticker based on company name.
- For stocks: Search company name → Find NSE/BSE ticker → Verify with yfinance → Return working ticker
- For mutual funds: 
  * **EVERY MUTUAL FUND HAS A UNIQUE AMFI CODE** - NEVER use the same code for different funds!
  * Search AMFI website, Value Research, or MoneyControl for the EXACT scheme code
  * Each fund name gets its OWN unique 6-digit code (like 101305, 100104, 120760, 134997, 125497)
  * Return numeric AMFI code, NOT scheme name
  * DOUBLE-CHECK: If you're returning the same code twice, YOU ARE WRONG!
- For PMS/AIF: Search name → Find registration code (INP/AIF format) → Return code
- For Bonds: Search name → Find ISIN or NSE ticker → Verify with yfinance
- Always verify ticker actually works with yfinance/mftool before returning
- If ticker already has .NS or .BO, verify it still works (don't assume it's correct)
- If you're unsure, prefer NSE for major stocks, BSE for smaller/regional stocks

IMPORTANT: Don't just add .NS to existing ticker! Use the NAME to search and find the correct working ticker!

**MUTUAL FUND WARNING**: Each mutual fund scheme has its own unique AMFI code. NEVER return duplicate codes. If you don't know the exact code, search for it online before responding!

**JSON FORMAT REQUIREMENTS:**
- Return ONLY valid JSON - no comments, no trailing commas, no extra text
- Each object must have ALL fields: ticker, name, sector, source, verified
- Ensure proper comma separation between objects
- Do not include any explanatory text before or after the JSON
- Test your JSON is valid before returning

Return ONLY the JSON object, nothing else."""

        # Try OpenAI first, fallback to Gemini
        ai_response = None
        
        if not use_gemini:
            try:
                st.caption(f"   🔄 Using OpenAI (gpt-5) for ticker resolution...")
                response = client.chat.completions.create(
                    model="gpt-5",  # GPT-5 for better accuracy and reasoning
                    messages=[
                        {"role": "system", "content": "You are a financial ticker verification expert with access to real-time data. For each ticker, search online databases and verify it works with yfinance or mftool APIs. Return ONLY valid JSON with unique tickers for each holding."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=2000
                    # Note: GPT-5 only supports default temperature (1)
                )
                ai_response = response.choices[0].message.content
                st.caption(f"   ✅ OpenAI response received")
            except Exception as e:
                error_msg = str(e)
                st.caption(f"   ❌ OpenAI error: {error_msg[:100]}")
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.caption(f"   ⚠️ OpenAI quota exceeded, trying Gemini...")
                    use_gemini = True
                else:
                    st.caption(f"   ⚠️ OpenAI failed, trying Gemini...")
                    use_gemini = True
        
        # Fallback to Gemini
        if use_gemini and ai_response is None:
            try:
                import google.generativeai as genai
                gemini_key = st.secrets["api_keys"].get("gemini_api_key")
                if gemini_key:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    full_prompt = f"""You are a financial ticker verification expert. Verify tickers for yfinance and mftool APIs. Return ONLY valid JSON.

{prompt}"""
                    
                    response = model.generate_content(full_prompt)
                    ai_response = response.text
                    st.caption(f"   ✅ Using Gemini for ticker resolution")
            except Exception as e:
                st.caption(f"   ❌ Gemini also failed: {str(e)[:50]}")
                return {}

        if not ai_response:
            return {}
        
        # Extract JSON with better error handling
        import json
        import re
        
        response_text = ai_response.strip()
        
        # Remove markdown code blocks
        if response_text.startswith('```'):
            # Remove ```json or ``` at start
            response_text = re.sub(r'^```(?:json)?\s*\n', '', response_text)
            # Remove ``` at end
            response_text = re.sub(r'\n```\s*$', '', response_text)
            response_text = response_text.strip()
        
        # Find JSON object
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            
            try:
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError as je:
                # Try to fix common JSON errors
                st.caption(f"   ⚠️ JSON parse error, attempting auto-fix...")
                
                # Fix 1: Remove trailing commas before } or ]
                fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
                
                # Fix 2: Remove comments (// or /* */)
                fixed_json = re.sub(r'//.*$', '', fixed_json, flags=re.MULTILINE)
                fixed_json = re.sub(r'/\*.*?\*/', '', fixed_json, flags=re.DOTALL)
                
                # Fix 3: Fix unquoted keys
                fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
                
                try:
                    parsed = json.loads(fixed_json)
                    st.caption(f"   ✅ Auto-fixed JSON successfully")
                    return parsed
                except:
                    # Show details about original error
                    st.caption(f"   ⚠️ JSON parse error at line {je.lineno}, column {je.colno}")
                    st.caption(f"   Error: {je.msg}")
                    # Show snippet of problematic JSON
                    lines = json_str.split('\n')
                    if je.lineno <= len(lines):
                        problem_line = lines[je.lineno - 1] if je.lineno > 0 else ""
                        st.caption(f"   Problem: {problem_line[:100]}")
                    return {}
        
        st.caption(f"   ⚠️ Could not find valid JSON in AI response")
        return {}
        
    except Exception as e:
        st.caption(f"   ⚠️ AI ticker resolution error: {str(e)[:100]}")
        import traceback
        st.caption(f"   Traceback: {traceback.format_exc()[:200]}")
        return {}

def _legacy_process_uploaded_files(uploaded_files, user_id, portfolio_id):
    """
    Process uploaded files and store to DB using AI
    Supports CSV, PDF, Excel, Images, and any file format
    """
    total_imported = 0
    processing_log = []
    
    # Always show CSV processing message (AI is used only for ticker resolution)
    st.info(f"📊 Processing {len(uploaded_files)} file(s)...")
    
    for file_idx, uploaded_file in enumerate(uploaded_files, 1):
        st.caption(f"📁 [{file_idx}/{len(uploaded_files)}] Processing {uploaded_file.name}...")
        
        # Initialize error tracking variables at the start
        imported = 0
        skipped = 0
        errors = 0
        tickers_in_this_file = set()  # Track tickers from this file to avoid re-fetching old data
        
        try:
            # Direct CSV/Excel processing (AI file extraction is disabled)
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext in ['csv', 'xlsx', 'xls']:
                st.caption(f"   📊 Processing {file_ext.upper()} file directly (no AI needed)...")
                
                # Read file with pandas
                if file_ext == 'csv':
                    df = pd.read_csv(uploaded_file)
                else:  # xlsx or xls
                    try:
                        engine = 'openpyxl' if file_ext == 'xlsx' else None
                        df = pd.read_excel(uploaded_file, engine=engine)
                    except ImportError as e:
                        st.error(f"❌ Missing Excel dependency. Install with: `pip install openpyxl xlrd`")
                        return
                    except Exception as e:
                        if 'openpyxl' in str(e).lower() or 'xlrd' in str(e).lower():
                            st.error(f"❌ Missing Excel dependency. Install with: `pip install openpyxl xlrd`")
                        else:
                            st.error(f"❌ Error reading Excel file: {e}")
                        return
                
                st.caption(f"   ✅ Read {len(df)} rows from {file_ext.upper()}")
                
                # Helper functions to handle NaN/None values
                def safe_value(val, default=''):
                    """Convert pandas NaN to safe value"""
                    if pd.isna(val) or val is None:
                        return default
                    return val
                
                def safe_float(val, default=0.0):
                    """Convert to float, handling NaN/None"""
                    try:
                        if pd.isna(val) or val is None or val == '':
                            return default
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                # Detect dominant date format in CSV by sampling dates
                def detect_date_format(df, date_column='date'):
                    """
                    Detect the dominant date format in the CSV by sampling dates.
                    Returns 'dayfirst' (DD-MM-YYYY) or 'monthfirst' (MM-DD-YYYY) or None (ambiguous)
                    """
                    if date_column not in df.columns:
                        return None
                    
                    # Sample up to 50 non-null dates from the CSV
                    sample_dates = df[date_column].dropna().head(50).tolist()
                    if not sample_dates:
                        return None
                    
                    dayfirst_count = 0
                    monthfirst_count = 0
                    unambiguous_count = 0
                    
                    for date_str in sample_dates:
                        try:
                            date_str = str(date_str).strip()
                            # Remove time part if present
                            if ' AM' in date_str.upper() or ' PM' in date_str.upper():
                                date_str = date_str.split()[0]
                            
                            # Check if date has numeric parts separated by - or /
                            parts = date_str.replace('/', '-').replace('.', '-').split('-')
                            if len(parts) == 3:
                                try:
                                    first_part = int(parts[0])
                                    second_part = int(parts[1])
                                    
                                    # If first part > 12, it MUST be DD-MM-YYYY (dayfirst)
                                    if first_part > 12:
                                        dayfirst_count += 1
                                        unambiguous_count += 1
                                    # If second part > 12, it MUST be MM-DD-YYYY (monthfirst)
                                    elif second_part > 12:
                                        monthfirst_count += 1
                                        unambiguous_count += 1
                                    # If both <= 12, it's ambiguous - try both and see which makes sense
                                    else:
                                        # Try dayfirst
                                        dt1 = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                                        # Try monthfirst
                                        dt2 = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
                                        
                                        if pd.notna(dt1) and pd.notna(dt2):
                                            # Both parse successfully - check which is more reasonable
                                            # Prefer dates that are not too far in the future or past
                                            from datetime import datetime
                                            today = datetime.now()
                                            diff1 = abs((dt1 - today).days)
                                            diff2 = abs((dt2 - today).days)
                                            
                                            # If one is clearly more reasonable (closer to today), use that
                                            if diff1 < diff2:
                                                dayfirst_count += 0.5  # Half weight for ambiguous
                                            elif diff2 < diff1:
                                                monthfirst_count += 0.5
                                except:
                                    pass
                        except:
                            pass
                    
                    # Determine dominant format
                    if unambiguous_count > 0:
                        # If we have unambiguous dates, trust them
                        if dayfirst_count > monthfirst_count:
                            return 'dayfirst'
                        elif monthfirst_count > dayfirst_count:
                            return 'monthfirst'
                    
                    # If no clear winner, default to dayfirst for Indian CSVs
                    return 'dayfirst'
                
                # Detect format once for this CSV
                detected_format = detect_date_format(df)
                if detected_format:
                    st.caption(f"   📅 Detected date format: {'DD-MM-YYYY' if detected_format == 'dayfirst' else 'MM-DD-YYYY'}")
                
                def normalize_date(date_str, preferred_format=None):
                    """Convert ANY date format to YYYY-MM-DD with robust format detection"""
                    try:
                        if pd.isna(date_str) or not date_str:
                            return ''
                        
                        date_str = str(date_str).strip()
                        
                        # Remove time part ONLY if AM/PM is present (e.g., "18-03-2021 09:34 AM" -> "18-03-2021")
                        # Preserve formats like "10 Oct 2025" (don't split these!)
                        if ' AM' in date_str.upper() or ' PM' in date_str.upper():
                            date_str = date_str.split()[0]
                        
                        # Try multiple parsing strategies for maximum compatibility
                        dt = None
                        
                        # First, check if date is unambiguous (can determine format from values)
                        parts = date_str.replace('/', '-').replace('.', '-').split('-')
                        is_ambiguous = False
                        if len(parts) == 3:
                            try:
                                first_part = int(parts[0])
                                second_part = int(parts[1])
                                # If first part > 12, it MUST be DD-MM-YYYY (unambiguous)
                                # If second part > 12, it MUST be MM-DD-YYYY (unambiguous)
                                # If both <= 12, it's ambiguous
                                is_ambiguous = (first_part <= 12 and second_part <= 12)
                            except:
                                pass
                        
                        # For unambiguous dates, use the correct format
                        # For ambiguous dates, use preferred_format (detected from CSV)
                        use_dayfirst = preferred_format == 'dayfirst' if preferred_format else True
                        
                        # Strategy 1: Try preferred format first (for ambiguous dates) or dayfirst (for unambiguous)
                        if not is_ambiguous or use_dayfirst:
                            dt = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                            if pd.notna(dt):
                                # Validate unambiguous dates
                                if len(parts) == 3:
                                    try:
                                        first_part = int(parts[0])
                                        if first_part > 12:
                                            return dt.strftime('%Y-%m-%d')  # Definitely DD-MM-YYYY
                                    except:
                                        pass
                                # For ambiguous dates, trust preferred format
                                if is_ambiguous and use_dayfirst:
                                    return dt.strftime('%Y-%m-%d')
                                elif not is_ambiguous:
                                    return dt.strftime('%Y-%m-%d')
                        
                        # Strategy 2: Try opposite format (for ambiguous dates with different preference)
                        if is_ambiguous and not use_dayfirst:
                            dt = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
                            if pd.notna(dt):
                                return dt.strftime('%Y-%m-%d')
                        
                        # Strategy 3: Try both formats and validate (fallback for ambiguous dates)
                        if is_ambiguous:
                            dt1 = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                            dt2 = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
                            
                            if pd.notna(dt1) and pd.notna(dt2):
                                # Both parse successfully - use preferred format
                                if use_dayfirst:
                                    return dt1.strftime('%Y-%m-%d')
                                else:
                                    return dt2.strftime('%Y-%m-%d')
                            elif pd.notna(dt1):
                                return dt1.strftime('%Y-%m-%d')
                            elif pd.notna(dt2):
                                return dt2.strftime('%Y-%m-%d')
                        
                        # Strategy 4: Try dayfirst=False for unambiguous dates (if Strategy 1 failed)
                        if not is_ambiguous:
                            dt = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
                            if pd.notna(dt):
                                # Validate
                                if len(parts) == 3:
                                    try:
                                        second_part = int(parts[1])
                                        if second_part > 12:
                                            return dt.strftime('%Y-%m-%d')  # Definitely MM-DD-YYYY
                                    except:
                                        pass
                                return dt.strftime('%Y-%m-%d')
                        
                        # Strategy 5: Try infer_datetime_format (auto-detect)
                        dt = pd.to_datetime(date_str, infer_datetime_format=True, errors='coerce')
                        if pd.notna(dt):
                            return dt.strftime('%Y-%m-%d')
                        
                        # Strategy 6: Try common explicit formats
                        common_formats = [
                            '%d-%m-%Y',      # DD-MM-YYYY
                            '%d/%m/%Y',      # DD/MM/YYYY
                            '%m-%d-%Y',      # MM-DD-YYYY
                            '%m/%d/%Y',      # MM/DD/YYYY
                            '%Y-%m-%d',      # YYYY-MM-DD
                            '%Y/%m/%d',      # YYYY/MM/DD
                            '%d-%m-%y',      # DD-MM-YY
                            '%d/%m/%y',      # DD/MM/YY
                            '%m-%d-%y',      # MM-DD-YY
                            '%m/%d/%y',      # MM/DD/YY
                            '%d %b %Y',      # DD Mon YYYY (e.g., "10 Oct 2025")
                            '%d %B %Y',      # DD Month YYYY (e.g., "10 October 2025")
                            '%b %d, %Y',     # Mon DD, YYYY (e.g., "Oct 10, 2025")
                            '%B %d, %Y',     # Month DD, YYYY (e.g., "October 10, 2025")
                            '%Y-%m-%d %H:%M:%S',  # With timestamp
                            '%d-%m-%Y %H:%M:%S',  # With timestamp
                        ]
                        
                        for fmt in common_formats:
                            try:
                                dt = pd.to_datetime(date_str, format=fmt, errors='coerce')
                                if pd.notna(dt):
                                    return dt.strftime('%Y-%m-%d')
                            except:
                                continue
                        
                        # Final fallback: try python-dateutil parser with different assumptions
                        for dayfirst_flag in (
                            preferred_format != 'monthfirst',
                            True,
                            False,
                        ):
                            try:
                                dt = dateutil_parser.parse(
                                    date_str,
                                    dayfirst=dayfirst_flag,
                                    fuzzy=True,
                                )
                                return dt.strftime('%Y-%m-%d')
                            except Exception:
                                continue

                        # Final resort: strip non-date characters and retry
                        cleaned = re.sub(r'[^0-9A-Za-z:/\\ -]', ' ', date_str)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

                        for dayfirst_flag in (True, False):
                            try:
                                dt = dateutil_parser.parse(
                                    cleaned,
                                    dayfirst=dayfirst_flag,
                                    fuzzy=True,
                                )
                                return dt.strftime('%Y-%m-%d')
                            except Exception:
                                continue

                        # If all strategies failed, return empty
                        return ''
                    except Exception as e:
                        # Log error for debugging (but don't break the import)
                        print(f"[DATE_PARSE_ERROR] Failed to parse date '{date_str}': {str(e)}")
                        return ''
                    finally:
                        pass
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_rows = len(df)
                
                for idx, (_, row) in enumerate(df.iterrows()):
                    try:
                        # Update progress every 50 rows
                        if idx % 50 == 0:
                            progress_bar.progress((idx + 1) / total_rows)
                            status_text.text(f"📝 Processing row {idx + 1}/{total_rows}... ({imported} imported, {skipped} skipped)")
                        
                        # Extract channel from filename if not provided
                        channel = safe_value(row.get('channel'), None)
                        if not channel:
                            # Use filename without extension
                            import os
                            channel = os.path.splitext(uploaded_file.name)[0]
                        
                        # Smart mapping - ticker can be NaN (AI will resolve later)
                        ticker = safe_value(row.get('ticker'), '')
                        stock_name = safe_value(row.get('stock_name'), '')
                        
                        # Clean ticker - remove $ and other special characters
                        ticker = str(ticker).replace('$', '').strip()
                        
                        # If ticker is NaN/empty, use stock_name as placeholder
                        # AI will resolve this to proper ticker later
                        if not ticker:
                            ticker = stock_name  # Temporary - AI will fix
                        
                        # Determine asset type
                        asset_type = safe_value(row.get('asset_type'), None)
                        if not asset_type:
                            # Auto-detect from stock name
                            name_lower = stock_name.lower()
                            if 'pms' in name_lower or 'portfolio management' in name_lower or ticker.startswith('INP'):
                                asset_type = 'pms'
                            elif 'aif' in name_lower or 'alternative investment' in name_lower or ticker.startswith('AIF'):
                                asset_type = 'aif'
                            elif 'bond' in name_lower or 'debenture' in name_lower or 'sgb' in name_lower:
                                asset_type = 'bond'
                            elif 'fund' in name_lower or 'scheme' in name_lower or 'growth' in name_lower:
                                asset_type = 'mutual_fund'
                            else:
                                asset_type = 'stock'
                        else:
                            asset_type = str(asset_type)
                        
                        # For PMS, AIF, Bonds - ignore CSV ticker, use stock_name as placeholder
                        # AI will fetch proper ticker/code later
                        if asset_type in ['pms', 'aif', 'bond']:
                            ticker = stock_name  # Force AI to resolve
                        
                        # SMART PRICE DETECTION (Priority Order):
                        # 1. If CSV has 'price' column → use it directly
                        # 2. If CSV has 'amount' column → calculate price = amount ÷ quantity
                        # 3. If CSV has only 'date' → fetch historical price for that date
                        # 4. If nothing works → price = 0 (will show as missing)
                        
                        csv_price = safe_float(row.get('price', 0), 0)
                        
                        # Try calculating from amount if price is missing (check both 'amount' and 'Amount')
                        if csv_price == 0:
                            # Check both lowercase and capital A (Groww uses different cases)
                            csv_amount = safe_float(row.get('amount', 0), 0)
                            if csv_amount == 0:
                                csv_amount = safe_float(row.get('Amount', 0), 0)
                            
                            csv_quantity = safe_float(row.get('quantity', 0), 0)
                            
                            if csv_amount > 0 and csv_quantity > 0:
                                csv_price = csv_amount / csv_quantity
                        
                        transaction_date = normalize_date(row['date'], preferred_format=detected_format)
                        
                        # Last resort: Fetch historical price for transaction date if price still missing
                        if csv_price == 0 and transaction_date:
                            try:
                                from enhanced_price_fetcher import EnhancedPriceFetcher
                                price_fetcher = EnhancedPriceFetcher()
                                
                                # Fetch historical price for transaction date
                                hist_price = price_fetcher.get_historical_price(
                                    ticker=ticker,
                                    asset_type=asset_type,
                                    date=transaction_date,
                                    fund_name=stock_name
                                )
                                
                                if hist_price and hist_price > 0:
                                    csv_price = hist_price
                                    st.caption(f"      📅 Fetched price for {ticker} on {transaction_date}: ₹{hist_price:.2f}")
                            except Exception as e:
                                pass  # Keep price as 0, will show in logs
                        
                        # Track ticker from this file
                        if ticker:
                            tickers_in_this_file.add(ticker)
                        
                        # Import with placeholder ticker (AI will resolve later)
                        transaction_data = {
                            'user_id': user_id,
                            'portfolio_id': portfolio_id,
                            'ticker': str(ticker),  # May be stock_name if ticker was NaN
                            'stock_name': str(stock_name),
                            'scheme_name': str(stock_name) if asset_type == 'mutual_fund' else None,
                            'quantity': safe_float(row['quantity'], 0),
                            'price': csv_price,  # Use fetched historical price if CSV price was missing
                            'transaction_date': transaction_date,
                            'transaction_type': str(safe_value(row['transaction_type'], 'buy')).lower(),
                            'asset_type': str(asset_type),
                            'channel': str(safe_value(channel, 'Direct')),
                            # For bonds, always set sector to "bond"
                            'sector': 'bond' if asset_type == 'bond' else str(safe_value(row.get('sector', 'Unknown'), 'Unknown')),
                            'filename': uploaded_file.name
                        }

                        result = db.add_transaction(transaction_data)
                        if result.get('success'):
                            imported += 1

                            # Add small delay every 50 transactions to avoid overwhelming database
                            if imported % 50 == 0:
                                import time
                                time.sleep(0.5)  # 500ms pause
                        else:
                            # Show why it failed
                            error_msg = result.get('error', 'Unknown error')
                            if 'duplicate' in error_msg.lower():
                                skipped += 1  # Count duplicates as skipped
                            elif '502' in error_msg or 'bad gateway' in error_msg.lower():
                                # 502 error - Supabase server issue
                                st.warning("   ⚠️ Database server error (502). Retrying in 2 seconds...")
                                import time
                                time.sleep(2)
                                # Retry once
                                retry_result = db.add_transaction(transaction_data)
                                if retry_result.get('success'):
                                    imported += 1
                                else:
                                    st.caption(f"   ⚠️ Retry failed: {retry_result.get('error', 'Unknown')[:100]}")
                                    errors += 1
                            else:
                                st.caption(f"   ⚠️ Failed: {error_msg[:100]}")
                                errors += 1
                    except Exception as e:
                        errors += 1
                        st.caption(f"   ❌ Error processing row {idx + 1}: {str(e)[:100]}")
                        continue
                
                # Clear progress bar
                progress_bar.empty()
                status_text.empty()
                
                # Show detailed import summary
                if file_ext in ['csv', 'xlsx', 'xls']:
                    if imported > 0:
                        st.success(f"   ✅ Imported {imported} transactions from {uploaded_file.name}")
                    if skipped > 0:
                        st.info(f"   ⏭️  Skipped {skipped} duplicate transactions")
                    if errors > 0:
                        st.warning(f"   ⚠️  {errors} errors encountered")
                    if imported == 0 and skipped == 0 and errors == 0:
                        st.error(f"   ❌ No transactions imported from {uploaded_file.name} - check CSV format!")
                
                # Log summary
                print(f"[CSV_IMPORT] File: {uploaded_file.name} | Rows read: {total_rows} | Imported: {imported} | Skipped: {skipped} | Errors: {errors}")
                
                # Use AI to resolve tickers based on stock names
                # Run even if imported=0 to fix existing data!
                if AI_TICKER_RESOLUTION_ENABLED:
                    st.caption(f"   🤖 Using AI to resolve tickers from stock names...")
                    
                    try:
                        # Get ALL transactions to fix tickers
                        all_transactions = db.get_user_transactions(user_id)
                        
                        # Filter: Get transactions that need ticker resolution
                        # Resolve ALL tickers to ensure they have correct .NS/.BO suffix and AMFI codes
                        file_transactions = []
                        for t in all_transactions:
                            ticker = t.get('ticker', '')
                            asset_type = t.get('asset_type', 'stock')
                            
                            needs_resolution = False
                            
                            # STOCKS: Resolve ALL to get proper .NS/.BO suffix
                            if asset_type == 'stock':
                                # Skip if already has .NS or .BO suffix
                                if not (ticker.endswith('.NS') or ticker.endswith('.BO')):
                                    needs_resolution = True
                            
                            # If ticker contains spaces, $, or is very long (scheme names)
                            if ' ' in ticker or '$' in ticker or len(ticker) > 50:
                                needs_resolution = True
                            
                            # PMS, AIF, Bonds ALWAYS need AI resolution to find proper codes
                            if asset_type in ['pms', 'aif', 'bond']:
                                needs_resolution = True
                            
                            # Mutual funds: SKIP AI resolution (AI doesn't have accurate AMFI codes)
                            # Will fetch prices using scheme_name with mftool later
                            if asset_type == 'mutual_fund':
                                needs_resolution = False  # Don't resolve MF with AI
                            
                            if needs_resolution:
                                file_transactions.append(t)
                        
                        st.caption(f"   Found {len(file_transactions)} transactions needing ticker resolution...")
                        
                        if file_transactions:
                            # Separate by asset type for different processing strategies
                            stocks_to_resolve = {}
                            others_to_resolve = {}
                            
                            for trans in file_transactions:
                                ticker = trans.get('ticker')
                                stock_name = trans.get('stock_name')
                                asset_type = trans.get('asset_type', 'stock')
                                
                                if ticker and stock_name:
                                    if asset_type == 'stock':
                                        # Stocks - will batch process
                                        stocks_to_resolve[ticker] = {
                                            'name': stock_name,
                                            'asset_type': asset_type
                                        }
                                    else:
                                        # PMS/AIF/Bonds - process one by one
                                        others_to_resolve[ticker] = {
                                            'name': stock_name,
                                            'asset_type': asset_type
                                        }
                            
                            all_resolved = {}
                            
                            # Process STOCKS in larger batches (10 at a time - AI is very accurate for stocks)
                            if stocks_to_resolve:
                                st.caption(f"   🔍 Resolving {len(stocks_to_resolve)} stock tickers (batch size: 10)...")
                                
                                batch_size = 10  # Larger batch for stocks
                                stock_items = list(stocks_to_resolve.items())
                                
                                for batch_start in range(0, len(stock_items), batch_size):
                                    batch_end = min(batch_start + batch_size, len(stock_items))
                                    batch = dict(stock_items[batch_start:batch_end])
                                    
                                    st.caption(f"      Stock batch {batch_start//batch_size + 1} ({len(batch)} tickers)...")
                                    
                                    # Use AI to get verified tickers and sectors
                                    batch_resolved = ai_resolve_tickers_from_names(batch)
                                    if batch_resolved:
                                        for original_key, info in batch.items():
                                            resolved_entry = batch_resolved.get(original_key)
                                            if not resolved_entry:
                                                continue
                                            resolved_entry.setdefault('name', info.get('name'))
                                            resolved_entry['asset_type'] = info.get('asset_type', resolved_entry.get('asset_type', 'stock'))
                                            batch_resolved[original_key] = refine_resolved_identifier_with_ai(original_key, resolved_entry)
                                    if batch_resolved:
                                        all_resolved.update(batch_resolved)
                            
                            # Process PMS/AIF/BONDS one by one (need more careful verification)
                            if others_to_resolve:
                                st.caption(f"   🔍 Resolving {len(others_to_resolve)} PMS/AIF/Bond tickers (one-by-one)...")
                                
                                for ticker, info in others_to_resolve.items():
                                    single_pair = {ticker: info}
                                    
                                    # Use AI to get verified ticker
                                    single_resolved = ai_resolve_tickers_from_names(single_pair)
                                    if single_resolved:
                                        resolved_entry = single_resolved.get(ticker)
                                        if resolved_entry:
                                            resolved_entry.setdefault('name', info.get('name'))
                                            resolved_entry['asset_type'] = info.get('asset_type', resolved_entry.get('asset_type'))
                                            single_resolved[ticker] = refine_resolved_identifier_with_ai(ticker, resolved_entry)
                                    if single_resolved:
                                        all_resolved.update(single_resolved)
                            
                            resolved_tickers = all_resolved
                            
                            if resolved_tickers:
                                st.caption(f"   ✅ AI resolved {len(resolved_tickers)} tickers with verified sources")
                                
                                # CRITICAL: Check for duplicate tickers (AI bug detection)
                                verified_ticker_list = [data.get('ticker') for data in resolved_tickers.values() if data.get('ticker')]
                                unique_verified = set(verified_ticker_list)
                                
                                if len(verified_ticker_list) != len(unique_verified):
                                    # DUPLICATES DETECTED - This is EXPECTED for same stocks with different CSV ticker variations
                                    st.info(f"   ℹ️ Found {len(verified_ticker_list) - len(unique_verified)} ticker merges (same stocks, different CSV tickers)")
                                    
                                    # Find which tickers are duplicated
                                    from collections import Counter
                                    ticker_counts = Counter(verified_ticker_list)
                                    duplicates = {ticker: count for ticker, count in ticker_counts.items() if count > 1}
                                    
                                    for dup_ticker, count in duplicates.items():
                                        # Show which CSV tickers will be merged
                                        affected = [orig for orig, data in resolved_tickers.items() if data.get('ticker') == dup_ticker]
                                        st.caption(f"      ✓ Merging: {', '.join(affected[:5])} → {dup_ticker}")
                                    
                                    st.caption("   ℹ️ These are legitimate merges (e.g., rights issues, bonus shares, name variations)")
                                else:
                                    # All tickers are unique
                                    st.caption(f"   ✅ All {len(unique_verified)} tickers are unique")
                                
                                # Proceed with updates (duplicates are valid merges!)
                                updated_count = 0
                                for original_ticker, resolved_data in resolved_tickers.items():
                                    resolved_data = refine_resolved_identifier_with_ai(original_ticker, resolved_data)
                                    verified_ticker = resolved_data.get('ticker')
                                    sector = resolved_data.get('sector', 'Unknown')
                                    source = resolved_data.get('source', 'ai')
                                    
                                    if verified_ticker:
                                        # Update stock_master record with verified ticker
                                        # Handle merges (when multiple CSV tickers resolve to same verified ticker)
                                        
                                        if verified_ticker != original_ticker:
                                            # Find old stock_master record with original ticker
                                            old_stock = db.supabase.table('stock_master').select('id, ticker, stock_name').eq(
                                                'ticker', original_ticker
                                            ).execute()
                                            
                                            if old_stock.data:
                                                old_stock_id = old_stock.data[0]['id']
                                                stock_name = resolved_data.get('name', old_stock.data[0].get('stock_name'))
                                                
                                                # Check if target ticker already exists
                                                existing_stock = db.supabase.table('stock_master').select('id').eq(
                                                    'ticker', verified_ticker
                                                ).eq('stock_name', stock_name).execute()
                                                
                                                if existing_stock.data:
                                                    # Target ticker already exists - MERGE!
                                                    # Point transactions to existing stock and delete old record
                                                    target_stock_id = existing_stock.data[0]['id']
                                                    
                                                    # Update all transactions to point to existing stock
                                                    db.supabase.table('user_transactions').update({
                                                        'stock_id': target_stock_id
                                                    }).eq('stock_id', old_stock_id).execute()
                                                    
                                                    # Delete old stock_master record
                                                    db.supabase.table('stock_master').delete().eq('id', old_stock_id).execute()
                                                    
                                                    st.caption(f"      ✓ Merged: {original_ticker} → {verified_ticker}")
                                                    updated_count += 1
                                                else:
                                                    # Target doesn't exist - safe to update
                                                    db.supabase.table('stock_master').update({
                                                        'ticker': verified_ticker,
                                                        'sector': sector
                                                    }).eq('id', old_stock_id).execute()
                                                    
                                                    st.caption(f"      ✓ Updated: {original_ticker} → {verified_ticker}")
                                                    updated_count += 1
                                            else:
                                                # Create new stock_master record with verified ticker
                                                # Get asset_type from original groups
                                                asset_type = resolved_data.get('asset_type') or \
                                                    stocks_to_resolve.get(original_ticker, {}).get('asset_type') or \
                                                    others_to_resolve.get(original_ticker, {}).get('asset_type', 'stock')
                                                
                                                # For bonds, always set sector to "bond"
                                                if asset_type == 'bond':
                                                    sector = 'bond'
                                                
                                                new_stock_id = db.get_or_create_stock(
                                                    ticker=verified_ticker,
                                                    stock_name=resolved_data.get('name', original_ticker),
                                                    asset_type=asset_type,
                                                    sector=sector
                                                )
                                                
                                                st.caption(f"      ✓ Created: {verified_ticker}")
                                                updated_count += 1
                                        else:
                                            # Just update sector if ticker is same
                                            update_payload = {'sector': sector}
                                            if resolved_data.get('asset_type'):
                                                update_payload['asset_type'] = resolved_data['asset_type']
                                            db.supabase.table('stock_master').update(update_payload).eq('ticker', verified_ticker).execute()
                                            updated_count += 1
                                
                                if updated_count > 0:
                                    st.caption(f"   ✅ Updated {updated_count} transactions with verified tickers")
                    
                    except Exception as e:
                        st.caption(f"   ⚠️ AI ticker resolution skipped: {str(e)[:100]}")
                
                # Resolve MUTUAL FUNDS using mftool search (more reliable than AI)
                st.caption(f"   🔍 Resolving mutual fund AMFI codes using mftool search...")
                try:
                    # Get all mutual fund transactions
                    mf_transactions = [t for t in db.get_user_transactions(user_id) if t.get('asset_type') == 'mutual_fund']
                    
                    if mf_transactions:
                        mf_updated = 0
                        amfi_dataset = get_amfi_dataset()
                        code_lookup = amfi_dataset.get('code_lookup', {})
                        for trans in mf_transactions:
                            scheme_name = trans.get('stock_name', '') or ''
                            current_ticker = str(trans.get('ticker', '') or '').strip()

                            final_code = None
                            final_name = None
                            final_source = None
                            ai_confidence = None

                            resolution = resolve_mutual_fund_with_amfi(scheme_name, current_ticker, amfi_dataset)
                            status = resolution.get('status')
                            direct_scheme = resolution.get('direct_scheme')
                            matches = resolution.get('matches', [])

                            if status == 'direct' and direct_scheme:
                                final_code = direct_scheme['code']
                                final_name = direct_scheme['name']
                                final_source = 'amfi_direct'
                            elif status in {'direct_mismatch', 'name_matches'}:
                                top_match = matches[0] if matches else None
                                if top_match and top_match.get('score', 0) >= 0.92:
                                    final_code = top_match['code']
                                    final_name = top_match['name']
                                    final_source = 'amfi_name_match'
                                else:
                                    ai_choice = ai_select_amfi_code(scheme_name, current_ticker, matches)
                                    if ai_choice and ai_choice.get('code'):
                                        candidate_code = str(ai_choice['code']).strip()
                                        scheme = code_lookup.get(candidate_code)
                                        if scheme:
                                            final_code = candidate_code
                                            final_name = scheme['name']
                                            final_source = 'ai_amfi'
                                            ai_confidence = ai_choice.get('confidence')
                                    elif status == 'direct_mismatch' and direct_scheme:
                                        final_code = direct_scheme['code']
                                        final_name = direct_scheme['name']
                                        final_source = 'amfi_direct_mismatch'
                            else:
                                if matches:
                                    ai_choice = ai_select_amfi_code(scheme_name, current_ticker, matches)
                                    if ai_choice and ai_choice.get('code'):
                                        candidate_code = str(ai_choice['code']).strip()
                                        scheme = code_lookup.get(candidate_code)
                                        if scheme:
                                            final_code = candidate_code
                                            final_name = scheme['name']
                                            final_source = 'ai_amfi'
                                            ai_confidence = ai_choice.get('confidence')

                            if not final_code:
                                result = search_mftool_for_amfi_code(scheme_name)
                                if result:
                                    final_code = result['ticker']
                                    final_name = result['name']
                                    final_source = 'mftool'
                                    ai_confidence = result.get('match_confidence')

                            if not final_code:
                                continue

                            if final_code == current_ticker and final_source != 'amfi_direct_mismatch':
                                continue

                            old_stock = db.supabase.table('stock_master').select('id').eq(
                                'ticker', current_ticker
                            ).execute()

                            confidence_note = ""
                            if isinstance(ai_confidence, (int, float)):
                                confidence_note = f" ({ai_confidence:.0%})"

                            if final_code == current_ticker:
                                if old_stock.data and final_name:
                                    db.supabase.table('stock_master').update({
                                        'stock_name': final_name,
                                        'sector': 'Mutual Fund'
                                    }).eq('id', old_stock.data[0]['id']).execute()
                                    st.caption(
                                        f"      ✓ MF: {scheme_name[:40]} name harmonized [{final_source or 'amfi'}{confidence_note}]"
                                    )
                                    mf_updated += 1
                                continue

                            existing_stock = db.supabase.table('stock_master').select('id').eq(
                                'ticker', final_code
                            ).execute()

                            if existing_stock.data:
                                target_stock_id = existing_stock.data[0]['id']
                                if old_stock.data:
                                    old_stock_id = old_stock.data[0]['id']
                                    db.supabase.table('user_transactions').update({
                                        'stock_id': target_stock_id
                                    }).eq('stock_id', old_stock_id).execute()
                                    db.supabase.table('stock_master').delete().eq('id', old_stock_id).execute()
                                    st.caption(
                                        f"      ✓ MF: {scheme_name[:40]} → {final_code} [{final_source or 'amfi'}{confidence_note}] (merged)"
                                    )
                                    mf_updated += 1
                            else:
                                if old_stock.data:
                                    db.supabase.table('stock_master').update({
                                        'ticker': final_code,
                                        'stock_name': final_name or scheme_name,
                                        'sector': 'Mutual Fund'
                                    }).eq('id', old_stock.data[0]['id']).execute()
                                    st.caption(
                                        f"      ✓ MF: {scheme_name[:40]} → {final_code} [{final_source or 'amfi'}{confidence_note}]"
                                    )
                                    mf_updated += 1

                        if mf_updated > 0:
                            st.caption(f"   ✅ Updated {mf_updated} mutual fund AMFI codes")
                    
                except Exception as e:
                    st.caption(f"   ⚠️ Mutual fund resolution skipped: {str(e)[:100]}")
                
                # Recalculate holdings from transactions (CRITICAL for MFs to show up!)
                if imported > 0:
                    st.caption(f"   📊 Recalculating holdings from transactions...")
                    try:
                        holdings_count = db.recalculate_holdings(user_id, portfolio_id)
                        st.caption(f"   ✅ Calculated {holdings_count} holdings")
                    except Exception as e:
                        st.caption(f"   ⚠️ Holdings recalculation skipped: {str(e)[:100]}")
                
                # Auto-fetch prices AND 52-week historical data (only for tickers in this file)
                if imported > 0 and tickers_in_this_file:
                    st.caption(f"   📊 Fetching current prices and 52-week historical data for {len(tickers_in_this_file)} ticker(s) from this file...")
                    
                    try:
                        # Get asset types for tickers in this file
                        new_asset_types = {}
                        holdings = db.get_user_holdings(user_id)
                        if holdings:
                            # Map tickers to asset types from holdings
                            for h in holdings:
                                ticker = h.get('ticker')
                                if ticker and ticker in tickers_in_this_file:
                                    new_asset_types[ticker] = h.get('asset_type', 'stock')
                        
                        # If some tickers don't have asset types yet, default to 'stock'
                        for ticker in tickers_in_this_file:
                            if ticker not in new_asset_types:
                                new_asset_types[ticker] = 'stock'
                        
                        if tickers_in_this_file:
                            new_tickers_list = list(tickers_in_this_file)
                            st.caption(f"      📈 Fetching comprehensive data for {len(new_tickers_list)} ticker(s)...")
                            
                            # Fetch current prices + 52-week historical data (only for tickers in this file)
                            db.bulk_process_new_stocks_with_comprehensive_data(
                                tickers=new_tickers_list,
                                asset_types=new_asset_types
                            )
                            
                            st.caption(f"   ✅ Updated current prices and 52-week data for {len(new_tickers_list)} ticker(s)")
                        else:
                            st.caption(f"   ℹ️  No tickers to fetch")
                            
                    except Exception as e:
                        st.caption(f"   ⚠️ Price/historical fetching skipped: {str(e)[:50]}")
                elif imported > 0:
                    st.caption(f"   ℹ️  No tickers identified in this file (skipping price fetch)")
                
                    processing_log.append({
                'file': uploaded_file.name,
                'imported': imported,
                    'skipped': 0,
                    'errors': len(df) - imported
                    })
                    
                    total_imported += imported
                    
                else:
                # Non-CSV/Excel files require AI
                    st.caption(f"   ⚠️ Only CSV/Excel files supported in direct mode. {uploaded_file.name} requires AI extraction.")
                processing_log.append({
                    'file': uploaded_file.name,
                    'imported': 0,
                    'skipped': 0,
                    'errors': 1
                })
        
        except Exception as e:
            st.caption(f"   ❌ Error processing {uploaded_file.name}: {e}")
            processing_log.append({
                'file': uploaded_file.name,
                'imported': 0,
                'skipped': 0,
                'errors': 1
            })
    
    # Final holdings recalculation AFTER all files processed (ensures MFs + stocks are combined)
    if total_imported > 0:
        st.caption(f"📊 Final holdings recalculation (combining all assets)...")
        try:
            final_holdings_count = db.recalculate_holdings(user_id, portfolio_id)
            st.caption(f"✅ Total portfolio: {final_holdings_count} unique holdings")
        except Exception as e:
            st.caption(f"⚠️ Final recalculation skipped: {str(e)[:100]}")
    
    # Show summary
    if processing_log:
        st.success(f"✅ Processing complete! Imported {total_imported} transactions from {len(uploaded_files)} files.")
        
        # Show detailed log
        with st.expander("📋 Processing Details", expanded=False):
            for log in processing_log:
                st.caption(f"📄 {log['file']}: {log['imported']} imported, {log['skipped']} skipped, {log['errors']} errors")
    
    return total_imported


def _tx_prepare_preview_row(
    tx: Dict[str, Any],
    source_file: str,
    method: str,
    *,
    use_price_fetcher: bool = False,
    price_fetcher: Optional["EnhancedPriceFetcher"] = None,
) -> Dict[str, Any]:
    """Normalize a transaction dict for CSV preview/download (and DB import when use_price_fetcher=True)."""
    date_value = tx.get('transaction_date') or tx.get('date') or tx.get('Date') or ''
    ticker_value = tx.get('ticker') or ''
    stock_name = tx.get('stock_name') or tx.get('scheme_name') or ticker_value or ''

    normalized = {
        'source_file': source_file,
        'extraction_method': method,
        'date': _tx_safe_str(date_value),
        'ticker': _tx_normalize_ticker(ticker_value) or _tx_fallback_ticker_from_name(stock_name),
        'stock_name': _tx_safe_str(stock_name),
        'scheme_name': _tx_safe_str(tx.get('scheme_name')),
        'quantity': _tx_safe_float(tx.get('quantity') or tx.get('units')),
        'price': _tx_safe_float(tx.get('price')),
        'amount': _tx_safe_float(tx.get('amount') or tx.get('value')),
        'transaction_type': _tx_safe_str(tx.get('transaction_type') or tx.get('type') or 'buy').lower(),
        'asset_type': _tx_safe_str(tx.get('asset_type') or tx.get('category') or ''),
        'channel': _tx_safe_str(tx.get('channel') or tx.get('platform') or ''),
        'sector': _tx_safe_str(tx.get('sector')),
        'notes': _tx_safe_str(tx.get('notes') or tx.get('remarks')),
    }

    asset_type_lower = normalized['asset_type'].lower()

    if asset_type_lower == 'mutual_fund':
        resolved_code, _source = _preview_resolve_mutual_fund_ticker(stock_name, normalized['ticker'])
        if resolved_code:
            normalized['ticker'] = resolved_code
            normalized['sector'] = normalized['sector'] or 'Mutual Fund'

    # CRITICAL: For file uploads, calculate price from amount/quantity FIRST
    # This ensures we use the transaction price from the file, not current market prices
    # NOTE: This logic ONLY applies during file uploads (when use_price_fetcher=True)
    # Historical price fetching for 52 weeks/live prices works normally in other functions
    
    # Get values from both raw transaction dict and normalized dict to check if they exist
    # Check normalized first (already processed), then fall back to raw transaction dict
    quantity_val = normalized.get('quantity', 0) or _tx_safe_float(tx.get('quantity') or tx.get('units') or tx.get('__original_quantity') or 0)
    amount_val = normalized.get('amount', 0) or _tx_safe_float(tx.get('amount') or tx.get('value') or tx.get('__original_amount') or 0)
    price_val = normalized.get('price', 0)
    
    # Check if we have amount and quantity but price is missing/zero
    # Only do this during file uploads (use_price_fetcher=True means we're processing a file)
    price_was_calculated = False
    if use_price_fetcher and price_val <= 0 and quantity_val > 0 and amount_val > 0:
        # Calculate price from amount/quantity for file uploads
        calculated_price = amount_val / quantity_val
        normalized['price'] = round(calculated_price, 4)
        price_was_calculated = True
        print(f"[PREVIEW] ✅ Calculated price from amount/quantity: {calculated_price} (qty={quantity_val}, amt={amount_val})")
        print(f"[PREVIEW]   Transaction: {normalized.get('ticker')} on {normalized.get('date')} - SKIPPING historical price fetch")
    
    # CRITICAL: Only fetch historical price if we're in file upload mode AND price wasn't calculated
    # If price was calculated from amount/quantity, we MUST skip fetching to avoid yfinance errors
    if normalized['ticker'] and normalized['date'] and use_price_fetcher and not price_was_calculated:
        # Fetch historical price (for file uploads when calculation not possible)
        key = (
            normalized['ticker'],
            normalized['date'],
            asset_type_lower or 'stock',
        )
        fetched_price = _preview_price_cache.get(key)
        if fetched_price is None and normalized['price'] <= 0:
            fetcher = price_fetcher or _preview_get_price_fetcher()
            if fetcher:
                # CRITICAL: Use the transaction date from the file, NOT current date
                transaction_date = normalized['date']
                print(f"[PREVIEW] 🔍 Fetching historical price for {normalized['ticker']} on transaction date: {transaction_date} (from file)")
                try:
                    fetched_price = fetcher.get_historical_price(
                        normalized['ticker'],
                        asset_type_lower or 'stock',
                        transaction_date,  # Use date from file, not current date
                        fund_name=normalized['stock_name'],
                    )
                    if fetched_price:
                        print(f"[PREVIEW] ✅ Fetched historical price: ₹{fetched_price} for date {transaction_date} (from file)")
                    else:
                        print(f"[PREVIEW] ⚠️ No historical price found for {normalized['ticker']} on {transaction_date} (from file)")
                except Exception as e:
                    fetched_price = None
                    print(f"[PREVIEW] ⚠️ Error fetching historical price for {normalized['ticker']} on {transaction_date}: {str(e)}")
                # CRITICAL: Do NOT fall back to current price if historical price is not available
                # This would use today's date instead of the transaction date, which is incorrect
                # If historical price is not available, try these fallbacks in order:
                # 1. Original price from file
                # 2. Calculate from quantity and amount (if both available) - but we already did this above
                if not fetched_price:
                    # Fallback: Use original price from transaction if it exists
                    original_price = _tx_safe_float(tx.get('price') or tx.get('__original_price'))
                    if original_price and original_price > 0:
                        fetched_price = original_price
                        print(f"[PREVIEW] Using original price from file: {fetched_price} (historical price not available for date {normalized['date']})")
                    else:
                        print(f"[PREVIEW] ⚠️ Historical price not available for {normalized['ticker']} on {normalized['date']}, and cannot calculate from quantity/amount. Leaving as 0.")
                        fetched_price = None
            _preview_price_cache[key] = fetched_price

            if fetched_price and fetched_price > 0:
                normalized['price'] = round(float(fetched_price), 4)
                print(f"[PREVIEW] ✅ Fetched historical price: {normalized['price']} for {normalized['ticker']} on {normalized['date']}")
    
    # After price is set, calculate missing values:
    # 1. Calculate amount if missing (when we have quantity and price)
    # 2. Calculate quantity if missing (when we have amount and price)
    # But only if the values weren't originally present in the file
    quantity_was_present = tx.get('__original_quantity_present', False) or (tx.get('quantity') and _tx_safe_float(tx.get('quantity')) > 0)
    amount_was_present = tx.get('__original_amount_present', False) or (tx.get('amount') and _tx_safe_float(tx.get('amount')) > 0)
    
    if normalized['price'] > 0:
        if normalized['quantity'] > 0 and normalized['amount'] <= 0 and not amount_was_present:
            # Calculate amount from quantity and price
            calculated_amount = normalized['quantity'] * normalized['price']
            normalized['amount'] = round(calculated_amount, 4)
            print(f"[PREVIEW] Calculated amount from quantity/price: {calculated_amount} (qty={normalized['quantity']}, price={normalized['price']})")
        elif normalized['amount'] > 0 and normalized['quantity'] <= 0 and not quantity_was_present:
            # Calculate quantity from amount and price
            calculated_quantity = normalized['amount'] / normalized['price']
            normalized['quantity'] = round(calculated_quantity, 4)
            print(f"[PREVIEW] Calculated quantity from amount/price: {calculated_quantity} (amt={normalized['amount']}, price={normalized['price']})")

    # Get original values to check if they were present
    # Check metadata first (from Python extraction), then fall back to transaction dict
    quantity_present = tx.get('__original_quantity_present', False)
    amount_present = tx.get('__original_amount_present', False)
    price_present = tx.get('__original_price_present', False)
    
    # Get original values if stored, otherwise check transaction dict
    original_quantity = tx.get('__original_quantity')
    if original_quantity is None:
        original_quantity = tx.get('quantity') or tx.get('units')
        # If not explicitly marked as present, check if it's a valid value
        if not quantity_present and original_quantity and _tx_safe_float(original_quantity) > 0:
            quantity_present = True
    
    original_price = tx.get('__original_price')
    if original_price is None:
        original_price = tx.get('price')
        if not price_present and original_price and _tx_safe_float(original_price) > 0:
            price_present = True
    
    original_amount = tx.get('__original_amount')
    if original_amount is None:
        original_amount = tx.get('amount') or tx.get('value')
        if not amount_present and original_amount and _tx_safe_float(original_amount) > 0:
            amount_present = True
    
    # CRITICAL: Never recalculate quantity if it was present in the file
    if quantity_present:
        # Quantity was in the file - use it as-is, don't recalculate
        pass
    elif not quantity_present and amount_present and price_present:
        # Only calculate quantity if it's truly missing AND we have amount+price
        if normalized['amount'] > 0 and normalized['price'] > 0:
            normalized['quantity'] = round(normalized['amount'] / normalized['price'], 4)
            print(f"[TX_NORMALIZE] Calculated quantity from amount/price: {normalized['quantity']}")
    
    # Calculate price from amount/quantity if price is missing
    if not price_present and amount_present and quantity_present:
        if normalized['amount'] > 0 and normalized['quantity'] > 0:
            normalized['price'] = round(normalized['amount'] / normalized['quantity'], 4)
            print(f"[TX_NORMALIZE] Calculated price from amount/quantity: {normalized['price']}")
    
    # Calculate amount from quantity/price if amount is missing
    if not amount_present and quantity_present and price_present:
        if normalized['quantity'] > 0 and normalized['price'] > 0:
            normalized['amount'] = round(normalized['quantity'] * normalized['price'], 4)
            print(f"[TX_NORMALIZE] Calculated amount from quantity/price: {normalized['amount']}")
    
    # Validate calculated values
    if normalized['quantity'] > 0 and normalized['price'] > 0:
        computed_amount = round(normalized['quantity'] * normalized['price'], 4)
        if normalized['amount'] <= 0 or abs(normalized['amount'] - computed_amount) > 0.01 * max(1.0, computed_amount):
            normalized['amount'] = computed_amount

    return normalized


def extract_transactions_for_csv(uploaded_file, file_name: str, user_id: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract transactions from any supported file type without inserting into the database.
    Returns normalized rows ready for CSV download along with the method used.
    """
    transactions: List[Dict[str, Any]] = []
    method_used = 'python'

    # File size validation for Streamlit Cloud
    try:
        uploaded_file.seek(0, 2)  # Seek to end
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)  # Reset
        
        file_size_mb = file_size / (1024 * 1024)
        MAX_FILE_SIZE_MB = 10  # Streamlit Cloud limit
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            error_msg = f"File too large: {file_size_mb:.2f} MB. Maximum allowed: {MAX_FILE_SIZE_MB} MB for Streamlit Cloud."
            print(f"[FILE_PARSE] ❌ {error_msg}")
            try:
                st.error(f"❌ **File Size Error**: {error_msg}\n\nPlease use a smaller file or split it into multiple files.")
            except:
                pass
            return [], 'error'
    except Exception:
        pass  # Skip validation if can't check size

    # For Excel files, use AI extraction first (better column mapping)
    # For CSV/PDF, try Python first then AI
    file_ext = Path(file_name).suffix.lower()
    is_excel = file_ext in {'.xlsx', '.xls'}
    
    if is_excel and AI_AGENTS_AVAILABLE:
        # Excel files: Use AI extraction first for better column mapping
        method_used = 'ai'
        print(f"[FILE_PARSE] Excel file detected - using AI extraction first for better column mapping: {file_name}")
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        transactions = process_file_with_ai(uploaded_file, file_name, user_id or '') or []
        if transactions:
            print(f"[FILE_PARSE] ✅ AI extraction found {len(transactions)} transactions from Excel file")
        else:
            print(f"[FILE_PARSE] ⚠️ AI extraction found no transactions, trying Python extraction as fallback...")
            # Fallback to Python if AI fails
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            extraction_result = extract_transactions_python(uploaded_file, file_name)
            if isinstance(extraction_result, tuple) and len(extraction_result) == 2:
                python_transactions, _python_log = extraction_result
            else:
                python_transactions = extraction_result or []
            if python_transactions:
                method_used = 'python'
                transactions = python_transactions
                print(f"[FILE_PARSE] ✅ Python extraction found {len(transactions)} transactions as fallback")
    else:
        # CSV/PDF: Try Python first, then AI
        method_used = 'python'
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        extraction_result = extract_transactions_python(uploaded_file, file_name)
        if isinstance(extraction_result, tuple) and len(extraction_result) == 2:
            python_transactions, _python_log = extraction_result
        else:
            python_transactions = extraction_result or []
        transactions = python_transactions or []

        # Fall back to AI only when Python parser finds nothing
        if not transactions:
            if AI_AGENTS_AVAILABLE:
                method_used = 'ai'
                print(f"[FILE_PARSE] Python extraction found no transactions, trying AI extraction for {file_name}...")
                # Check if Vision API text was pre-extracted
                if hasattr(uploaded_file, '_vision_api_text'):
                    print(f"[FILE_PARSE] ✅ Pre-extracted Vision API text found ({len(uploaded_file._vision_api_text)} characters) - will reuse it")
                else:
                    print(f"[FILE_PARSE] ⚠️ No pre-extracted Vision API text found")
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                transactions = process_file_with_ai(uploaded_file, file_name, user_id or '') or []
                if transactions:
                    print(f"[FILE_PARSE] ✅ AI extraction found {len(transactions)} transactions")
                else:
                    print(f"[FILE_PARSE] ⚠️ AI extraction also found no transactions")
                    # Check if this is a PDF that might be image-based
                    if file_name.lower().endswith('.pdf'):
                        st.warning("""
                        **⚠️ PDF Processing Issue**
                        
                        The PDF file appears to be **image-based (scanned)** and contains no extractable text.
                        
                        **Solutions:**
                        1. **Use OCR software** (e.g., Adobe Acrobat, online OCR tools) to convert the PDF to text-selectable format
                        2. **Provide the original source file** (CSV, Excel, or text-selectable PDF) instead
                        3. **Check terminal logs** for detailed extraction diagnostics
                        
                        The system detected table structures but all cells were empty, and text extraction returned no content.
                        """)
            else:
                print(f"[FILE_PARSE] ⚠️ Python extraction found no transactions and AI agents are not available")
                print(f"[FILE_PARSE]   Check terminal logs above for details on why extraction failed")

    normalized_rows = [
        _tx_prepare_preview_row(tx, file_name, method_used, use_price_fetcher=True)
        for tx in transactions
        if isinstance(tx, dict)
    ]

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    return normalized_rows, method_used


@st.cache_data(ttl=600)
def _preview_get_amfi_dataset() -> Dict[str, Any]:
    """Cached AMFI dataset for preview conversions."""
    return get_amfi_dataset()


@st.cache_resource(show_spinner=False)
def _preview_get_price_fetcher() -> Optional["EnhancedPriceFetcher"]:
    """Create a shared EnhancedPriceFetcher instance for preview conversions."""
    try:
        return EnhancedPriceFetcher()
    except Exception:
        return None


def _preview_resolve_mutual_fund_ticker(scheme_name: str, current_ticker: str) -> Tuple[str, Optional[str]]:
    """
    Resolve mutual fund ticker to AMFI code for CSV preview without mutating the database.
    Returns (resolved_code, source) where resolved_code may equal current_ticker if no better match is found.
    """
    dataset = _preview_get_amfi_dataset()
    if not dataset.get("schemes"):
        return current_ticker, None

    cleaned_current = str(current_ticker or "").strip()
    if cleaned_current.isdigit() and len(cleaned_current) >= 5:
        return cleaned_current, "current"

    resolution = resolve_mutual_fund_with_amfi(scheme_name, cleaned_current, dataset)
    status = resolution.get("status")
    direct_scheme = resolution.get("direct_scheme")
    matches = resolution.get("matches", [])
    code_lookup = dataset.get("code_lookup", {})

    if status == "direct" and direct_scheme:
        return direct_scheme.get("code") or cleaned_current, "amfi_direct"

    if status == "direct_mismatch" and direct_scheme:
        return direct_scheme.get("code") or cleaned_current, "amfi_direct_mismatch"

    if matches:
        top_match = matches[0]
        code = top_match.get("code")
        if code:
            return code, "amfi_name_match"

    result = search_mftool_for_amfi_code(scheme_name)
    if result:
        return result.get("ticker") or cleaned_current, "mftool"

    # As a last resort, try AI selection if available (without committing changes)
    if matches:
        ai_choice = ai_select_amfi_code(scheme_name, cleaned_current, matches)
        if ai_choice:
            code = str(ai_choice.get("code") or "").strip()
            if code and code_lookup.get(code):
                return code, "ai_amfi"

    return cleaned_current or _tx_fallback_ticker_from_name(scheme_name), None


def process_uploaded_files(uploaded_files, user_id, portfolio_id):
    """
    Process uploaded files and store transactions in the database.
    Currently uses AI-first extraction for better column mapping.
    """
    if not uploaded_files:
        return 0

    db = st.session_state.db

    if 'upload_price_fetcher' not in st.session_state:
        try:
            st.session_state.upload_price_fetcher = EnhancedPriceFetcher()
        except Exception:
            st.session_state.upload_price_fetcher = None
    price_fetcher = st.session_state.upload_price_fetcher
    price_cache: Dict[Tuple[str, str, str], Optional[float]] = {}

    total_imported = 0
    processing_log: List[Dict[str, Any]] = []
    collected_tickers: Dict[str, str] = {}

    st.info(f"📊 Processing {len(uploaded_files)} file(s)...")

    for file_idx, uploaded_file in enumerate(uploaded_files, 1):
        file_name = uploaded_file.name
        st.caption(f"📁 [{file_idx}/{len(uploaded_files)}] Processing {file_name}...")

        imported = skipped = errors = 0
        method_used = 'python'

        try:
            transactions = []
            python_debug_log: List[Dict[str, Any]] = []
            
            # For Excel files, use AI extraction first (better column mapping)
            # For CSV/PDF, try Python first then AI
            file_ext = Path(file_name).suffix.lower()
            is_excel = file_ext in {'.xlsx', '.xls'}
            
            if is_excel and AI_AGENTS_AVAILABLE:
                # Excel files: Use AI extraction first for better column mapping
                method_used = 'ai'
                print(f"[FILE_PARSE] Excel file detected - using AI extraction first for better column mapping: {file_name}")
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                ai_transactions = process_file_with_ai(uploaded_file, file_name, user_id) or []
                if ai_transactions:
                    transactions = ai_transactions
                    print(f"[FILE_PARSE] ✅ AI extraction found {len(transactions)} transactions from Excel file")
                else:
                    print(f"[FILE_PARSE] ⚠️ AI extraction found no transactions, trying Python extraction as fallback...")
                    # Fallback to Python if AI fails
                    try:
                        uploaded_file.seek(0)
                    except Exception:
                        pass
                    extraction_result = extract_transactions_python(uploaded_file, file_name)
                    if isinstance(extraction_result, tuple) and len(extraction_result) == 2:
                        python_transactions, python_debug_log = extraction_result
                    else:
                        python_transactions = extraction_result or []
                        python_debug_log = []
                    if python_transactions:
                        method_used = 'python'
                        transactions = python_transactions
                        print(f"[FILE_PARSE] ✅ Python extraction found {len(transactions)} transactions as fallback")
            else:
                # CSV/PDF: Try Python first, then AI
                method_used = 'python'
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                
                extraction_result = extract_transactions_python(uploaded_file, file_name)
                if isinstance(extraction_result, tuple) and len(extraction_result) == 2:
                    python_transactions, python_debug_log = extraction_result
                else:
                    python_transactions = extraction_result or []
                    python_debug_log = []
                transactions = python_transactions or []
                
                # Fallback to AI only if Python found nothing (or for PDFs)
                if not transactions:
                    method_used = 'ai'
                    try:
                        uploaded_file.seek(0)
                    except Exception:
                        pass
                    
                    ai_transactions = process_file_with_ai(uploaded_file, file_name, user_id) or []
                    transactions = ai_transactions

            if not transactions:
                st.error(f"   ❌ Could not extract transactions from {file_name}.")
                processing_log.append({
                    'file': file_name,
                    'imported': 0,
                    'skipped': 0,
                    'errors': 1,
                    'method': method_used,
                    'debug': python_debug_log,
                })
                continue

            fallback_channel = (Path(file_name).stem or 'Direct').title()

            for tx in transactions:
                normalized_tx = _tx_prepare_preview_row(
                    tx,
                    file_name,
                    method_used,
                    use_price_fetcher=True,
                    price_fetcher=price_fetcher,
                )
                payload = {
                    'user_id': user_id,
                    'portfolio_id': portfolio_id,
                    'ticker': normalized_tx['ticker'],
                    'stock_name': normalized_tx['stock_name'],
                    'scheme_name': normalized_tx.get('scheme_name'),
                    'quantity': normalized_tx['quantity'],
                    'price': normalized_tx['price'],
                    'transaction_date': normalized_tx['date'],
                    'transaction_type': normalized_tx['transaction_type'],
                    'asset_type': normalized_tx['asset_type'] or 'stock',
                    'channel': normalized_tx['channel'] or fallback_channel,
                    'sector': normalized_tx['sector'] or ('Mutual Fund' if (normalized_tx['asset_type'] or '').lower() == 'mutual_fund' else 'Unknown'),
                    'amount': normalized_tx['amount'],
                    'filename': file_name,
                    'notes': normalized_tx.get('notes'),
                }
                try:
                    result = db.add_transaction(payload)
                except Exception as exc:
                    st.caption(f"   ⚠️ Database error: {str(exc)[:100]}")
                    errors += 1
                    continue

                if result.get('success'):
                    imported += 1
                    ticker_key = payload.get('ticker')
                    asset_type_value = payload.get('asset_type', 'stock')
                    collected_tickers[ticker_key] = asset_type_value or 'stock'
                else:
                    error_msg = (result.get('error') or '').lower()
                    if 'duplicate' in error_msg:
                        skipped += 1
                    else:
                        errors += 1

            if imported > 0:
                st.success(f"   ✅ Imported {imported} transaction(s) from {file_name} (AI mapper)")
                if skipped:
                    st.info(f"   ⏭️  Skipped {skipped} duplicate transaction(s)")
                if errors:
                    st.warning(f"   ⚠️ Encountered {errors} row error(s)")
            else:
                if skipped and not errors:
                    st.info(f"   ⏭️  All {skipped} rows were duplicates.")
                else:
                    st.error(f"   ❌ Failed to import transactions from {file_name}.")

            processing_log.append({
                'file': file_name,
                'imported': imported,
                'skipped': skipped,
                'errors': errors,
                'method': method_used,
                'debug': python_debug_log,
            })
            total_imported += imported

        except Exception as exc:
            st.error(f"   ❌ Error processing {file_name}: {str(exc)[:120]}")
            processing_log.append({
                'file': file_name,
                'imported': 0,
                'skipped': 0,
                'errors': 1,
                'method': method_used,
            })

    if total_imported > 0:
        st.caption("📊 Recalculating holdings...")
        try:
            holdings_count = db.recalculate_holdings(user_id, portfolio_id)
            st.caption(f"   ✅ Updated {holdings_count} holding record(s)")
        except Exception as exc:
            st.caption(f"   ⚠️ Holdings recalculation skipped: {str(exc)[:80]}")

        if collected_tickers:
            st.caption(f"📈 Refreshing market data for {len(collected_tickers)} ticker(s)...")
            try:
                db.bulk_process_new_stocks_with_comprehensive_data(
                    tickers=list(collected_tickers.keys()),
                    asset_types=collected_tickers,
                )
                st.caption("   ✅ Latest prices and weekly history refreshed.")
            except Exception as exc:
                st.caption(f"   ⚠️ Market data refresh skipped: {str(exc)[:80]}")

    if processing_log:
        st.success(f"✅ Processing complete! Imported {total_imported} transactions from {len(uploaded_files)} file(s).")
        with st.expander("📋 Processing Details", expanded=False):
            for log in processing_log:
                method_label = "Python" if log.get('method') == 'python' else "AI"
                st.caption(
                    f"📄 {log['file']} • {method_label} • "
                    f"{log['imported']} imported, {log['skipped']} skipped, {log['errors']} errors"
                )
                debug_entries = log.get('debug') or []
                if debug_entries:
                    for entry in debug_entries:
                        sheet_name = entry.get('sheet', 'Unknown sheet')
                        columns = ", ".join(entry.get('columns', []))
                        st.caption(f"      -> {sheet_name}: {columns}")

    return total_imported

def main_dashboard():
    """Main dashboard after login"""
    user = st.session_state.user
    
    st.title(f"💰 Welcome, {user['full_name']}")

    if st.session_state.get('needs_initial_refresh', False) and not st.session_state.get('_auto_refresh_running', False):
        st.session_state['_auto_refresh_running'] = True
        try:
            run_portfolio_refresh(user['id'], auto=True)
        finally:
            st.session_state['_auto_refresh_running'] = False
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    
    if st.session_state.needs_initial_refresh:
        st.info("🔄 Portfolio data has not been refreshed this session. Use **Refresh All Prices** in the sidebar (or the button below) to load live prices and weekly history.")
        if st.button("🚀 Refresh portfolio data now"):
            run_portfolio_refresh(user['id'])
    elif st.session_state.last_fetch_time:
        time_since_fetch = (datetime.now() - st.session_state.last_fetch_time).total_seconds() / 60
        st.sidebar.caption(f"✅ Prices checked {int(time_since_fetch)} min ago")
    else:
        st.sidebar.caption("📊 No refresh history for this session.")

    if st.sidebar.button("🔄 Refresh All Prices"):
        run_portfolio_refresh(user['id'])
    
    # Navigation
    navigation_options = [
        "🏠 Portfolio Overview",
        "📊 P&L Analysis",
        "📡 Channel Analytics",
        "🏭 Sector Analytics",
        "📈 Charts & Analytics",
        "🤖 AI Assistant",
        "📁 Upload More Files"
    ]
    
    # Add AI Insights page if agents are available
    if AI_AGENTS_AVAILABLE:
        navigation_options.insert(-1, "🧠 AI Insights")
    
    # Add User Profile page
    navigation_options.insert(-1, "👤 Profile Settings")
    
    page = st.sidebar.radio(
        "Choose a page:",
        navigation_options
    )
    
    st.sidebar.markdown("---")
    
    # Simplified sidebar - AI Assistant now has its own page
    with st.sidebar:
        st.markdown("**💡 Quick Access**")
        st.caption("🤖 AI Assistant now has its own dedicated page!")
        st.caption("📊 Access all portfolio insights and chat features")
        st.caption("📚 Upload and analyze PDF documents")
        st.caption("💬 Chat with AI about your portfolio")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**👤 {user['full_name']}**")
    st.sidebar.markdown(f"📧 {user['email']}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.missing_weeks_fetched = False
        st.rerun()
    
    # Route to pages
    if page == "🏠 Portfolio Overview":
        portfolio_overview_page()
    elif page == "📊 P&L Analysis":
        pnl_analysis_page()
    elif page == "📡 Channel Analytics":
        channel_analytics_page()
    elif page == "🏭 Sector Analytics":
        sector_analytics_page()
    elif page == "📈 Charts & Analytics":
        charts_page()
    elif page == "🤖 AI Assistant":
        ai_assistant_page()
    elif page == "🧠 AI Insights" and AI_AGENTS_AVAILABLE:
        ai_insights_page()
    elif page == "👤 Profile Settings":
        user_profile_page()
    elif page == "📁 Upload More Files":
        upload_files_page()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_portfolio_metrics(holdings: List[Dict]) -> Dict[str, Any]:
    """Cache expensive portfolio calculations"""
    if not holdings:
        return {}
    
    # Calculate all metrics in one pass
    total_investment = 0
    total_current = 0
    gainers = 0
    momentum_stocks = 0
    returns = []
    
    for holding in holdings:
        # CRITICAL: Skip holdings with zero or negative quantity (fully sold positions)
        # Use a small epsilon to handle floating point precision issues (e.g., 0.0001)
        quantity = float(holding.get('total_quantity', 0))
        if quantity <= 0.0001:  # Treat anything <= 0.0001 as effectively zero
            continue  # Skip fully sold positions - they shouldn't count in invested amount or P&L
        
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        
        current_value = float(current_price) * quantity
        investment = quantity * float(holding['average_price'])
        
        total_investment += investment
        total_current += current_value
        
        pnl = current_value - investment
        pnl_pct = (pnl / investment * 100) if investment > 0 else 0
        returns.append(pnl_pct)
        
        if pnl > 0:
            gainers += 1
        if pnl_pct > 10:
            momentum_stocks += 1
    
    total_pnl = total_current - total_investment
    total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
    market_breadth = (gainers / len(holdings) * 100) if holdings else 0
    momentum_score = (momentum_stocks / len(holdings) * 100) if holdings else 0
    volatility = np.std(returns) if returns else 0
    
    return {
        'total_investment': total_investment,
        'total_current': total_current,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'market_breadth': market_breadth,
        'momentum_score': momentum_score,
        'volatility': volatility,
        'gainers': gainers,
        'momentum_stocks': momentum_stocks,
        'total_holdings': len(holdings)
    }

def portfolio_overview_page():
    """Portfolio overview with current week prices"""
    st.header("🏠 Portfolio Overview")
    
    user = st.session_state.user
    db = st.session_state.db  # Get database manager from session state
    
    # Show Corporate Actions Alert (Stock Splits/Bonus)
    if 'corporate_actions_detected' in st.session_state and st.session_state.corporate_actions_detected:
        corporate_actions = st.session_state.corporate_actions_detected
        
        st.warning(f"📊 **{len(corporate_actions)} Stock Splits/Bonus Shares Detected!**")
        
        with st.expander(f"🔧 View and Fix Corporate Actions ({len(corporate_actions)} stocks)", expanded=True):
            st.markdown("""
            **Corporate actions detected!** Your portfolio has stocks that underwent splits or bonus issues.  
            Click the "Fix" button to automatically adjust quantities and prices.
            """)
            
            # Create a table of detected corporate actions
            for idx, action in enumerate(corporate_actions):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    st.write(f"**{action['stock_name']}** (`{action['ticker']}`)")
                
                with col2:
                    st.caption(f"Your Avg: ₹{action['avg_price']:,.2f}")
                    st.caption(f"Current: ₹{action['current_price']:,.2f}")
                
                with col3:
                    st.info(f"**1:{action['split_ratio']} Split**")
                    st.caption(f"({action['ratio']:.1f}x difference)")
                
                with col4:
                    if st.button(f"✅ Fix", key=f"fix_split_{idx}_{action['ticker']}"):
                        with st.spinner(f"Adjusting {action['ticker']}..."):
                            adjusted = adjust_for_corporate_action(
                                user['id'], 
                                action['stock_id'], 
                                action['split_ratio'],
                                db
                            )
                            
                            if adjusted > 0:
                                st.success(f"✅ Adjusted {adjusted} transactions for {action['ticker']}")
                                # Clear from session state
                                st.session_state.corporate_actions_detected = [
                                    a for a in corporate_actions if a['ticker'] != action['ticker']
                                ]
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to adjust {action['ticker']}")
            
            # Add "Fix All" button
            st.markdown("---")
            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                if st.button("✅ Fix All Splits", type="primary", use_container_width=True):
                    with st.spinner("Adjusting all stocks..."):
                        total_adjusted = 0
                        for action in corporate_actions:
                            adjusted = adjust_for_corporate_action(
                                user['id'],
                                action['stock_id'],
                                action['split_ratio'],
                                db
                            )
                            total_adjusted += adjusted
                        
                        if total_adjusted > 0:
                            st.success(f"✅ Adjusted {total_adjusted} total transactions across {len(corporate_actions)} stocks!")
                            st.session_state.corporate_actions_detected = None
                            time.sleep(2)
                            st.rerun()
            
            with col_b:
                if st.button("❌ Dismiss", use_container_width=True):
                    st.session_state.corporate_actions_detected = None
                    st.rerun()
        
        st.markdown("---")
    
    # Add AI-powered proactive alerts if available
    if AI_AGENTS_AVAILABLE:
        try:
            alerts = get_ai_alerts()
            if alerts:
                st.markdown("### 🚨 AI Alerts")
                for alert in alerts[:3]:  # Show top 3 alerts
                    severity_emoji = {
                        "high": "🔴",
                        "medium": "🟡",
                        "low": "🟢"
                    }.get(alert.get("severity", "low"), "🟢")
                    
                    with st.expander(f"{severity_emoji} {alert.get('title', 'Alert')}", expanded=(alert.get("severity") == "high")):
                        st.markdown(f"**{alert.get('description', 'No description')}**")
                        st.markdown(f"*Recommendation: {alert.get('recommendation', 'No recommendation')}*")
                st.markdown("---")
        except Exception as e:
            # Silently handle errors to not disrupt the main page
            pass
    
    # Add smart manual update prices button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        # Check how many holdings need updating
        holdings = get_cached_holdings(user['id'])
        
        # Find ALL holdings with stale prices (current_price == avg_price, 0% return, or missing)
        stale_prices = []
        bonds_to_update = []
        for h in holdings:
            # Check both current_price and live_price fields
            cp = h.get('current_price') or h.get('live_price') or 0
            ap = h.get('average_price', 0)
            asset_type = h.get('asset_type', '')
            
            # If current_price is None, 0, or exactly equals avg_price → likely stale (for ALL asset types)
            is_stale = False
            
            if cp is None or cp == 0:
                # Missing price - definitely stale
                is_stale = True
            elif ap > 0 and abs(cp - ap) < 0.01:
                # Current price equals average (within 1 paisa) = stale (price wasn't fetched)
                is_stale = True
            
            # Special handling for bonds - check if price seems wrong
            if asset_type == 'bond':
                ticker = (h.get('ticker') or '').lower()
                name = (h.get('stock_name') or '').lower()
                is_sgb = any(k in ticker or k in name for k in ['sgb', 'gold bond', 'goldbond', 'sovereign', 'sr-'])
                
                # SGBs should be around ₹12,000-15,000, not ₹6,000
                if is_sgb and cp < 10000:
                    bonds_to_update.append(h)
                    is_stale = True
                elif cp is None or cp == 0 or (ap > 0 and cp < ap * 0.5):  # Price is < 50% of avg = likely wrong
                    bonds_to_update.append(h)
                    is_stale = True
                elif ap > 0 and cp < ap * 0.7:  # Current price < 70% of average = likely wrong
                    bonds_to_update.append(h)
                    is_stale = True
            
            # Add to stale_prices if stale (for ALL asset types)
            if is_stale:
                stale_prices.append(h)
    
        # Show "Refresh All Prices" button if there are stale prices
        if stale_prices:
            if st.button(f"🔄 Refresh {len(stale_prices)} Price(s)", help=f"Refresh prices for holdings with missing or stale prices"):
                with st.spinner(f"Refreshing prices for {len(stale_prices)} holdings..."):
                    from enhanced_price_fetcher import EnhancedPriceFetcher
                    price_fetcher = EnhancedPriceFetcher()
                    price_fetcher.update_live_prices_for_holdings(stale_prices, db)
                    st.success(f"✅ Refreshed prices for {len(stale_prices)} holdings!")
                    st.rerun()
        
        # Show bond update button if bonds need updating
        if bonds_to_update:
            if st.button(f"💰 Update {len(bonds_to_update)} Bond Price(s)", help=f"Update bond prices using AI (SGBs need market prices)"):
                with st.spinner(f"Fetching bond prices for {len(bonds_to_update)} bonds..."):
                    from enhanced_price_fetcher import EnhancedPriceFetcher
                    price_fetcher = EnhancedPriceFetcher()
                    
                    updated = 0
                    for bond in bonds_to_update:
                        ticker = bond.get('ticker')
                        stock_name = bond.get('stock_name')
                        
                        print(f"[BOND_UPDATE] Manual update: {stock_name} ({ticker})")
                        price, source = price_fetcher._get_bond_price(ticker, stock_name)
                        
                        if price and price > 0:
                            db._store_current_price(ticker, price, 'bond')
                            print(f"[BOND_UPDATE] ✅ Updated {ticker}: ₹{price:.2f} (from {source})")
                            updated += 1
                        else:
                            print(f"[BOND_UPDATE] ❌ Failed to get price for {ticker}")
                    
                    if updated > 0:
                        st.success(f"✅ Updated {updated}/{len(bonds_to_update)} bond price(s)!")
                        st.rerun()
                    else:
                        st.warning("⚠️ No bond prices updated (AI may be unavailable)")
                        # Don't rerun if update failed - prevent infinite loop
    
    # Show general "Refresh All Prices" button
    with col3:
            if st.button("🔄 Refresh All Prices", help="Refresh current prices for all holdings"):
                with st.spinner("Refreshing all prices (this may take a minute)..."):
                    holdings = get_cached_holdings(user['id'])
                    if holdings:
                        from enhanced_price_fetcher import EnhancedPriceFetcher
                        price_fetcher = EnhancedPriceFetcher()
                        price_fetcher.update_live_prices_for_holdings(holdings, db)
                        st.success(f"✅ Refreshed prices for {len(holdings)} holdings!")
                        st.rerun()
                    else:
                        st.warning("No holdings found to update.")
    
    # Use cached holdings data (with zero-quantity filtering)
    holdings = get_cached_holdings(user['id'])
    
    # AUTOMATIC: Resolve tickers and fetch prices for all asset types (PMS, stock, MF, AIF, bond)
    if holdings:
        try:
            from enhanced_price_fetcher import EnhancedPriceFetcher
            price_fetcher = EnhancedPriceFetcher()
            
            # Automatically resolve mutual fund names from AMFI
            amfi_data = get_amfi_dataset()
            mf_updates = []
            for h in holdings:
                asset_type = h.get('asset_type', '')
                ticker = str(h.get('ticker', '')).strip()
                stock_name = str(h.get('stock_name', '')).strip()
                stock_id = h.get('stock_id')
                
                # Resolve mutual fund names from AMFI
                if asset_type == 'mutual_fund' and ticker.isdigit() and stock_name == ticker:
                    if amfi_data and 'code_lookup' in amfi_data:
                        scheme = amfi_data['code_lookup'].get(ticker)
                        if scheme and scheme.get('name'):
                            try:
                                db.supabase.table('stock_master').update({
                                    'stock_name': scheme.get('name')
                                }).eq('id', stock_id).execute()
                                mf_updates.append(ticker)
                            except Exception:
                                pass
            
            # Automatically update prices for all holdings (background, non-blocking)
            # This will resolve tickers and fetch prices for all asset types
            try:
                # Update prices in background (non-blocking)
                price_fetcher.update_live_prices_for_holdings(holdings, db)
                if mf_updates:
                    print(f"[AUTO_RESOLVE] ✅ Resolved {len(mf_updates)} mutual fund name(s) from AMFI")
                print(f"[AUTO_RESOLVE] ✅ Automatically updated prices for {len(holdings)} holdings")
            except Exception as e:
                print(f"[AUTO_RESOLVE] ⚠️ Auto price update failed: {e}")
            
            # Clear cache to refresh holdings with updated names/prices
            if mf_updates:
                get_cached_holdings.clear()
                # Reload holdings to get updated names
                holdings = get_cached_holdings(user['id'])
        except Exception as e:
            print(f"[AUTO_RESOLVE] ⚠️ Auto resolution failed: {e}")
            pass  # Continue even if auto-resolution fails
    
    # CRITICAL: Double-check filtering at display level (safety net)
    # Filter out any zero-quantity holdings that might have slipped through cache
    filtered_holdings = []
    zero_qty_found = []
    for h in holdings:
        quantity_raw = h.get('total_quantity', 0)
        
        # Convert to float, handling various formats
        if quantity_raw is None:
            quantity = 0.0
        elif isinstance(quantity_raw, str):
            # Handle string "0", "0.0", "", etc.
            quantity_raw = quantity_raw.strip()
            if not quantity_raw or quantity_raw in ['0', '0.0', '0.00', '']:
                quantity = 0.0
            else:
                try:
                    quantity = float(quantity_raw)
                except (ValueError, TypeError):
                    quantity = 0.0
        else:
            try:
                quantity = float(quantity_raw)
            except (ValueError, TypeError):
                quantity = 0.0
        
        # CRITICAL: Filter out anything <= 0.0001 (fully sold position)
        if quantity > 0.0001:  # Only include holdings with positive quantity
            filtered_holdings.append(h)
        else:
            zero_qty_found.append(f"{h.get('ticker')} ({h.get('stock_id')}) - {h.get('stock_name')} (qty={quantity}, raw={quantity_raw}, type={type(quantity_raw).__name__})")
            print(f"[HOLDINGS_PAGE] ⚠️ Filtering out zero-quantity holding at display level: {h.get('ticker')} - {h.get('stock_name')} (quantity={quantity}, raw={quantity_raw}, type={type(quantity_raw).__name__})")
    holdings = filtered_holdings
    
    # Show warning and clear cache button if zero-quantity holdings were found
    if zero_qty_found:
        st.error(f"⚠️ **CRITICAL: Found {len(zero_qty_found)} zero-quantity holdings that should not appear!**")
        st.warning("These holdings have been filtered out from display, but you need to recalculate holdings to update the database.")
        
        # Show details
        with st.expander(f"View {len(zero_qty_found)} filtered zero-quantity holdings"):
            for zq in zero_qty_found:
                st.text(f"  • {zq}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Recalculate & Clear Cache", help="Recalculate holdings from transactions and clear cache", type="primary"):
                from database_shared import SharedDatabaseManager
                db = SharedDatabaseManager()
                with st.spinner("Recalculating holdings from transactions..."):
                    try:
                        count = db.recalculate_holdings(user['id'])
                        st.success(f"✅ Recalculated {count} holdings!")
                    except Exception as e:
                        st.error(f"❌ Error recalculating: {e}")
                        st.exception(e)
                # Clear all caches
                get_cached_holdings.clear()
                get_portfolio_metrics.clear()
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Cache Only", help="Just clear cached holdings data"):
                get_cached_holdings.clear()
                get_portfolio_metrics.clear()
                st.success("✅ Cache cleared!")
                st.rerun()
    
    # CRITICAL: Log holdings count for debugging
    if zero_qty_found:
        st.warning(f"⚠️ **Found {len(zero_qty_found)} zero-quantity holdings that were filtered out:**")
        for zq in zero_qty_found[:5]:  # Show first 5
            st.text(f"  • {zq}")
        if len(zero_qty_found) > 5:
            st.text(f"  ... and {len(zero_qty_found) - 5} more")
    
    # Get cached metrics (only for filtered holdings)
    metrics = get_portfolio_metrics(holdings)
    
    if not holdings:
        if zero_qty_found:
            st.info("⚠️ All holdings have zero quantity (fully sold). Recalculate holdings to update the database.")
        else:
            st.info("No holdings found. Upload transaction files to see your portfolio.")
        return
    
    st.success(f"📊 Loaded {len(holdings)} holdings")
    
    # Calculate portfolio metrics (clean, no logs)
    total_investment = 0
    total_current = 0
    total_pnl = 0
    
    for holding in holdings:
        # CRITICAL: Skip holdings with zero or negative quantity (fully sold positions)
        # Use a small epsilon to handle floating point precision issues (e.g., 0.0001)
        quantity = float(holding.get('total_quantity', 0))
        if quantity <= 0.0001:  # Treat anything <= 0.0001 as effectively zero
            continue  # Skip fully sold positions - they shouldn't count in invested amount or P&L
        
        # Calculate investment value
        investment_value = quantity * float(holding['average_price'])
        total_investment += investment_value
        
        # Get current price - handle None values (check both current_price and live_price)
        current_price = holding.get('current_price') or holding.get('live_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        
        if current_price and current_price != holding['average_price']:
            # Price was fetched successfully
            current_value = quantity * float(current_price)
            total_current += current_value
            
            pnl = current_value - investment_value
            total_pnl += pnl
        else:
            # Using average price as fallback
            current_value = investment_value
            total_current += current_value
            total_pnl += 0  # No P&L if using average price
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Investment", f"₹{total_investment:,.0f}")
    
    with col2:
        st.metric("Current Value", f"₹{total_current:,.0f}")
    
    with col3:
        st.metric("Total P&L", f"₹{total_pnl:,.0f}")
    
    with col4:
        pnl_percent = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        st.metric("P&L %", f"{pnl_percent:+.1f}%")
    
    # Holdings table
    st.subheader("📊 Your Holdings")
    
    holdings_data = []
    for holding in holdings:
        # CRITICAL: Skip holdings with zero or negative quantity (fully sold positions)
        # Handle both string and numeric quantities, and account for floating point precision
        quantity_raw = holding.get('total_quantity', 0)
        
        # Convert to float, handling various formats
        if quantity_raw is None:
            quantity = 0.0
        elif isinstance(quantity_raw, str):
            # Handle string "0", "0.0", "", etc.
            quantity_raw = quantity_raw.strip()
            if not quantity_raw or quantity_raw in ['0', '0.0', '0.00', '']:
                quantity = 0.0
            else:
                try:
                    quantity = float(quantity_raw)
                except (ValueError, TypeError):
                    quantity = 0.0
        else:
            try:
                quantity = float(quantity_raw)
            except (ValueError, TypeError):
                quantity = 0.0
        
        # Use a small epsilon to handle floating point precision issues (e.g., 0.0001)
        # Also handle cases where quantity might be exactly 0, negative, or very small
        # CRITICAL: Filter out anything <= 0.0001 (fully sold position)
        if quantity <= 0.0001:  # Treat anything <= 0.0001 as effectively zero
            # Debug: Log skipped holdings to help diagnose issues
            print(f"[HOLDINGS_TABLE] ⚠️ Skipping zero-quantity holding: {holding.get('ticker')} - {holding.get('stock_name')} (quantity={quantity}, raw={quantity_raw}, type={type(quantity_raw)})")
            continue  # Skip fully sold positions - they shouldn't appear in holdings table
        
        # Handle None current_price - check both current_price and live_price fields
        asset_type = holding.get('asset_type', 'stock')
        current_price = holding.get('current_price') or holding.get('live_price')
        
        # For PMS/AIF, if price is missing or equals average, try to calculate using CAGR
        if asset_type in ['pms', 'aif'] and (current_price is None or current_price == 0 or abs(current_price - holding.get('average_price', 0)) < 0.01):
            try:
                from enhanced_price_fetcher import EnhancedPriceFetcher
                price_fetcher = EnhancedPriceFetcher()
                ticker = holding.get('ticker')
                if ticker:
                    # Use the holding-specific calculation method which has access to transaction data
                    calculated_price, source = price_fetcher._calculate_pms_aif_live_price(
                        ticker, asset_type, db, holding
                    )
                    if calculated_price and calculated_price > 0:
                        current_price = calculated_price
                        # Update the holding's live_price in database
                        try:
                            db.supabase.table('stock_master').update({
                                'live_price': calculated_price,
                                'last_updated': datetime.now().isoformat()
                            }).eq('id', holding.get('stock_id')).execute()
                            print(f"[PMS_CALC] ✅ Updated {ticker} ({asset_type}): ₹{calculated_price:,.2f} (source: {source})")
                        except Exception as e:
                            print(f"[PMS_CALC] ⚠️ Failed to update DB for {ticker}: {e}")
            except Exception as e:
                print(f"[PMS_CALC] ⚠️ Failed to calculate PMS price for {holding.get('ticker')}: {e}")
                pass  # Fall back to average_price if calculation fails
        
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        current_value = quantity * float(current_price) if current_price else 0
        investment_value = quantity * float(holding['average_price'])
        pnl = current_value - investment_value
        pnl_percent = (pnl / investment_value * 100) if investment_value > 0 else 0
        
        # Get rating for this stock
        stars, grade, rating = get_performance_rating(pnl_percent)
        
        holdings_data.append({
            'Ticker': holding['ticker'],
            'Name': holding['stock_name'],
            'Rating': stars,
            'Grade': grade,
            'Type': holding['asset_type'],
            'Quantity': f"{quantity:,.0f}",  # Use the filtered quantity, not raw holding['total_quantity']
            'Invested': f"₹{investment_value:,.0f}",  # CRITICAL: Investment amount = quantity * average_price
            'Avg Price': f"₹{holding['average_price']:,.2f}",
            'Current Price': f"₹{current_price:,.2f}" if current_price else "N/A",
            'Current Value': f"₹{current_value:,.0f}",
            'P&L': f"₹{pnl:,.0f}",
            'P&L %': f"{pnl_percent:+.1f}%",
            'Performance': rating
        })
    
    df_holdings = pd.DataFrame(holdings_data)
    st.dataframe(df_holdings, use_container_width=True)
    
    # Market Sentiment Section
    st.markdown("---")
    st.subheader("📈 Market Sentiment & Trends")
    
    # Market Sentiment Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Market Breadth (using cached metrics)
        market_breadth = metrics.get('market_breadth', 0)
        gainers = metrics.get('gainers', 0)
        total_holdings = metrics.get('total_holdings', 0)
        
        st.metric(
            "Market Breadth",
            f"{market_breadth:.1f}%",
            delta=f"{gainers}/{total_holdings}",
            help="Percentage of holdings in positive territory"
        )
        
        if market_breadth > 70:
            st.success("Strong Bullish Sentiment")
        elif market_breadth < 30:
            st.error("Bearish Sentiment")
        else:
            st.info("Mixed Sentiment")
    
    with col2:
        # Average P&L (using cached metrics)
        total_pnl_pct = metrics.get('total_pnl_pct', 0)
        
        st.metric(
            "Avg Portfolio Return",
            f"{total_pnl_pct:.2f}%",
            help="Average return across all holdings"
        )
        
        if total_pnl_pct > 5:
            st.success("Strong Performance")
        elif total_pnl_pct < -5:
            st.error("Underperforming")
        else:
            st.info("Moderate Performance")
    
    with col3:
        # Volatility Index (using cached metrics)
        volatility = metrics.get('volatility', 0)
        
        st.metric(
            "Portfolio Volatility",
            f"{volatility:.2f}%",
            help="Standard deviation of returns"
        )
        
        if volatility > 20:
            st.error("High Volatility")
        elif volatility < 10:
            st.success("Low Volatility")
        else:
            st.warning("Moderate Volatility")
    
    with col4:
        # Momentum Score (using cached metrics)
        momentum_score = metrics.get('momentum_score', 0)
        momentum_stocks = metrics.get('momentum_stocks', 0)
        
        st.metric(
            "Momentum Score",
            f"{momentum_score:.1f}%",
            delta=f"{momentum_stocks} stocks >10%",
            help="Percentage of holdings with strong momentum"
        )
        
        if momentum_score > 50:
            st.success("Strong Momentum")
        elif momentum_score < 20:
            st.error("Weak Momentum")
        else:
            st.info("Moderate Momentum")
    
    # Market Sentiment Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector Performance Heatmap
        sector_data = {}
        for holding in holdings:
            sector = holding.get('sector', holding.get('asset_type', 'Unknown'))
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            
            pnl_pct = ((current_price - holding.get('average_price', 0)) / holding.get('average_price', 1) * 100) if holding.get('average_price', 0) > 0 else 0
            
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append(pnl_pct)
        
        # Calculate average sector performance
        sector_performance = {sector: np.mean(returns) for sector, returns in sector_data.items()}
        
        if sector_performance:
            fig_sector_heatmap = go.Figure(data=go.Heatmap(
                z=[[list(sector_performance.values())[i]] for i in range(len(sector_performance))],
                x=['Performance'],
                y=list(sector_performance.keys()),
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return %")
            ))
            
            fig_sector_heatmap.update_layout(
                title="Sector Performance Heatmap",
                height=300
            )
            
            st.plotly_chart(fig_sector_heatmap, use_container_width=True)
    
    with col2:
        # Market Cap Distribution
        market_cap_data = []
        for holding in holdings:
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            
            market_value = float(current_price) * float(holding['total_quantity'])
            
            # Categorize by market cap (simplified)
            if market_value > 1000000:  # > 10L
                cap_category = "Large Cap"
            elif market_value > 100000:  # > 1L
                cap_category = "Mid Cap"
            else:
                cap_category = "Small Cap"
            
            market_cap_data.append({
                'ticker': holding['ticker'],
                'market_value': market_value,
                'cap_category': cap_category
            })
        
        if market_cap_data:
            df_market_cap = pd.DataFrame(market_cap_data)
            cap_distribution = df_market_cap.groupby('cap_category')['market_value'].sum()
            
            fig_market_cap = px.pie(
                values=cap_distribution.values,
                names=cap_distribution.index,
                title="Market Cap Distribution",
                hole=0.4
            )
            fig_market_cap.update_layout(height=300)
            st.plotly_chart(fig_market_cap, use_container_width=True)
    
    # Advanced Charts Section
    st.markdown("---")
    st.subheader("📊 Advanced Portfolio Analytics")
    
    # Top Performers and Underperformers
    col1, col2 = st.columns(2)
    
    # Prepare performance data
    perf_data = []
    for holding in holdings:
        # CRITICAL: Skip holdings with zero or negative quantity (fully sold positions)
        # Use a small epsilon to handle floating point precision issues (e.g., 0.0001)
        quantity = float(holding.get('total_quantity', 0))
        if quantity <= 0.0001:  # Treat anything <= 0.0001 as effectively zero
            continue  # Skip fully sold positions - they shouldn't count in invested amount or P&L
        
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        current_value = quantity * float(current_price) if current_price else 0
        investment_value = quantity * float(holding['average_price'])
        pnl = current_value - investment_value
        pnl_percent = (pnl / investment_value * 100) if investment_value > 0 else 0
        
        perf_data.append({
            'ticker': holding['ticker'],
            'stock_name': holding['stock_name'],
            'invested_amount': investment_value,
            'unrealized_pnl': pnl,
            'pnl_percentage': pnl_percent
        })
    
    df_perf = pd.DataFrame(perf_data)
    
    with col1:
        st.markdown("### 📈 Top 5 Performers")
        top_performers = df_perf.nlargest(5, 'pnl_percentage')
        
        if not top_performers.empty:
            top_performers = top_performers.sort_values('pnl_percentage', ascending=True)
            top_performers['display_label'] = top_performers['ticker'] + ' - ' + top_performers['stock_name'].str[:20]
            
            fig_top = go.Figure()
            fig_top.add_trace(go.Bar(
                x=top_performers['pnl_percentage'],
                y=top_performers['display_label'],
                orientation='h',
                marker=dict(
                    color=top_performers['pnl_percentage'],
                    colorscale='Greens',
                    showscale=False
                ),
                text=top_performers['pnl_percentage'].round(2),
                texttemplate='%{text:.2f}%',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Return: %{x:.2f}%<br>P&L: ₹%{customdata:,.0f}<extra></extra>',
                customdata=top_performers['unrealized_pnl']
            ))
            
            fig_top.update_layout(
                title="",
                xaxis_title="Return %",
                yaxis_title="",
                height=300,
                showlegend=False,
                margin=dict(l=200)
            )
            
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No gainers yet")
    
    with col2:
        st.markdown("### 📉 Bottom 5 Performers")
        underperformers = df_perf.nsmallest(5, 'pnl_percentage')
        
        if not underperformers.empty:
            underperformers = underperformers.sort_values('pnl_percentage', ascending=False)
            underperformers['display_label'] = underperformers['ticker'] + ' - ' + underperformers['stock_name'].str[:20]
            
            fig_bottom = go.Figure()
            fig_bottom.add_trace(go.Bar(
                x=underperformers['pnl_percentage'],
                y=underperformers['display_label'],
                orientation='h',
                marker=dict(
                    color=underperformers['pnl_percentage'],
                    colorscale='Reds',
                    showscale=False,
                    reversescale=True
                ),
                text=underperformers['pnl_percentage'].round(2),
                texttemplate='%{text:.2f}%',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Return: %{x:.2f}%<br>P&L: ₹%{customdata:,.0f}<extra></extra>',
                customdata=underperformers['unrealized_pnl']
            ))
            
            fig_bottom.update_layout(
                title="",
                xaxis_title="Return %",
                yaxis_title="",
                height=300,
                showlegend=False,
                margin=dict(l=200)
            )
            
            st.plotly_chart(fig_bottom, use_container_width=True)
        else:
            st.info("No underperformers - Great portfolio!")
    
    # Investment Distribution Treemap
    st.markdown("---")
    st.subheader("🗺️ Portfolio Allocation Treemap")
    
    # Create treemap data
    treemap_data = []
    for holding in holdings:
        # CRITICAL: Skip holdings with zero or negative quantity (fully sold positions)
        quantity = float(holding.get('total_quantity', 0))
        if quantity <= 0:
            continue  # Skip fully sold positions - they shouldn't appear in treemap
        
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        current_value = quantity * float(current_price) if current_price else 0
        
        treemap_data.append({
            'labels': f"{holding['ticker']}<br>{holding['stock_name'][:20]}",
            'parents': holding.get('asset_type', 'Unknown'),
            'values': current_value,
            'text': f"₹{current_value:,.0f}"
        })
    
    # Add parent categories
    asset_types = set([holding.get('asset_type', 'Unknown') for holding in holdings])
    for asset_type in asset_types:
        treemap_data.append({
            'labels': asset_type,
            'parents': '',
            'values': 0,
            'text': asset_type
        })
    
    df_treemap = pd.DataFrame(treemap_data)
    
    fig_treemap = go.Figure(go.Treemap(
        labels=df_treemap['labels'],
        parents=df_treemap['parents'],
        values=df_treemap['values'],
        text=df_treemap['text'],
        textposition='middle center',
        marker=dict(
            colorscale='Viridis',
            cmid=df_treemap['values'].mean()
        )
    ))
    
    fig_treemap.update_layout(
        title="Portfolio Allocation by Asset Type and Holdings",
        height=500
    )
    
    st.plotly_chart(fig_treemap, use_container_width=True)

def get_performance_rating(pnl_percent):
    """Get star rating and grade based on P&L percentage"""
    if pnl_percent >= 50:
        return "⭐⭐⭐⭐⭐", "A+", "Excellent"
    elif pnl_percent >= 30:
        return "⭐⭐⭐⭐", "A", "Very Good"
    elif pnl_percent >= 15:
        return "⭐⭐⭐", "B+", "Good"
    elif pnl_percent >= 5:
        return "⭐⭐", "B", "Average"
    elif pnl_percent >= 0:
        return "⭐", "C", "Below Average"
    elif pnl_percent >= -10:
        return "❌", "D", "Poor"
    else:
        return "❌❌", "F", "Very Poor"

def get_risk_score(volatility):
    """Get risk rating based on volatility"""
    if volatility < 10:
        return "🟢 Low Risk", "Conservative"
    elif volatility < 20:
        return "🟡 Moderate Risk", "Balanced"
    elif volatility < 30:
        return "🟠 High Risk", "Aggressive"
    else:
        return "🔴 Very High Risk", "Speculative"

def pnl_analysis_page():
    """P&L Analysis based on current week price (as per your image)"""
    st.header("📊 P&L Analysis")
    st.caption("Calculated based on current week prices")
    
    user = st.session_state.user
    holdings = db.get_user_holdings(user['id'])
    
    if not holdings:
        st.info("No holdings found.")
        return
    
    # Group by sector and channel
    sector_data = {}
    channel_data = {}
    
    for holding in holdings:
        # CRITICAL: Skip holdings with zero or negative quantity (fully sold positions)
        # Use a small epsilon to handle floating point precision issues (e.g., 0.0001)
        quantity = float(holding.get('total_quantity', 0))
        if quantity <= 0.0001:  # Treat anything <= 0.0001 as effectively zero
            continue  # Skip fully sold positions - they shouldn't count in invested amount or P&L
        
        sector = holding.get('sector', 'Unknown')
        channel = holding.get('channel', 'Direct')
        
        # Handle None current_price
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        current_value = quantity * float(current_price) if current_price else 0
        investment_value = quantity * float(holding['average_price'])
        pnl = current_value - investment_value
        
        # Sector analysis
        if sector not in sector_data:
            sector_data[sector] = {'investment': 0, 'current': 0, 'pnl': 0}
        sector_data[sector]['investment'] += investment_value
        sector_data[sector]['current'] += current_value
        sector_data[sector]['pnl'] += pnl
        
        # Channel analysis
        if channel not in channel_data:
            channel_data[channel] = {'investment': 0, 'current': 0, 'pnl': 0}
        channel_data[channel]['investment'] += investment_value
        channel_data[channel]['current'] += current_value
        channel_data[channel]['pnl'] += pnl
    
    # Sector analysis
    st.subheader("📊 Sector Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector pie chart
        sectors = list(sector_data.keys())
        values = [sector_data[s]['current'] for s in sectors]
        
        fig = px.pie(values=values, names=sectors, title="Portfolio by Sector")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sector table with ratings
        sector_table = []
        for sector, data in sector_data.items():
            pnl_percent = (data['pnl'] / data['investment'] * 100) if data['investment'] > 0 else 0
            stars, grade, rating = get_performance_rating(pnl_percent)
            
            sector_table.append({
                'Sector': sector,
                'Rating': stars,
                'Grade': grade,
                'Investment': f"₹{data['investment']:,.0f}",
                'Current': f"₹{data['current']:,.0f}",
                'P&L': f"₹{data['pnl']:,.0f}",
                'P&L %': f"{pnl_percent:+.1f}%",
                'Performance': rating
            })
        
        df_sectors = pd.DataFrame(sector_table)
        st.dataframe(df_sectors, use_container_width=True)
    
    # Channel analysis
    st.subheader("📊 Channel Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Channel pie chart
        channels = list(channel_data.keys())
        values = [channel_data[c]['current'] for c in channels]
        
        fig = px.pie(values=values, names=channels, title="Portfolio by Channel")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Channel table with ratings
        channel_table = []
        for channel, data in channel_data.items():
            pnl_percent = (data['pnl'] / data['investment'] * 100) if data['investment'] > 0 else 0
            stars, grade, rating = get_performance_rating(pnl_percent)
            
            channel_table.append({
                'Channel': channel,
                'Rating': stars,
                'Grade': grade,
                'Investment': f"₹{data['investment']:,.0f}",
                'Current': f"₹{data['current']:,.0f}",
                'P&L': f"₹{data['pnl']:,.0f}",
                'P&L %': f"{pnl_percent:+.1f}%",
                'Performance': rating
            })
        
        df_channels = pd.DataFrame(channel_table)
        st.dataframe(df_channels, use_container_width=True)

def charts_page():
    """Comprehensive charts and analytics page"""
    st.header("📈 Charts & Analytics")
    
    user = st.session_state.user
    
    # Use cached holdings data (same as portfolio overview)
    holdings = get_cached_holdings(user['id'])
    
    if not holdings:
        st.info("No holdings found. Upload transaction files to see charts.")
        return
    
    # Add a small loading indicator for better UX
    with st.spinner("📊 Loading charts..."):
        pass  # This creates a brief loading state
    
    # Add info about page behavior
    with st.expander("ℹ️ About Page Refreshing", expanded=False):
        st.info("""
        **Why does the page refresh when changing comparisons?**
        
        This is normal Streamlit behavior - when you change dropdown selections, the entire page reruns to update the charts with new data. 
        
        **Tips for smoother experience:**
        - ✅ Your selections are preserved between refreshes
        - ✅ Data is cached for faster loading
        - ✅ Only the comparison section refreshes, not the entire app
        
        **What's optimized:**
        - Session state maintains your selections
        - Cached data reduces loading time
        - Smart defaults for common selections
        """)
    
    # Create tabs for different chart types
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Portfolio Allocation", 
        "💰 Performance", 
        "📅 52-Week NAVs",
        "🔍 Advanced Analytics",
        "📈 Technical Analysis",
        "⚡ Risk Metrics",
        "🔄 Compare Holdings"
    ])
    
    with tab1:
        st.subheader("📊 Portfolio Allocation")
        
        # Asset Type Distribution
        asset_types = {}
        for holding in holdings:
            asset_type = holding.get('asset_type', 'Unknown')
            # Handle None current_price
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            current_value = float(current_price) * float(holding.get('total_quantity', 0))
            asset_types[asset_type] = asset_types.get(asset_type, 0) + current_value
        
        if asset_types:
            fig_pie = px.pie(
                values=list(asset_types.values()),
                names=list(asset_types.keys()),
                title="Asset Type Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Top Holdings
        holdings_data = []
        for holding in holdings:
            # Handle None current_price
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            current_value = float(current_price) * float(holding.get('total_quantity', 0))
            holdings_data.append({
                'Ticker': holding['ticker'],
                'Name': holding['stock_name'],
                'Current Value': current_value,
                'Quantity': holding['total_quantity'],
                'Avg Price': holding['average_price']
            })
        
        if holdings_data:
            df_holdings = pd.DataFrame(holdings_data)
            df_holdings = df_holdings.sort_values('Current Value', ascending=False).head(10)
            
            fig_bar = px.bar(
                df_holdings, 
                x='Ticker', 
                y='Current Value',
                title="Top 10 Holdings by Value"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Performance Analysis")
        
        # Calculate P&L for each holding
        performance_data = []
        for holding in holdings:
            investment = float(holding['total_quantity']) * float(holding['average_price'])
            # Handle None current_price
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            current_value = float(current_price) * float(holding['total_quantity'])
            pnl = current_value - investment
            pnl_pct = (pnl / investment * 100) if investment > 0 else 0
            
            performance_data.append({
                'Ticker': holding['ticker'],
                'Name': holding['stock_name'],
                'Investment': investment,
                'Current Value': current_value,
                'P&L': pnl,
                'P&L %': pnl_pct,
                'Asset Type': holding.get('asset_type', 'Unknown'),
                'Channel': holding.get('channel', 'Direct')
            })
        
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            
            # Top Gainers and Losers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Top 10 Gainers")
                gainers = df_performance[df_performance['P&L'] > 0].nlargest(10, 'P&L %')
                if not gainers.empty:
                    for idx, row in gainers.iterrows():
                        with st.container():
                            st.markdown(f"**{row['Name']}** ({row['Ticker']})")
                            st.markdown(f"💰 P&L: ₹{row['P&L']:,.0f} | 📊 {row['P&L %']:.2f}%")
                            st.progress(min(row['P&L %'] / 100, 1.0))
                            st.markdown("---")
                else:
                    st.info("No gainers yet")
            
            with col2:
                st.subheader("📉 Top 10 Losers")
                losers = df_performance[df_performance['P&L'] < 0].nsmallest(10, 'P&L %')
                if not losers.empty:
                    for idx, row in losers.iterrows():
                        with st.container():
                            st.markdown(f"**{row['Name']}** ({row['Ticker']})")
                            st.markdown(f"💸 Loss: ₹{row['P&L']:,.0f} | 📊 {row['P&L %']:.2f}%")
                            st.progress(min(abs(row['P&L %']) / 100, 1.0))
                            st.markdown("---")
                else:
                    st.info("No losers - Excellent performance!")
            
            st.markdown("---")
            
            # Performance by Asset Type and Channel
            st.subheader("📊 Performance Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏢 By Asset Type")
                asset_perf = df_performance.groupby('Asset Type').agg({
                    'Investment': 'sum',
                    'Current Value': 'sum',
                    'P&L': 'sum'
                }).reset_index()
                asset_perf['P&L %'] = (asset_perf['P&L'] / asset_perf['Investment'] * 100)
                asset_perf = asset_perf.sort_values('P&L %', ascending=False)
                
                fig_asset_bar = go.Figure()
                fig_asset_bar.add_trace(go.Bar(
                    x=asset_perf['Asset Type'],
                    y=asset_perf['P&L %'],
                    marker_color=['green' if x >= 0 else 'red' for x in asset_perf['P&L %']],
                    text=asset_perf['P&L %'].round(2),
                    textposition='auto'
                ))
                fig_asset_bar.update_layout(
                    title="Asset Type Performance %",
                    xaxis_title="Asset Type",
                    yaxis_title="P&L %",
                    height=400
                )
                st.plotly_chart(fig_asset_bar, use_container_width=True)
            
            with col2:
                st.markdown("### 📡 By Channel")
                channel_perf = df_performance.groupby('Channel').agg({
                    'Investment': 'sum',
                    'Current Value': 'sum',
                    'P&L': 'sum'
                }).reset_index()
                channel_perf['P&L %'] = (channel_perf['P&L'] / channel_perf['Investment'] * 100)
                channel_perf = channel_perf.sort_values('P&L %', ascending=False)
                
                fig_channel_bar = go.Figure()
                fig_channel_bar.add_trace(go.Bar(
                    x=channel_perf['Channel'],
                    y=channel_perf['P&L %'],
                    marker_color=['green' if x >= 0 else 'red' for x in channel_perf['P&L %']],
                    text=channel_perf['P&L %'].round(2),
                    textposition='auto'
                ))
                fig_channel_bar.update_layout(
                    title="Channel Performance %",
                    xaxis_title="Channel",
                    yaxis_title="P&L %",
                    height=400
                )
                st.plotly_chart(fig_channel_bar, use_container_width=True)
            
            st.markdown("---")
            
            # Portfolio Statistics
            st.subheader("📊 Portfolio Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_investment = df_performance['Investment'].sum()
                st.metric("Total Investment", f"₹{total_investment:,.0f}")
            
            with col2:
                total_current = df_performance['Current Value'].sum()
                st.metric("Current Value", f"₹{total_current:,.0f}")
            
            with col3:
                total_pnl = df_performance['P&L'].sum()
                st.metric("Total P&L", f"₹{total_pnl:,.0f}")
            
            with col4:
                total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
                st.metric("Total P&L %", f"{total_pnl_pct:+.1f}%")
            
            # Performance Table
            st.subheader("📋 Detailed Performance")
            df_display = df_performance.copy()
            df_display['Investment'] = df_display['Investment'].apply(lambda x: f"₹{x:,.0f}")
            df_display['Current Value'] = df_display['Current Value'].apply(lambda x: f"₹{x:,.0f}")
            df_display['P&L'] = df_display['P&L'].apply(lambda x: f"₹{x:,.0f}")
            df_display['P&L %'] = df_display['P&L %'].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(df_display, use_container_width=True)
    
    with tab3:
        st.subheader("📅 52-Week NAVs")
        
        # Get ticker options
        ticker_options = [h['ticker'] for h in holdings]
        
        # Select ticker for NAV view
        selected_ticker_nav = st.selectbox("Select Ticker for NAV History", ticker_options, key="nav_ticker")
        
        if selected_ticker_nav:
            stock_id = next(h['stock_id'] for h in holdings if h['ticker'] == selected_ticker_nav)
            prices = db.get_historical_prices_for_stock_silent(stock_id)
            
            if prices:
                df_navs = pd.DataFrame(prices)
                df_navs['date'] = pd.to_datetime(df_navs['price_date'])
                df_navs = df_navs.sort_values('date')

                if len(df_navs) > 52:
                    df_navs = df_navs.tail(52)
                
                fig_nav = px.line(
                    df_navs, 
                    x='date', 
                    y='price',
                    title=f"{selected_ticker_nav} - 52-Week NAVs"
                )
                fig_nav.update_layout(xaxis_title="Date", yaxis_title="NAV (₹)")
                st.plotly_chart(fig_nav, use_container_width=True)
                
                st.subheader(f"{selected_ticker_nav} - NAV History")
                df_display = df_navs[['date', 'price', 'iso_week', 'iso_year']].copy()
                df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
                df_display.columns = ['Date', 'NAV', 'Week', 'Year']
                st.dataframe(df_display.tail(20), use_container_width=True)
            else:
                st.info(f"No NAV data available for {selected_ticker_nav}")
    
    with tab4:
        st.subheader("🔍 Advanced Analytics")
        
        # Prepare comprehensive analytics data
        analytics_data = []
        for holding in holdings:
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            current_value = float(current_price) * float(holding['total_quantity'])
            investment = float(holding['total_quantity']) * float(holding['average_price'])
            pnl = current_value - investment
            pnl_pct = (pnl / investment * 100) if investment > 0 else 0
            
            analytics_data.append({
                'ticker': holding['ticker'],
                'stock_name': holding['stock_name'],
                'asset_type': holding.get('asset_type', 'Unknown'),
                'channel': holding.get('channel', 'Direct'),
                'investment': investment,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'quantity': float(holding['total_quantity']),
                'avg_price': float(holding['average_price']),
                'current_price': float(current_price) if current_price else 0
            })
        
        df_analytics = pd.DataFrame(analytics_data)
        
        # Create sub-tabs for different analytics
        analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
            "📊 Risk Metrics",
            "💰 Value Analysis", 
            "📈 Performance Distribution",
            "🎯 Portfolio Insights"
        ])
        
        with analytics_tab1:
            st.markdown("### 📊 Risk & Return Metrics")
            
            # Calculate portfolio metrics
            total_investment = df_analytics['investment'].sum()
            total_current = df_analytics['current_value'].sum()
            portfolio_return = (total_current - total_investment) / total_investment * 100 if total_investment > 0 else 0
            individual_returns = df_analytics['pnl_pct'].tolist()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Return", f"{portfolio_return:+.2f}%")
            
            with col2:
                if individual_returns:
                    volatility = np.std(individual_returns)
                    risk_label, risk_strategy = get_risk_score(volatility)
                    st.metric("Volatility", f"{volatility:.2f}%", delta=risk_label)
                else:
                    st.metric("Volatility", "N/A")
            
            with col3:
                if individual_returns and volatility > 0:
                    sharpe_ratio = portfolio_return / volatility
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                else:
                    st.metric("Sharpe Ratio", "N/A")
            
            with col4:
                max_drawdown = min(individual_returns) if individual_returns else 0
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            
            st.markdown("---")
            
            # Additional risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Win rate
                winners = len(df_analytics[df_analytics['pnl'] > 0])
                total_holdings = len(df_analytics)
                win_rate = (winners / total_holdings * 100) if total_holdings > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{winners}/{total_holdings} holdings")
            
            with col2:
                # Average gain/loss
                avg_gain = df_analytics[df_analytics['pnl'] > 0]['pnl_pct'].mean() if winners > 0 else 0
                st.metric("Avg Gain", f"{avg_gain:+.2f}%")
            
            with col3:
                losers = len(df_analytics[df_analytics['pnl'] < 0])
                avg_loss = df_analytics[df_analytics['pnl'] < 0]['pnl_pct'].mean() if losers > 0 else 0
                st.metric("Avg Loss", f"{avg_loss:.2f}%")
            
            # Risk-Return Scatter Plot
            st.markdown("#### 🎯 Risk-Return Profile")
            
            fig_risk_return = px.scatter(
                df_analytics,
                x='investment',
                y='pnl_pct',
                size='current_value',
                color='asset_type',
                hover_data=['ticker', 'stock_name'],
                title="Risk-Return Analysis (Size = Current Value)",
                labels={'investment': 'Investment Amount (₹)', 'pnl_pct': 'Return (%)'}
            )
            
            # Add quadrant lines
            fig_risk_return.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_risk_return.add_vline(x=df_analytics['investment'].median(), line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_risk_return, use_container_width=True)
        
        with analytics_tab2:
            st.markdown("### 💰 Value & Allocation Analysis")
            
            # Portfolio composition by value
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Top 10 Holdings by Value")
                top_10_value = df_analytics.nlargest(10, 'current_value')
                
                fig_top_value = px.bar(
                    top_10_value,
                    x='current_value',
                    y='ticker',
                    orientation='h',
                    title="Top 10 Holdings by Current Value",
                    color='pnl_pct',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    labels={'current_value': 'Current Value (₹)', 'ticker': 'Ticker'}
                )
                st.plotly_chart(fig_top_value, use_container_width=True)
            
            with col2:
                st.markdown("#### 💸 Top 10 Holdings by Investment")
                top_10_inv = df_analytics.nlargest(10, 'investment')
                
                fig_top_inv = px.bar(
                    top_10_inv,
                    x='investment',
                    y='ticker',
                    orientation='h',
                    title="Top 10 Holdings by Investment",
                    color='pnl_pct',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    labels={'investment': 'Investment (₹)', 'ticker': 'Ticker'}
                )
                st.plotly_chart(fig_top_inv, use_container_width=True)
            
            # Concentration metrics
            st.markdown("#### 🎯 Concentration Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Top holding concentration
                top_holding_pct = (df_analytics['current_value'].max() / total_current * 100) if total_current > 0 else 0
                st.metric("Top Holding", f"{top_holding_pct:.1f}%")
            
            with col2:
                # Top 5 concentration
                top_5_value = df_analytics.nlargest(5, 'current_value')['current_value'].sum()
                top_5_pct = (top_5_value / total_current * 100) if total_current > 0 else 0
                st.metric("Top 5 Holdings", f"{top_5_pct:.1f}%")
            
            with col3:
                # Top 10 concentration
                top_10_value_sum = df_analytics.nlargest(10, 'current_value')['current_value'].sum()
                top_10_pct = (top_10_value_sum / total_current * 100) if total_current > 0 else 0
                st.metric("Top 10 Holdings", f"{top_10_pct:.1f}%")
            
            with col4:
                # Number of holdings
                st.metric("Total Holdings", len(df_analytics))
            
            # Herfindahl Index (concentration measure)
            holdings_shares = (df_analytics['current_value'] / total_current * 100) ** 2
            herfindahl_index = holdings_shares.sum()
            
            st.markdown(f"**Herfindahl Index:** {herfindahl_index:.2f}")
            if herfindahl_index > 2500:
                st.warning("⚠️ Highly concentrated portfolio (HHI > 2500)")
            elif herfindahl_index > 1500:
                st.info("ℹ️ Moderately concentrated portfolio (HHI 1500-2500)")
            else:
                st.success("✅ Well diversified portfolio (HHI < 1500)")
        
        with analytics_tab3:
            st.markdown("### 📈 Performance Distribution Analysis")
            
            # Performance histogram
            fig_hist = px.histogram(
                df_analytics,
                x='pnl_pct',
                nbins=20,
                title="Distribution of Returns Across Holdings",
                labels={'pnl_pct': 'Return (%)', 'count': 'Number of Holdings'},
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
            fig_hist.add_vline(x=df_analytics['pnl_pct'].median(), line_dash="dash", line_color="green", annotation_text="Median")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Performance quartiles
            st.markdown("#### 📊 Performance Quartiles")
            
            q1 = df_analytics['pnl_pct'].quantile(0.25)
            q2 = df_analytics['pnl_pct'].quantile(0.50)  # Median
            q3 = df_analytics['pnl_pct'].quantile(0.75)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Q1 (25th %ile)", f"{q1:+.2f}%")
            with col2:
                st.metric("Q2 (Median)", f"{q2:+.2f}%")
            with col3:
                st.metric("Q3 (75th %ile)", f"{q3:+.2f}%")
            with col4:
                iqr = q3 - q1
                st.metric("IQR", f"{iqr:.2f}%")
            
            # Box plot by asset type
            st.markdown("#### 📦 Performance by Asset Type (Box Plot)")
            
            fig_box = px.box(
                df_analytics,
                x='asset_type',
                y='pnl_pct',
                title="Return Distribution by Asset Type",
                labels={'asset_type': 'Asset Type', 'pnl_pct': 'Return (%)'},
                color='asset_type'
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistical summary
            st.markdown("#### 📊 Statistical Summary")
            
            summary_stats = df_analytics['pnl_pct'].describe()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{summary_stats['mean']:+.2f}%",
                        f"{summary_stats['std']:.2f}%",
                        f"{summary_stats['min']:+.2f}%",
                        f"{summary_stats['max']:+.2f}%"
                    ]
                }), use_container_width=True, hide_index=True)
            
            with col2:
                # Skewness and kurtosis
                from scipy import stats
                skewness = stats.skew(df_analytics['pnl_pct'])
                kurtosis = stats.kurtosis(df_analytics['pnl_pct'])
                
                st.dataframe(pd.DataFrame({
                    'Metric': ['Skewness', 'Kurtosis', 'Range', 'CV'],
                    'Value': [
                        f"{skewness:.2f}",
                        f"{kurtosis:.2f}",
                        f"{summary_stats['max'] - summary_stats['min']:.2f}%",
                        f"{(summary_stats['std'] / abs(summary_stats['mean']) * 100):.2f}%" if summary_stats['mean'] != 0 else "N/A"
                    ]
                }), use_container_width=True, hide_index=True)
        
        with analytics_tab4:
            st.markdown("### 🎯 Portfolio Insights & Recommendations")
            
            # Key insights
            st.markdown("#### 💡 Key Insights")
            
            # 1. Best and worst performers
            best_performer = df_analytics.nlargest(1, 'pnl_pct').iloc[0]
            worst_performer = df_analytics.nsmallest(1, 'pnl_pct').iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **🏆 Best Performer:**
                - {best_performer['ticker']} - {best_performer['stock_name']}
                - Return: {best_performer['pnl_pct']:+.2f}%
                - P&L: ₹{best_performer['pnl']:,.0f}
                """)
            
            with col2:
                st.error(f"""
                **📉 Worst Performer:**
                - {worst_performer['ticker']} - {worst_performer['stock_name']}
                - Return: {worst_performer['pnl_pct']:+.2f}%
                - P&L: ₹{worst_performer['pnl']:,.0f}
                """)
            
            # 2. Asset allocation insights
            st.markdown("#### 📊 Asset Allocation Insights")
            
            asset_allocation = df_analytics.groupby('asset_type').agg({
                'current_value': 'sum',
                'pnl_pct': 'mean'
            }).reset_index()
            asset_allocation['allocation_pct'] = (asset_allocation['current_value'] / total_current * 100)
            
            for _, row in asset_allocation.iterrows():
                asset_type = row['asset_type']
                allocation = row['allocation_pct']
                avg_return = row['pnl_pct']
                
                if allocation > 50:
                    st.warning(f"⚠️ {asset_type.upper()}: {allocation:.1f}% allocation (High concentration)")
                elif allocation > 30:
                    st.info(f"ℹ️ {asset_type.upper()}: {allocation:.1f}% allocation | Avg Return: {avg_return:+.1f}%")
                else:
                    st.success(f"✅ {asset_type.upper()}: {allocation:.1f}% allocation | Avg Return: {avg_return:+.1f}%")
            
            # 3. Channel performance insights
            st.markdown("#### 📡 Channel Performance Insights")
            
            channel_performance = df_analytics.groupby('channel').agg({
                'current_value': 'sum',
                'pnl': 'sum',
                'pnl_pct': 'mean'
            }).reset_index()
            channel_performance = channel_performance.sort_values('pnl_pct', ascending=False)
            
            best_channel = channel_performance.iloc[0]
            st.info(f"""
            **🏆 Best Performing Channel:** {best_channel['channel']}
            - Average Return: {best_channel['pnl_pct']:+.2f}%
            - Total P&L: ₹{best_channel['pnl']:,.0f}
            - Portfolio Value: ₹{best_channel['current_value']:,.0f}
            """)
            
            # 4. Recommendations
            st.markdown("#### 🎯 Recommendations")
            
            recommendations = []
            
            # Check for underperformers
            underperformers = df_analytics[df_analytics['pnl_pct'] < -10]
            if len(underperformers) > 0:
                recommendations.append(f"⚠️ Review {len(underperformers)} holdings with losses > 10%")
            
            # Check for concentration
            if top_holding_pct > 25:
                recommendations.append(f"⚠️ Top holding represents {top_holding_pct:.1f}% of portfolio - consider diversification")
            
            # Check for winners
            big_winners = df_analytics[df_analytics['pnl_pct'] > 50]
            if len(big_winners) > 0:
                recommendations.append(f"✅ {len(big_winners)} holdings with returns > 50% - consider profit booking")
            
            # Check volatility
            if volatility > 30:
                recommendations.append(f"⚠️ High portfolio volatility ({volatility:.1f}%) - consider adding stable assets")
            
            # Check win rate
            if win_rate < 50:
                recommendations.append(f"⚠️ Win rate below 50% ({win_rate:.1f}%) - review investment strategy")
            
            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("✅ Portfolio is well balanced with no immediate concerns!")
    
    with tab5:
        st.subheader("📈 Technical Analysis")
        st.caption("Professional-grade technical indicators and analysis")
        
        # Technical Analysis Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_ticker_tech = st.selectbox(
                "Select Stock for Technical Analysis:",
                [h['ticker'] for h in holdings],
                key="tech_ticker_select"
            )
        
        with col2:
            time_period = st.selectbox(
                "Time Period:",
                ["1M", "3M", "6M", "1Y", "2Y"],
                index=3,
                key="tech_time_period"
            )
        
        with col3:
            show_indicators = st.multiselect(
                "Technical Indicators:",
                ["RSI", "MACD", "Moving Averages", "Bollinger Bands", "Volume"],
                default=["RSI", "MACD", "Moving Averages"],
                key="tech_indicators"
            )
        
        # Only load technical analysis if user has made selections
        if selected_ticker_tech and show_indicators:
            st.markdown("---")
            
            # Get historical data for technical analysis
            try:
                import yfinance as yf
                
                # Convert time period to days
                period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}
                period = period_map.get(time_period, "1y")
                
                # Try multiple ticker formats
                hist = pd.DataFrame()
                ticker_formats = [
                    selected_ticker_tech,  # Original (e.g., IDFCFIRSTB.NS)
                    selected_ticker_tech.replace('.NS', '').replace('.BO', ''),  # Without suffix
                    selected_ticker_tech + '.NS' if '.NS' not in selected_ticker_tech and '.BO' not in selected_ticker_tech else selected_ticker_tech,  # Add .NS
                    selected_ticker_tech.replace('.NS', '.BO') if '.NS' in selected_ticker_tech else selected_ticker_tech.replace('.BO', '.NS')  # Switch exchange
                ]
                
                for ticker_format in ticker_formats:
                    try:
                        stock = yf.Ticker(ticker_format)
                        hist = stock.history(period=period)
                        if not hist.empty and len(hist) > 20:  # Need at least 20 days for indicators
                            st.caption(f"✅ Data fetched using ticker: {ticker_format}")
                            break
                    except:
                        continue
                
                if not hist.empty and len(hist) > 20:
                    # Price Chart with Technical Indicators
                    fig_tech = go.Figure()
                    
                    # Add candlestick chart
                    fig_tech.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name="Price",
                        increasing_line_color='#00ff00',
                        decreasing_line_color='#ff0000'
                    ))
                    
                    # Add technical indicators
                    if "Moving Averages" in show_indicators:
                        # 20-day MA
                        hist['MA20'] = hist['Close'].rolling(window=20).mean()
                        fig_tech.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['MA20'],
                            mode='lines',
                            name='MA 20',
                            line=dict(color='orange', width=2)
                        ))
                        
                        # 50-day MA
                        hist['MA50'] = hist['Close'].rolling(window=50).mean()
                        fig_tech.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['MA50'],
                            mode='lines',
                            name='MA 50',
                            line=dict(color='blue', width=2)
                        ))
                    
                    if "Bollinger Bands" in show_indicators:
                        # Bollinger Bands
                        hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
                        bb_std = hist['Close'].rolling(window=20).std()
                        hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
                        hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)
                        
                        fig_tech.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['BB_Upper'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash')
                        ))
                        fig_tech.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['BB_Lower'],
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty'
                        ))
                    
                    fig_tech.update_layout(
                        title=f"{selected_ticker_tech} - Technical Analysis ({time_period})",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_tech, use_container_width=True)
                    
                    # Technical Indicators Panel
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if "RSI" in show_indicators:
                            # Calculate RSI
                            delta = hist['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            current_rsi = rsi.iloc[-1] if not rsi.empty else 0
                            
                            st.metric(
                                "RSI (14)",
                                f"{current_rsi:.1f}",
                                delta=f"{rsi.iloc[-1] - rsi.iloc[-2]:.1f}" if len(rsi) > 1 else "0.0"
                            )
                            
                            # RSI interpretation
                            if current_rsi > 70:
                                st.error("Overbought")
                            elif current_rsi < 30:
                                st.success("Oversold")
                            else:
                                st.info("Neutral")
                    
                    with col2:
                        if "MACD" in show_indicators:
                            # Calculate MACD
                            exp1 = hist['Close'].ewm(span=12).mean()
                            exp2 = hist['Close'].ewm(span=26).mean()
                            macd = exp1 - exp2
                            signal = macd.ewm(span=9).mean()
                            histogram = macd - signal
                            
                            current_macd = macd.iloc[-1] if not macd.empty else 0
                            current_signal = signal.iloc[-1] if not signal.empty else 0
                            
                            st.metric(
                                "MACD",
                                f"{current_macd:.2f}",
                                delta=f"{current_macd - current_signal:.2f}"
                            )
                            
                            # MACD interpretation
                            if current_macd > current_signal:
                                st.success("Bullish")
                            else:
                                st.error("Bearish")
                    
                    with col3:
                        # Volume Analysis
                        if "Volume" in show_indicators:
                            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
                            current_volume = hist['Volume'].iloc[-1]
                            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                            
                            st.metric(
                                "Volume Ratio",
                                f"{volume_ratio:.1f}x",
                                delta=f"{((current_volume - avg_volume) / avg_volume * 100):.1f}%" if avg_volume > 0 else "0%"
                            )
                            
                            if volume_ratio > 1.5:
                                st.success("High Volume")
                            elif volume_ratio < 0.5:
                                st.warning("Low Volume")
                            else:
                                st.info("Normal Volume")
                    
                    with col4:
                        # Price Change
                        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
                        price_change_pct = (price_change / hist['Close'].iloc[-2] * 100) if len(hist) > 1 and hist['Close'].iloc[-2] > 0 else 0
                        
                        st.metric(
                            "Price Change",
                            f"{price_change:.2f}",
                            delta=f"{price_change_pct:.2f}%"
                        )
                        
                        if price_change > 0:
                            st.success("Gaining")
                        else:
                            st.error("Declining")
                    
                    # Technical Analysis Summary (only if we have data)
                    if show_indicators and len(hist) > 0:
                        st.markdown("---")
                        st.subheader("📊 Technical Analysis Summary")
                        
                        # Build summary based on available indicators
                        tech_summary_parts = [
                            f"Stock: {selected_ticker_tech}",
                            f"Current Price: ₹{hist['Close'].iloc[-1]:.2f}",
                            f"Time Period: {time_period}"
                        ]
                        
                        # Add indicator-specific data only if calculated
                        if "RSI" in show_indicators:
                            delta = hist['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            current_rsi = rsi.iloc[-1] if not rsi.empty else 0
                            tech_summary_parts.append(f"RSI (14): {current_rsi:.1f}")
                        
                        if "MACD" in show_indicators:
                            exp1 = hist['Close'].ewm(span=12).mean()
                            exp2 = hist['Close'].ewm(span=26).mean()
                            macd = exp1 - exp2
                            signal = macd.ewm(span=9).mean()
                            current_macd = macd.iloc[-1] if not macd.empty else 0
                            current_signal = signal.iloc[-1] if not signal.empty else 0
                            tech_summary_parts.append(f"MACD: {current_macd:.2f} (Signal: {current_signal:.2f})")
                        
                        if "Volume" in show_indicators and 'Volume' in hist.columns:
                            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
                            current_volume = hist['Volume'].iloc[-1]
                            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                            tech_summary_parts.append(f"Volume Ratio: {volume_ratio:.1f}x")
                        
                        # Price change
                        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
                        price_change_pct = (price_change / hist['Close'].iloc[0] * 100) if hist['Close'].iloc[0] > 0 else 0
                        tech_summary_parts.append(f"Price Change: {price_change_pct:+.2f}%")
                        
                        tech_summary = "\n".join(tech_summary_parts)
                        
                        # Generate AI-powered technical analysis
                        try:
                            import openai
                            openai.api_key = st.secrets["api_keys"]["open_ai"]
                            
                            response = openai.chat.completions.create(
                                model="gpt-5",  # Upgraded to GPT-5 for better results
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a professional technical analyst. Provide a brief technical analysis "
                                            "summary based on the indicators provided. Focus on key signals and trading "
                                            "implications. Use emojis and be concise."
                                        )
                                    },
                                    {"role": "user", "content": tech_summary},
                                ]
                                # Note: GPT-5 only supports default temperature (1)
                                # Removed max_completion_tokens to allow full response
                            )
                            
                            if response and response.choices and len(response.choices) > 0:
                                ai_tech_analysis = response.choices[0].message.content
                            if ai_tech_analysis and ai_tech_analysis.strip():
                                st.markdown(
                                    f'<div class="ai-response-box"><strong>🤖 Technical Analysis:</strong><br><br>{ai_tech_analysis}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                # Show fallback if AI response is empty
                                st.info("📊 **Technical Indicators Summary:**\n\n" + tech_summary.replace("\n", "\n- "))
                            
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate AI analysis: {str(e)[:100]}")
                            # Show fallback summary even if AI fails
                            st.info("📊 **Technical Indicators Summary:**\n\n" + tech_summary.replace("\n", "\n- "))
                
                else:
                    st.warning(f"⚠️ No sufficient data available for {selected_ticker_tech}")
                    st.info("""
                    **Possible reasons:**
                    - Stock may be delisted or suspended
                    - Ticker format may be incorrect
                    - Insufficient trading history
                    
                    **Suggestions:**
                    - Try a different stock from your portfolio
                    - Check if the stock is actively trading
                    - Use a shorter time period (1M or 3M)
                    """)
                    
            except Exception as e:
                st.error(f"⚠️ Error fetching technical data: {str(e)[:100]}")
                st.info("Try selecting a different stock or time period.")
    
    with tab6:
        st.subheader("⚡ Risk Metrics")
        st.caption("Advanced risk analysis and portfolio risk management")
        
        # Risk Analysis Controls
        col1, col2 = st.columns(2)
        
        with col1:
            risk_timeframe = st.selectbox(
                "Risk Analysis Timeframe:",
                ["1M", "3M", "6M", "1Y"],
                index=2,
                key="risk_timeframe"
            )
        
        with col2:
            confidence_level = st.selectbox(
                "VaR Confidence Level:",
                ["90%", "95%", "99%"],
                index=1,
                key="var_confidence"
            )
        
        # Calculate Risk Metrics
        try:
            # Prepare risk data with robust null handling
            risk_data = []
            total_portfolio_value = 0
            
            # First pass: calculate total portfolio value
            for holding in holdings:
                try:
                    current_price = holding.get('current_price')
                    if current_price is None or current_price == 0:
                        current_price = holding.get('average_price', 0)
                    
                    if current_price is None or current_price == 0:
                        continue  # Skip holdings with no price data
                    
                    quantity = holding.get('total_quantity', 0)
                    if quantity is None or quantity == 0:
                        continue
                    
                    current_value = float(current_price) * float(quantity)
                    total_portfolio_value += current_value
                except (TypeError, ValueError):
                    continue
            
            # Second pass: build risk data
            for holding in holdings:
                try:
                    current_price = holding.get('current_price')
                    if current_price is None or current_price == 0:
                        current_price = holding.get('average_price', 0)
                    
                    if current_price is None or current_price == 0:
                        continue
                    
                    quantity = holding.get('total_quantity', 0)
                    avg_price = holding.get('average_price', 0)
                    
                    if quantity is None or quantity == 0 or avg_price is None or avg_price == 0:
                        continue
                    
                    current_value = float(current_price) * float(quantity)
                    investment = float(quantity) * float(avg_price)
                    pnl = current_value - investment
                    pnl_pct = (pnl / investment * 100) if investment > 0 else 0
                    weight = (current_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                    
                    risk_data.append({
                        'ticker': holding.get('ticker', 'Unknown'),
                        'stock_name': holding.get('stock_name', 'Unknown'),
                        'current_value': current_value,
                        'pnl_pct': pnl_pct,
                        'weight': weight
                    })
                except (TypeError, ValueError, KeyError) as e:
                    continue
            
            df_risk = pd.DataFrame(risk_data)
            
            if not df_risk.empty:
                # Risk Metrics Dashboard
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Portfolio Volatility (simplified)
                    portfolio_volatility = df_risk['pnl_pct'].std()
                    st.metric(
                        "Portfolio Volatility",
                        f"{portfolio_volatility:.2f}%",
                        help="Standard deviation of returns"
                    )
                
                with col2:
                    # Value at Risk (VaR) - Simplified calculation
                    var_95 = np.percentile(df_risk['pnl_pct'], 5)  # 5th percentile for 95% VaR
                    st.metric(
                        f"VaR ({confidence_level})",
                        f"{var_95:.2f}%",
                        help="Maximum expected loss with given confidence"
                    )
                
                with col3:
                    # Maximum Drawdown
                    max_drawdown = df_risk['pnl_pct'].min()
                    st.metric(
                        "Max Drawdown",
                        f"{max_drawdown:.2f}%",
                        help="Maximum peak-to-trough decline"
                    )
                
                with col4:
                    # Sharpe Ratio (simplified)
                    risk_free_rate = 6.0  # Assume 6% risk-free rate
                    excess_return = df_risk['pnl_pct'].mean() - risk_free_rate
                    sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
                    st.metric(
                        "Sharpe Ratio",
                        f"{sharpe_ratio:.2f}",
                        help="Risk-adjusted return measure"
                    )
                
                # Risk Analysis Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk-Return Scatter Plot
                    fig_risk_return = px.scatter(
                        df_risk,
                        x='pnl_pct',
                        y='current_value',
                        size='weight',
                        color='pnl_pct',
                        hover_data=['ticker', 'stock_name'],
                        title="Risk-Return Analysis",
                        labels={'pnl_pct': 'Return (%)', 'current_value': 'Current Value (₹)'},
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_risk_return.update_layout(height=400)
                    st.plotly_chart(fig_risk_return, use_container_width=True)
                
                with col2:
                    # Portfolio Concentration
                    fig_concentration = px.pie(
                        df_risk,
                        values='current_value',
                        names='ticker',
                        title="Portfolio Concentration",
                        hole=0.4
                    )
                    fig_concentration.update_layout(height=400)
                    st.plotly_chart(fig_concentration, use_container_width=True)
                
                # Risk Summary Table
                st.markdown("---")
                st.subheader("📊 Individual Stock Risk Analysis")
                
                risk_summary = df_risk.copy()
                risk_summary['Risk Level'] = risk_summary['pnl_pct'].apply(
                    lambda x: 'High' if abs(x) > 20 else 'Medium' if abs(x) > 10 else 'Low'
                )
                risk_summary['Recommendation'] = risk_summary['pnl_pct'].apply(
                    lambda x: 'Hold' if x > 5 else 'Review' if x < -10 else 'Monitor'
                )
                
                st.dataframe(
                    risk_summary[['ticker', 'stock_name', 'current_value', 'pnl_pct', 'weight', 'Risk Level', 'Recommendation']].style.format({
                        'current_value': '₹{:,.0f}',
                        'pnl_pct': '{:+.2f}%',
                        'weight': '{:.1%}'
                    }),
                    use_container_width=True
                )
                
                # AI Risk Analysis
                try:
                    import openai
                    openai.api_key = st.secrets["api_keys"]["open_ai"]
                    
                    risk_summary_text = f"""
                    Portfolio Risk Analysis:
                    - Portfolio Volatility: {portfolio_volatility:.2f}%
                    - VaR (95%): {var_95:.2f}%
                    - Max Drawdown: {max_drawdown:.2f}%
                    - Sharpe Ratio: {sharpe_ratio:.2f}
                    - Number of Holdings: {len(holdings)}
                    - Top Risk Holdings: {', '.join(df_risk.nsmallest(3, 'pnl_pct')['ticker'].tolist())}
                    """
                    
                    response = openai.chat.completions.create(
                        model="gpt-5-mini",  # GPT-5-mini for faster risk analysis
                        messages=[
                            {"role": "system", "content": "You are a professional risk analyst. Analyze the portfolio risk metrics and provide actionable risk management recommendations. Focus on diversification, position sizing, and risk mitigation strategies. Use emojis and be practical."},
                            {"role": "user", "content": risk_summary_text}
                        ],
                        max_completion_tokens=400,
                        # Note: GPT-5-mini only supports default temperature (1)
                    )
                    
                    ai_risk_analysis = response.choices[0].message.content
                    st.markdown(f'<div class="ai-response-box"><strong>🤖 Risk Analysis:</strong><br><br>{ai_risk_analysis}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.warning(f"Could not generate AI risk analysis: {str(e)[:50]}")
            
        except Exception as e:
            st.error(f"Error calculating risk metrics: {str(e)[:100]}")
    
    with tab7:
        st.subheader("🔄 Compare Holdings")
        st.caption("Compare performance across different dimensions")
        
        # Prepare comparison data
        comparison_data = []
        if not holdings:
            st.warning("⚠️ No holdings found. Please upload transaction files first.")
            return
        
        for holding in holdings:
            try:
                current_price = holding.get('current_price')
                if current_price is None or current_price == 0:
                    current_price = holding.get('average_price', 0)
                
                # Ensure we have valid numeric values
                if current_price is None or current_price == 0:
                    continue  # Skip holdings with no price data
                
                current_value = float(current_price) * float(holding['total_quantity'])
                investment = float(holding['total_quantity']) * float(holding['average_price'])
                pnl = current_value - investment
                pnl_pct = (pnl / investment * 100) if investment > 0 else 0
                
                stars, grade, rating = get_performance_rating(pnl_pct)
                
                comparison_data.append({
                    'ticker': holding['ticker'],
                    'stock_name': holding['stock_name'],
                    'asset_type': holding.get('asset_type', 'Unknown'),
                    'channel': holding.get('channel', 'Unknown'),
                    'sector': holding.get('sector', 'Unknown'),
                    'investment': investment,
                    'current_value': current_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'rating': stars,
                    'grade': grade
                })
            except (ValueError, TypeError) as e:
                st.warning(f"⚠️ Skipping holding {holding.get('ticker', 'Unknown')}: Invalid data")
                continue
        
        df_compare = pd.DataFrame(comparison_data)
        
        if df_compare.empty:
            st.warning("⚠️ No valid holdings data available for comparison. Please check if prices are being fetched correctly.")
            # Debug information
            #st.caption(f"Debug: Found {len(holdings)} holdings, but none had valid price data")
            if holdings:
                #st.caption("Sample holding data:")
                sample_holding = holdings[0]
                #st.caption(f"Ticker: {sample_holding.get('ticker')}, Current Price: {sample_holding.get('current_price')}, Average Price: {sample_holding.get('average_price')}")
            return
        
        # Debug information
        #st.caption(f"✅ Loaded {len(df_compare)} holdings for comparison")
        
        # Enhanced Comparison Options with Better UI
        st.markdown("### 📊 Select Comparison Type")
        
        comparison_type = st.radio(
            "Compare by:",
            ["By Channel", "By Sector", "By Asset Type", "By Individual Holdings", "Multi-Comparison"],
            horizontal=True,
            help="Choose how you want to compare your holdings"
        )
        
        # Add some spacing
        st.markdown("---")
        
        if comparison_type == "By Channel":
            st.markdown("#### 📡 Channel Comparison")
            
            # Get unique channels with counts
            channels = df_compare['channel'].unique().tolist()
            channel_counts = df_compare['channel'].value_counts().to_dict()
            
            # Enhanced multi-select for channels with search
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Initialize session state for channels
                if 'selected_channels_state' not in st.session_state:
                    st.session_state.selected_channels_state = channels[:min(3, len(channels))]
                
                selected_channels = st.multiselect(
                    "🔍 Select channels to compare:",
                    channels,
                    default=st.session_state.selected_channels_state,
                    help="Search and select multiple channels. Use Ctrl+Click for multiple selections.",
                    placeholder="Type to search channels...",
                    key="channel_comparison_multiselect"
                )
                
                # Update session state
                st.session_state.selected_channels_state = selected_channels
            
            with col2:
                st.markdown("**📊 Available Channels:**")
                for channel in channels:
                    count = channel_counts.get(channel, 0)
                    st.caption(f"• {channel}: {count} holdings")
            
            # Quick select buttons
            if len(channels) > 1:
                st.markdown("**⚡ Quick Select:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Select All", key="select_all_channels"):
                        st.session_state.selected_channels_state = channels
                        st.rerun()
                with col2:
                    if st.button("Select Top 3", key="select_top3_channels"):
                        top_channels = df_compare.groupby('channel')['current_value'].sum().nlargest(3).index.tolist()
                        st.session_state.selected_channels_state = top_channels
                        st.rerun()
                with col3:
                    if st.button("Clear All", key="clear_channels"):
                        st.session_state.selected_channels_state = []
                        st.rerun()
            
            if selected_channels:
                # Filter data
                channel_comparison = df_compare[df_compare['channel'].isin(selected_channels)].groupby('channel').agg({
                    'investment': 'sum',
                    'current_value': 'sum',
                    'pnl': 'sum'
                }).reset_index()
                
                channel_comparison['pnl_pct'] = (channel_comparison['pnl'] / channel_comparison['investment'] * 100)
                channel_comparison['rating'] = channel_comparison['pnl_pct'].apply(lambda x: get_performance_rating(x)[0])
                channel_comparison['grade'] = channel_comparison['pnl_pct'].apply(lambda x: get_performance_rating(x)[1])
                
                # Comparison Chart
                fig_channel_compare = go.Figure()
                
                fig_channel_compare.add_trace(go.Bar(
                    name='Investment',
                    x=channel_comparison['channel'],
                    y=channel_comparison['investment'],
                    marker_color='lightblue'
                ))
                
                fig_channel_compare.add_trace(go.Bar(
                    name='Current Value',
                    x=channel_comparison['channel'],
                    y=channel_comparison['current_value'],
                    marker_color='lightgreen'
                ))
                
                fig_channel_compare.update_layout(
                    title="Channel Investment vs Current Value",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_channel_compare, use_container_width=True)
                
                # Performance comparison
                fig_perf = px.bar(
                    channel_comparison,
                    x='channel',
                    y='pnl_pct',
                    title="Channel Performance Comparison (%)",
                    color='pnl_pct',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    text='pnl_pct'
                )
                fig_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Detailed table
                st.dataframe(channel_comparison.style.format({
                    'investment': '₹{:,.0f}',
                    'current_value': '₹{:,.0f}',
                    'pnl': '₹{:,.0f}',
                    'pnl_pct': '{:+.2f}%'
                }), use_container_width=True)
        
        elif comparison_type == "By Sector":
            st.markdown("#### 🏢 Sector Comparison")
            
            # Get unique sectors (from holdings data we need to fetch sector info)
            # For now, we'll use asset_type as a proxy, but you can enhance this by adding sector to holdings
            st.info("📊 Sector analysis based on stock data")
            
            # Get holdings with sector information
            sector_holdings = {}
            for holding in holdings:
                sector = holding.get('sector', holding.get('asset_type', 'Unknown'))
                current_price = holding.get('current_price')
                if current_price is None or current_price == 0:
                    current_price = holding.get('average_price', 0)
                current_value = float(current_price) * float(holding['total_quantity'])
                investment = float(holding['total_quantity']) * float(holding['average_price'])
                pnl = current_value - investment
                
                if sector not in sector_holdings:
                    sector_holdings[sector] = {'investment': 0, 'current_value': 0, 'pnl': 0, 'count': 0}
                
                sector_holdings[sector]['investment'] += investment
                sector_holdings[sector]['current_value'] += current_value
                sector_holdings[sector]['pnl'] += pnl
                sector_holdings[sector]['count'] += 1
            
            # Create sector comparison dataframe
            sector_data = []
            for sector, data in sector_holdings.items():
                pnl_pct = (data['pnl'] / data['investment'] * 100) if data['investment'] > 0 else 0
                stars, grade, rating = get_performance_rating(pnl_pct)
                
                sector_data.append({
                    'Sector': sector,
                    'Holdings': data['count'],
                    'Rating': stars,
                    'Grade': grade,
                    'Investment': data['investment'],
                    'Current Value': data['current_value'],
                    'P&L': data['pnl'],
                    'P&L %': pnl_pct
                })
            
            df_sectors = pd.DataFrame(sector_data).sort_values('P&L %', ascending=False)
            
            # Sector comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_sector_pie = px.pie(
                    df_sectors,
                    values='Current Value',
                    names='Sector',
                    title="Portfolio Allocation by Sector"
                )
                st.plotly_chart(fig_sector_pie, use_container_width=True)
            
            with col2:
                fig_sector_perf = px.bar(
                    df_sectors,
                    x='Sector',
                    y='P&L %',
                    title="Sector Performance (%)",
                    color='P&L %',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    text='P&L %'
                )
                fig_sector_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_sector_perf, use_container_width=True)
            
            # Detailed sector table
            st.dataframe(df_sectors.style.format({
                'Investment': '₹{:,.0f}',
                'Current Value': '₹{:,.0f}',
                'P&L': '₹{:,.0f}',
                'P&L %': '{:+.2f}%'
            }), use_container_width=True)
        
        elif comparison_type == "By Asset Type":
            st.markdown("#### 💼 Asset Type Comparison")
            
            # Get unique asset types with counts and values
            asset_types = df_compare['asset_type'].unique().tolist()
            type_counts = df_compare['asset_type'].value_counts().to_dict()
            type_values = df_compare.groupby('asset_type')['current_value'].sum().to_dict()
            
            # Enhanced multi-select for asset types
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Initialize session state for asset types
                if 'selected_types_state' not in st.session_state:
                    st.session_state.selected_types_state = asset_types[:min(3, len(asset_types))]
                
                selected_types = st.multiselect(
                    "🔍 Select asset types to compare:",
                    asset_types,
                    default=st.session_state.selected_types_state,
                    help="Search and select multiple asset types. Use Ctrl+Click for multiple selections.",
                    placeholder="Type to search asset types...",
                    key="asset_type_comparison_multiselect"
                )
                
                # Update session state
                st.session_state.selected_types_state = selected_types
            
            with col2:
                st.markdown("**📊 Available Asset Types:**")
                for asset_type in asset_types:
                    count = type_counts.get(asset_type, 0)
                    value = type_values.get(asset_type, 0)
                    st.caption(f"• {asset_type}: {count} holdings (₹{value:,.0f})")
            
            # Quick select buttons for asset types
            if len(asset_types) > 1:
                st.markdown("**⚡ Quick Select:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Select All", key="select_all_types"):
                        st.session_state.selected_types = asset_types
                        st.rerun()
                with col2:
                    if st.button("Select Top 3", key="select_top3_types"):
                        top_types = df_compare.groupby('asset_type')['current_value'].sum().nlargest(3).index.tolist()
                        st.session_state.selected_types = top_types
                        st.rerun()
                with col3:
                    if st.button("Clear All", key="clear_types"):
                        st.session_state.selected_types = []
                        st.rerun()
            
            if selected_types:
                # Filter data
                type_comparison = df_compare[df_compare['asset_type'].isin(selected_types)].groupby('asset_type').agg({
                    'investment': 'sum',
                    'current_value': 'sum',
                    'pnl': 'sum'
                }).reset_index()
                
                type_comparison['pnl_pct'] = (type_comparison['pnl'] / type_comparison['investment'] * 100)
                type_comparison['rating'] = type_comparison['pnl_pct'].apply(lambda x: get_performance_rating(x)[0])
                type_comparison['grade'] = type_comparison['pnl_pct'].apply(lambda x: get_performance_rating(x)[1])
                
                # Comparison Chart
                fig_type_compare = go.Figure()
                
                fig_type_compare.add_trace(go.Bar(
                    name='Investment',
                    x=type_comparison['asset_type'],
                    y=type_comparison['investment'],
                    marker_color='lightblue'
                ))
                
                fig_type_compare.add_trace(go.Bar(
                    name='Current Value',
                    x=type_comparison['asset_type'],
                    y=type_comparison['current_value'],
                    marker_color='lightgreen'
                ))
                
                fig_type_compare.update_layout(
                    title="Asset Type Investment vs Current Value",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_type_compare, use_container_width=True)
                
                # Performance comparison
                fig_perf = px.bar(
                    type_comparison,
                    x='asset_type',
                    y='pnl_pct',
                    title="Asset Type Performance Comparison (%)",
                    color='pnl_pct',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    text='pnl_pct'
                )
                fig_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Detailed table
                st.dataframe(type_comparison.style.format({
                    'investment': '₹{:,.0f}',
                    'current_value': '₹{:,.0f}',
                    'pnl': '₹{:,.0f}',
                    'pnl_pct': '{:+.2f}%'
                }), use_container_width=True)
        
        elif comparison_type == "By Individual Holdings":
            st.markdown("#### 📈 Individual Holdings Comparison")
            
            # Create enhanced holdings list with performance info
            holdings_list = []
            for _, row in df_compare.iterrows():
                pnl_pct = row['pnl_pct']
                emoji = "🚀" if pnl_pct > 10 else "📈" if pnl_pct > 0 else "📉" if pnl_pct < -5 else "➡️"
                holdings_list.append(f"{emoji} {row['ticker']} - {row['stock_name']} ({pnl_pct:+.1f}%)")
            
            # Enhanced multi-select for holdings
            # Initialize session state for holdings
            if 'selected_holdings_state' not in st.session_state:
                st.session_state.selected_holdings_state = holdings_list[:min(3, len(holdings_list))]
            
            selected_holdings = st.multiselect(
                "🔍 Select holdings to compare (up to 10):",
                holdings_list,
                default=st.session_state.selected_holdings_state,
                help="Search and select multiple holdings. Use Ctrl+Click for multiple selections.",
                placeholder="Type to search holdings...",
                key="holdings_comparison_multiselect"
            )
            
            # Update session state
            st.session_state.selected_holdings_state = selected_holdings
            
            if selected_holdings:
                # Extract tickers from selection (handle emoji and formatting)
                selected_tickers = []
                for h in selected_holdings:
                    # Split by space and get the part after emoji (ticker)
                    parts = h.split(' ')
                    if len(parts) >= 2:
                        ticker = parts[1]  # Get ticker (e.g., "HCLTECH.NS")
                        selected_tickers.append(ticker)
                
                holding_comparison = df_compare[df_compare['ticker'].isin(selected_tickers)]
                
                # Historical Performance Line Chart
                fig_holdings = go.Figure()
                
                # Get historical data for each selected holding
                for _, holding in holding_comparison.iterrows():
                    ticker = holding['ticker']
                    stock_name = holding['stock_name']
                    
                    # Get stock_id from the original holdings data
                    stock_id = None
                    for h in holdings:
                        if h['ticker'] == ticker:
                            stock_id = h['stock_id']
                            break
                    
                    if not stock_id:
                        continue  # Skip if stock_id not found
                    
                    # Get historical prices for this stock_id
                    historical_prices = db.get_historical_prices_for_stock_silent(stock_id)
                    
                    # Debug information
                    #st.caption(f"Debug: {ticker} - Found {len(historical_prices) if historical_prices else 0} historical prices")
                    
                    if historical_prices and len(historical_prices) > 0:
                        # Sort by date
                        historical_prices.sort(key=lambda x: x['price_date'])
                        
                        # Prepare data for line chart
                        dates = [price['price_date'] for price in historical_prices]
                        prices = [price['price'] for price in historical_prices]
                        
                        # Calculate percentage change from first price
                        if len(prices) > 0:
                            first_price = prices[0]
                            pct_changes = [((price - first_price) / first_price) * 100 for price in prices]
                            
                            # Add line trace
                            fig_holdings.add_trace(go.Scatter(
                                x=dates,
                                y=pct_changes,
                                mode='lines+markers',
                                name=f"{ticker} - {stock_name[:20]}{'...' if len(stock_name) > 20 else ''}",
                                line=dict(width=2),
                                marker=dict(size=4),
                                hovertemplate=f"<b>{ticker}</b><br>" +
                                            "Date: %{x}<br>" +
                                            "Price: ₹%{customdata:.2f}<br>" +
                                            "Change: %{y:.2f}%<br>" +
                                            "<extra></extra>",
                                customdata=prices
                            ))
                
                # Update layout for line chart
                fig_holdings.update_layout(
                    title="📈 Historical Performance Comparison - Selected Holdings",
                    xaxis_title="Date",
                    yaxis_title="Price Change (%)",
                    height=500,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        zeroline=True,
                        zerolinecolor='rgba(128,128,128,0.5)'
                    )
                )
                
                if fig_holdings.data:
                    st.plotly_chart(fig_holdings, use_container_width=True)
                else:
                    st.warning("⚠️ No historical data available for selected holdings. This could be because:")
                    st.caption("• Historical prices haven't been fetched yet")
                    st.caption("• The holdings don't have price history in the database")
                    st.caption("• Try running 'Update Prices' to fetch historical data")
                
                # Add summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Holdings Selected", len(selected_holdings))
                with col2:
                    avg_performance = holding_comparison['pnl_pct'].mean()
                    st.metric("📈 Average Performance", f"{avg_performance:+.1f}%")
                with col3:
                    best_performer = holding_comparison.loc[holding_comparison['pnl_pct'].idxmax()]
                    st.metric("🏆 Best Performer", f"{best_performer['ticker']} ({best_performer['pnl_pct']:+.1f}%)")
                
                # Detailed comparison table
                comparison_table = holding_comparison[['ticker', 'stock_name', 'rating', 'grade', 'investment', 'current_value', 'pnl', 'pnl_pct']]
                st.dataframe(comparison_table.style.format({
                    'investment': '₹{:,.0f}',
                    'current_value': '₹{:,.0f}',
                    'pnl': '₹{:,.0f}',
                    'pnl_pct': '{:+.2f}%'
                }), use_container_width=True)
        
        else:  # Multi-Comparison
            st.markdown("#### 🔀 Multi-Dimensional Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Select channel
                channels = ['All'] + df_compare['channel'].unique().tolist()
                selected_channel = st.selectbox("Filter by Channel:", channels)
            
            with col2:
                # Select asset type
                asset_types = ['All'] + df_compare['asset_type'].unique().tolist()
                selected_type = st.selectbox("Filter by Asset Type:", asset_types)
            
            # Apply filters
            filtered_df = df_compare.copy()
            if selected_channel != 'All':
                filtered_df = filtered_df[filtered_df['channel'] == selected_channel]
            if selected_type != 'All':
                filtered_df = filtered_df[filtered_df['asset_type'] == selected_type]
            
            if not filtered_df.empty:
                # Summary metrics
                total_inv = filtered_df['investment'].sum()
                total_curr = filtered_df['current_value'].sum()
                total_pnl = filtered_df['pnl'].sum()
                total_pnl_pct = (total_pnl / total_inv * 100) if total_inv > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Investment", f"₹{total_inv:,.0f}")
                with col2:
                    st.metric("Current Value", f"₹{total_curr:,.0f}")
                with col3:
                    st.metric("P&L", f"₹{total_pnl:,.0f}")
                with col4:
                    st.metric("P&L %", f"{total_pnl_pct:+.1f}%")
                
                # Holdings in this filter
                st.markdown(f"**{len(filtered_df)} holdings match your filters**")
                
                # Top performers in this filter
                top_3 = filtered_df.nlargest(3, 'pnl_pct')
                st.markdown("**🏆 Top 3 Performers:**")
                for _, row in top_3.iterrows():
                    st.write(f"{row['rating']} {row['ticker']} - {row['stock_name'][:30]}: {row['pnl_pct']:+.1f}%")
                
                # Chart
                fig_multi = px.scatter(
                    filtered_df,
                    x='investment',
                    y='pnl_pct',
                    size='current_value',
                    color='asset_type',
                    hover_data=['ticker', 'stock_name', 'channel'],
                    title="Investment vs Performance (bubble size = current value)"
                )
                st.plotly_chart(fig_multi, use_container_width=True)
            else:
                st.info("No holdings match the selected filters")

def channel_analytics_page():
    """Dedicated channel analytics dashboard"""
    st.header("📡 Channel Analytics")

    user = st.session_state.user
    holdings = get_cached_holdings(user['id'])

    if not holdings:
        st.info("No holdings available. Upload transactions to view channel analytics.")
        return

    df = pd.DataFrame(holdings)
    if df.empty:
        st.info("Channel analytics unavailable because holdings data could not be processed.")
        return

    for col in ['channel', 'total_quantity', 'average_price', 'current_price', 'investment', 'current_value', 'pnl']:
        if col not in df.columns:
            df[col] = 0

    df['channel'] = df['channel'].fillna('Unknown').replace('', 'Unknown').astype(str)
    df['total_quantity'] = pd.to_numeric(df['total_quantity'], errors='coerce').fillna(0.0)
    df['average_price'] = pd.to_numeric(df['average_price'], errors='coerce').fillna(0.0)
    df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce').fillna(0.0)
    df['investment'] = pd.to_numeric(df['investment'], errors='coerce').fillna(0.0)
    df['current_value'] = pd.to_numeric(df['current_value'], errors='coerce').fillna(0.0)
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)

    df['effective_current_price'] = df['current_price'].where(df['current_price'] > 0, df['average_price'])
    df.loc[df['investment'] == 0, 'investment'] = df['total_quantity'] * df['average_price']
    df.loc[df['current_value'] == 0, 'current_value'] = df['total_quantity'] * df['effective_current_price']
    df.loc[df['pnl'] == 0, 'pnl'] = df['current_value'] - df['investment']

    df['pnl_pct'] = df.apply(
        lambda row: (row['pnl'] / row['investment'] * 100) if row['investment'] > 0 else 0.0,
        axis=1
    )

    channel_summary = df.groupby('channel').agg(
        total_positions=('ticker', 'count') if 'ticker' in df.columns else ('channel', 'count'),
        unique_assets=('ticker', 'nunique') if 'ticker' in df.columns else ('channel', 'count'),
        total_investment=('investment', 'sum'),
        current_value=('current_value', 'sum'),
        total_pnl=('pnl', 'sum')
    ).reset_index()

    channel_summary['pnl_pct'] = channel_summary.apply(
        lambda row: (row['total_pnl'] / row['total_investment'] * 100) if row['total_investment'] > 0 else 0.0,
        axis=1
    )
    total_current_value = channel_summary['current_value'].sum()
    if total_current_value > 0:
        channel_summary['allocation_pct'] = channel_summary['current_value'] / total_current_value * 100
    else:
        channel_summary['allocation_pct'] = 0.0

    total_channels = channel_summary.shape[0]
    total_value = channel_summary['current_value'].sum()
    total_investment = channel_summary['total_investment'].sum()
    total_pnl = channel_summary['total_pnl'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Channels", total_channels)
    col2.metric("Total Investment", f"₹{total_investment:,.0f}")
    col3.metric("Current Value", f"₹{total_value:,.0f}")
    col4.metric("Aggregate P&L", f"₹{total_pnl:,.0f}")

    if total_investment > 0:
        st.metric("Portfolio P&L %", f"{(total_pnl / total_investment) * 100:,.2f}%")

    st.markdown("### Channel Performance Overview")
    st.dataframe(
        channel_summary.assign(
            total_investment=lambda df_: df_['total_investment'].map(lambda v: f"₹{v:,.0f}"),
            current_value=lambda df_: df_['current_value'].map(lambda v: f"₹{v:,.0f}"),
            total_pnl=lambda df_: df_['total_pnl'].map(lambda v: f"₹{v:,.0f}"),
            pnl_pct=lambda df_: df_['pnl_pct'].map(lambda v: f"{v:+.2f}%"),
            allocation_pct=lambda df_: df_['allocation_pct'].map(lambda v: f"{v:.1f}%")
        ),
        use_container_width=True
    )

    st.markdown("### Allocation by Channel")
    try:
        fig_allocation = px.pie(
            channel_summary,
            values='current_value',
            names='channel',
            title="Current Value Distribution by Channel"
        )
        st.plotly_chart(fig_allocation, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render allocation chart: {exc}")

    st.markdown("### Channel P&L % Comparison")
    try:
        channel_summary_sorted = channel_summary.sort_values('pnl_pct', ascending=False).reset_index(drop=True)
        fig_pnl = px.bar(
            channel_summary_sorted,
            x='channel',
            y='pnl_pct',
            text=channel_summary_sorted['pnl_pct'].map(lambda v: f"{v:+.2f}%"),
            title="Channel Performance (Total P&L %)",
        )
        fig_pnl.update_traces(textposition='outside', texttemplate='%{text}')
        fig_pnl.update_layout(
            yaxis_title="Total P&L %",
            yaxis=dict(zeroline=True, zerolinecolor='rgba(128,128,128,0.5)'),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render total performance chart: {exc}")

    try:
        weekly_history = db.get_channel_weekly_history(user['id'])
        if weekly_history:
            tab_weeks = st.tabs(["All Channels"] + sorted(list({entry['channel'] for entry in weekly_history})))

            with tab_weeks[0]:
                all_df = pd.DataFrame([entry for entry in weekly_history])
                fig_all = px.line(
                    all_df,
                    x='date',
                    y='pnl_pct',
                    color='channel',
                    title="Channel Performance (Last 52 Weeks)",
                )
                fig_all.update_layout(
                    yaxis_title="Weekly P&L %",
                    xaxis_title="Date",
                    hovermode='x unified',
                )
                st.plotly_chart(fig_all, use_container_width=True)

            channels = sorted(list({entry['channel'] for entry in weekly_history}))
            for idx, channel_name in enumerate(channels, start=1):
                with tab_weeks[idx]:
                    channel_df = pd.DataFrame(
                        [entry for entry in weekly_history if entry['channel'] == channel_name]
                    )
                    if not channel_df.empty:
                        st.write(f"**{channel_name}** — 52-Week Trend (Select weeks as needed)")
                        channel_df['date'] = pd.to_datetime(channel_df['date'])
                        channel_df = channel_df.sort_values('date')
                        fig_channel = px.line(
                            channel_df,
                            x='date',
                            y='pnl_pct',
                            markers=True,
                            title=f"{channel_name} — Weekly P&L %",
                        )
                        fig_channel.update_layout(
                            yaxis_title="Weekly P&L %",
                            xaxis_title="Date",
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig_channel, use_container_width=True)
                        st.dataframe(
                            channel_df[['date', 'pnl_pct', 'investment', 'current_value']].assign(
                                date=lambda df_: df_['date'].dt.strftime('%Y-%m-%d'),
                                pnl_pct=lambda df_: df_['pnl_pct'].map(lambda v: f"{v:+.2f}%"),
                                investment=lambda df_: df_['investment'].map(lambda v: f"₹{v:,.0f}"),
                                current_value=lambda df_: df_['current_value'].map(lambda v: f"₹{v:,.0f}"),
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.info(f"No weekly history available for {channel_name}.")
        else:
            st.info("52-week channel performance not available yet—weekly NAV history is still being gathered.")
    except Exception as exc:
        st.warning(f"Could not render 52-week performance chart: {exc}")

    st.markdown("### Channel Details")
    for _, row in channel_summary.sort_values('current_value', ascending=False).iterrows():
        channel_name = row['channel']
        with st.expander(f"{channel_name} • Current Value ₹{row['current_value']:,.0f}", expanded=False):
            channel_df = df[df['channel'] == channel_name].copy()
            channel_df_display = channel_df[
                [col for col in ['ticker', 'stock_name', 'asset_type', 'total_quantity', 'investment', 'current_value', 'pnl', 'pnl_pct']
                 if col in channel_df.columns]
            ].copy()

            numeric_columns = ['total_quantity', 'investment', 'current_value', 'pnl', 'pnl_pct']
            for col in numeric_columns:
                if col in channel_df_display.columns:
                    if col == 'pnl_pct':
                        channel_df_display[col] = channel_df_display[col].map(lambda v: f"{v:+.2f}%")
                    elif col == 'total_quantity':
                        channel_df_display[col] = channel_df_display[col].map(lambda v: f"{v:,.2f}")
                    else:
                        channel_df_display[col] = channel_df_display[col].map(lambda v: f"₹{v:,.2f}")

            st.dataframe(channel_df_display, use_container_width=True)

            asset_breakdown = channel_df.groupby('asset_type').agg(
                current_value=('current_value', 'sum'),
                investment=('investment', 'sum')
            ).reset_index()
            if not asset_breakdown.empty:
                total_channel_value = asset_breakdown['current_value'].sum()
                if total_channel_value > 0:
                    asset_breakdown['allocation_pct'] = asset_breakdown['current_value'] / total_channel_value * 100
                else:
                    asset_breakdown['allocation_pct'] = 0.0
                st.markdown("**Asset Allocation within Channel**")
                st.dataframe(
                    asset_breakdown.assign(
                        current_value=lambda df_: df_['current_value'].map(lambda v: f"₹{v:,.0f}"),
                        investment=lambda df_: df_['investment'].map(lambda v: f"₹{v:,.0f}"),
                        allocation_pct=lambda df_: df_['allocation_pct'].map(lambda v: f"{v:.1f}%")
                    ),
                    use_container_width=True
                )

    st.markdown("---")
    st.caption("Tip: Keep channel metadata (e.g., broker/platform names) consistent while uploading transactions to unlock richer analytics.")


def sector_analytics_page():
    """Dedicated sector analytics dashboard"""
    st.header("🏭 Sector Analytics")

    user = st.session_state.user
    holdings = get_cached_holdings(user['id'])

    if not holdings:
        st.info("No holdings available. Upload transactions to view sector analytics.")
        return

    df = pd.DataFrame(holdings)
    if df.empty:
        st.info("Sector analytics unavailable because holdings data could not be processed.")
        return

    for col in ['sector', 'total_quantity', 'average_price', 'current_price', 'investment', 'current_value', 'pnl']:
        if col not in df.columns:
            df[col] = 0

    df['sector'] = df['sector'].fillna('Unknown').replace('', 'Unknown').astype(str)
    df['total_quantity'] = pd.to_numeric(df['total_quantity'], errors='coerce').fillna(0.0)
    df['average_price'] = pd.to_numeric(df['average_price'], errors='coerce').fillna(0.0)
    df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce').fillna(0.0)
    df['investment'] = pd.to_numeric(df['investment'], errors='coerce').fillna(0.0)
    df['current_value'] = pd.to_numeric(df['current_value'], errors='coerce').fillna(0.0)
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)

    df['effective_current_price'] = df['current_price'].where(df['current_price'] > 0, df['average_price'])
    df.loc[df['investment'] == 0, 'investment'] = df['total_quantity'] * df['average_price']
    df.loc[df['current_value'] == 0, 'current_value'] = df['total_quantity'] * df['effective_current_price']
    df.loc[df['pnl'] == 0, 'pnl'] = df['current_value'] - df['investment']

    df['pnl_pct'] = df.apply(
        lambda row: (row['pnl'] / row['investment'] * 100) if row['investment'] > 0 else 0.0,
        axis=1
    )

    sector_summary = df.groupby('sector').agg(
        total_positions=('ticker', 'count') if 'ticker' in df.columns else ('sector', 'count'),
        unique_assets=('ticker', 'nunique') if 'ticker' in df.columns else ('sector', 'count'),
        total_investment=('investment', 'sum'),
        current_value=('current_value', 'sum'),
        total_pnl=('pnl', 'sum')
    ).reset_index()

    sector_summary['pnl_pct'] = sector_summary.apply(
        lambda row: (row['total_pnl'] / row['total_investment'] * 100) if row['total_investment'] > 0 else 0.0,
        axis=1
    )
    total_current_value = sector_summary['current_value'].sum()
    if total_current_value > 0:
        sector_summary['allocation_pct'] = sector_summary['current_value'] / total_current_value * 100
    else:
        sector_summary['allocation_pct'] = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sectors", sector_summary.shape[0])
    col2.metric("Total Investment", f"₹{sector_summary['total_investment'].sum():,.0f}")
    col3.metric("Current Value", f"₹{sector_summary['current_value'].sum():,.0f}")
    col4.metric("Aggregate P&L", f"₹{sector_summary['total_pnl'].sum():,.0f}")

    total_investment = sector_summary['total_investment'].sum()
    if total_investment > 0:
        st.metric("Portfolio P&L %", f"{(sector_summary['total_pnl'].sum() / total_investment) * 100:,.2f}%")

    st.markdown("### Sector Performance Overview")
    st.dataframe(
        sector_summary.assign(
            total_investment=lambda df_: df_['total_investment'].map(lambda v: f"₹{v:,.0f}"),
            current_value=lambda df_: df_['current_value'].map(lambda v: f"₹{v:,.0f}"),
            total_pnl=lambda df_: df_['total_pnl'].map(lambda v: f"₹{v:,.0f}"),
            pnl_pct=lambda df_: df_['pnl_pct'].map(lambda v: f"{v:+.2f}%"),
            allocation_pct=lambda df_: df_['allocation_pct'].map(lambda v: f"{v:.1f}%")
        ),
        use_container_width=True
    )

    st.markdown("### Allocation by Sector")
    try:
        fig_allocation = px.pie(
            sector_summary,
            values='current_value',
            names='sector',
            title="Current Value Distribution by Sector"
        )
        st.plotly_chart(fig_allocation, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render allocation chart: {exc}")

    st.markdown("### Sector P&L % Comparison")
    try:
        sector_summary_sorted = sector_summary.sort_values('pnl_pct', ascending=False).reset_index(drop=True)
        fig_pnl = px.bar(
            sector_summary_sorted,
            x='sector',
            y='pnl_pct',
            text=sector_summary_sorted['pnl_pct'].map(lambda v: f"{v:+.2f}%"),
            title="Sector Performance (Total P&L %)",
        )
        fig_pnl.update_traces(textposition='outside', texttemplate='%{text}')
        fig_pnl.update_layout(
            yaxis_title="Total P&L %",
            yaxis=dict(zeroline=True, zerolinecolor='rgba(128,128,128,0.5)'),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render total performance chart: {exc}")

    try:
        weekly_history = db.get_sector_weekly_history(user['id'])
        if weekly_history:
            tab_weeks = st.tabs(["All Sectors"] + sorted(list({entry['sector'] for entry in weekly_history})))

            with tab_weeks[0]:
                all_df = pd.DataFrame([entry for entry in weekly_history])
                fig_all = px.line(
                    all_df,
                    x='date',
                    y='pnl_pct',
                    color='sector',
                    title="Sector Performance (Last 52 Weeks)",
                )
                fig_all.update_layout(
                    yaxis_title="Weekly P&L %",
                    xaxis_title="Date",
                    hovermode='x unified',
                )
                st.plotly_chart(fig_all, use_container_width=True)

            sectors = sorted(list({entry['sector'] for entry in weekly_history}))
            for idx, sector_name in enumerate(sectors, start=1):
                with tab_weeks[idx]:
                    sector_df = pd.DataFrame(
                        [entry for entry in weekly_history if entry['sector'] == sector_name]
                    )
                    if not sector_df.empty:
                        st.write(f"**{sector_name}** — 52-Week Trend (Select weeks as needed)")
                        sector_df['date'] = pd.to_datetime(sector_df['date'])
                        sector_df = sector_df.sort_values('date')
                        fig_sector = px.line(
                            sector_df,
                            x='date',
                            y='pnl_pct',
                            markers=True,
                            title=f"{sector_name} — Weekly P&L %",
                        )
                        fig_sector.update_layout(
                            yaxis_title="Weekly P&L %",
                            xaxis_title="Date",
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig_sector, use_container_width=True)
                        st.dataframe(
                            sector_df[['date', 'pnl_pct', 'investment', 'current_value']].assign(
                                date=lambda df_: df_['date'].dt.strftime('%Y-%m-%d'),
                                pnl_pct=lambda df_: df_['pnl_pct'].map(lambda v: f"{v:+.2f}%"),
                                investment=lambda df_: df_['investment'].map(lambda v: f"₹{v:,.0f}"),
                                current_value=lambda df_: df_['current_value'].map(lambda v: f"₹{v:,.0f}"),
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.info(f"No weekly history available for {sector_name}.")
        else:
            st.info("52-week sector performance not available yet—weekly NAV history is still being gathered.")
    except Exception as exc:
        st.warning(f"Could not render 52-week performance chart: {exc}")

    st.markdown("### Sector Details")
    for _, row in sector_summary.sort_values('current_value', ascending=False).iterrows():
        sector_name = row['sector']
        with st.expander(f"{sector_name} • Current Value ₹{row['current_value']:,.0f}", expanded=False):
            sector_df = df[df['sector'] == sector_name].copy()
            sector_df_display = sector_df[
                [col for col in ['ticker', 'stock_name', 'asset_type', 'total_quantity', 'investment', 'current_value', 'pnl', 'pnl_pct']
                 if col in sector_df.columns]
            ].copy()

            numeric_columns = ['total_quantity', 'investment', 'current_value', 'pnl', 'pnl_pct']
            for col in numeric_columns:
                if col in sector_df_display.columns:
                    if col == 'pnl_pct':
                        sector_df_display[col] = sector_df_display[col].map(lambda v: f"{v:+.2f}%")
                    elif col == 'total_quantity':
                        sector_df_display[col] = sector_df_display[col].map(lambda v: f"{v:,.2f}")
                    else:
                        sector_df_display[col] = sector_df_display[col].map(lambda v: f"₹{v:,.2f}")

            st.dataframe(sector_df_display, use_container_width=True)

    st.markdown("---")
    st.caption("Tip: Keep sector metadata consistent across uploads to unlock richer sector analytics.")

def ai_assistant_page():
    """Dedicated AI Assistant page"""
    st.header("🤖 AI Assistant")
    st.caption("Your intelligent portfolio advisor with access to all your data")
    
    user = st.session_state.user
    
    # Initialize chat history and PDF context
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Load chat history from database (user-specific)
    try:
        # Check if method exists (for backward compatibility)
        if hasattr(db, 'get_user_chat_history'):
            db_chat_history = db.get_user_chat_history(user['id'], limit=50)
            if db_chat_history:
                # Convert database format to session state format
                st.session_state.chat_history = [
                    {"q": chat['question'], "a": chat['answer']}
                    for chat in reversed(db_chat_history)  # Reverse to show oldest first
                ]
        else:
            # Method not available, use empty list
            st.session_state.chat_history = []
    except Exception as e:
        # If table doesn't exist yet, just use empty list
        st.session_state.chat_history = []
    
    # Always load PDF context from database to ensure older session PDFs are included
    st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
    
    # Show PDF context status and refresh option
    pdf_count = len(db.get_user_pdfs(user['id']))
    if pdf_count > 0:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"📚 Loaded {pdf_count} PDFs from shared library (all users) for AI context")
        with col2:
            if st.button("🔄", key="refresh_pdf_context_main", help="Refresh PDF context from database"):
                st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
                st.success("PDF context refreshed!")
                st.rerun()
    
    # Get portfolio context (cached)
    holdings = get_cached_holdings(user['id'])
    
    # Get cached portfolio summary
    portfolio_summary = get_cached_portfolio_summary(holdings)
    
    # Get user PDFs for the session
    user_pdfs = db.get_user_pdfs(user['id'])
    
    # Enhanced chat interface
    st.markdown("---")
    st.markdown("**💬 Chat with AI**")
    
    # Chat input
    user_question = st.text_input(
        "Ask me anything about your portfolio:",
        placeholder="e.g., 'How is my technology sector performing?' or 'Should I rebalance my portfolio?'",
        key="ai_chat_input"
    )
    
    if user_question:
        with st.spinner("🤖 AI is thinking..."):
            try:
                import openai
                from datetime import datetime
                import json
                openai.api_key = st.secrets["api_keys"]["open_ai"]
                
                # Safe float conversion helper
                def safe_float(value, default=0):
                    try:
                        return float(value) if value is not None else default
                    except (ValueError, TypeError):
                        return default
                
                # ===== DATABASE QUERY FUNCTIONS FOR AI =====
                # These functions give the AI direct access to query the database
                
                def get_holdings(user_id: str, asset_type: str = None, sector: str = None, limit: int = None) -> str:
                    """Get user holdings from database. Returns JSON string of holdings data."""
                    try:
                        query = db.supabase.table('user_holdings_detailed').select('*').eq('user_id', user_id)
                        if asset_type:
                            query = query.eq('asset_type', asset_type)
                        if sector:
                            query = query.eq('sector', sector)
                        if limit:
                            query = query.limit(limit)
                        response = query.execute()
                        return json.dumps(response.data if response.data else [], indent=2)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_transactions(user_id: str, date_from: str = None, date_to: str = None, transaction_type: str = None, ticker: str = None, limit: int = 200) -> str:
                    """Get user transactions from database. Returns JSON string of transactions data."""
                    try:
                        query = db.supabase.table('user_transactions_detailed').select('*').eq('user_id', user_id)
                        if date_from:
                            query = query.gte('transaction_date', date_from)
                        if date_to:
                            query = query.lte('transaction_date', date_to)
                        if transaction_type:
                            query = query.eq('transaction_type', transaction_type.lower())
                        if ticker:
                            query = query.eq('ticker', ticker)
                        if limit:
                            query = query.limit(limit)
                        response = query.order('transaction_date', desc=True).execute()
                        return json.dumps(response.data if response.data else [], indent=2)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_historical_prices(ticker: str, date_from: str = None, date_to: str = None, limit: int = 52) -> str:
                    """Get historical prices for a ticker. Returns JSON string of historical price data."""
                    try:
                        # Get stock_id first
                        stock_response = db.supabase.table('stock_master').select('id').eq('ticker', ticker).execute()
                        if not stock_response.data:
                            return json.dumps({"error": f"Ticker {ticker} not found"})
                        stock_id = stock_response.data[0]['id']
                        
                        # Get historical prices
                        query = db.supabase.table('historical_prices').select('*').eq('stock_id', stock_id)
                        if date_from:
                            query = query.gte('price_date', date_from)
                        if date_to:
                            query = query.lte('price_date', date_to)
                        query = query.order('price_date', desc=True)
                        if limit:
                            query = query.limit(limit)
                        response = query.execute()
                        return json.dumps(response.data if response.data else [], indent=2)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_stock_master(ticker: str = None, asset_type: str = None, limit: int = 100) -> str:
                    """Get stock master data. Returns JSON string of stock master records."""
                    try:
                        query = db.supabase.table('stock_master').select('*')
                        if ticker:
                            query = query.eq('ticker', ticker)
                        if asset_type:
                            query = query.eq('asset_type', asset_type)
                        if limit:
                            query = query.limit(limit)
                        response = query.execute()
                        return json.dumps(response.data if response.data else [], indent=2)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_pdfs(user_id: str = None, search_term: str = None, limit: int = 10) -> str:
                    """Get PDF documents and their summaries. Returns JSON string of PDF data."""
                    try:
                        pdfs = db.get_user_pdfs(user_id) if user_id else db.get_user_pdfs()
                        if search_term:
                            search_lower = search_term.lower()
                            pdfs = [p for p in pdfs if search_lower in (p.get('filename', '') + p.get('ai_summary', '')).lower()]
                        if limit:
                            pdfs = pdfs[:limit]
                        return json.dumps(pdfs, indent=2, default=str)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_financial_news(ticker: str = None, company_name: str = None, sector: str = None, limit: int = 10) -> str:
                    """Get latest financial news from Moneycontrol, Economic Times, and other sources. Returns JSON string of news articles."""
                    try:
                        import requests
                        from bs4 import BeautifulSoup
                        from datetime import datetime, timedelta
                        
                        news_articles = []
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        
                        # 1. Moneycontrol - Try multiple approaches
                        try:
                            moneycontrol_urls = []
                            
                            # Approach 1: Ticker-specific news (if ticker provided)
                            if ticker:
                                clean_ticker = ticker.replace('.NS', '').replace('.BO', '').upper()
                                # Try different Moneycontrol URL patterns
                                moneycontrol_urls.append(f"https://www.moneycontrol.com/news/tags/{clean_ticker.lower()}.html")
                                moneycontrol_urls.append(f"https://www.moneycontrol.com/news/tags/{clean_ticker}.html")
                                # Try company page news
                                moneycontrol_urls.append(f"https://www.moneycontrol.com/india/stockpricequote/{clean_ticker.lower()}")
                            
                            # Approach 2: General market news (always try)
                            moneycontrol_urls.append("https://www.moneycontrol.com/news/business/")
                            moneycontrol_urls.append("https://www.moneycontrol.com/news/business/markets/")
                            moneycontrol_urls.append("https://www.moneycontrol.com/news/business/stocks/")
                            
                            # Approach 3: Sector-specific news
                            if sector:
                                sector_lower = sector.lower().replace(' ', '-')
                                moneycontrol_urls.append(f"https://www.moneycontrol.com/news/tags/{sector_lower}.html")
                            
                            # Try each URL until we get results
                            for url in moneycontrol_urls[:3]:  # Limit to 3 URLs to avoid too many requests
                                try:
                                    response = requests.get(url, headers=headers)
                                    if response.status_code == 200:
                                        soup = BeautifulSoup(response.content, 'html.parser')
                                        
                                        # Try multiple selectors for Moneycontrol's article structure
                                        articles = []
                                        
                                        # Selector 1: Standard news list
                                        articles.extend(soup.find_all('li', class_='clearfix', limit=limit))
                                        
                                        # Selector 2: Article cards
                                        if not articles:
                                            articles.extend(soup.find_all('div', class_='newslist', limit=limit))
                                        
                                        # Selector 3: Generic article links
                                        if not articles:
                                            articles.extend(soup.find_all('a', href=lambda x: x and '/news/' in x, limit=limit))
                                        
                                        for article in articles[:limit]:
                                            try:
                                                # Try to find title
                                                title_elem = (article.find('h2') or 
                                                            article.find('h3') or 
                                                            article.find('a', class_=lambda x: x and 'title' in str(x).lower()) or
                                                            article.find('a'))
                                                
                                                # Try to find link
                                                link_elem = article.find('a') if not isinstance(article, type(soup.find('a'))) else article
                                                
                                                if title_elem:
                                                    title = title_elem.get_text(strip=True)
                                                    if title and len(title) > 10:  # Valid title
                                                        url_link = ""
                                                        if link_elem and hasattr(link_elem, 'get'):
                                                            url_link = link_elem.get('href', '')
                                                            # Make absolute URL if relative
                                                            if url_link and not url_link.startswith('http'):
                                                                url_link = f"https://www.moneycontrol.com{url_link}" if url_link.startswith('/') else f"https://www.moneycontrol.com/{url_link}"
                                                        
                                                        # Try to find date
                                                        date_elem = (article.find('span', class_='date') or 
                                                                   article.find('span', class_=lambda x: x and 'date' in str(x).lower()) or
                                                                   article.find('time'))
                                                        date_str = date_elem.get_text(strip=True) if date_elem else datetime.now().strftime("%Y-%m-%d")
                                                        
                                                        news_articles.append({
                                                            "title": title,
                                                            "url": url_link,
                                                            "source": "Moneycontrol",
                                                            "date": date_str,
                                                            "ticker": ticker if ticker else None,
                                                            "sector": sector if sector else None
                                                        })
                                                        
                                                        if len(news_articles) >= limit:
                                                            break
                                            except Exception:
                                                continue
                                        
                                        if len(news_articles) >= limit:
                                            break
                                except Exception:
                                    continue
                                    
                        except Exception as e:
                            pass  # Moneycontrol failed, try other sources
                        
                        # 2. Use AI to fetch news if web scraping fails or for general queries
                        if len(news_articles) < limit:
                            try:
                                # Use OpenAI to get recent financial news (GPT-5 has knowledge up to its training date)
                                news_query = f"Latest financial news"
                                if ticker:
                                    news_query += f" about {ticker}"
                                if company_name:
                                    news_query += f" ({company_name})"
                                if sector:
                                    news_query += f" in {sector} sector"

                                # Note: This uses AI's training knowledge, not real-time web access
                                # For true real-time news, you'd need a news API like NewsAPI, Alpha Vantage, etc.
                                ai_news_response = openai.chat.completions.create(
                                    model="gpt-5",
                                    messages=[{
                                        "role": "user",
                                        "content": f"Provide the latest financial news and market updates {news_query}. Include recent developments, market trends, and any significant events. Format as a list of news items with titles and brief summaries."
                                    }],
                                    max_completion_tokens=1000
                                )
                                
                                if ai_news_response.choices:
                                    ai_news_text = ai_news_response.choices[0].message.content
                                    # Parse AI response into structured format
                                news_articles.append({
                                    "title": "AI-Generated Market Update",
                                    "content": ai_news_text,
                                        "source": "AI Analysis (Training Data)",
                                    "date": datetime.now().strftime("%Y-%m-%d"),
                                        "note": "Based on AI training data - may not include very recent news"
                                })
                            except Exception as e:
                                pass
                        
                        # If no news found, return a helpful message
                        if not news_articles:
                            return json.dumps({
                                "message": "No recent news found. For real-time financial news, consider integrating NewsAPI, Alpha Vantage News API, or other financial news services.",
                                "suggestions": [
                                    "Use NewsAPI.org for real-time news",
                                    "Use Alpha Vantage News & Sentiment API",
                                    "Scrape Moneycontrol, Economic Times, or Business Standard",
                                    "Use RSS feeds from financial news sources"
                                ]
                            }, indent=2)
                        
                        return json.dumps(news_articles[:limit], indent=2, default=str)
                    except Exception as e:
                        return json.dumps({"error": str(e), "message": "Financial news fetching failed. Consider using a dedicated news API for real-time updates."})
                
                # Define OpenAI function tools
                functions = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_holdings",
                            "description": "Get user holdings from the database. Use this to query portfolio holdings, filter by asset_type (stock, mutual_fund, bond, pms, aif) or sector, and limit results.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string", "description": "User ID (required)"},
                                    "asset_type": {"type": "string", "description": "Filter by asset type: stock, mutual_fund, bond, pms, aif"},
                                    "sector": {"type": "string", "description": "Filter by sector"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return"}
                                },
                                "required": ["user_id"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_transactions",
                            "description": "Get user transactions from the database. Use this to query transaction history, filter by date range, transaction type (buy/sell), or ticker.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string", "description": "User ID (required)"},
                                    "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD format)"},
                                    "date_to": {"type": "string", "description": "End date (YYYY-MM-DD format)"},
                                    "transaction_type": {"type": "string", "description": "Filter by transaction type: buy or sell"},
                                    "ticker": {"type": "string", "description": "Filter by ticker symbol"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return (default 200)"}
                                },
                                "required": ["user_id"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_historical_prices",
                            "description": "Get historical price data for a ticker. Use this to analyze price trends, 52-week highs/lows, and price movements over time.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string", "description": "Ticker symbol (required)"},
                                    "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD format)"},
                                    "date_to": {"type": "string", "description": "End date (YYYY-MM-DD format)"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return (default 52 for 52 weeks)"}
                                },
                                "required": ["ticker"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_stock_master",
                            "description": "Get stock master data including ticker information, asset types, sectors, and current prices. Use this to get metadata about stocks, mutual funds, bonds, etc.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string", "description": "Filter by specific ticker"},
                                    "asset_type": {"type": "string", "description": "Filter by asset type: stock, mutual_fund, bond, pms, aif"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return"}
                                }
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_pdfs",
                            "description": "Get PDF documents and their AI summaries from the shared library. Use this to access research documents, reports, and analysis that can inform recommendations.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string", "description": "User ID (optional, if not provided returns all shared PDFs)"},
                                    "search_term": {"type": "string", "description": "Search term to filter PDFs by filename or content"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return"}
                                }
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_financial_news",
                            "description": "Get latest financial news and market updates from Moneycontrol, Economic Times, and other sources. Use this to access real-time financial news, market trends, company updates, and sector news. Note: Currently uses web scraping and AI knowledge - for true real-time news, consider integrating NewsAPI or Alpha Vantage News API.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., 'RELIANCE.NS', 'TCS.NS') to get news about a specific stock"},
                                    "company_name": {"type": "string", "description": "Company name to search for news"},
                                    "sector": {"type": "string", "description": "Sector name (e.g., 'Technology', 'Banking') to get sector-specific news"},
                                    "limit": {"type": "integer", "description": "Maximum number of news articles to return (default 10)"}
                                }
                            }
                        }
                    }
                ]
                
                # Get current date for context
                current_date = datetime.now().strftime("%Y-%m-%d")
                current_year = datetime.now().year
                
                # Prepare session dataframes summary (available in memory)
                session_data_summary = f"""
📊 SESSION DATA AVAILABLE (Already loaded in memory):
- Holdings DataFrame: {len(holdings)} holdings loaded
- Portfolio Summary: Available
- PDF Context: {len(user_pdfs)} PDFs loaded
- User ID: {user['id']}

💡 INSTRUCTIONS:
- For data-related questions, use the database query functions (get_holdings, get_transactions, etc.) to fetch the exact data you need
- The session dataframes are already loaded, but you can query the database for more specific or filtered data
- Always query the database when you need:
  * Filtered transactions (by date, type, ticker)
  * Historical prices for specific tickers
  * Stock master metadata
  * PDF documents matching search terms
- Use the functions based on what the user is asking - don't query everything, only what's needed
"""
                
                # Build system prompt with full database access instructions
                system_prompt = f"""You are an expert portfolio analyst and financial advisor with DIRECT ACCESS to the complete database for user_id: {user['id']}.

📅 CURRENT DATE: {current_date} (Year: {current_year})
⚠️ CRITICAL: Today's date is {current_date}. Always use this date when:
- Calculating time periods (e.g., "1 year ago" means {current_year - 1}-{datetime.now().strftime('%m-%d')})
- Referencing current market conditions
- Making time-based predictions
- Analyzing transaction dates and holding periods
Do NOT use 2024 or any other year - use {current_year}.

🔑 DATABASE ACCESS:
You have DIRECT ACCESS to query the database using these functions:
1. get_holdings(user_id, asset_type, sector, limit) - Query holdings
2. get_transactions(user_id, date_from, date_to, transaction_type, ticker, limit) - Query transactions
3. get_historical_prices(ticker, date_from, date_to, limit) - Query historical prices
4. get_stock_master(ticker, asset_type, limit) - Query stock metadata
5. get_pdfs(user_id, search_term, limit) - Query PDF documents
6. get_financial_news(ticker, company_name, sector, limit) - Get latest financial news from Moneycontrol, Economic Times, and other sources

📰 FINANCIAL NEWS ACCESS:
- Use get_financial_news() to fetch latest market news, company updates, and sector trends
- You can search by ticker, company name, or sector
- This helps you provide recommendations based on current market conditions and news
- Note: Currently uses web scraping and AI knowledge - for true real-time news, consider integrating NewsAPI or Alpha Vantage News API

📊 SESSION DATA:
{session_data_summary}

🎯 HOW TO USE:
1. Analyze the user's question to determine what data you need
2. Use the appropriate function(s) to query the database for the exact data needed
3. Don't query everything - only query what's relevant to the question
4. For example:
   - "Show my tech stocks" → get_holdings(user_id="{user['id']}", asset_type="stock", sector="Technology")
   - "1 year buy transactions" → get_transactions(user_id="{user['id']}", date_from="{datetime.now().replace(year=current_year-1).strftime('%Y-%m-%d')}", transaction_type="buy")
   - "Price history of RELIANCE" → get_historical_prices(ticker="RELIANCE.NS")
   - "PDFs about banking" → get_pdfs(search_term="banking")

💡 CAPABILITIES:
- ✅ Suggest BUY recommendations based on PDF research, market analysis, and portfolio gaps
- ✅ Suggest SELL recommendations based on overvaluation, poor performance, or risk concerns
- ✅ Analyze when transactions would have been more profitable using historical price data
- ✅ Provide actionable investment recommendations with specific tickers and reasoning
- ✅ Make PREDICTIONS and FORECASTS about stock prices, market trends, and economic indicators
- ✅ Calculate P&L, returns, and other metrics from transaction data
- ✅ Compare filtered results with overall portfolio when relevant

Always:
- Use the database functions to get the exact data you need based on the question
- Cite specific tickers, dates, and amounts from the queried data
- Reference PDF research documents when making recommendations
- Provide data-driven recommendations based on actual numbers from the database"""
                
                # Start conversation with chat history (if available) and user question
                messages = [{"role": "system", "content": system_prompt}]
                
                # Add chat history to context (last 10 conversations)
                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history[-10:]:  # Last 10 conversations
                        messages.append({"role": "user", "content": chat.get("q", "")})
                        messages.append({"role": "assistant", "content": chat.get("a", "")})
                
                # Add current user question
                messages.append({"role": "user", "content": user_question})
                
                # Function calling loop - allow AI to query database multiple times
                max_iterations = 5
                for iteration in range(max_iterations):
                    response = openai.chat.completions.create(
                        model="gpt-5",
                        messages=messages,
                        tools=functions,
                        tool_choice="auto"  # Let AI decide when to use functions
                    )
                    
                    choice = response.choices[0]
                    # Convert message object to dict format for consistency
                    message_dict = {
                        "role": choice.message.role,
                        "content": choice.message.content if choice.message.content else None
                    }
                    if choice.message.tool_calls:
                        message_dict["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in choice.message.tool_calls
                        ]
                    messages.append(message_dict)
                    
                    # Check if AI wants to call a function
                    if choice.message.tool_calls:
                        # Execute function calls
                        for tool_call in choice.message.tool_calls:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            
                            # Add user_id to function calls that need it
                            if function_name in ['get_holdings', 'get_transactions'] and 'user_id' not in function_args:
                                function_args['user_id'] = user['id']
                            
                            # Execute the function
                            if function_name == 'get_holdings':
                                function_result = get_holdings(**function_args)
                            elif function_name == 'get_transactions':
                                function_result = get_transactions(**function_args)
                            elif function_name == 'get_historical_prices':
                                function_result = get_historical_prices(**function_args)
                            elif function_name == 'get_stock_master':
                                function_result = get_stock_master(**function_args)
                            elif function_name == 'get_pdfs':
                                function_result = get_pdfs(**function_args)
                            elif function_name == 'get_financial_news':
                                function_result = get_financial_news(**function_args)
                            else:
                                function_result = json.dumps({"error": f"Unknown function: {function_name}"})
                            
                            # Add function result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": function_result
                            })
                    else:
                        # AI provided final answer, break loop
                        break
                
                # Get final AI response
                # Handle both dict and object formats
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    ai_response = last_message.get("content", "") if last_message.get("content") else "I apologize, but I couldn't generate a response."
                else:
                    ai_response = last_message.content if hasattr(last_message, 'content') and last_message.content else "I apologize, but I couldn't generate a response."
                
                # Check if response was truncated
                if not ai_response or ai_response.strip() == "":
                    st.error("❌ Empty response from AI. Please try again.")
                    st.stop()
                
                # Display the response immediately
                st.markdown("---")
                st.markdown("### 💬 AI Response:")
                st.success(ai_response)
                
                # Store in chat history (session state)
                st.session_state.chat_history.append({
                    "q": user_question,
                    "a": ai_response
                })
                
                # Save to database (user-specific, persistent)
                # Check if method exists before calling
                if hasattr(db, 'save_chat_history'):
                    try:
                        db.save_chat_history(user['id'], user_question, ai_response)
                    except Exception as e:
                        # If table doesn't exist, just continue without saving
                        pass
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:100]}")
                st.error(f"Full error: {str(e)}")  # Show full error for debugging
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("**💭 Chat History**")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q: {chat['q'][:50]}..."):
                st.markdown(f"**Question:** {chat['q']}")
                st.markdown(f"**Answer:** {chat['a']}")
    
    # PDF Library Section (Shared across all users)
    st.markdown("---")
    st.markdown("**📚 Shared PDF Library (Available to All Users)**")
    st.caption("💡 PDFs uploaded by any user are visible to everyone")
    
    if user_pdfs and len(user_pdfs) > 0:
        for pdf in user_pdfs:
            with st.expander(f"📄 {pdf['filename']}"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"📅 Uploaded: {pdf['uploaded_at'][:10]}")
                    
                    if pdf.get('ai_summary'):
                        st.markdown("**🤖 AI Summary:**")
                        st.info(pdf['ai_summary'])
                    
                    # Add button to use this PDF for analysis
                    if st.button(f"🔍 Analyze {pdf['filename'][:20]}...", key=f"analyze_{pdf['id']}", help="Use this PDF for AI analysis"):
                        try:
                            import openai
                            openai.api_key = st.secrets["api_keys"]["open_ai"]
                            
                            # Get portfolio context
                            portfolio_summary = get_cached_portfolio_summary(holdings)
                            
                            # Analyze the stored PDF
                            analysis_prompt = f"""
                            Analyze this stored PDF document for portfolio management insights.
                            
                            📄 DOCUMENT INFO:
                            - Filename: {pdf['filename']}
                            - Uploaded: {pdf['uploaded_at'][:10]}
                            
                            💼 USER'S PORTFOLIO:
                            {portfolio_summary}
                            
                            📝 PDF CONTENT:
                            {pdf.get('pdf_text', '')[:5000]}...
                            
                            🤖 PREVIOUS AI SUMMARY:
                            {pdf.get('ai_summary', 'No previous summary')}
                            
                            Please provide a fresh analysis focusing on:
                            1. Key insights from the document
                            2. How it relates to the user's current portfolio
                            3. Actionable recommendations
                            
                            Be specific and actionable. Use emojis and clear formatting.
                            """
                            
                            response = openai.chat.completions.create(
                                model="gpt-5",  # Upgraded to GPT-5 for better results
                                messages=[{"role": "user", "content": analysis_prompt}],
                                max_completion_tokens=800,
                                # Note: GPT-5 only supports default temperature (1)
                            )
                            
                            fresh_analysis = response.choices[0].message.content
                            
                            # Display the fresh analysis
                            st.markdown("### 🔍 Fresh Analysis")
                            st.markdown(fresh_analysis)
                            
                            # Store in chat history (session state)
                            st.session_state.chat_history.append({
                                "q": f"Analyze PDF: {pdf['filename']}", 
                                "a": fresh_analysis
                            })
                            # Save to database (user-specific, persistent)
                            if hasattr(db, 'save_chat_history'):
                                try:
                                    db.save_chat_history(user['id'], f"Analyze PDF: {pdf['filename']}", fresh_analysis)
                                except Exception:
                                    pass
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing PDF: {str(e)[:100]}")
                
                with col2:
                    if st.button("🗑️", key=f"del_{pdf['id']}", help="Delete this PDF"):
                        if db.delete_pdf(pdf['id']):
                            st.success("Deleted!")
                            st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
                            st.rerun()
    else:
        st.caption("No PDFs uploaded yet")
    
    _render_document_upload_section(
        section_key="document_ai_primary",
        user=user,
        holdings=holdings,
        db=db,
    )
    
    # Quick Tips Section
    st.markdown("---")
    st.markdown("**💡 Quick Tips**")
    st.caption("Try asking me:")
    st.caption("• 'How is my portfolio performing overall?'")
    st.caption("• 'Which sectors are my best performers?'")
    st.caption("• 'How can I reduce portfolio risk?'")
    st.caption("• 'Which channels are giving me the best returns?'")
    st.caption("• 'Should I rebalance my portfolio?'")
    st.caption("• 'Upload a research report for analysis'")

def ai_insights_page():
    """AI Insights page with agent analysis and recommendations"""
    st.header("🧠 AI Insights")
    st.caption("Powered by specialized AI agents for portfolio analysis")
    
    if not AI_AGENTS_AVAILABLE:
        st.error("AI agents are not available. Please check the installation.")
        return
    
    # Get user data
    user = st.session_state.user
    db = st.session_state.db
    
    # Get holdings data
    holdings = get_cached_holdings(user['id'])
    
    if not holdings:
        st.info("No holdings found. Upload transaction files to see AI insights.")
        return
    
    # Get comprehensive context for AI analysis (same as AI Assistant has)
    pdf_context = db.get_all_pdfs_text(user['id'])
    pdf_count = len(db.get_user_pdfs(user['id']))
    
    # Get all transactions for the user
    try:
        all_transactions_response = db.supabase.table('user_transactions_detailed').select('*').eq('user_id', user['id']).order('transaction_date', desc=False).limit(1000).execute()
        all_transactions = all_transactions_response.data if all_transactions_response.data else []
    except:
        all_transactions = []
    
    # Get historical prices for all tickers in holdings (increase limit)
    historical_prices = {}
    historical_prices_fetched = 0
    historical_prices_missing = 0
    try:
        # Get unique tickers from all holdings
        unique_tickers = set()
        for holding in holdings:
            ticker = holding.get('ticker')
            if ticker:
                unique_tickers.add(ticker)
        
        # Fetch historical prices for all unique tickers (not just top 30)
        for ticker in unique_tickers:
            try:
                # Get stock_id
                stock_response = db.supabase.table('stock_master').select('id').eq('ticker', ticker).execute()
                if stock_response.data:
                    stock_id = stock_response.data[0]['id']
                    # Get 52 weeks of historical prices
                    hist_response = db.supabase.table('historical_prices').select('price_date, price').eq('stock_id', stock_id).order('price_date', desc=True).limit(52).execute()
                    if hist_response.data and len(hist_response.data) > 0:
                        historical_prices[ticker] = hist_response.data
                        historical_prices_fetched += 1
                    else:
                        historical_prices_missing += 1
            except:
                historical_prices_missing += 1
    except Exception as e:
        pass
    
    # Get stock master data for all holdings
    try:
        # Get all stock master records for holdings (get all unique tickers)
        tickers = list(set([h.get('ticker') for h in holdings if h.get('ticker')]))
        if tickers:
            # Fetch in batches if needed (Supabase .in_() has limits)
            stock_master = []
            for i in range(0, len(tickers), 100):  # Process 100 at a time
                batch = tickers[i:i+100]
                try:
                    stock_master_response = db.supabase.table('stock_master').select('*').in_('ticker', batch).execute()
                    if stock_master_response.data:
                        stock_master.extend(stock_master_response.data)
                except:
                    pass
        else:
            stock_master = []
    except:
        stock_master = []
    
    # Show context status with detailed information
    context_info = []
    if pdf_count > 0:
        context_info.append(f"📚 {pdf_count} PDF(s)")
    if all_transactions:
        context_info.append(f"📝 {len(all_transactions)} transactions")
    if historical_prices:
        context_info.append(f"📊 Historical prices: {len(historical_prices)} tickers (52 weeks each)")
    if stock_master:
        context_info.append(f"🏢 {len(stock_master)} stock master records")
    
    if context_info:
        st.info(f"**AI Context Active:** {' | '.join(context_info)}")
        if historical_prices_missing > 0:
            st.caption(f"ℹ️ Note: {historical_prices_missing} ticker(s) don't have historical price data yet. They will be automatically fetched during next login. Click below to fetch now if needed.")
            if st.button("📊 Fetch Historical Prices Now", key="fetch_missing_historical", help="Manually fetch 52 weeks of historical prices for all tickers missing data (normally done automatically during login)"):
                with st.spinner(f"Fetching historical prices for {historical_prices_missing} ticker(s)..."):
                    try:
                        # Get tickers missing historical data
                        missing_tickers = []
                        for holding in holdings:
                            ticker = holding.get('ticker')
                            if ticker and ticker not in historical_prices:
                                missing_tickers.append(ticker)
                        
                        if missing_tickers:
                            # Get asset types
                            asset_types = {h.get('ticker'): h.get('asset_type', 'stock') for h in holdings if h.get('ticker')}
                            
                            # Fetch comprehensive data (includes historical prices)
                            if 'bulk_ai_fetcher' in st.session_state and st.session_state.bulk_ai_fetcher.available:
                                db.bulk_process_new_stocks_with_comprehensive_data(
                                    tickers=list(set(missing_tickers)),
                                    asset_types=asset_types
                                )
                                st.success(f"✅ Fetched historical prices for {len(set(missing_tickers))} ticker(s)!")
                                st.rerun()
                            else:
                                st.warning("⚠️ Bulk AI fetcher not available. Historical prices will be fetched automatically during next login.")
                                st.rerun()
                        else:
                            st.info("All tickers already have historical price data.")
                    except Exception as e:
                        st.error(f"❌ Error fetching historical prices: {str(e)[:100]}")
    else:
        st.caption("💡 Upload PDFs and transactions to enhance insights")
    
    # Create tabs for different AI insights
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Smart Recommendations",
        "📊 Portfolio Analysis",
        "🔍 Market Insights",
        "🔮 Scenario Analysis",
        "💡 Investment Recommendations",
        "⚙️ Agent Status"
    ])

    analysis_cache_key = f"ai_analysis_{user['id']}_{len(holdings)}_{len(all_transactions)}"
    analysis_result = st.session_state.get(analysis_cache_key)

    if analysis_result is None or not st.session_state.get('ai_analysis_complete', False):
        with st.spinner("🤖 AI agents analyzing your portfolio (this may take 30-60 seconds)..."):
            try:
                user_profile_data = db.get_user_profile(user['id'])
                user_profile = {
                    "user_id": user['id'],
                    "risk_tolerance": user_profile_data.get('risk_tolerance', 'moderate'),
                    "goals": user_profile_data.get('investment_goals', []),
                    "rebalancing_frequency": user_profile_data.get('rebalancing_frequency', 'quarterly'),
                    "tax_optimization": user_profile_data.get('tax_optimization', True),
                    "esg_investing": user_profile_data.get('esg_investing', False),
                    "international_exposure": user_profile_data.get('international_exposure', 20)
                }

                from ai_agents.agent_manager import run_ai_analysis
                analysis_result = run_ai_analysis(
                    holdings,
                    user_profile,
                    pdf_context,
                    all_transactions,
                    historical_prices,
                    stock_master
                )
                st.session_state[analysis_cache_key] = analysis_result
                st.session_state['ai_analysis_complete'] = True
            except Exception as e:
                st.error(f"Error running AI analysis: {str(e)}")
                analysis_result = {"error": str(e)}
                st.session_state[analysis_cache_key] = analysis_result
    else:
        analysis_result = st.session_state[analysis_cache_key]

    with tab1:
        st.subheader("🎯 Smart Recommendations")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                recommendations = analysis_result.get("investment_recommendations", [])
                if not recommendations:
                    try:
                        from ai_agents.agent_manager import get_ai_recommendations
                        recommendations = get_ai_recommendations(5)
                    except Exception:
                        recommendations = []

                if recommendations:
                    st.success(f"✅ Found {len(recommendations)} AI recommendations")
                    severity_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    for rec in recommendations:
                        icon = severity_icons.get(rec.get("severity", "low"), "🟢")
                        with st.expander(f"{icon} {rec.get('title', 'Recommendation')}", expanded=rec.get("severity") == "high"):
                            st.markdown(f"**Description:** {rec.get('description', 'No description')}")
                            st.markdown(f"**Recommendation:** {rec.get('recommendation', 'No recommendation')}")
                            if rec.get("data"):
                                st.json(rec["data"])
                else:
                    st.info("🎉 No urgent recommendations found. Your portfolio looks well-balanced!")
        except Exception as e:
            st.error(f"Error displaying recommendations: {str(e)}")

    with tab2:
        st.subheader("📊 Portfolio Analysis")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                insights = analysis_result.get("portfolio_insights", [])
                if insights:
                    st.success(f"📈 Portfolio analysis complete - {len(insights)} insights found")
                    for insight in insights:
                        with st.expander(f"💡 {insight.get('title', 'Insight')}", expanded=False):
                            st.markdown(insight.get('description', 'No description'))
                            if insight.get('data'):
                                st.json(insight['data'])
                else:
                    st.info("No portfolio insights available yet.")
        except Exception as e:
            st.error(f"Error displaying portfolio analysis: {str(e)}")

    with tab3:
        st.subheader("🔍 Market Insights")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                market_insights = analysis_result.get("market_insights", [])
                if market_insights:
                    st.success(f"🌍 Market analysis complete - {len(market_insights)} insights found")
                    for insight in market_insights:
                        with st.expander(f"📊 {insight.get('title', 'Market Insight')}", expanded=False):
                            st.markdown(insight.get('description', 'No description'))
                            if insight.get('data'):
                                st.json(insight['data'])
                else:
                    st.info("No market insights available yet.")
        except Exception as e:
            st.error(f"Error displaying market insights: {str(e)}")

    with tab4:
        st.subheader("🔮 Scenario Analysis")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                scenario_insights = analysis_result.get("scenario_insights", [])
                if scenario_insights:
                    st.success(f"🔮 Scenario analysis complete - {len(scenario_insights)} scenarios analyzed")
                    for scenario in scenario_insights:
                        with st.expander(f"🎯 {scenario.get('title', 'Scenario')}", expanded=False):
                            st.markdown(scenario.get('description', 'No description'))
                            if scenario.get('data'):
                                st.json(scenario['data'])
                else:
                    st.info("No scenario insights available yet.")
        except Exception as e:
            st.error(f"Error displaying scenario analysis: {str(e)}")

    with tab5:
        st.subheader("💡 Investment Recommendations")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                recommendations = analysis_result.get("investment_recommendations", [])
                if recommendations:
                    st.success(f"💡 Investment recommendations complete - {len(recommendations)} recommendations found")
                    for rec in recommendations:
                        with st.expander(f"💼 {rec.get('title', 'Recommendation')}", expanded=False):
                            st.markdown(rec.get('description', 'No description'))
                            st.markdown(f"**Action:** {rec.get('recommendation', 'No specific action')}")
                            if rec.get('data'):
                                st.json(rec['data'])
                else:
                    st.info("No investment recommendations available yet.")
        except Exception as e:
            st.error(f"Error displaying investment recommendations: {str(e)}")

    with tab6:
        st.subheader("⚙️ Agent Status")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                st.info("✅ All AI agents have completed their analysis")
                st.json(analysis_result)
        except Exception as e:
            st.error(f"Error displaying agent status: {str(e)}")

    _render_document_upload_section(
        section_key="document_ai_secondary",
        user=user,
        holdings=holdings,
        db=db,
    )
    
    # Quick Tips Section
    st.markdown("---")
    st.markdown("**💡 Quick Tips**")
    st.caption("Try asking me:")
    st.caption("• 'How is my portfolio performing overall?'")
    st.caption("• 'Which sectors are my best performers?'")
    st.caption("• 'How can I reduce portfolio risk?'")
    st.caption("• 'Which channels are giving me the best returns?'")
    st.caption("• 'Should I rebalance my portfolio?'")
    st.caption("• 'Upload a research report for analysis'")

def ai_insights_page():
    """AI Insights page with agent analysis and recommendations"""
    st.header("🧠 AI Insights")
    st.caption("Powered by specialized AI agents for portfolio analysis")
    
    if not AI_AGENTS_AVAILABLE:
        st.error("AI agents are not available. Please check the installation.")
        return
    
    # Get user data
    user = st.session_state.user
    db = st.session_state.db
    
    # Get holdings data
    holdings = get_cached_holdings(user['id'])
    
    if not holdings:
        st.info("No holdings found. Upload transaction files to see AI insights.")
        return
    
    # Get comprehensive context for AI analysis (same as AI Assistant has)
    pdf_context = db.get_all_pdfs_text(user['id'])
    pdf_count = len(db.get_user_pdfs(user['id']))
    
    # Get all transactions for the user
    try:
        all_transactions_response = db.supabase.table('user_transactions_detailed').select('*').eq('user_id', user['id']).order('transaction_date', desc=False).limit(1000).execute()
        all_transactions = all_transactions_response.data if all_transactions_response.data else []
    except:
        all_transactions = []
    
    # Get historical prices for all tickers in holdings (increase limit)
    historical_prices = {}
    historical_prices_fetched = 0
    historical_prices_missing = 0
    try:
        # Get unique tickers from all holdings
        unique_tickers = set()
        for holding in holdings:
            ticker = holding.get('ticker')
            if ticker:
                unique_tickers.add(ticker)
        
        # Fetch historical prices for all unique tickers (not just top 30)
        for ticker in unique_tickers:
            try:
                # Get stock_id
                stock_response = db.supabase.table('stock_master').select('id').eq('ticker', ticker).execute()
                if stock_response.data:
                    stock_id = stock_response.data[0]['id']
                    # Get 52 weeks of historical prices
                    hist_response = db.supabase.table('historical_prices').select('price_date, price').eq('stock_id', stock_id).order('price_date', desc=True).limit(52).execute()
                    if hist_response.data and len(hist_response.data) > 0:
                        historical_prices[ticker] = hist_response.data
                        historical_prices_fetched += 1
                    else:
                        historical_prices_missing += 1
            except:
                historical_prices_missing += 1
    except Exception as e:
        pass
    
    # Get stock master data for all holdings
    try:
        # Get all stock master records for holdings (get all unique tickers)
        tickers = list(set([h.get('ticker') for h in holdings if h.get('ticker')]))
        if tickers:
            # Fetch in batches if needed (Supabase .in_() has limits)
            stock_master = []
            for i in range(0, len(tickers), 100):  # Process 100 at a time
                batch = tickers[i:i+100]
                try:
                    stock_master_response = db.supabase.table('stock_master').select('*').in_('ticker', batch).execute()
                    if stock_master_response.data:
                        stock_master.extend(stock_master_response.data)
                except:
                    pass
        else:
            stock_master = []
    except:
        stock_master = []
    
    # Show context status with detailed information
    context_info = []
    if pdf_count > 0:
        context_info.append(f"📚 {pdf_count} PDF(s)")
    if all_transactions:
        context_info.append(f"📝 {len(all_transactions)} transactions")
    if historical_prices:
        context_info.append(f"📊 Historical prices: {len(historical_prices)} tickers (52 weeks each)")
    if stock_master:
        context_info.append(f"🏢 {len(stock_master)} stock master records")
    
    if context_info:
        st.info(f"**AI Context Active:** {' | '.join(context_info)}")
        if historical_prices_missing > 0:
            st.caption(f"ℹ️ Note: {historical_prices_missing} ticker(s) don't have historical price data yet. Use 'Fetch Historical Prices' button to add them.")
    else:
        st.caption("💡 Upload PDFs and transactions to enhance insights")
    
    # Create tabs for different AI insights
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Smart Recommendations", 
        "📊 Portfolio Analysis", 
        "🔍 Market Insights",
        "🔮 Scenario Analysis",
        "💡 Investment Recommendations",
        "⚙️ Agent Status"
    ])
    
    # Run AI analysis ONCE and cache it (all tabs will use the same result)
    # This prevents running 5 separate analyses (one per tab)
    analysis_cache_key = f"ai_analysis_{user['id']}_{len(holdings)}_{len(all_transactions)}"
    
    if analysis_cache_key not in st.session_state or not st.session_state.get('ai_analysis_complete', False):
        # Run analysis once for all tabs
        with st.spinner("🤖 AI agents analyzing your portfolio (this may take 30-60 seconds)..."):
            try:
                # Get user profile from database
                user_profile_data = db.get_user_profile(user['id'])
                user_profile = {
                    "user_id": user['id'],
                    "risk_tolerance": user_profile_data.get('risk_tolerance', 'moderate'),
                    "goals": user_profile_data.get('investment_goals', []),
                    "rebalancing_frequency": user_profile_data.get('rebalancing_frequency', 'quarterly'),
                    "tax_optimization": user_profile_data.get('tax_optimization', True),
                    "esg_investing": user_profile_data.get('esg_investing', False),
                    "international_exposure": user_profile_data.get('international_exposure', 20)
                }
                
                # Run comprehensive AI analysis with all context data (agents run in parallel)
                analysis_result = run_ai_analysis(holdings, user_profile, pdf_context, all_transactions, historical_prices, stock_master)
                
                # Cache the result
                st.session_state[analysis_cache_key] = analysis_result
                st.session_state['ai_analysis_complete'] = True
            except Exception as e:
                st.error(f"Error running AI analysis: {str(e)}")
                st.session_state[analysis_cache_key] = {"error": str(e)}
    else:
        # Use cached result
        analysis_result = st.session_state[analysis_cache_key]
    
    with tab1:
        st.subheader("🎯 Smart Recommendations")
        
        try:
            # Use cached analysis result
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                # Get recommendations from analysis result
                recommendations = analysis_result.get("investment_recommendations", [])
                
                # If no recommendations in result, try getting from agent manager cache
                if not recommendations:
                    try:
                        recommendations = get_ai_recommendations(5)
                    except:
                        recommendations = []
                
                if recommendations:
                    st.success(f"✅ Found {len(recommendations)} AI recommendations")
                    
                    for i, rec in enumerate(recommendations, 1):
                        severity_color = {
                            "high": "🔴",
                            "medium": "🟡", 
                            "low": "🟢"
                        }.get(rec.get("severity", "low"), "🟢")
                        
                        with st.expander(f"{severity_color} {rec.get('title', 'Recommendation')}", expanded=(rec.get("severity") == "high")):
                            st.markdown(f"**Description:** {rec.get('description', 'No description')}")
                            st.markdown(f"**Recommendation:** {rec.get('recommendation', 'No recommendation')}")
                            
                            if rec.get("data"):
                                st.json(rec["data"])
                else:
                    st.info("🎉 No urgent recommendations found. Your portfolio looks well-balanced!")
        except Exception as e:
            st.error(f"Error running AI analysis: {str(e)}")
    
    with tab2:
        st.subheader("📊 Portfolio Analysis")
        
        try:
            # Use cached analysis result (no need to run again)
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                if "portfolio_insights" in analysis_result:
                    # Get portfolio insights directly
                    portfolio_insights = analysis_result["portfolio_insights"]
                    
                    if portfolio_insights:
                        st.success(f"📈 Portfolio analysis complete - {len(portfolio_insights)} insights found")
                        
                        for insight in portfolio_insights:
                            severity_emoji = {
                                "high": "🚨",
                                "medium": "⚠️",
                                "low": "ℹ️"
                            }.get(insight.get("severity", "low"), "ℹ️")
                            
                            st.markdown(f"**{severity_emoji} {insight.get('title', 'Insight')}**")
                            st.markdown(f"{insight.get('description', 'No description')}")
                            st.markdown(f"*Recommendation: {insight.get('recommendation', 'No recommendation')}*")
                            
                            # Show additional data if available
                            if insight.get("data"):
                                with st.expander("📊 Detailed Analysis", expanded=False):
                                    st.json(insight["data"])
                            
                            st.markdown("---")
                    else:
                        st.info("No specific portfolio insights at this time.")
                else:
                    st.info("No specific portfolio insights at this time.")
                
        except Exception as e:
            st.error(f"Error getting portfolio insights: {str(e)}")
    
    with tab3:
        st.subheader("🔍 Market Insights")
        
        try:
            # Use cached analysis result (no need to run again)
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                if "market_insights" in analysis_result:
                    # Get market insights directly
                    market_insights = analysis_result["market_insights"]
                    
                    if market_insights:
                        st.success(f"📊 Market analysis complete - {len(market_insights)} insights found")
                        
                        for insight in market_insights:
                            severity_emoji = {
                                "high": "🚨",
                                "medium": "⚠️",
                                "low": "ℹ️"
                            }.get(insight.get("severity", "low"), "ℹ️")
                            
                            st.markdown(f"**{severity_emoji} {insight.get('title', 'Market Insight')}**")
                            st.markdown(f"{insight.get('description', 'No description')}")
                            st.markdown(f"*Recommendation: {insight.get('recommendation', 'No recommendation')}*")
                            
                            # Show additional data if available
                            if insight.get("data"):
                                with st.expander("📊 Market Data", expanded=False):
                                    st.json(insight["data"])
                            
                            st.markdown("---")
                    else:
                        st.info("No specific market insights at this time.")
                else:
                    st.info("No specific market insights at this time.")
                
        except Exception as e:
            st.error(f"Error getting market insights: {str(e)}")
    
    with tab4:
        st.subheader("🔮 Scenario Analysis")
        
        try:
            # Use cached analysis result (no need to run again)
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                if "scenario_insights" in analysis_result:
                    # Get scenario insights directly
                    scenario_insights = analysis_result["scenario_insights"]
                    
                    if scenario_insights:
                        st.success(f"🔮 Scenario analysis complete - {len(scenario_insights)} scenarios analyzed")
                        
                        # Group scenarios by type
                        scenario_types = {}
                        for insight in scenario_insights:
                            scenario_type = insight.get("type", "unknown")
                            if scenario_type not in scenario_types:
                                scenario_types[scenario_type] = []
                            scenario_types[scenario_type].append(insight)
                        
                        # Display scenarios by type
                        for scenario_type, scenarios in scenario_types.items():
                            st.markdown(f"**{scenario_type.replace('_', ' ').title()} Scenarios:**")
                            
                            for scenario in scenarios:
                                severity_emoji = {
                                    "high": "🚨",
                                    "medium": "⚠️",
                                    "low": "ℹ️"
                                }.get(scenario.get("severity", "low"), "ℹ️")
                                
                                with st.expander(f"{severity_emoji} {scenario.get('title', 'Scenario')}", expanded=(scenario.get("severity") == "high")):
                                    st.markdown(f"**{scenario.get('description', 'No description')}**")
                                    st.markdown(f"*Recommendation: {scenario.get('recommendation', 'No recommendation')}*")
                                    
                                    if scenario.get("data"):
                                        st.json(scenario["data"])
                                st.markdown("---")
                    else:
                        st.info("No scenario analysis available at this time.")
                else:
                    st.info("No scenario analysis available at this time.")
                
        except Exception as e:
            st.error(f"Error getting scenario insights: {str(e)}")
    
    with tab5:
        st.subheader("💡 Investment Recommendations")
        st.caption("AI-powered suggestions for new holdings to complement your portfolio")
        
        try:
            # Use cached analysis result (no need to run again)
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                if "investment_recommendations" in analysis_result:
                    # Get investment recommendations directly
                    investment_recommendations = analysis_result["investment_recommendations"]
                    
                    if investment_recommendations:
                        st.success(f"💼 Found {len(investment_recommendations)} investment opportunities for you")
                        
                        # Group recommendations by type
                        recommendation_types = {
                            "sell_recommendation": {"title": "🔴 SELL Recommendations", "icon": "📉", "recommendations": [], "color": "red"},
                            "stock_recommendation": {"title": "📈 BUY: Stock Recommendations", "icon": "🏢", "recommendations": [], "color": "green"},
                            "mutual_fund_recommendation": {"title": "📊 BUY: Mutual Fund Recommendations", "icon": "💰", "recommendations": [], "color": "green"},
                            "pms_recommendation": {"title": "🎯 BUY: PMS Recommendations", "icon": "💼", "recommendations": [], "color": "green"},
                            "bond_recommendation": {"title": "🔐 BUY: Bond Recommendations", "icon": "📜", "recommendations": [], "color": "green"},
                            "diversification_opportunity": {"title": "🌈 Diversification Opportunities", "icon": "🎨", "recommendations": [], "color": "blue"},
                            "investment_recommendation": {"title": "💡 General Investment Opportunities", "icon": "✨", "recommendations": [], "color": "blue"}
                        }
                        
                        for rec in investment_recommendations:
                            rec_type = rec.get("type", "investment_recommendation")
                            if rec_type in recommendation_types:
                                recommendation_types[rec_type]["recommendations"].append(rec)
                        
                        # Display recommendations by type
                        for rec_type, rec_group in recommendation_types.items():
                            if rec_group["recommendations"]:
                                st.markdown(f"### {rec_group['icon']} {rec_group['title']}")
                                
                                for rec in rec_group["recommendations"]:
                                    # Create expandable card for each recommendation
                                    with st.expander(f"**{rec.get('title', 'Investment Opportunity')}**", expanded=True):
                                        # Severity indicator
                                        severity_emoji = {
                                            "high": "🔥 High Priority",
                                            "medium": "⚡ Medium Priority", 
                                            "low": "💡 Consider"
                                        }.get(rec.get("severity", "medium"), "💡 Consider")
                                        
                                        st.markdown(f"**Priority:** {severity_emoji}")
                                        
                                        # Description
                                        st.markdown("**📝 Why This Investment:**")
                                        st.markdown(rec.get('description', 'No description available'))
                                        
                                        # Recommendation/Action
                                        st.markdown("**🎯 Action Plan:**")
                                        st.markdown(rec.get('recommendation', 'No recommendation available'))
                                        
                                        # Investment details
                                        if rec.get('data'):
                                            data = rec['data']
                                            
                                            # Check if this is a SELL recommendation
                                            if data.get('action') == 'SELL':
                                                st.markdown("**🔴 SELL Details:**")
                                                
                                                col1, col2, col3 = st.columns(3)
                                                
                                                with col1:
                                                    if data.get('ticker'):
                                                        st.metric("Ticker to Sell", data['ticker'])
                                                    if data.get('current_holding_quantity'):
                                                        st.metric("Current Holding", f"{data['current_holding_quantity']:,} shares")
                                                
                                                with col2:
                                                    if data.get('suggested_sell_quantity'):
                                                        st.metric("Sell Quantity", f"{data['suggested_sell_quantity']:,} shares", 
                                                                delta=f"-{data.get('percentage_to_sell', 0)}%", delta_color="inverse")
                                                    if data.get('funds_freed'):
                                                        st.metric("Funds Freed", f"₹{data['funds_freed']:,.0f}")
                                                
                                                with col3:
                                                    if data.get('value_after_sale') is not None:
                                                        st.metric("Value After Sale", f"₹{data['value_after_sale']:,.0f}")
                                                    if data.get('current_loss'):
                                                        st.metric("Current Loss", f"₹{abs(data['current_loss']):,.0f}", 
                                                                delta=f"{data.get('loss_percentage', 0):.1f}%", delta_color="inverse")
                                                
                                                # Reason for sell
                                                if data.get('reason'):
                                                    st.error(f"⚠️ **Reason to Sell:** {data['reason']}")
                                                
                                                # Rebalancing strategy
                                                if data.get('rebalancing_strategy'):
                                                    st.success(f"♻️ **Rebalancing Plan:** {data['rebalancing_strategy']}")
                                                
                                                # Tax consideration
                                                if data.get('tax_consideration'):
                                                    st.info(f"💰 **Tax Impact:** {data['tax_consideration']}")
                                                
                                                # Why now
                                                if data.get('why_now'):
                                                    st.warning(f"⏰ **Why Sell Now:** {data['why_now']}")
                                            
                                            else:
                                                # BUY recommendation details
                                                st.markdown("**📊 Investment Details:**")
                                                
                                                col1, col2, col3 = st.columns(3)
                                                
                                                with col1:
                                                    if data.get('ticker'):
                                                        st.metric("Ticker", data['ticker'])
                                                    if data.get('asset_type'):
                                                        st.markdown(f"**Asset Type:** {data['asset_type']}")
                                                
                                                with col2:
                                                    if data.get('sector'):
                                                        st.markdown(f"**Sector:** {data['sector']}")
                                                    if data.get('risk_level'):
                                                        st.markdown(f"**Risk Level:** {data['risk_level']}")
                                                
                                                with col3:
                                                    if data.get('suggested_allocation_percentage'):
                                                        st.metric("Suggested Allocation", f"{data['suggested_allocation_percentage']}%")
                                                    if data.get('expected_return'):
                                                        st.markdown(f"**Expected Return:** {data['expected_return']}")
                                                
                                                # Investment thesis
                                                if data.get('investment_thesis'):
                                                    st.info(f"💡 **Investment Thesis:** {data['investment_thesis']}")
                                                
                                                # Why now
                                                if data.get('why_now'):
                                                    st.success(f"⏰ **Why Now:** {data['why_now']}")
                                                
                                                # Suggested amount
                                                if data.get('suggested_amount'):
                                                    st.markdown(f"**💵 Suggested Investment:** ₹{data['suggested_amount']:,.0f}")
                                        
                                        st.markdown("---")
                                
                                st.markdown("")  # Add spacing between groups
                    else:
                        st.info("No specific investment recommendations at this time. Your portfolio appears well-balanced!")
                else:
                    st.info("No investment recommendations available. Run analysis to generate recommendations.")
                
        except Exception as e:
            st.error(f"Error getting investment recommendations: {str(e)}")
    
    with tab6:
        st.subheader("⚙️ Agent Status")
        
        try:
            # Get agent status
            agent_manager = get_agent_manager()
            agent_status = agent_manager.get_agent_status()
            
            st.markdown("**🤖 AI Agent Status:**")
            
            for agent_id, status in agent_status.items():
                status_emoji = {
                    "active": "🟢",
                    "analyzing": "🟡",
                    "error": "🔴",
                    "initialized": "🔵"
                }.get(status.get("status", "unknown"), "⚪")
                
                st.markdown(f"**{status_emoji} {status.get('agent_name', agent_id)}**")
                st.markdown(f"Status: {status.get('status', 'unknown')}")
                st.markdown(f"Last Update: {status.get('last_update', 'never')}")
                
                if status.get("capabilities"):
                    st.markdown(f"Capabilities: {', '.join(status['capabilities'])}")
                
                st.markdown("---")
            
            # Show analysis summary
            summary = agent_manager.get_recommendations_summary()
            if summary and "summary" in summary:
                st.markdown("**📊 Analysis Summary:**")
                st.json(summary["summary"])
            
            # Show performance metrics
            try:
                from ai_agents.performance_optimizer import performance_optimizer
                perf_report = performance_optimizer.get_performance_report()
                
                st.markdown("**⚡ Performance Metrics:**")
                cache_stats = perf_report["cache_statistics"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1f}%")
                with col2:
                    st.metric("Avg Response Time", f"{cache_stats['avg_response_time']:.2f}s")
                with col3:
                    st.metric("Cache Size", cache_stats['cache_size'])
                with col4:
                    st.metric("Total Requests", cache_stats['total_requests'])
                
                # Show optimization recommendations
                recommendations = perf_report.get("recommendations", [])
                if recommendations:
                    st.markdown("**💡 Optimization Recommendations:**")
                    for rec in recommendations:
                        st.info(rec)
                        
            except ImportError:
                pass
                
        except Exception as e:
            st.error(f"Error getting agent status: {str(e)}")
    
    # Add refresh button
    if st.button("🔄 Refresh AI Analysis"):
        # Clear any cached analysis
        if 'agent_manager' in st.session_state:
            del st.session_state.agent_manager
        st.rerun()

def user_profile_page():
    """User Profile Settings page"""
    st.header("👤 Profile Settings")
    st.caption("Manage your investment preferences and goals")
    
    user = st.session_state.user
    db = st.session_state.db
    
    # Get current user profile
    user_profile = db.get_user_profile(user['id'])
    if not user_profile:
        st.error("Could not load user profile")
        return
    
    # Add info box at the top explaining personalization
    st.info("""
    💡 **Your profile directly impacts AI recommendations!** 
    
    The AI analyzes your risk tolerance, goals, and preferences to suggest investments that match YOUR needs.
    Update your settings below to get more personalized recommendations.
    """)
    
    # Create tabs for different settings
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Investment Goals",
        "⚖️ Risk Preferences", 
        "📊 Profile Summary",
        "🤖 How AI Uses Your Profile"
    ])
    
    with tab1:
        st.subheader("🎯 Investment Goals")
        st.caption("Set and manage your financial goals")
        
        # Display current goals
        current_goals = user_profile.get('investment_goals', [])
        
        if current_goals:
            st.markdown("**Current Goals:**")
            for i, goal in enumerate(current_goals):
                with st.expander(f"Goal {i+1}: {goal.get('type', 'Unknown').title()}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Type:** {goal.get('type', 'Unknown')}")
                        st.markdown(f"**Target Amount:** ₹{goal.get('target_amount', 0):,.0f}")
                        st.markdown(f"**Timeline:** {goal.get('timeline_years', 0)} years")
                        st.markdown(f"**Current Progress:** ₹{goal.get('current_progress', 0):,.0f}")
                        
                        if goal.get('description'):
                            st.markdown(f"**Description:** {goal.get('description')}")
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_goal_{i}"):
                            result = db.delete_investment_goal(user['id'], goal.get('id'))
                            if result['success']:
                                st.success("Goal deleted!")
                                st.rerun()
                            else:
                                st.error(f"Error: {result['error']}")
        else:
            st.info("No investment goals set yet. Add your first goal below!")
        
        st.markdown("---")
        
        # Add new goal form
        st.markdown("**Add New Goal:**")
        
        with st.form("add_goal_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                goal_type = st.selectbox(
                    "Goal Type:",
                    ["retirement", "education", "emergency_fund", "home_purchase", "vacation", "other"],
                    key="new_goal_type"
                )
                
                target_amount = st.number_input(
                    "Target Amount (₹):",
                    min_value=10000,
                    max_value=100000000,
                    value=1000000,
                    step=10000,
                    key="new_target_amount"
                )
            
            with col2:
                timeline_years = st.number_input(
                    "Timeline (Years):",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key="new_timeline"
                )
                
                current_progress = st.number_input(
                    "Current Progress (₹):",
                    min_value=0,
                    max_value=target_amount,
                    value=0,
                    step=10000,
                    key="new_progress"
                )
            
            description = st.text_area(
                "Description (Optional):",
                placeholder="Describe your goal...",
                key="new_description"
            )
            
            if st.form_submit_button("➕ Add Goal"):
                new_goal = {
                    "type": goal_type,
                    "target_amount": target_amount,
                    "timeline_years": timeline_years,
                    "current_progress": current_progress,
                    "description": description
                }
                
                result = db.add_investment_goal(user['id'], new_goal)
                if result['success']:
                    st.success("Goal added successfully!")
                    st.rerun()
                else:
                    st.error(f"Error: {result['error']}")
    
    with tab2:
        st.subheader("⚖️ Risk Preferences")
        st.caption("Configure your risk tolerance and investment preferences")
        
        # Risk tolerance settings
        current_risk = user_profile.get('risk_tolerance', 'moderate')
        
        with st.form("risk_settings_form"):
            st.markdown("**Risk Tolerance:**")
            
            risk_options = {
                "conservative": {
                    "name": "Conservative",
                    "description": "Low risk, stable returns. Focus on capital preservation.",
                    "allocation": "40% Stocks, 50% Bonds, 10% Alternatives"
                },
                "moderate": {
                    "name": "Moderate", 
                    "description": "Balanced risk and return. Growth with some stability.",
                    "allocation": "60% Stocks, 30% Bonds, 10% Alternatives"
                },
                "aggressive": {
                    "name": "Aggressive",
                    "description": "High risk, high potential returns. Growth-focused.",
                    "allocation": "80% Stocks, 15% Bonds, 5% Alternatives"
                }
            }
            
            selected_risk = st.radio(
                "Select your risk tolerance:",
                list(risk_options.keys()),
                format_func=lambda x: f"{risk_options[x]['name']}: {risk_options[x]['description']}",
                index=list(risk_options.keys()).index(current_risk)
            )
            
            # Show allocation preview
            st.markdown("**Recommended Allocation:**")
            st.info(risk_options[selected_risk]['allocation'])
            
            # Additional preferences
            st.markdown("**Additional Preferences:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                rebalancing_frequency = st.selectbox(
                    "Rebalancing Frequency:",
                    ["monthly", "quarterly", "semi_annually", "annually"],
                    index=1  # Default to quarterly
                )
                
                tax_optimization = st.checkbox(
                    "Enable Tax Optimization",
                    value=True,
                    help="AI will suggest tax-efficient strategies"
                )
            
            with col2:
                esg_investing = st.checkbox(
                    "ESG Investing Preference",
                    value=False,
                    help="Consider Environmental, Social, and Governance factors"
                )
                
                international_exposure = st.slider(
                    "International Exposure (%)",
                    min_value=0,
                    max_value=50,
                    value=20,
                    help="Percentage of portfolio in international markets"
                )
            
            if st.form_submit_button("💾 Save Risk Preferences"):
                # Update user profile
                profile_data = {
                    "risk_tolerance": selected_risk,
                    "rebalancing_frequency": rebalancing_frequency,
                    "tax_optimization": tax_optimization,
                    "esg_investing": esg_investing,
                    "international_exposure": international_exposure
                }
                
                result = db.update_user_profile(user['id'], profile_data)
                if result['success']:
                    st.success("Risk preferences updated successfully!")
                    # Update session state
                    st.session_state.user = result['user']
                    st.rerun()
                else:
                    st.error(f"Error: {result['error']}")
    
    with tab3:
        st.subheader("📊 Profile Summary")
        st.caption("Overview of your investment profile")
        
        # Display current profile
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information:**")
            st.markdown(f"**Name:** {user_profile.get('full_name', 'N/A')}")
            st.markdown(f"**Username:** {user_profile.get('username', 'N/A')}")
            st.markdown(f"**Email:** {user_profile.get('email', 'N/A')}")
            st.markdown(f"**Member Since:** {user_profile.get('created_at', 'N/A')[:10] if user_profile.get('created_at') else 'N/A'}")
        
        with col2:
            st.markdown("**Investment Preferences:**")
            st.markdown(f"**Risk Tolerance:** {user_profile.get('risk_tolerance', 'moderate').title()}")
            st.markdown(f"**Rebalancing:** {user_profile.get('rebalancing_frequency', 'quarterly').title()}")
            st.markdown(f"**Tax Optimization:** {'Yes' if user_profile.get('tax_optimization') else 'No'}")
            st.markdown(f"**ESG Investing:** {'Yes' if user_profile.get('esg_investing') else 'No'}")
            st.markdown(f"**International Exposure:** {user_profile.get('international_exposure', 20)}%")
        
        # Goals summary
        goals = user_profile.get('investment_goals', [])
        if goals:
            st.markdown("**Investment Goals Summary:**")
            
            total_target = sum(goal.get('target_amount', 0) for goal in goals)
            total_progress = sum(goal.get('current_progress', 0) for goal in goals)
            progress_pct = (total_progress / total_target * 100) if total_target > 0 else 0
            
            st.markdown(f"**Total Goals:** {len(goals)}")
            st.markdown(f"**Total Target:** ₹{total_target:,.0f}")
            st.markdown(f"**Total Progress:** ₹{total_progress:,.0f} ({progress_pct:.1f}%)")
            
            # Progress bar
            st.progress(progress_pct / 100)
        else:
            st.info("No investment goals set yet.")
        
        # AI Agent Integration
        if AI_AGENTS_AVAILABLE:
            st.markdown("**AI Agent Integration:**")
            st.success("✅ AI agents are using your profile for personalized recommendations")
            
            # Show how profile affects AI recommendations
            st.markdown("**How your profile affects AI recommendations:**")
            st.markdown(f"• **Risk-based allocation:** AI uses your {user_profile.get('risk_tolerance', 'moderate')} risk tolerance")
            st.markdown(f"• **Goal-based analysis:** AI considers your {len(goals)} investment goals")
            st.markdown(f"• **Tax optimization:** {'Enabled' if user_profile.get('tax_optimization') else 'Disabled'}")
            st.markdown(f"• **ESG preferences:** {'Considered' if user_profile.get('esg_investing') else 'Not considered'}")
        else:
            st.warning("⚠️ AI agents not available - profile settings won't affect recommendations")
            
            with st.expander("ℹ️ How to enable AI agents"):
                st.markdown("""
                **To enable AI agents, ensure:**
                
                1. ✅ All files in `ai_agents/` directory exist:
                   - `__init__.py`
                   - `base_agent.py`
                   - `communication.py`
                   - `portfolio_agent.py`
                   - `market_agent.py`
                   - `strategy_agent.py`
                   - `scenario_agent.py`
                   - `agent_manager.py`
                   - `performance_optimizer.py`
                
                2. ✅ Required packages installed:
                   - `pandas`, `numpy` (already in requirements.txt)
                
                3. ✅ Restart the Streamlit application:
                   ```bash
                   streamlit run web_agent.py
                   ```
                
                4. ✅ Check the terminal for any import errors
                
                **Note**: If you're on Streamlit Cloud, the app will auto-restart after deployment.
                """)

def process_file_with_ai(uploaded_file, filename, user_id):
    """
    Universal AI-powered file processor
    Handles CSV, PDF, Excel, and any other file format
    Returns extracted transactions ready for database storage
    """
    if not AI_AGENTS_AVAILABLE:
        st.error("🚫 AI agents not available. Please check your configuration.")
        return None
    
    try:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        # Initialize AI File Processor
        file_processor = AIFileProcessor()
        
        # Process file with AI
        with st.spinner(f"🤖 AI is analyzing {filename} and extracting transactions..."):
            transactions = file_processor.process_file(uploaded_file, filename)
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        
        if not transactions:
            st.warning(f"⚠️ No transactions found in {filename}")
            return None
        
        # Enhance transactions with metadata
        for trans in transactions:
            # If price is 0 or missing, it will be fetched automatically later
            if not trans.get('price') or trans['price'] == 0:
                trans['price'] = 0  # Will trigger automatic price fetching
        
        return transactions
        
    except Exception as e:
        st.error(f"❌ Error processing file {filename}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []  # Return empty list instead of None

def upload_files_page():
    """Enhanced upload more files page with AI PDF extraction"""
    st.header("📁 Upload More Files")
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        color: white;
    }
    .upload-info {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .file-preview {
        background: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border: 1px solid #2196f3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    user = st.session_state.user
    portfolios = db.get_user_portfolios(user['id'])
    
    if not portfolios:
        st.error("No portfolios found.")
        return
    
    # Portfolio selection with enhanced UI
    st.markdown("### 📊 Select Target Portfolio")
    portfolio = st.selectbox(
        "Choose portfolio:", 
        portfolios, 
        format_func=lambda x: f"{x['portfolio_name']} (ID: {x['id'][:8]}...)",
        help="Select which portfolio to add the new transactions to"
    )
    
    st.markdown("---")
    
    # Enhanced file upload section
    st.markdown("### 📤 Upload Transaction Files")
    
    # Show sample format
    with st.expander("📋 Sample CSV Format (Click to expand)", expanded=False):
        st.code("""date,ticker,quantity,transaction_type,price,stock_name,sector,channel
2024-01-15,RELIANCE,100,buy,2500,Reliance Industries,Oil & Gas,Zerodha
2024-02-01,120760,50,buy,250.75,Quant Flexi Cap Fund,Flexi Cap,Groww
2024-03-10,TCS,25,buy,3600,Tata Consultancy Services,IT Services,Zerodha""", language="csv")
        
        st.markdown("""
        **📝 Required Columns:**
        - `date`: Transaction date (YYYY-MM-DD)
        - `ticker`: Stock/MF ticker symbol
        - `quantity`: Number of shares/units
        - `transaction_type`: buy/sell
        - `price`: Price per share/unit (optional - will be fetched if missing)
        - `stock_name`: Name of the security (optional)
        - `sector`: Sector classification (optional)
        - `channel`: Channel/platform name (optional - will use filename if missing)
        """)
    
    uploaded_files = st.file_uploader(
        "📁 Choose CSV, Excel, or PDF files to upload",
        type=['csv', 'tsv', 'xlsx', 'xls', 'pdf'],
        accept_multiple_files=True,
        key="upload_files_main",
        help="Python first maps standard columns; AI fallback handles unstructured layouts automatically."
    )
    
    # Show file preview
    if uploaded_files:
        st.markdown("### 📋 File Preview")
        
        for i, uploaded_file in enumerate(uploaded_files):
            with st.container():
                st.markdown(f"""
                <div class="file-preview">
                    <strong>📄 File {i+1}:</strong> {uploaded_file.name}<br>
                    <strong>📊 Size:</strong> {uploaded_file.size:,} bytes<br>
                    <strong>📅 Type:</strong> {uploaded_file.type}
                </div>
                """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        convert_to_csv_clicked = False
        
        with col1:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                st.info(f"🚀 Processing {len(uploaded_files)} file(s) for portfolio: {portfolio['portfolio_name']}")
                progress_bar = st.progress(0)
                status_text = st.empty()
                try:
                    process_uploaded_files(uploaded_files, user['id'], portfolio['id'])
                    progress_bar.progress(1.0)
                    status_text.success("✅ All files processed successfully!")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    progress_bar.progress(0)
                    status_text.error(f"❌ Error processing files: {str(e)}")
        
        with col2:
            convert_to_csv_clicked = st.button("📝 Convert to CSV", use_container_width=True)

        with col3:
            if st.button("🗑️ Clear Files", use_container_width=True):
                if "upload_files_main" in st.session_state:
                    del st.session_state["upload_files_main"]
                st.rerun()

        if convert_to_csv_clicked:
            with st.spinner("🔄 Converting files..."):
                csv_rows: List[Dict[str, Any]] = []
                summary_messages: List[str] = []

                for uploaded_file in uploaded_files:
                    rows, method_used = extract_transactions_for_csv(uploaded_file, uploaded_file.name, user['id'])
                    if rows:
                        csv_rows.extend(rows)
                        summary_messages.append(f"📄 {uploaded_file.name}: {len(rows)} rows ({method_used.upper()})")
                    else:
                        # Check if it's an image-based PDF
                        if uploaded_file.name.lower().endswith('.pdf'):
                            summary_messages.append(f"⚠️ {uploaded_file.name}: Image-based PDF detected. Use OCR or provide text-selectable PDF/CSV/Excel.")
                        else:
                            summary_messages.append(f"⚠️ {uploaded_file.name}: No transactions detected.")

            if csv_rows:
                df_preview = pd.DataFrame(csv_rows)
                st.success(f"✅ Extracted {len(csv_rows)} transactions. Review below or download as CSV.")
                for msg in summary_messages:
                    st.caption(msg)
                st.dataframe(df_preview, use_container_width=True)

                csv_data = df_preview.to_csv(index=False).encode('utf-8')
                timestamp = int(time.time())
                st.download_button(
                    "⬇️ Download Converted CSV",
                    csv_data,
                    file_name=f"converted_transactions_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"csv_download_{timestamp}"
                )
            else:
                for msg in summary_messages:
                    st.caption(msg)
                st.warning("No transactions extracted from the selected files. Please review the input.")
    
    # Show transactions with week info
    st.subheader("📊 Your Transactions (with Week Info)")
    
    try:
        with st.spinner("🔄 Loading transactions..."):
            transactions = db.get_user_transactions(user['id'])
        
        if transactions:
            st.success(f"📊 Loaded {len(transactions)} transactions")
            
            # Analyze transaction data
            st.caption("🔍 Analyzing transaction data...")
            
            # Count by asset type
            asset_types = {}
            channels = {}
            missing_weeks = []
            
            for trans in transactions:
                # Asset type count
                asset_type = trans.get('asset_type', 'Unknown')
                asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
                
                # Channel count
                channel = trans.get('channel', 'Direct')
                channels[channel] = channels.get(channel, 0) + 1
                
                # Check week info
                if not trans.get('week_label'):
                    missing_weeks.append(trans)
            
            # Show summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Transactions", len(transactions))
            with col2:
                st.metric("📅 Missing Week Info", len(missing_weeks))
            with col3:
                st.metric("🎯 Unique Channels", len(channels))
            
            # Asset type breakdown
            st.caption("📈 Asset Type Breakdown:")
            for asset_type, count in asset_types.items():
                st.caption(f"   • {asset_type}: {count} transactions")
            # Channel breakdown
            st.caption("📁 Channel Breakdown:")
            for channel, count in channels.items():
                st.caption(f"   • {channel}: {count} transactions")
            
            # Create display table
            st.caption("📋 Creating transaction table...")
            trans_data = []
            for trans in transactions:
                week_status = "✅" if trans.get('week_label') else "❌"
                trans_data.append({
                    'Ticker': trans['ticker'],
                    'Name': trans['stock_name'],
                    'Date': trans['transaction_date'],
                    'Week': trans.get('week_label', 'Not calculated'),
                    'Status': week_status,
                    'Type': trans['transaction_type'],
                    'Quantity': f"{trans['quantity']:,.0f}",
                    'Price': f"₹{trans['price']:,.2f}",
                    'Channel': trans.get('channel', 'Direct')
                })
            
            df_transactions = pd.DataFrame(trans_data)
            st.dataframe(df_transactions, use_container_width=True)
            
            # Week calculation status
            if missing_weeks:
                st.warning(f"⚠️ {len(missing_weeks)} transactions missing week information")
                #st.caption("🔧 To fix this, run: `streamlit run fix_week_calculation.py`")
                
                # Show which transactions are missing week info
                with st.expander("🔍 Transactions Missing Week Info"):
                    missing_data = []
                    for trans in missing_weeks:
                        missing_data.append({
                            'Ticker': trans['ticker'],
                            'Date': trans['transaction_date'],
                            'Type': trans['transaction_type'],
                            'Channel': trans.get('channel', 'Direct')
                        })
                    df_missing = pd.DataFrame(missing_data)
                    st.dataframe(df_missing, use_container_width=True)
            else:
                st.success("✅ All transactions have week information calculated!")
        else:
            st.info("No transactions found. Upload CSV files to see your transaction history.")
    
    except Exception as e:
        st.error(f"❌ Error loading transactions: {str(e)}")
        #st.caption("🔧 This might be a database connection issue. Check your Supabase connection.")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app function"""
    if st.session_state.user is None:
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()
