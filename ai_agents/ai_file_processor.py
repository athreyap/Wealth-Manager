"""
Unified AI-Powered File Processor
Handles CSV, PDF, Excel, and any other file format intelligently
"""

import openai
import json
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .base_agent import BaseAgent

try:
    from enhanced_price_fetcher import EnhancedPriceFetcher
except ImportError:  # pragma: no cover - runtime dependency
    EnhancedPriceFetcher = None  # type: ignore[misc]


class AIFileProcessor(BaseAgent):
    """
    Universal AI agent that processes ANY file type and extracts transactions
    Supports: CSV, PDF, Excel, TXT, and more
    """
    
    def __init__(self):
        super().__init__(
            agent_id="ai_file_processor",
            agent_name="AI Universal File Processor"
        )
        
        self.capabilities = [
            "csv_processing",
            "pdf_processing",
            "excel_processing",
            "transaction_extraction",
            "data_validation",
            "missing_data_inference",
            "sector_classification",
            "asset_type_detection",
            "price_handling"
        ]
        
        # Defer OpenAI client initialization until it's actually needed
        # This avoids issues with st.secrets not being available during import
        self.openai_client = None
        self._openai_initialized = False
        
        # Defer price fetcher initialization until it's actually needed
        # EnhancedPriceFetcher.__init__ may access st.secrets which isn't available during import
        self.price_fetcher = None
        self._price_fetcher_initialized = False

        self._price_cache: Dict[Tuple[str, str, str], Optional[float]] = {}
        self.processed_transactions = []
    
    def _get_openai_client(self):
        """Lazy initialization of OpenAI client - only when needed"""
        if self._openai_initialized:
            return self.openai_client
        
        self._openai_initialized = True
        try:
            import streamlit as st
            # Check if secrets are available
            if "api_keys" not in st.secrets:
                raise KeyError("'api_keys' not found in st.secrets")
            if "open_ai" not in st.secrets.get("api_keys", {}):
                raise KeyError("'open_ai' not found in st.secrets['api_keys']")
            self.openai_client = openai.OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
            self.logger.info("OpenAI client initialized successfully")
            return self.openai_client
        except KeyError as e:
            self.logger.error(f"Failed to initialize OpenAI client - missing secret key: {e}")
            self.logger.error("   Please ensure st.secrets['api_keys']['open_ai'] is set in .streamlit/secrets.toml")
            self.openai_client = None
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.logger.error(f"   Error type: {type(e).__name__}")
            self.openai_client = None
            return None
    
    def process_file(self, file_data: Any, filename: str) -> List[Dict[str, Any]]:
        """
        Universal file processor - handles ANY file type
        
        Args:
            file_data: File object or file content
            filename: Name of the file
            
        Returns:
            List of extracted transactions
        """
        
        self.update_status("analyzing")
        
        # Initialize OpenAI client lazily (only when needed)
        openai_client = self._get_openai_client()
        
        try:
            if not openai_client:
                self.logger.error("OpenAI client not available")
                return []
            
            # Check if Vision API text was already extracted (to avoid re-extraction)
            vision_text_pre_extracted = None
            if hasattr(file_data, '_vision_api_text'):
                vision_text_pre_extracted = file_data._vision_api_text
                print(f"[AI_FILE_PROCESSOR] ✅ Found pre-extracted Vision API text ({len(vision_text_pre_extracted)} characters)")
                self.logger.info(f"Using pre-extracted Vision API text ({len(vision_text_pre_extracted)} characters)")
            else:
                print(f"[AI_FILE_PROCESSOR] ⚠️ No pre-extracted Vision API text found on file_data object")
                print(f"[AI_FILE_PROCESSOR]   file_data type: {type(file_data)}, hasattr check: {hasattr(file_data, '_vision_api_text')}")
            
            # Detect file type and extract content
            if vision_text_pre_extracted:
                # Use the already-extracted Vision API text
                file_content = vision_text_pre_extracted
                file_type = 'pdf'
                print(f"[AI_FILE_PROCESSOR] ✅ Using pre-extracted Vision API text for {filename}")
                self.logger.info(f"Using pre-extracted Vision API text for {filename}")
            else:
                print(f"[AI_FILE_PROCESSOR] ⚠️ No pre-extracted text, calling _extract_file_content...")
                file_content, file_type = self._extract_file_content(file_data, filename)
            
            if not file_content:
                self.logger.error(f"Could not extract content from {filename} (file_type: {file_type})")
                print(f"[AI_FILE_PROCESSOR] ❌ Could not extract content from {filename} (file_type: {file_type})")
                # If it's a PDF that failed, try Vision API directly as last resort
                if filename.lower().endswith('.pdf') and file_type == 'error':
                    self.logger.info("Attempting direct Vision API call for PDF...")
                    file_data.seek(0)
                    vision_text = self._extract_pdf_with_vision(file_data, filename)
                    if vision_text and vision_text.strip():
                        self.logger.info(f"Vision API successfully extracted {len(vision_text)} characters")
                        file_content = vision_text
                        file_type = 'pdf'
                    else:
                        self.logger.error("Vision API also failed - cannot process this PDF")
                        return []
                else:
                    print(f"[AI_FILE_PROCESSOR] ❌ File content extraction failed for {filename}")
                    return []
            
            # Log content preview for debugging
            content_preview = file_content[:200] if len(file_content) > 200 else file_content
            print(f"[AI_FILE_PROCESSOR] ✅ Extracted {len(file_content)} characters from {filename}")
            print(f"[AI_FILE_PROCESSOR]   Content preview: {content_preview}...")
            
            # Use AI to extract transactions
            transactions = self._ai_extract_transactions(file_content, filename, file_type)
            
            # Validate and enhance transactions
            validated_transactions = self._validate_and_enhance(transactions, filename)
            
            # Cache transactions
            self.processed_transactions = validated_transactions
            
            self.update_status("active")
            return validated_transactions
            
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            self.update_status("error")
            return []
    
    def _extract_pdf_with_ocr(self, file_data: Any) -> str:
        """
        Extract text from image-based PDF using OCR.
        Tries multiple OCR methods:
        1. Tesseract (if available - works locally)
        2. EasyOCR (pure Python - works on Streamlit Cloud)
        
        Returns extracted text or empty string if OCR fails or is unavailable.
        """
        file_data.seek(0)
        pdf_bytes = file_data.read()
        
        # Method 1: Try Tesseract (works locally, may not work on Streamlit Cloud)
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            from PIL import Image
            
            self.logger.info("Attempting Tesseract OCR...")
            images = convert_from_bytes(pdf_bytes, dpi=300, fmt='RGB')
            self.logger.info(f"Converted {len(images)} pages to images")
            
            all_text = []
            for page_num, image in enumerate(images, 1):
                try:
                    self.logger.info(f"Processing page {page_num}/{len(images)} with Tesseract...")
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    
                    if page_text and page_text.strip():
                        all_text.append(f"--- Page {page_num} ---\n{page_text}\n")
                        self.logger.info(f"Page {page_num}: Extracted {len(page_text)} characters")
                except Exception as e:
                    # Tesseract not available or failed - try EasyOCR
                    self.logger.warning(f"Tesseract failed for page {page_num}: {str(e)}")
                    break
            
            if all_text:
                combined_text = "\n".join(all_text)
                if combined_text.strip():
                    self.logger.info(f"Tesseract extracted {len(combined_text)} total characters")
                    return combined_text
        except Exception as e:
            self.logger.warning(f"Tesseract not available: {str(e)}")
            self.logger.info("Falling back to EasyOCR (works on Streamlit Cloud)...")
        
        # Method 2: Try EasyOCR (pure Python - works on Streamlit Cloud)
        try:
            import easyocr
            import numpy as np
            
            # Try to convert PDF to images - may fail on Streamlit Cloud if Poppler not available
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(pdf_bytes, dpi=300, fmt='RGB')
            except Exception as pdf_conv_error:
                self.logger.warning(f"pdf2image failed (may need Poppler): {str(pdf_conv_error)}")
                self.logger.warning("Note: On Streamlit Cloud, Poppler may not be available")
                return ""
            
            self.logger.info("Attempting EasyOCR (Streamlit Cloud compatible)...")
            # Initialize EasyOCR reader (English only for speed)
            # This will download models on first run (~100MB, cached after first use)
            reader = easyocr.Reader(['en'], gpu=False)
            self.logger.info(f"Converted {len(images)} pages to images")
            
            all_text = []
            for page_num, image in enumerate(images, 1):
                try:
                    self.logger.info(f"Processing page {page_num}/{len(images)} with EasyOCR...")
                    # Convert PIL image to numpy array
                    img_array = np.array(image)
                    # Run OCR
                    results = reader.readtext(img_array)
                    # Extract text from results
                    page_text = '\n'.join([result[1] for result in results])
                    
                    if page_text and page_text.strip():
                        all_text.append(f"--- Page {page_num} ---\n{page_text}\n")
                        self.logger.info(f"Page {page_num}: Extracted {len(page_text)} characters")
                    else:
                        self.logger.warning(f"Page {page_num}: No text extracted")
                except Exception as e:
                    self.logger.warning(f"Page {page_num}: EasyOCR failed: {str(e)}")
                    continue
            
            combined_text = "\n".join(all_text)
            if combined_text.strip():
                self.logger.info(f"EasyOCR extracted {len(combined_text)} total characters from {len(images)} pages")
                return combined_text
            else:
                self.logger.warning("EasyOCR completed but no text was extracted")
                return ""
                
        except ImportError as e:
            self.logger.warning(f"EasyOCR not available: {e}")
            self.logger.warning("Install with: pip install easyocr")
            return ""
        except Exception as e:
            self.logger.error(f"EasyOCR extraction failed: {str(e)}")
            return ""
        
        return ""
    
    def _extract_pdf_with_vision(self, file_data: Any, filename: str) -> str:
        """
        Extract text from image-based PDF using GPT-4 Vision API.
        This is the final fallback when OCR is not available.
        """
        self.logger.info(f"[VISION_API] Starting GPT-4 Vision extraction for {filename}...")
        
        openai_client = self._get_openai_client()
        if not openai_client:
            self.logger.error("[VISION_API] ❌ OpenAI client not available for Vision API")
            self.logger.error("[VISION_API] Check if OpenAI API key is configured in Streamlit secrets")
            return ""
        
        self.logger.info("[VISION_API] ✅ OpenAI client is available, proceeding with Vision API...")
        
        try:
            import base64
            import io
            
            # Convert PDF to images using PyMuPDF (fitz) - doesn't require Poppler
            from PIL import Image
            try:
                # Try standard import first
                import fitz  # PyMuPDF
                self.logger.info("[VISION_API] Using PyMuPDF to convert PDF to images (no system dependencies required)...")
                use_fitz = True
            except ImportError:
                self.logger.warning("[VISION_API] PyMuPDF not available, trying pdf2image...")
                use_fitz = False
                # Fallback to pdf2image if PyMuPDF not available
                try:
                    from pdf2image import convert_from_bytes
                    self.logger.info("[VISION_API] Using pdf2image to convert PDF to images (requires Poppler)...")
                    use_fitz = False
                except ImportError:
                    self.logger.error("[VISION_API] ❌ Neither PyMuPDF nor pdf2image available for Vision API")
                    self.logger.info("[VISION_API] Install PyMuPDF with: pip install pymupdf (recommended, no system dependencies)")
                    self.logger.info("[VISION_API] Or install pdf2image with: pip install pdf2image (requires Poppler)")
                    self.logger.info("[VISION_API] Note: After installing, restart Streamlit to load the module")
                    return ""
            
            # Read PDF bytes
            file_data.seek(0)
            if hasattr(file_data, 'read'):
                pdf_bytes = file_data.read()
            elif hasattr(file_data, 'getvalue'):
                pdf_bytes = file_data.getvalue()
                if isinstance(pdf_bytes, str):
                    pdf_bytes = pdf_bytes.encode('utf-8')
            else:
                pdf_bytes = file_data
            
            # Convert PDF pages to images
            images = []
            try:
                if use_fitz:
                    self.logger.info(f"[VISION_API] Opening PDF with PyMuPDF (filename: {filename})...")
                    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    total_pages = len(pdf_doc)
                    self.logger.info(f"[VISION_API] ✅ PDF opened successfully with {total_pages} pages")
                    for page_num in range(total_pages):
                        try:
                            page = pdf_doc[page_num]
                            # Render page to image (150 DPI optimized for Streamlit Cloud - 4x less memory)
                            mat = fitz.Matrix(150/72, 150/72)  # 150 DPI (reduced from 300 for memory optimization)
                            pix = page.get_pixmap(matrix=mat)
                            # Convert to PIL Image
                            img_data = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_data))
                            
                            # Compress image if too large (max 2048px dimension)
                            max_dimension = 2048
                            if img.width > max_dimension or img.height > max_dimension:
                                ratio = min(max_dimension / img.width, max_dimension / img.height)
                                new_size = (int(img.width * ratio), int(img.height * ratio))
                                img = img.resize(new_size, Image.Resampling.LANCZOS)
                                self.logger.info(f"[VISION_API] Page {page_num + 1}: Resized from {img.width}x{img.height} to {new_size[0]}x{new_size[1]}")
                            
                            images.append(img)
                            # Free pixmap memory immediately
                            pix = None
                            if (page_num + 1) % 3 == 0 or page_num == 0:  # Log every 3rd page or first page
                                self.logger.info(f"[VISION_API] Converted page {page_num + 1}/{total_pages} to image")
                        except Exception as page_error:
                            self.logger.warning(f"[VISION_API] Failed to convert page {page_num + 1} to image: {str(page_error)}")
                            continue
                    pdf_doc.close()
                    self.logger.info(f"[VISION_API] ✅ Successfully converted {len(images)}/{total_pages} pages to images")
                else:
                    # Use pdf2image
                    self.logger.info("[VISION_API] Converting PDF to images using pdf2image...")
                    images = convert_from_bytes(pdf_bytes, dpi=150, fmt='RGB')  # Reduced from 300 for memory optimization
                    self.logger.info(f"[VISION_API] ✅ pdf2image converted {len(images)} pages")
            except Exception as conv_error:
                self.logger.error(f"[VISION_API] ❌ Failed to convert PDF to images: {str(conv_error)}")
                import traceback
                self.logger.error(f"[VISION_API] Conversion error traceback: {traceback.format_exc()}")
                return ""
            
            if not images:
                self.logger.error("[VISION_API] ❌ No images extracted from PDF for Vision API")
                return ""
            
            self.logger.info(f"[VISION_API] ✅ Ready to process {len(images)} PDF pages with GPT-4 Vision")
            
            # Process images with GPT-4 Vision (process first 10 pages to avoid token limits)
            max_pages = min(10, len(images))
            all_text = []
            
            for page_num, image in enumerate(images[:max_pages], 1):
                try:
                    # Convert PIL Image to base64 with compression
                    buffered = io.BytesIO()
                    # Use optimize=True and compress_level for smaller file size
                    image.save(buffered, format="PNG", optimize=True, compress_level=6)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Call GPT-4 Vision API
                    self.logger.info(f"[VISION_API] Processing page {page_num}/{max_pages} with GPT-4 Vision API...")
                    openai_client = self._get_openai_client()
                    if not openai_client:
                        raise Exception("OpenAI client not available")
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

CRITICAL: Do NOT use filename "{filename}" or channel/broker name as stock_name. Extract the actual fund/security name from the document. If stock_name is missing or looks like a channel name, set it to null.

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
                    
                    page_text = response.choices[0].message.content
                    if page_text and page_text.strip() and "No transactions" not in page_text:
                        all_text.append(f"--- Page {page_num} ---\n{page_text}\n")
                        self.logger.info(f"[VISION_API] ✅ Page {page_num}: Extracted transaction data ({len(page_text)} chars)")
                    else:
                        self.logger.info(f"[VISION_API] Page {page_num}: No transaction data found")
                        
                except Exception as page_error:
                    self.logger.error(f"[VISION_API] ❌ GPT-4 Vision failed for page {page_num}: {str(page_error)}")
                    import traceback
                    self.logger.error(f"[VISION_API] Page error traceback: {traceback.format_exc()}")
                    continue
            
            if all_text:
                combined_text = "\n".join(all_text)
                self.logger.info(f"[VISION_API] ✅ Successfully extracted {len(combined_text)} characters from {len(all_text)} pages")
                return combined_text
            else:
                self.logger.warning("[VISION_API] ⚠️ GPT-4 Vision completed but no transaction data extracted from any pages")
                return ""
                
        except ImportError as import_error:
            self.logger.error(f"Required library not available for Vision API: {import_error}")
            self.logger.info("Install PyMuPDF with: pip install pymupdf (recommended, no system dependencies)")
            import traceback
            self.logger.debug(f"Import error traceback: {traceback.format_exc()}")
            return ""
        except Exception as e:
            self.logger.error(f"GPT-4 Vision extraction failed: {str(e)}")
            import traceback
            self.logger.error(f"Vision API error traceback: {traceback.format_exc()}")
            return ""
    
    def _extract_file_content(self, file_data: Any, filename: str) -> tuple:
        """Extract content from any file type"""
        
        try:
            file_ext = filename.lower().split('.')[-1]
            
            # CSV Files
            if file_ext == 'csv':
                content = file_data.getvalue().decode('utf-8')
                return content, 'csv'
            
            # Excel Files
            elif file_ext in ['xlsx', 'xls']:
                # Read all sheets from Excel file
                try:
                    # Check if openpyxl is available for .xlsx files
                    if file_ext == 'xlsx':
                        try:
                            import openpyxl
                        except ImportError:
                            error_msg = "Missing dependency 'openpyxl'. Install with: pip install openpyxl"
                            self.logger.error(f"{error_msg} (for {filename})")
                            print(f"[AI_FILE_PROCESSOR] ❌ {error_msg}")
                            # Try to read anyway - pandas might have a fallback
                    
                    # Check if xlrd is available for .xls files
                    if file_ext == 'xls':
                        try:
                            import xlrd
                        except ImportError:
                            error_msg = "Missing dependency 'xlrd'. Install with: pip install xlrd"
                            self.logger.warning(f"{error_msg} (for {filename})")
                            print(f"[AI_FILE_PROCESSOR] ⚠️ {error_msg}")
                    
                    all_sheets = pd.read_excel(file_data, sheet_name=None, engine='openpyxl' if file_ext == 'xlsx' else None)
                    if not all_sheets:
                        self.logger.warning(f"Excel file {filename} has no sheets")
                        print(f"[AI_FILE_PROCESSOR] ⚠️ Excel file {filename} has no sheets")
                        return None, 'error'
                    
                    # Combine all sheets into CSV format with clear structure
                    sheet_contents = []
                    for sheet_name, df in all_sheets.items():
                        if df.empty:
                            self.logger.warning(f"Sheet '{sheet_name}' in {filename} is empty")
                            continue
                        
                        # Log column names for debugging
                        print(f"[AI_FILE_PROCESSOR] Sheet '{sheet_name}' columns: {list(df.columns)}")
                        print(f"[AI_FILE_PROCESSOR] Sheet '{sheet_name}' shape: {df.shape}")
                        if not df.empty:
                            print(f"[AI_FILE_PROCESSOR] Sheet '{sheet_name}' first row sample: {df.iloc[0].to_dict()}")
                        
                        # Add sheet name as header comment
                        sheet_contents.append(f"=== Sheet: {sheet_name} ===")
                        # Convert to CSV with headers - this is what AI will see
                        csv_content = df.to_csv(index=False)
                        sheet_contents.append(csv_content)
                    
                    if not sheet_contents:
                        self.logger.warning(f"Excel file {filename} has no data in any sheet")
                        print(f"[AI_FILE_PROCESSOR] ⚠️ Excel file {filename} has no data in any sheet")
                        return None, 'error'
                    
                    content = "\n".join(sheet_contents)
                    self.logger.info(f"Excel file {filename}: Extracted {len(all_sheets)} sheet(s), {len(content)} characters")
                    print(f"[AI_FILE_PROCESSOR] ✅ Excel file {filename}: Extracted {len(all_sheets)} sheet(s), {len(content)} characters")
                    print(f"[AI_FILE_PROCESSOR] CSV content preview (first 500 chars): {content[:500]}")
                    return content, 'excel'
                except ImportError as e:
                    error_msg = f"Missing Excel dependency. For .xlsx files, install: pip install openpyxl. For .xls files, install: pip install xlrd"
                    self.logger.error(f"{error_msg} (error: {e})")
                    print(f"[AI_FILE_PROCESSOR] ❌ {error_msg}")
                    return None, 'error'
                except Exception as e:
                    error_msg = f"Error reading Excel file {filename}: {e}"
                    self.logger.error(error_msg)
                    print(f"[AI_FILE_PROCESSOR] ❌ {error_msg}")
                    # If it's a dependency error, provide helpful message
                    if 'openpyxl' in str(e).lower() or 'xlrd' in str(e).lower():
                        print(f"[AI_FILE_PROCESSOR] 💡 Install missing dependency: pip install openpyxl xlrd")
                    return None, 'error'
            
            # PDF Files
            elif file_ext == 'pdf':
                # Store filename for Vision API
                pdf_filename = filename
                
                # Try pdfplumber first (better text extraction)
                pdf_text = ""
                
                try:
                    import pdfplumber
                    import io
                    
                    # Convert file_data to BytesIO for pdfplumber compatibility
                    file_data.seek(0)
                    if hasattr(file_data, 'read'):
                        pdf_bytes = file_data.read()
                    elif hasattr(file_data, 'getvalue'):
                        pdf_bytes = file_data.getvalue()
                        if isinstance(pdf_bytes, str):
                            pdf_bytes = pdf_bytes.encode('utf-8')
                    else:
                        pdf_bytes = file_data
                    
                    # Ensure we have bytes
                    if isinstance(pdf_bytes, str):
                        pdf_bytes = pdf_bytes.encode('latin-1')
                    
                    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                        total_pages = len(pdf.pages)
                        self.logger.info(f"[PDF_EXTRACT] PDF opened with {total_pages} pages")
                        
                        for page_num, page in enumerate(pdf.pages, 1):
                            try:
                                page_text = page.extract_text()
                                if not page_text or not page_text.strip():
                                    words = page.extract_words()
                                    if words:
                                        page_text = ' '.join([w.get('text', '') for w in words if w.get('text')])
                                
                                if page_text and page_text.strip():
                                    pdf_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                                    self.logger.info(f"[PDF_EXTRACT] Page {page_num}: Extracted {len(page_text)} chars")
                            except Exception as page_error:
                                self.logger.warning(f"[PDF_EXTRACT] Page {page_num}: Error - {str(page_error)}")
                                continue
                    
                    if pdf_text.strip():
                        self.logger.info(f"[PDF_EXTRACT] ✅ pdfplumber extracted {len(pdf_text)} characters")
                        return pdf_text, 'pdf'
                    else:
                        self.logger.warning("[PDF_EXTRACT] pdfplumber returned empty text")
                        
                except ImportError as import_error:
                    self.logger.warning(f"[PDF_EXTRACT] pdfplumber not installed: {import_error}")
                except Exception as e:
                    self.logger.warning(f"[PDF_EXTRACT] pdfplumber failed: {e}")
                
                # Try PyPDF2 as fallback
                if not pdf_text.strip():
                    try:
                        import PyPDF2
                        import io
                        
                        file_data.seek(0)
                        if hasattr(file_data, 'read'):
                            pdf_bytes = file_data.read()
                        elif hasattr(file_data, 'getvalue'):
                            pdf_bytes = file_data.getvalue()
                            if isinstance(pdf_bytes, str):
                                pdf_bytes = pdf_bytes.encode('utf-8')
                        else:
                            pdf_bytes = file_data
                        
                        if isinstance(pdf_bytes, str):
                            pdf_bytes = pdf_bytes.encode('latin-1')
                        
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                        pdf_text = ""
                        
                        for page_num, page in enumerate(pdf_reader.pages, 1):
                            try:
                                page_text = page.extract_text()
                                if page_text and page_text.strip():
                                    pdf_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                            except Exception as page_error:
                                continue
                        
                        if pdf_text.strip():
                            self.logger.info(f"[PDF_EXTRACT] ✅ PyPDF2 extracted {len(pdf_text)} characters")
                            return pdf_text, 'pdf'
                        else:
                            self.logger.warning("[PDF_EXTRACT] PyPDF2 also returned empty text")
                    except Exception as e:
                        self.logger.warning(f"[PDF_EXTRACT] PyPDF2 failed: {e}")
                
                # If both pdfplumber and PyPDF2 failed, try OCR
                if not pdf_text.strip():
                    self.logger.warning("[PDF_EXTRACT] Trying OCR as fallback...")
                    try:
                        file_data.seek(0)
                        ocr_text = self._extract_pdf_with_ocr(file_data)
                        if ocr_text and ocr_text.strip():
                            self.logger.info(f"[PDF_EXTRACT] ✅ OCR extracted {len(ocr_text)} characters")
                            return ocr_text, 'pdf'
                        else:
                            self.logger.warning("[PDF_EXTRACT] OCR failed or not available")
                    except Exception as ocr_error:
                        self.logger.warning(f"[PDF_EXTRACT] OCR error: {ocr_error}")
                
                # Final fallback: GPT-4 Vision API
                if not pdf_text.strip():
                    self.logger.warning("[PDF_EXTRACT] All text extraction methods failed - trying GPT-4 Vision API...")
                    try:
                        file_data.seek(0)
                        self.logger.info(f"[PDF_EXTRACT] 🚀 Calling GPT-4 Vision API for {pdf_filename}...")
                        vision_text = self._extract_pdf_with_vision(file_data, pdf_filename)
                        if vision_text and vision_text.strip():
                            self.logger.info(f"[PDF_EXTRACT] ✅ GPT-4 Vision extracted {len(vision_text)} characters")
                            return vision_text, 'pdf'
                        else:
                            self.logger.error("[PDF_EXTRACT] ❌ GPT-4 Vision also returned empty text")
                            return None, 'error'
                    except Exception as vision_error:
                        self.logger.error(f"[PDF_EXTRACT] ❌ GPT-4 Vision failed: {str(vision_error)}")
                        import traceback
                        self.logger.error(f"[PDF_EXTRACT] Vision error: {traceback.format_exc()}")
                        return None, 'error'
                
                # If we get here, all methods failed
                if not pdf_text.strip():
                    self.logger.error("[PDF_EXTRACT] ❌ All PDF extraction methods failed")
                    return None, 'error'
                else:
                    return pdf_text, 'pdf'
            
            # Text Files
            elif file_ext in ['txt', 'text']:
                content = file_data.getvalue().decode('utf-8')
                return content, 'txt'
            
            # Unknown format - try to read as text
            else:
                try:
                    content = file_data.getvalue().decode('utf-8')
                    return content, 'unknown'
                except:
                    return None, 'unsupported'
                    
        except Exception as e:
            self.logger.error(f"Error extracting file content: {e}")
            return None, 'error'
    
    def _ai_extract_transactions(self, file_content: str, filename: str, file_type: str) -> List[Dict[str, Any]]:
        """Use AI to extract transactions from any file format"""
        
        try:
            # Detect if this is Vision API extracted text (has pipe separators or structured format)
            # Only treat as Vision API text if it's from a PDF and has the characteristic format
            # CSV/Excel content will have commas/tabs as delimiters, not pipes with "Date:" labels
            # Check first 500 chars to avoid false positives
            sample = file_content[:500] if len(file_content) > 500 else file_content
            is_vision_api_text = (
                file_type == 'pdf' and 
                (
                    ('Date:' in sample and 'Ticker:' in sample and '|' in sample) or
                    ('Date:' in sample and 'Quantity:' in sample and '|' in sample)
                ) and
                # CSV/Excel would have comma or tab delimiters, not pipe with labels
                not (',' in sample and sample.count(',') > sample.count('|'))
            )
            
            if is_vision_api_text:
                # Vision API format: "Date: YYYY-MM-DD | Ticker: SYMBOL | Name: ... | Quantity: ..."
                prompt = f"""
You are extracting financial transactions from Vision API extracted text. The text may be in structured format with pipe separators or line-by-line format.

Extract ALL transactions from the following text and convert to JSON array. The text may contain:
- Structured lines like: "Date: 2024-01-15 | Ticker: RELIANCE.NS | Name: Reliance Industries | Quantity: 100 | Price: 2500 | Amount: 250000 | Type: buy"
- Or table-like text with headers and rows
- Or unstructured transaction descriptions

Schema (JSON array of objects):
[
  {{
    "date": "YYYY-MM-DD",
    "ticker": "exchange-ready ticker",
    "stock_name": "full security name",
    "scheme_name": null,
    "quantity": 0.0,
    "price": 0.0,
    "amount": 0.0,
    "transaction_type": "buy" | "sell",
    "asset_type": "stock" | "mutual_fund" | "bond" | "pms" | "aif",
    "sector": "Sector name or Unknown",
    "channel": "{filename}"
  }}
]

Rules:
- Extract date in any format and normalize to YYYY-MM-DD
- Extract ticker/symbol (normalize to exchange format like RELIANCE.NS if needed)
- Extract stock name/security name (NOT filename or channel name - if missing, set to null)
- Extract quantity (shares/units)
- Extract price per unit
- Extract amount (total value)
- Determine transaction_type: "buy" for purchases/buys, "sell" for sales/redemptions
- Infer asset_type: numeric codes → mutual_fund, .NS/.BO → stock, etc.
- Set sector to "Unknown" if not found
- Set channel to filename: "{filename}"

IMPORTANT:
- Extract EVERY transaction you find, even if some fields are missing
- If quantity or amount is missing, try to calculate from other fields
- If date is missing, skip that transaction
- Return empty array [] ONLY if NO transactions are found

Extracted text:
{file_content}
"""
            else:
                # CSV/Excel format - AI needs to intelligently map columns
                prompt = f"""
You are an expert at analyzing financial transaction files. The file below is in CSV format (converted from Excel).

YOUR TASK:
1. **Analyze the column headers** in the first row
2. **Understand what each column represents** by looking at column names AND sample data
3. **Map columns intelligently** to the standard schema below
4. **Extract ALL transactions** from the data

IMPORTANT COLUMN MAPPING RULES:
- **Date columns**: Look for "Date", "Execution date", "Txn Date", "Transaction Date", "Trade Date", etc. → map to `"date"`
- **Ticker/Symbol columns**: Look for "Symbol", "Ticker", "Scrip Symbol", "Trading Symbol", "ISIN" (use ISIN if no symbol) → map to `"ticker"`
- **Stock Name columns**: Look for "Stock name", "Stock Name", "Security Name", "Scrip Name", "Name" → map to `"stock_name"`
  - CRITICAL: Do NOT use filename, channel, or broker name as stock_name
  - If stock_name column is missing or contains channel/broker name, set it to null (will be fetched from AMFI/mftool)
- **Quantity columns**: Look for "Quantity", "Qty", "Units", "No. of Units", "Number of Units", "Units Held", "Unit Balance", "Total Units", "Current Units", "Holding Quantity", "Balance Units" → map to `"quantity"` (THIS IS CRITICAL - quantity must be extracted correctly!)
- **Amount/Value columns**: Look for "Value", "Amount", "Total Value", "Consideration", "Trade Value" → map to `"amount"`
- **Price columns**: Look for "Price", "Rate", "NAV", "Per Unit Price" → map to `"price"` (if missing, we'll calculate from amount/quantity)
- **Transaction Type**: Look for "Type", "Transaction Type", "Action", "Buy/Sell" → map to `"transaction_type"` ("BUY" → "buy", "SELL" → "sell")
- **Exchange**: Look for "Exchange", "Exchange Name" → use for context
- **Channel**: Use filename `{filename}` or any "Broker"/"Platform" column

CRITICAL INSTRUCTIONS:
- **Quantity is the MOST IMPORTANT field** - if the file has a "Quantity", "Units", "Number of Units", "Units Held", or similar column, extract it EXACTLY as shown, do NOT calculate it
- If quantity is present in the file (even if the column is named "Units", "No. of Units", "Units Held", etc.), use it as-is (even if it's 0)
- Only calculate quantity if it's completely missing AND you have amount+price
- If you see "Quantity" or "Units" column with values like 5, 3000, 151218, etc. - extract those EXACT values, do NOT recalculate
- For mutual funds, quantity is often called "Units" or "No. of Units" - map these to `"quantity"`
- If you see "Value" column - that's the total amount, map it to `"amount"`
- If price column is missing but quantity and amount exist, set price to 0 (we'll calculate it later)

OUTPUT SCHEMA (JSON array):
[
  {{
    "date": "YYYY-MM-DD",
    "ticker": "exchange-ready ticker",
    "stock_name": "full security name",
    "scheme_name": null,
    "quantity": <EXACT VALUE FROM FILE IF PRESENT, OR 0>,
    "price": <VALUE FROM FILE OR 0>,
    "amount": <VALUE FROM FILE OR 0>,
    "transaction_type": "buy" | "sell",
    "asset_type": "stock" | "mutual_fund" | "bond" | "pms" | "aif",
    "sector": "Sector name or Unknown",
    "channel": "{filename}"
  }}
]

EXAMPLE MAPPING:
If CSV has: "Stock name, Symbol, Quantity, Value, Execution date and time, Type"
Map to:
- "Stock name" → stock_name
- "Symbol" → ticker  
- "Quantity" → quantity (EXTRACT EXACT VALUE!)
- "Value" → amount
- "Execution date and time" → date
- "Type" → transaction_type

File content (CSV format with headers):
{file_content[:12000] if len(file_content) <= 12000 else file_content[:10000] + '\n... (truncated) ...'}

Output ONLY the JSON array—no commentary or explanation.
"""

            content_length = len(file_content)
            print(f"[AI_EXTRACT] Content length: {content_length} chars, processing with AI...")
            
            # Call OpenAI without timeout - let it take as long as needed
            openai_client = self._get_openai_client()
            if not openai_client:
                self.logger.error("OpenAI client not available")
                return []
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # GPT-4o for better file processing
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial data extraction AI specializing in parsing investment transaction files.

CAPABILITIES:
- Extract transactions from ANY file format (CSV, PDF, Excel, Text)
- Intelligently map columns and fields
- Infer missing data using domain knowledge
- Detect asset types automatically
- Normalize dates and formats
- Handle messy, unstructured data

INTELLIGENCE FEATURES:
- Recognize stock tickers vs mutual fund codes
- Infer sectors from company names
- Detect transaction types from context
- Handle missing prices gracefully
- Smart column mapping even with different naming

CRITICAL RULES:
1. Extract ALL transactions from the file
2. Return ONLY valid JSON array
3. Use 0 for missing prices (system handles fetching)
4. Infer missing data intelligently
5. Validate all data types"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=4000
                # Note: No timeout - let the API take as long as needed
            )
            
            # Extract JSON from response
            ai_response = response.choices[0].message.content.strip()
            print(f"[AI_EXTRACT] AI response length: {len(ai_response)} chars")
            print(f"[AI_EXTRACT] AI response preview: {ai_response[:500]}...")
            
            # Check if response is empty or just whitespace
            if not ai_response or len(ai_response.strip()) == 0:
                print(f"[AI_EXTRACT] ❌ AI returned empty response")
                print(f"[AI_EXTRACT]   Response object: {response}")
                print(f"[AI_EXTRACT]   Response choices: {response.choices if hasattr(response, 'choices') else 'N/A'}")
                return []
            
            transactions = self._extract_json(ai_response)
            
            if not transactions:
                print(f"[AI_EXTRACT] ⚠️ No transactions extracted from AI response")
                print(f"[AI_EXTRACT]   Full AI response: {ai_response}")
                print(f"[AI_EXTRACT]   Trying to parse response as-is...")
                # Try to see if there's any content that might be JSON
                if '[' in ai_response or '{' in ai_response:
                    print(f"[AI_EXTRACT]   Response contains JSON-like characters, but extraction failed")
            else:
                print(f"[AI_EXTRACT] ✅ Successfully extracted {len(transactions)} transactions")
                # Log first transaction as sample
                if transactions:
                    print(f"[AI_EXTRACT]   Sample transaction: {transactions[0]}")
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"AI extraction failed: {e}")
            print(f"[AI_EXTRACT] ❌ AI extraction failed: {e}")
            # Try fallback parsing
            return self._fallback_extraction(file_content, file_type)
    
    def _extract_json(self, ai_response: str) -> List[Dict[str, Any]]:
        """Extract JSON array from AI response"""
        
        try:
            # Find JSON array in response
            start_idx = ai_response.find('[')
            end_idx = ai_response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = ai_response[start_idx:end_idx]
                transactions = json.loads(json_str)
                
                if isinstance(transactions, list):
                    return transactions
            
            return []
            
        except Exception as e:
            self.logger.error(f"JSON extraction error: {e}")
            return []
    
    def _validate_and_enhance(self, transactions: List[Dict[str, Any]], source_filename: str) -> List[Dict[str, Any]]:
        """Validate and enhance transaction data"""
        
        validated = []
        skipped_count = 0
        for trans in transactions:
            try:
                # Skip if missing critical fields
                if not trans.get('ticker') or not trans.get('date'):
                    skipped_count += 1
                    if skipped_count <= 3:  # Log first 3 skipped transactions
                        print(f"[VALIDATE] ⚠️ Skipping transaction (missing ticker or date): {trans}")
                    continue
                
                # Normalize and validate fields
                transaction_type_raw = (
                    trans.get('transaction_type')
                    or trans.get('type')
                    or trans.get('action')
                    or trans.get('side')
                )
                amount_value = trans.get('amount', trans.get('value', 0))

                # Check original values to see if they were actually present (not just zero)
                original_quantity = trans.get('quantity') or trans.get('units')
                original_price = trans.get('price')
                original_amount = trans.get('amount') or trans.get('value')
                
                # Determine which fields were originally present
                quantity_present = original_quantity is not None and str(original_quantity).strip() != ''
                price_present = original_price is not None and str(original_price).strip() != ''
                amount_present = original_amount is not None and str(original_amount).strip() != ''
                
                # CRITICAL: Validate stock_name - don't use filename/channel as stock_name
                raw_stock_name = trans.get('stock_name', '') or trans.get('scheme_name', '')
                if raw_stock_name:
                    raw_stock_name_lower = str(raw_stock_name).lower().strip()
                    # Check if it looks like a channel/filename
                    is_invalid_name = (
                        len(raw_stock_name_lower) < 10 and ' ' not in raw_stock_name_lower or
                        raw_stock_name_lower in ['pornima', 'zerodha', 'groww', 'paytm', 'upstox', 'angel', 'icici', 'hdfc', 'sbi', 'direct'] or
                        (raw_stock_name_lower.islower() and len(raw_stock_name_lower.split()) == 1 and raw_stock_name_lower not in ['infosys', 'reliance', 'tcs', 'hdfc', 'icici'])
                    )
                    if is_invalid_name:
                        print(f"[VALIDATE] ⚠️ Ignoring invalid stock_name '{raw_stock_name}' (looks like channel/filename), will fetch from AMFI/mftool")
                        raw_stock_name = None
                
                validated_trans = {
                    'date': self._normalize_date(trans.get('date', '')),
                    'ticker': str(trans.get('ticker', '')).strip(),
                    'stock_name': raw_stock_name if raw_stock_name else None,  # None instead of ticker - let database fetch it
                    'scheme_name': trans.get('scheme_name'),
                    'quantity': self._safe_float(original_quantity or 0),
                    'price': self._safe_float(original_price or 0),  # 0 if missing - will be fetched
                    'transaction_type': str(transaction_type_raw or 'buy').lower(),
                    'asset_type': self._detect_asset_type(trans),
                    'sector': trans.get('sector') or 'Unknown',
                    'channel': self._infer_channel_from_filename(
                        source_filename,
                        trans.get('channel')
                    ),
                    # Store metadata about which fields were originally present
                    '__original_quantity_present': quantity_present,
                    '__original_amount_present': amount_present,
                    '__original_price_present': price_present,
                    '__original_quantity': original_quantity if quantity_present else None,
                    '__original_amount': original_amount if amount_present else None,
                    '__original_price': original_price if price_present else None,
                }

                # CRITICAL: Never recalculate quantity if it was present in the AI response
                original_quantity_value = validated_trans['quantity']
                
                # Calculate missing values from available data
                # Priority 1: Calculate price from amount/quantity if price is missing OR zero
                # CRITICAL: If AI returns price=0, we should still calculate from amount/quantity
                price_was_calculated = False
                price_val = validated_trans.get('price', 0)
                if (not price_present or price_val <= 0) and amount_present and quantity_present:
                    amount_float = self._safe_float(original_amount or 0)
                    if amount_float > 0 and validated_trans['quantity'] > 0:
                        calculated_price = amount_float / validated_trans['quantity']
                        validated_trans['price'] = calculated_price
                        price_was_calculated = True
                        print(f"[VALIDATE] ✅ Calculated price from amount/quantity: {calculated_price} (qty={validated_trans['quantity']}, amt={amount_float})")
                        print(f"[VALIDATE]   Transaction: {validated_trans['ticker']} on {validated_trans['date']} - SKIPPING historical price fetch")
                
                # Priority 2: Only calculate quantity if it's TRULY missing (not present in AI response)
                if not quantity_present and amount_present and price_present:
                    amount_float = self._safe_float(original_amount or 0)
                    price_float = self._safe_float(original_price or 0)
                    if amount_float > 0 and price_float > 0:
                        validated_trans['quantity'] = amount_float / price_float
                        print(f"[VALIDATE] ✅ Calculated quantity from amount/price: {validated_trans['quantity']}")
                
                # Priority 3: Calculate amount from quantity/price if amount is missing
                if not amount_present and quantity_present and price_present:
                    if validated_trans['quantity'] > 0 and validated_trans['price'] > 0:
                        validated_trans['amount'] = validated_trans['quantity'] * validated_trans['price']
                        print(f"[VALIDATE] ✅ Calculated amount from quantity/price: {validated_trans['amount']}")
                
                # Safety check: If quantity was present, never override it
                if quantity_present and validated_trans['quantity'] != original_quantity_value:
                    print(f"[VALIDATE] ⚠️ WARNING: Quantity was present ({original_quantity_value}) but got changed to {validated_trans['quantity']}. Restoring original.")
                    validated_trans['quantity'] = original_quantity_value
                
                # Validate quantity is positive (after calculations)
                if validated_trans['quantity'] <= 0:
                    skipped_count += 1
                    if skipped_count <= 3:  # Log first 3 skipped transactions
                        print(f"[VALIDATE] ⚠️ Skipping transaction (quantity <= 0 after calculations): {validated_trans}")
                    continue
                
                # Ensure price is non-negative
                if validated_trans['price'] < 0:
                    validated_trans['price'] = 0

                # CRITICAL: Fetch historical price if:
                # 1. Price was NOT calculated from amount/quantity (use calculated price if available)
                # 2. OR price is still 0/missing even after calculations (amount/quantity may be 0/invalid)
                # Always use transaction date from file, NOT current date
                should_fetch_historical = False
                if price_was_calculated:
                    # If we calculated price, only fetch if calculated price is 0 or invalid
                    if validated_trans['price'] <= 0:
                        should_fetch_historical = True
                        print(f"[VALIDATE] ⚠️ Calculated price is 0/invalid, fetching historical price as fallback")
                    else:
                        print(f"[VALIDATE] ✅ Using calculated price from amount/quantity: {validated_trans['price']} (NOT fetching historical price)")
                elif validated_trans['price'] <= 0 or not price_present:
                    # Price is missing/zero and wasn't calculated - definitely fetch
                    should_fetch_historical = True
                
                if should_fetch_historical:
                    transaction_date = validated_trans.get('date', '')
                    print(f"[VALIDATE] 🔍 Fetching historical price for {validated_trans['ticker']} on transaction date: {transaction_date} (from file)")
                    fetched_price, price_source = self._fetch_price_for_transaction_with_resolution(validated_trans)
                    if fetched_price and fetched_price > 0:
                        validated_trans['price'] = round(float(fetched_price), 4)
                        print(f"[VALIDATE] ✅ Fetched historical price: ₹{validated_trans['price']} for {validated_trans['ticker']} on {transaction_date} (from file)")
                        
                        # Check if ticker was resolved via name-based resolution
                        # For mutual funds: source format is "amfi_name_resolved:148520"
                        # For stocks: source format is "yfinance_nse_resolved:RELIANCE.NS"
                        if price_source and ('name_resolved:' in price_source or '_resolved:' in price_source):
                            # Extract resolved ticker from source string
                            try:
                                # Try both formats
                                if 'name_resolved:' in price_source:
                                    resolved_ticker = price_source.split('name_resolved:')[1].strip()
                                elif '_resolved:' in price_source:
                                    resolved_ticker = price_source.split('_resolved:')[1].strip()
                                else:
                                    resolved_ticker = None
                                
                                if resolved_ticker:
                                    # Remove any trailing characters (commas, parentheses, etc.)
                                    resolved_ticker = resolved_ticker.split(',')[0].split(')')[0].split('(')[0].strip()
                                    if resolved_ticker and resolved_ticker != validated_trans['ticker']:
                                        print(f"[VALIDATE] 🔄 Updating ticker from {validated_trans['ticker']} to {resolved_ticker} (name-based resolution)")
                                        validated_trans['ticker'] = resolved_ticker
                            except Exception as e:
                                self.logger.warning(f"Failed to extract resolved ticker from source '{price_source}': {e}")
                    else:
                        print(f"[VALIDATE] ⚠️ No historical price found for {validated_trans['ticker']} on {transaction_date} (from file)")
                        # Try one more time with expanded date range (up to 90 days)
                        if not fetched_price or fetched_price <= 0:
                            print(f"[VALIDATE] 🔄 Retrying with expanded date range (up to 90 days) for {validated_trans['ticker']}...")
                            try:
                                from datetime import datetime, timedelta
                                from dateutil import parser
                                
                                # Parse transaction date
                                trans_dt = parser.parse(transaction_date) if transaction_date else None
                                if trans_dt:
                                    # Try dates up to 90 days before/after
                                    for days_offset in [1, 2, 3, 7, 14, 30, 60, 90]:
                                        # Try before
                                        try_date = (trans_dt - timedelta(days=days_offset)).strftime('%Y-%m-%d')
                                        retry_price, _ = self._fetch_price_for_transaction_with_resolution({
                                            **validated_trans,
                                            'date': try_date
                                        })
                                        if retry_price and retry_price > 0:
                                            validated_trans['price'] = round(float(retry_price), 4)
                                            print(f"[VALIDATE] ✅ Found price on {try_date} ({(trans_dt - parser.parse(try_date)).days} days before): ₹{validated_trans['price']}")
                                            break
                                        
                                        # Try after
                                        try_date = (trans_dt + timedelta(days=days_offset)).strftime('%Y-%m-%d')
                                        retry_price, _ = self._fetch_price_for_transaction_with_resolution({
                                            **validated_trans,
                                            'date': try_date
                                        })
                                        if retry_price and retry_price > 0:
                                            validated_trans['price'] = round(float(retry_price), 4)
                                            print(f"[VALIDATE] ✅ Found price on {try_date} ({(parser.parse(try_date) - trans_dt).days} days after): ₹{validated_trans['price']}")
                                            break
                                    
                                    if validated_trans['price'] <= 0:
                                        print(f"[VALIDATE] ❌ Could not find price even with 90-day range for {validated_trans['ticker']}")
                            except Exception as e:
                                self.logger.warning(f"Expanded date range retry failed: {e}")
                
                # Final check: Don't save transactions with price=0 if we have a ticker and date
                # (unless it's explicitly a zero-price transaction like bonus shares)
                if validated_trans['price'] <= 0 and validated_trans.get('ticker') and validated_trans.get('date'):
                    # Only skip if we really couldn't find any price
                    if not should_fetch_historical or (should_fetch_historical and not fetched_price):
                        print(f"[VALIDATE] ⚠️ WARNING: Transaction {validated_trans['ticker']} on {validated_trans['date']} has price=0 - will be saved but may need manual price entry")
                
                validated.append(validated_trans)
                
            except Exception as e:
                self.logger.error(f"Validation error: {e}")
                print(f"[VALIDATE] ❌ Validation error: {e}")
                continue
        
        if skipped_count > 0:
            print(f"[VALIDATE] ⚠️ Skipped {skipped_count} transactions during validation")
        print(f"[VALIDATE] ✅ Validated {len(validated)} transactions from {len(transactions)} raw transactions")
        return validated
    
    def _detect_asset_type(self, trans: Dict[str, Any]) -> str:
        """Intelligently detect asset type by checking actual data sources"""
        
        ticker = str(trans.get('ticker', '')).strip()
        name = str(trans.get('stock_name', '')).lower()
        
        # Check explicit asset_type first
        if trans.get('asset_type'):
            return str(trans['asset_type']).lower()
        
        # Contains .NS or .BO = stock
        if '.NS' in ticker.upper() or '.BO' in ticker.upper():
            return 'stock'
        
        # For numeric tickers, check actual data sources to determine type
        if ticker.isdigit():
            # First, check if it's in AMFI dataset (mutual fund)
            try:
                from web_agent import get_amfi_dataset
                amfi_data = get_amfi_dataset()
                if amfi_data and 'code_lookup' in amfi_data:
                    if ticker in amfi_data['code_lookup']:
                        return 'mutual_fund'  # Found in AMFI = mutual fund
            except Exception:
                pass  # AMFI check failed, continue to stock check
            
            # Try to fetch from yfinance (for stocks)
            try:
                import yfinance as yf
                # Try NSE first - only add .NS if ticker doesn't already have .NS or .BO
                if ticker.endswith('.NS'):
                    nse_ticker = ticker
                elif ticker.endswith('.BO'):
                    nse_ticker = ticker.replace('.BO', '.NS')
                else:
                    nse_ticker = f"{ticker}.NS"
                stock = yf.Ticker(nse_ticker)
                hist = stock.history(period='1d')
                if not hist.empty:
                    return 'stock'  # Found in yfinance NSE = stock
                
                # Try BSE - only add .BO if ticker doesn't already have .NS or .BO
                if ticker.endswith('.BO'):
                    bse_ticker = ticker
                elif ticker.endswith('.NS'):
                    bse_ticker = ticker.replace('.NS', '.BO')
                else:
                    bse_ticker = f"{ticker}.BO"
                stock = yf.Ticker(bse_ticker)
                hist = stock.history(period='1d')
                if not hist.empty:
                    return 'stock'  # Found in yfinance BSE = stock
            except Exception:
                pass  # yfinance check failed, fall back to heuristics
            
            # Fallback to heuristics if both checks failed
            # Most 6-digit numbers starting with '1' are mutual funds
            if len(ticker) == 6 and ticker.startswith('1'):
                return 'mutual_fund'
            else:
                # Other numeric codes are likely BSE stocks
                return 'stock'
        
        # Check name for clues
        # CRITICAL: Check for "fund" or "scheme" FIRST before "bond"
        # This ensures mutual funds with "bond" in the name (e.g., "Bond Fund", "AAA Bond Fund") are correctly identified as mutual_fund
        if 'fund' in name or 'scheme' in name:
            return 'mutual_fund'
        
        if 'pms' in name or 'portfolio management' in name:
            return 'pms'
        
        if 'aif' in name or 'alternative investment' in name:
            return 'aif'
        
        # Only check for bond if it's NOT a fund/scheme (bond funds are already handled above)
        if 'bond' in name or 'debenture' in name:
            return 'bond'
        
        # Default to stock
        return 'stock'
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to YYYY-MM-DD"""
        
        try:
            formats = [
                '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d', '%d.%m.%Y',
                '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(str(date_str).strip(), fmt)
                    return dt.strftime('%Y-%m-%d')
                except:
                    continue
            
            return str(date_str)
            
        except:
            return str(date_str)
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert to float"""
        
        try:
            if value is None or value == '':
                return 0.0
            
            # Remove common currency symbols and commas
            if isinstance(value, str):
                cleaned = value.replace('₹', '').replace('Rs', '').replace(',', '').strip()
                cleaned = re.sub(r'[^0-9\.\-]', '', cleaned)
                if cleaned in {'', '.', '-', '-.'}:
                    return 0.0
                value = cleaned
            
            return float(value)
        except:
            return 0.0
    
    def _fallback_extraction(self, content: str, file_type: str) -> List[Dict[str, Any]]:
        """Fallback extraction using pandas"""
        
        try:
            if file_type == 'csv':
                import io
                df = pd.read_csv(io.StringIO(content))
                
                transactions = []
                for _, row in df.iterrows():
                    # CRITICAL: Validate stock_name - don't use channel/filename
                    raw_stock_name = row.get('stock_name', row.get('Stock Name', ''))
                    if raw_stock_name:
                        raw_stock_name_lower = str(raw_stock_name).lower().strip()
                        is_invalid_name = (
                            len(raw_stock_name_lower) < 10 and ' ' not in raw_stock_name_lower or
                            raw_stock_name_lower in ['pornima', 'zerodha', 'groww', 'paytm', 'upstox', 'angel', 'icici', 'hdfc', 'sbi', 'direct'] or
                            (raw_stock_name_lower.islower() and len(raw_stock_name_lower.split()) == 1 and raw_stock_name_lower not in ['infosys', 'reliance', 'tcs', 'hdfc', 'icici'])
                        )
                        if is_invalid_name:
                            raw_stock_name = None  # Will be fetched from AMFI/mftool
                    
                    trans = {
                        'date': row.get('date', row.get('Date', '')),
                        'ticker': row.get('ticker', row.get('Ticker', row.get('Symbol', ''))),
                        'stock_name': raw_stock_name if raw_stock_name else None,  # None instead of empty string
                        'quantity': self._safe_float(row.get('quantity', row.get('Quantity', 0))),
                        'price': self._safe_float(row.get('price', row.get('Price', 0))),
                        'transaction_type': str(row.get('transaction_type', row.get('Type', 'buy'))).lower(),
                        'asset_type': row.get('asset_type', row.get('Asset Type', 'stock')),
                        'sector': row.get('sector', row.get('Sector', 'Unknown')),
                        'channel': row.get('channel', row.get('Channel', 'Direct'))
                    }
                    transactions.append(trans)
                
                return transactions
        except:
            pass
        
        return []

    def _infer_channel_from_filename(self, filename: str, explicit_channel: Optional[str] = None) -> str:
        """Infer channel/platform from explicit value or fallback to filename stem."""
        if explicit_channel:
            candidate = str(explicit_channel).strip()
            if candidate:
                return candidate

        if not filename:
            return "Direct"

        stem = Path(filename).stem
        clean = re.sub(r'[_\-\s]+', ' ', stem).strip()
        return clean.title() if clean else "Direct"

    def _get_price_fetcher(self):
        """Lazy initialization of price fetcher - only when needed"""
        if self._price_fetcher_initialized:
            return self.price_fetcher
        
        self._price_fetcher_initialized = True
        if EnhancedPriceFetcher is not None:
            try:
                self.price_fetcher = EnhancedPriceFetcher()
                self.logger.info("EnhancedPriceFetcher initialized successfully")
            except Exception as exc:
                self.logger.error(f"Failed to initialize EnhancedPriceFetcher: {exc}")
                self.price_fetcher = None
        else:
            self.price_fetcher = None
        
        return self.price_fetcher
    
    def _fetch_price_for_transaction(self, trans: Dict[str, Any]) -> Optional[float]:
        """Fetch historical price for transaction date when price missing/zero."""
        price, _ = self._fetch_price_for_transaction_with_resolution(trans)
        return price
    
    def _fetch_price_for_transaction_with_resolution(self, trans: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
        """Fetch historical price for transaction date when price missing/zero. Returns price and source."""
        price_fetcher = self._get_price_fetcher()
        if not price_fetcher:
            return None, None

        price = trans.get('price') or 0
        if price and price > 0:
            return price, None

        ticker = trans.get('ticker')
        date = trans.get('date')
        asset_type = (trans.get('asset_type') or 'stock').lower()

        if not ticker or not date:
            return None, None

        cache_key = (ticker, date, asset_type)
        if cache_key in self._price_cache:
            cached = self._price_cache[cache_key]
            if isinstance(cached, tuple):
                return cached
            return cached, None

        fund_name = trans.get('stock_name') or trans.get('scheme_name')
        fetched_price = None
        price_source = None

        try:
            # Ensure date is in YYYY-MM-DD format
            if isinstance(date, str):
                try:
                    # Try parsing and reformatting
                    from dateutil import parser
                    date_obj = parser.parse(date)
                    date = date_obj.strftime('%Y-%m-%d')
                except:
                    # If parsing fails, try to clean the date string
                    date = date.strip()
            
            print(f"[FETCH_PRICE] Fetching historical price for {ticker} ({asset_type}) on {date}")
            fetched_price = price_fetcher.get_historical_price(
                ticker,
                asset_type,
                date,
                fund_name=fund_name
            )

            if fetched_price and fetched_price > 0:
                price_source = 'historical'
                print(f"[FETCH_PRICE] ✅ Got historical price: ₹{fetched_price:.2f} for {ticker} on {date}")
            elif not fetched_price:
                print(f"[FETCH_PRICE] ⚠️ Historical price not found, trying current price...")
                current_price, source = price_fetcher.get_current_price(
                    ticker,
                    asset_type,
                    fund_name=fund_name
                )
                fetched_price = current_price
                price_source = source
                
                # For stocks, check if ticker was resolved via name-based lookup
                # The resolved ticker is stored in _last_resolved_ticker
                if asset_type == 'stock' and hasattr(price_fetcher, '_last_resolved_ticker'):
                    resolved_ticker = getattr(price_fetcher, '_last_resolved_ticker', None)
                    if resolved_ticker and resolved_ticker != ticker:
                        # Add resolved ticker info to source
                        price_source = f"{source}_resolved:{resolved_ticker}"

            if fetched_price and fetched_price > 0:
                self._price_cache[cache_key] = (fetched_price, price_source)
                return fetched_price, price_source
        except Exception as exc:
            self.logger.warning(f"Price backfill failed for {ticker} on {date}: {exc}")

        self._price_cache[cache_key] = None
        return None, None
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze method required by BaseAgent"""
        file_data = data.get('file_data')
        filename = data.get('filename', 'unknown')
        
        transactions = self.process_file(file_data, filename)
        
        return self.format_response([{"transactions": transactions}], "high")
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get insights (required by BaseAgent)"""
        return self.processed_transactions
    
    def get_processed_transactions(self) -> List[Dict[str, Any]]:
        """Get processed transactions"""
        return self.processed_transactions

