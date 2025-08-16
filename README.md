

## 📁 Project Structure

```
Copyy/
├── main.py                          # Main processing pipeline
├── main_2.py                        # Metadata extraction utility
├── requirements.txt                 # Python dependencies
├── book_metadata.json              # Book metadata storage
├── Dataset_PDFs/                   # Input PDF files
├── Processed_Data/                 # Output processed data
│   ├── all_processed_books.csv     # Combined analysis results
├── New_Dataset/                    # New dataset processing
│   └── metadata.json              # Extracted metadata
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Copyy
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install spaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Install additional system dependencies** (for OCR functionality):
   ```bash
   sudo apt-get install tesseract-ocr
   sudo apt-get install poppler-utils

   ```

## 📖 Usage

### Basic Processing

1. **Place your PDF files** in the `Dataset_PDFs/` directory

2. **Run the main processing pipeline**:
   ```bash
   python main.py
   ```

3. **Extract metadata** (optional):
   ```bash
   python main_2.py
   ```


## 🔧 Configuration

### Main Processing Settings

The main processing pipeline can be configured by modifying the paths in `main.py`:

```python
dataset_path = "/path/to/your/pdf/files"
output_path = "/path/to/output/directory"
```

---


