

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
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   sudo apt-get install poppler-utils
   
   # Arch Linux
   sudo pacman -S tesseract
   sudo pacman -S poppler
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

### Copyright Analysis

Run the copyright analysis engine:
```bash
python Idk/analyze_copyright.py
```

### Adding New Books - if you want to break your pc 

Use the book addition utility:
```bash
python Idk/add_book.py
```

## 🔧 Configuration

### Main Processing Settings

The main processing pipeline can be configured by modifying the paths in `main.py`:

```python
dataset_path = "/path/to/your/pdf/files"
output_path = "/path/to/output/directory"
```


For questions and support, please open an issue on the GitHub repository.

---

**Note**: This system is designed to assist in identifying potential copyright issues but should not replace professional legal analysis. 
