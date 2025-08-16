# Copyy - AI-Generated Text Copyright Infringement Detection System

A comprehensive system for detecting potential copyright infringement in text documents using advanced stylometric analysis and copy detection algorithms. This project processes PDF books, extracts stylometric features, and identifies potential similarities that may indicate copyright issues.

## 🚀 Features

- **PDF Text Extraction**: Robust extraction of text content from PDF files
- **Stylometric Analysis**: Advanced linguistic feature extraction including:
  - Sentence and word count statistics
  - Readability scores (Flesch Reading Ease, Gunning Fog Index, SMOG Index)
  - Type-token ratios and vocabulary diversity
  - Punctuation and capitalization patterns
  - Function vs. content word analysis
  - Syllable counting and complexity metrics

- **Copy Detection**: 
  - TF-IDF vectorization with n-gram analysis
  - Cosine similarity calculations
  - Sequence matching for exact text comparisons
  - Configurable similarity thresholds

- **Metadata Extraction**: 
  - PDF metadata parsing
  - OCR-based title and author detection
  - Genre classification

- **Comprehensive Reporting**: 
  - Detailed analysis results in JSON format
  - Statistical summaries and visualizations
  - CSV exports for further analysis

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
│   └── processing_summary.json     # Statistical summary
├── New_Dataset/                    # New dataset processing
│   └── metadata.json              # Extracted metadata
└── Idk/                           # Additional utilities
    ├── analyze_copyright.py        # Copyright analysis engine
    ├── extract_metadata.py         # Enhanced metadata extraction
    ├── add_book.py                # Book addition utility
    ├── extra_books_xlsx.py        # Excel processing
    └── ext_meta.py                # Metadata utilities
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

### Adding New Books

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

### Analysis Parameters

In `Idk/analyze_copyright.py`, you can adjust:

- **Similarity threshold**: Minimum similarity score to flag potential copies
- **N-gram range**: Text analysis granularity (1-3 words)
- **Document frequency**: Minimum/maximum word frequency thresholds

## 📊 Output Files

### Processed Data (`Processed_Data/`)

- **`all_processed_books.csv`**: Complete analysis results with stylometric features
- **`processing_summary.json`**: Statistical summary including:
  - Total books and paragraphs processed
  - Genre and author distributions
  - Feature statistics (mean, std, min, max)

### Analysis Results

- **Copy detection results**: High-similarity text pairs
- **Stylometric clustering**: Author and genre-based groupings
- **Visualization plots**: Similarity matrices and feature distributions

## 🧠 Technical Details

### Stylometric Features

The system extracts 13 key stylometric features:

1. **Sentence count** - Total number of sentences
2. **Word count** - Total number of words
3. **Average sentence length** - Mean words per sentence
4. **Average word length** - Mean characters per word
5. **Type-token ratio** - Vocabulary diversity measure
6. **Punctuation ratio** - Punctuation frequency
7. **Capitalization ratio** - Capital letter frequency
8. **Function word ratio** - Common words vs. content words
9. **Content word ratio** - Substantive vocabulary proportion
10. **Syllable count** - Total syllables in text
11. **Flesch Reading Ease** - Readability score (0-100)
12. **Gunning Fog Index** - Complexity measure
13. **SMOG Index** - Grade level readability

### Copy Detection Algorithm

1. **Text Preprocessing**: Cleaning and normalization
2. **TF-IDF Vectorization**: Convert text to numerical features
3. **Similarity Calculation**: Cosine similarity between all text pairs
4. **Threshold Filtering**: Identify high-similarity matches
5. **Sequence Matching**: Exact text comparison for verification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is designed for educational and research purposes. It should not be used as the sole basis for legal decisions regarding copyright infringement. Always consult with legal professionals for copyright-related matters.

## 🐛 Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **PDF extraction errors**: Ensure PDFs are not password-protected or corrupted

3. **Memory issues**: Reduce `nlp.max_length` in main.py for large documents

4. **OCR errors**: Install Tesseract and Poppler utilities

### Performance Tips

- Use SSD storage for faster file I/O
- Increase system RAM for large document processing
- Process documents in batches for memory efficiency

## 📞 Support

For questions and support, please open an issue on the GitHub repository.

---

**Note**: This system is designed to assist in identifying potential copyright issues but should not replace professional legal analysis. 