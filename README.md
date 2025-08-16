# Copyy - AI-Generated Text Copyright Infringement Detection System

A specialized system that analyzes AI-generated text to find evidence of potential copyright infringement by combining copy detection and stylometric analysis.

## 🎯 Project Overview

This system is designed to:
- **Copy Detection**: Find literal text matches between AI outputs and copyrighted reference texts
- **Stylometric Analysis**: Identify imitation of writing styles
- **Risk Assessment**: Provide comprehensive analysis with risk levels and recommendations
- **Scalable Processing**: Handle multiple books efficiently with automated metadata management

## 📁 Project Structure

```
Copyy/
├── Dataset_PDFs/           # Place your PDF books here
├── Processed_Data/         # Output directory for processed data and analysis
├── main.py                 # Main processing script (enhanced)
├── add_book.py            # Utility to add new books with metadata
├── analyze_copyright.py   # Copyright infringement analysis
├── book_metadata.json     # Book metadata configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Add Books to Dataset

Use the interactive book manager:

```bash
python add_book.py
```

Or manually place PDF files in the `Dataset_PDFs/` folder and update `book_metadata.json`.

### 3. Process All Books

```bash
python main.py
```

This will:
- Extract text from all PDFs
- Segment into paragraphs
- Extract comprehensive stylometric features
- Save processed data to CSV

### 4. Run Copyright Analysis

```bash
python analyze_copyright.py
```

This will:
- Perform copy detection analysis
- Conduct stylometric analysis
- Generate sequence matching results
- Create risk assessment
- Generate visualizations and reports

## 📊 Analysis Features

### Copy Detection
- **TF-IDF Vectorization**: Uses 1-3 word n-grams for robust text comparison
- **Cosine Similarity**: Measures text similarity between paragraphs
- **Configurable Thresholds**: Adjustable similarity thresholds for different use cases

### Stylometric Analysis
- **Comprehensive Features**:
  - Sentence and word counts
  - Average sentence/word lengths
  - Type-token ratio (vocabulary diversity)
  - Punctuation and capitalization ratios
  - Function vs content word ratios
  - Readability scores (Flesch, Gunning Fog, SMOG)

- **Style Clustering**: Groups similar writing styles using K-means clustering
- **Style Similarity Matrix**: Compares writing styles across books

### Sequence Matching
- **Exact Text Matching**: Finds identical or near-identical text sequences
- **Configurable Sensitivity**: Adjustable similarity ratios
- **Match Size Filtering**: Filters out very short matches

## 📈 Output Files

After running the analysis, you'll find these files in `Processed_Data/`:

- `all_processed_books.csv` - Complete processed dataset
- `processing_summary.json` - Processing statistics
- `copyright_analysis_report.json` - Detailed analysis results
- `analysis_summary.json` - High-level summary
- `copyright_analysis_visualizations.png` - Analysis charts

## 🔧 Configuration

### Book Metadata (`book_metadata.json`)

```json
{
  "books": {
    "filename.pdf": {
      "title": "Book Title",
      "author": "Author Name",
      "genre": "Genre",
      "publication_year": 2023,
      "source": "AI Generated",
      "notes": "Additional notes"
    }
  }
}
```

### Analysis Parameters

You can adjust these parameters in the analysis scripts:

- **Similarity Threshold**: Minimum similarity for copy detection (default: 0.8)
- **Sequence Match Ratio**: Minimum ratio for sequence matching (default: 0.7)
- **Clustering**: Number of style clusters (adaptive based on data size)

## 📋 Best Practices for Classification

### 1. **Organized File Naming**
Use descriptive filenames that include genre indicators:
- `scf_book1.pdf` (Science Fiction)
- `mystery_book2.pdf` (Mystery)
- `romance_book3.pdf` (Romance)

### 2. **Comprehensive Metadata**
For each book, provide:
- Accurate title and author
- Correct genre classification
- Publication year (if known)
- Source type (AI Generated, Human Written, Mixed)

### 3. **Quality Control**
- Ensure PDFs are text-based (not scanned images)
- Remove headers/footers that might interfere with analysis
- Use consistent formatting across books

### 4. **Dataset Diversity**
- Include books from different genres
- Mix AI-generated and human-written content
- Include various authors and time periods

## 🎯 Analysis Interpretation

### Risk Levels

- **LOW RISK**: No significant copyright concerns detected
- **HIGH RISK**: Multiple indicators of potential copyright issues

### Key Metrics

- **High Similarity Pairs**: Number of text pairs with high similarity scores
- **Style Clusters**: Number of distinct writing style groups
- **Sequence Matches**: Number of exact or near-exact text matches

### Recommendations

The system provides specific recommendations based on findings:
- Review flagged text pairs for context
- Consider legal consultation for high-similarity content
- Expand dataset for better style diversity

## 🔍 Advanced Usage

### Custom Analysis

You can modify the analysis parameters:

```python
# In analyze_copyright.py
analyzer = CopyrightAnalyzer(data_path, output_path)

# Custom copy detection threshold
copy_results = analyzer.copy_detection_analysis(similarity_threshold=0.9)

# Custom sequence matching
sequence_results = analyzer.sequence_matching_analysis(min_ratio=0.8)
```

### Batch Processing

For large datasets, you can process books in batches:

```python
# Process specific books
specific_books = ['book1.pdf', 'book2.pdf']
for book in specific_books:
    df = analyzer.process_single_book(book)
    # Custom processing...
```

## 🛠️ Troubleshooting

### Common Issues

1. **No PDFs Found**: Ensure PDFs are in `Dataset_PDFs/` folder
2. **Memory Issues**: Reduce `nlp.max_length` for very large documents
3. **Processing Errors**: Check PDF quality and text extraction
4. **Missing Dependencies**: Run `pip install -r requirements.txt`

### Performance Tips

- Use SSD storage for faster file I/O
- Increase RAM for large datasets
- Process books in smaller batches if needed
- Use multiprocessing for very large datasets

## 📚 Future Enhancements

Potential improvements for the system:

1. **Machine Learning Models**: Train custom models for better detection
2. **Database Integration**: Store results in a database for querying
3. **Web Interface**: Create a web-based analysis dashboard
4. **API Integration**: Connect with external copyright databases
5. **Real-time Analysis**: Process documents as they're uploaded

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with copyright laws when using this system.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue in the repository

---

**Note**: This system is designed for educational and research purposes. Always consult with legal professionals for actual copyright infringement cases. 