
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

### 2. Add Books to Dataset - probly will break you pc

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

```bash
python main_2.py
```
For author and title, into json file, to lazy to be in xlsx



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



## 🛠️ Troubleshooting

### Common Issues

1. **No PDFs Found**: Ensure PDFs are in `Dataset_PDFs/` folder
2. **Memory Issues**: Reduce `nlp.max_length` for very large documents
3. **Processing Errors**: Check PDF quality and text extraction
4. **Missing Dependencies**: Run `pip install -r requirements.txt`


