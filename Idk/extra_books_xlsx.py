import os
import re
import statistics
import nltk
import string
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from openpyxl import Workbook


# Download both required tokenizers
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Input/output directories
input_dir = "/home/orange/Documents/Projects/Copyy/Dataset_PDFs"
output_dir = "/home/orange/Documents/Projects/Copyy/New_Dataset"
output_file = os.path.join(output_dir, "books_dataset.xlsx")

os.makedirs(output_dir, exist_ok=True)

# === GENRE KEYWORDS ===
GENRE_KEYWORDS = {
    "Technology": ["computer", "programming", "data", "machine learning", "neural", "network", "software", "engineering", "algorithm"],
    "Science": ["physics", "chemistry", "biology", "experiment", "hypothesis", "astronomy", "universe", "genetics"],
    "History": ["empire", "king", "queen", "war", "battle", "revolution", "civilization", "dynasty", "colonial"],
    "Philosophy": ["ethics", "morality", "existence", "truth", "mind", "consciousness", "reason"],
    "Religion": ["god", "faith", "church", "bible", "quran", "spiritual", "hindu", "islam", "buddha"],
    "Literature": ["poem", "novel", "story", "prose", "drama", "character", "plot", "verse"],
    "Law": ["constitution", "legal", "court", "justice", "rights", "criminal", "civil", "penalty"],
    "Economics": ["market", "economy", "trade", "finance", "inflation", "gdp", "money", "investment"],
}

# OCR function for first page (Title & Author fallback)
def ocr_first_page(file_path):
    try:
        images = convert_from_path(file_path, first_page=1, last_page=1)
        text = pytesseract.image_to_string(images[0])
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        guessed_title, guessed_author = None, None
        for line in lines:
            if 5 < len(line) < 100 and line[0].isupper():
                guessed_title = line
                break
        for line in lines:
            if re.search(r"^by\s+.+", line, re.IGNORECASE):
                guessed_author = re.sub(r"^by\s+", "", line, flags=re.IGNORECASE).strip()
                break
            elif line.isupper() and len(line.split()) <= 4:
                guessed_author = line.title()
                break

        return guessed_title, guessed_author
    except Exception:
        return None, None

# Text statistics function
def paragraph_stats(text):
    sentences = nltk.sent_tokenize(text)
    words = [w for w in re.findall(r"\b\w+\b", text) if w not in string.punctuation]

    sentence_count = len(sentences)
    word_count = len(words)

    avg_sentence_length = (word_count / sentence_count) if sentence_count > 0 else 0
    avg_word_length = statistics.mean([len(w) for w in words]) if words else 0
    type_token_ratio = (len(set(words)) / word_count) if word_count > 0 else 0

    return sentence_count, word_count, avg_sentence_length, avg_word_length, type_token_ratio

# Genre detection from text
def detect_genre(paragraphs, sample_size=10):
    sample_text = " ".join(paragraphs[:sample_size]).lower()
    scores = {genre: 0 for genre in GENRE_KEYWORDS}

    for genre, keywords in GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw in sample_text:
                scores[genre] += 1

    best_genre = max(scores, key=scores.get)
    return best_genre if scores[best_genre] > 0 else "Unknown"

# Create Excel workbook
wb = Workbook()
wb.remove(wb.active)  # remove default sheet

book_id = 1  # incremental ID for books

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(input_dir, filename)

        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata

            title = str(metadata.get("/Title")) if metadata and metadata.get("/Title") else None
            author = str(metadata.get("/Author")) if metadata and metadata.get("/Author") else None

            if not title or not author:
                ocr_title, ocr_author = ocr_first_page(file_path)
                if not title and ocr_title:
                    title = ocr_title
                if not author and ocr_author:
                    author = ocr_author

            if not title:
                title = "Unknown"
            if not author:
                author = "Unknown"

            # Extract full text
            full_text = ""
            for page in reader.pages:
                try:
                    full_text += page.extract_text() + "\n"
                except:
                    continue

            # Split into paragraphs
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]

            # Detect genre
            genre = detect_genre(paragraphs)

            # Create new sheet for this PDF
            sheet_name = os.path.splitext(filename)[0][:30]  # Excel max sheet name length = 31
            ws = wb.create_sheet(title=sheet_name)

            # Header row
            ws.append([
                "book_id", "genre", "author", "title",
                "paragraph_num", "text_content", "sentence_count",
                "word_count", "avg_sentence_length", "avg_word_length", "type_token_ratio"
            ])

            # Add data rows
            for idx, para in enumerate(paragraphs, start=1):
                s_count, w_count, avg_s_len, avg_w_len, ttr = paragraph_stats(para)
                ws.append([
                    book_id, genre, author, title,
                    idx, para, s_count, w_count,
                    round(avg_s_len, 2), round(avg_w_len, 2), round(ttr, 3)
                ])

            book_id += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save Excel file
wb.save(output_file)
print(f"Excel dataset saved to {output_file}")
