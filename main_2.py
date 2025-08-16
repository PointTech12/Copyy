import os
import json
import re
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

# Input and output directories
input_dir = "/home/orange/Documents/Projects/Copyy/Dataset_PDFs"
output_dir = "/home/orange/Documents/Projects/Copyy/New_Dataset"
output_file = os.path.join(output_dir, "metadata.json")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to OCR first page of PDF and guess Title & Author
def ocr_first_page(file_path):
    try:
        images = convert_from_path(file_path, first_page=1, last_page=1)
        text = pytesseract.image_to_string(images[0])
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        guessed_title, guessed_author = None, None

        # Heuristic: first line with proper case & length = Title
        for line in lines:
            if 5 < len(line) < 100 and line[0].isupper():
                guessed_title = line
                break

        # Heuristic: look for 'by ...' or uppercase name for Author
        for line in lines:
            if re.search(r"^by\s+.+", line, re.IGNORECASE):
                guessed_author = re.sub(r"^by\s+", "", line, flags=re.IGNORECASE).strip()
                break
            elif line.isupper() and len(line.split()) <= 4:  # short uppercase line
                guessed_author = line.title()
                break

        return guessed_title, guessed_author
    except Exception:
        return None, None

# Dictionary to store metadata
all_metadata = {}

# Loop through PDFs
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(input_dir, filename)
        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata

            title, author = None, None

            # Try extracting from PDF metadata
            if metadata:
                raw_title = metadata.get("/Title")
                raw_author = metadata.get("/Author")
                title = str(raw_title) if raw_title else None
                author = str(raw_author) if raw_author else None

            # If missing, run OCR on first page
            if not title or not author:
                ocr_title, ocr_author = ocr_first_page(file_path)
                if not title and ocr_title:
                    title = ocr_title
                if not author and ocr_author:
                    author = ocr_author

            all_metadata[filename] = {
                "Title": title if title else "Unknown",
                "Author": author if author else "Unknown"
            }

        except Exception as e:
            all_metadata[filename] = {
                "Title": "Error",
                "Author": f"Error: {str(e)}"
            }

# Save to JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, indent=4, ensure_ascii=False)

print(f"Metadata extracted and saved to {output_file}")
