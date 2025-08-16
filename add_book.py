#!/usr/bin/env python3
"""
Utility script to add new books to the dataset with proper metadata.
"""

import json
import shutil
import os
from pathlib import Path
from datetime import datetime

def load_metadata():
    """Load existing book metadata."""
    try:
        with open('book_metadata.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"books": {}, "genre_mappings": {}, "processing_settings": {}}

def save_metadata(metadata):
    """Save updated book metadata."""
    with open('book_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def get_genre_mapping(filename):
    """Get genre from filename patterns."""
    filename_lower = filename.lower()
    genre_mappings = {
        'scf': 'Science Fiction',
        'mystery': 'Mystery',
        'romance': 'Romance',
        'fantasy': 'Fantasy',
        'thriller': 'Thriller',
        'horror': 'Horror',
        'historical': 'Historical Fiction',
        'literary': 'Literary Fiction',
        'nonfiction': 'Non-Fiction',
        'biography': 'Biography',
        'memoir': 'Memoir'
    }
    
    for pattern, genre in genre_mappings.items():
        if pattern in filename_lower:
            return genre
    return 'Unknown'

def add_book():
    """Interactive function to add a new book."""
    print("=== Add New Book to Dataset ===\n")
    
    # Get book file path
    while True:
        book_path = input("Enter the path to the PDF file: ").strip()
        if os.path.exists(book_path) and book_path.lower().endswith('.pdf'):
            break
        print("Error: File not found or not a PDF. Please try again.")
    
    # Get book metadata
    filename = os.path.basename(book_path)
    print(f"\nDetected filename: {filename}")
    
    # Auto-detect genre
    auto_genre = get_genre_mapping(filename)
    print(f"Auto-detected genre: {auto_genre}")
    
    title = input(f"Book title (or press Enter to use '{Path(filename).stem}'): ").strip()
    if not title:
        title = Path(filename).stem
    
    author = input("Author name: ").strip()
    if not author:
        author = "Unknown Author"
    
    genre = input(f"Genre (or press Enter to use '{auto_genre}'): ").strip()
    if not genre:
        genre = auto_genre
    
    publication_year = input("Publication year (or press Enter to skip): ").strip()
    if not publication_year:
        publication_year = None
    else:
        try:
            publication_year = int(publication_year)
        except ValueError:
            publication_year = None
    
    source = input("Source (e.g., 'AI Generated', 'Human Written', 'Mixed'): ").strip()
    if not source:
        source = "Unknown"
    
    notes = input("Additional notes (optional): ").strip()
    
    # Confirm details
    print(f"\n=== Book Details ===")
    print(f"Title: {title}")
    print(f"Author: {author}")
    print(f"Genre: {genre}")
    print(f"Publication Year: {publication_year}")
    print(f"Source: {source}")
    print(f"Notes: {notes}")
    
    confirm = input("\nProceed with adding this book? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Book addition cancelled.")
        return
    
    # Load existing metadata
    metadata = load_metadata()
    
    # Add book to dataset
    dataset_path = Path("Dataset_PDFs")
    dataset_path.mkdir(exist_ok=True)
    
    # Copy file to dataset
    dest_path = dataset_path / filename
    shutil.copy2(book_path, dest_path)
    
    # Add metadata
    metadata["books"][filename] = {
        "title": title,
        "author": author,
        "genre": genre,
        "publication_year": publication_year,
        "language": "English",
        "source": source,
        "notes": notes,
        "added_date": datetime.now().isoformat()
    }
    
    # Save updated metadata
    save_metadata(metadata)
    
    print(f"\n✅ Successfully added '{title}' to the dataset!")
    print(f"File copied to: {dest_path}")
    print(f"Metadata saved to: book_metadata.json")

def list_books():
    """List all books in the dataset."""
    metadata = load_metadata()
    
    print("=== Books in Dataset ===\n")
    
    if not metadata["books"]:
        print("No books found in dataset.")
        return
    
    for filename, book_info in metadata["books"].items():
        print(f"📖 {book_info['title']}")
        print(f"   Author: {book_info['author']}")
        print(f"   Genre: {book_info['genre']}")
        print(f"   File: {filename}")
        if book_info.get('publication_year'):
            print(f"   Year: {book_info['publication_year']}")
        if book_info.get('source'):
            print(f"   Source: {book_info['source']}")
        print()

def main():
    """Main function."""
    print("Book Dataset Manager")
    print("===================")
    
    while True:
        print("\nOptions:")
        print("1. Add new book")
        print("2. List all books")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            add_book()
        elif choice == '2':
            list_books()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main() 