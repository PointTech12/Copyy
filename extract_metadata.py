#!/usr/bin/env python3
"""
Extract Metadata from Multiple PDF Books
A comprehensive script to extract title and author information from all PDF files in a directory.
"""

import PyPDF2
import json
import re
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiBookMetadataExtractor:
    def __init__(self, input_dir: str = "Dataset_PDFs", output_dir: str = "New_Dataset"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Known book titles and authors for content-based detection
        self.known_books = {
            # Lord of the Rings series
            'concerning hobbits': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'long expected party': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'shadow of the past': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'three is company': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'minas tirith': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'mount doom': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'return of the king': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'fellowship of the ring': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'two towers': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'hobbit': {'title': 'The Hobbit', 'author': 'J.R.R. Tolkien'},
            'middle earth': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'gandalf': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'frodo': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'sam': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'aragorn': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'legolas': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'gimli': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'boromir': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'merry': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            'pippin': {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
            
            # Harry Potter series
            'harry potter': {'title': 'Harry Potter', 'author': 'J.K. Rowling'},
            'philosopher stone': {'title': 'Harry Potter and the Philosopher\'s Stone', 'author': 'J.K. Rowling'},
            'chamber secrets': {'title': 'Harry Potter and the Chamber of Secrets', 'author': 'J.K. Rowling'},
            'prisoner azkaban': {'title': 'Harry Potter and the Prisoner of Azkaban', 'author': 'J.K. Rowling'},
            'goblet fire': {'title': 'Harry Potter and the Goblet of Fire', 'author': 'J.K. Rowling'},
            'order phoenix': {'title': 'Harry Potter and the Order of the Phoenix', 'author': 'J.K. Rowling'},
            'half blood prince': {'title': 'Harry Potter and the Half-Blood Prince', 'author': 'J.K. Rowling'},
            'deathly hallows': {'title': 'Harry Potter and the Deathly Hallows', 'author': 'J.K. Rowling'},
            'hermione': {'title': 'Harry Potter', 'author': 'J.K. Rowling'},
            'ron weasley': {'title': 'Harry Potter', 'author': 'J.K. Rowling'},
            'dumbledore': {'title': 'Harry Potter', 'author': 'J.K. Rowling'},
            'voldemort': {'title': 'Harry Potter', 'author': 'J.K. Rowling'},
            'hogwarts': {'title': 'Harry Potter', 'author': 'J.K. Rowling'},
            
            # Game of Thrones series
            'game thrones': {'title': 'A Game of Thrones', 'author': 'George R.R. Martin'},
            'clash kings': {'title': 'A Clash of Kings', 'author': 'George R.R. Martin'},
            'storm swords': {'title': 'A Storm of Swords', 'author': 'George R.R. Martin'},
            'feast crows': {'title': 'A Feast for Crows', 'author': 'George R.R. Martin'},
            'dance dragons': {'title': 'A Dance with Dragons', 'author': 'George R.R. Martin'},
            'winter coming': {'title': 'A Game of Thrones', 'author': 'George R.R. Martin'},
            'jon snow': {'title': 'A Game of Thrones', 'author': 'George R.R. Martin'},
            'daenerys': {'title': 'A Game of Thrones', 'author': 'George R.R. Martin'},
            'tyrion': {'title': 'A Game of Thrones', 'author': 'George R.R. Martin'},
            'arya stark': {'title': 'A Game of Thrones', 'author': 'George R.R. Martin'},
            'westeros': {'title': 'A Game of Thrones', 'author': 'George R.R. Martin'},
            
            # Dune series
            'dune': {'title': 'Dune', 'author': 'Frank Herbert'},
            'paul atreides': {'title': 'Dune', 'author': 'Frank Herbert'},
            'arrakis': {'title': 'Dune', 'author': 'Frank Herbert'},
            'spice melange': {'title': 'Dune', 'author': 'Frank Herbert'},
            'fremen': {'title': 'Dune', 'author': 'Frank Herbert'},
            'bene gesserit': {'title': 'Dune', 'author': 'Frank Herbert'},
            
            # The Hunger Games
            'hunger games': {'title': 'The Hunger Games', 'author': 'Suzanne Collins'},
            'katniss everdeen': {'title': 'The Hunger Games', 'author': 'Suzanne Collins'},
            'peeta mellark': {'title': 'The Hunger Games', 'author': 'Suzanne Collins'},
            'panem': {'title': 'The Hunger Games', 'author': 'Suzanne Collins'},
            'district': {'title': 'The Hunger Games', 'author': 'Suzanne Collins'},
            
            # The Chronicles of Narnia
            'narnia': {'title': 'The Chronicles of Narnia', 'author': 'C.S. Lewis'},
            'aslan': {'title': 'The Chronicles of Narnia', 'author': 'C.S. Lewis'},
            'wardrobe': {'title': 'The Lion, the Witch and the Wardrobe', 'author': 'C.S. Lewis'},
            'peter pevensie': {'title': 'The Chronicles of Narnia', 'author': 'C.S. Lewis'},
            'susan pevensie': {'title': 'The Chronicles of Narnia', 'author': 'C.S. Lewis'},
            'edmund pevensie': {'title': 'The Chronicles of Narnia', 'author': 'C.S. Lewis'},
            'lucy pevensie': {'title': 'The Chronicles of Narnia', 'author': 'C.S. Lewis'},
            
            # The Hitchhiker's Guide to the Galaxy
            'hitchhiker guide': {'title': 'The Hitchhiker\'s Guide to the Galaxy', 'author': 'Douglas Adams'},
            'arthur dent': {'title': 'The Hitchhiker\'s Guide to the Galaxy', 'author': 'Douglas Adams'},
            'ford prefect': {'title': 'The Hitchhiker\'s Guide to the Galaxy', 'author': 'Douglas Adams'},
            'vogon': {'title': 'The Hitchhiker\'s Guide to the Galaxy', 'author': 'Douglas Adams'},
            'babel fish': {'title': 'The Hitchhiker\'s Guide to the Galaxy', 'author': 'Douglas Adams'},
            
            # 1984
            'nineteen eighty four': {'title': '1984', 'author': 'George Orwell'},
            'big brother': {'title': '1984', 'author': 'George Orwell'},
            'winston smith': {'title': '1984', 'author': 'George Orwell'},
            'thought police': {'title': '1984', 'author': 'George Orwell'},
            'newspeak': {'title': '1984', 'author': 'George Orwell'},
            'doublethink': {'title': '1984', 'author': 'George Orwell'},
            
            # Brave New World
            'brave new world': {'title': 'Brave New World', 'author': 'Aldous Huxley'},
            'bernard marx': {'title': 'Brave New World', 'author': 'Aldous Huxley'},
            'john savage': {'title': 'Brave New World', 'author': 'Aldous Huxley'},
            'soma': {'title': 'Brave New World', 'author': 'Aldous Huxley'},
            
            # Fahrenheit 451
            'fahrenheit 451': {'title': 'Fahrenheit 451', 'author': 'Ray Bradbury'},
            'guy montag': {'title': 'Fahrenheit 451', 'author': 'Ray Bradbury'},
            'fireman': {'title': 'Fahrenheit 451', 'author': 'Ray Bradbury'},
            
            # To Kill a Mockingbird
            'kill mockingbird': {'title': 'To Kill a Mockingbird', 'author': 'Harper Lee'},
            'scout finch': {'title': 'To Kill a Mockingbird', 'author': 'Harper Lee'},
            'atticus finch': {'title': 'To Kill a Mockingbird', 'author': 'Harper Lee'},
            'jem finch': {'title': 'To Kill a Mockingbird', 'author': 'Harper Lee'},
            'maycomb': {'title': 'To Kill a Mockingbird', 'author': 'Harper Lee'},
            
            # The Great Gatsby
            'great gatsby': {'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald'},
            'jay gatsby': {'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald'},
            'nick carraway': {'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald'},
            'daisy buchanan': {'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald'},
            'west egg': {'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald'},
            
            # Pride and Prejudice
            'pride prejudice': {'title': 'Pride and Prejudice', 'author': 'Jane Austen'},
            'elizabeth bennet': {'title': 'Pride and Prejudice', 'author': 'Jane Austen'},
            'mr darcy': {'title': 'Pride and Prejudice', 'author': 'Jane Austen'},
            'jane bennet': {'title': 'Pride and Prejudice', 'author': 'Jane Austen'},
            'longbourn': {'title': 'Pride and Prejudice', 'author': 'Jane Austen'},
            
            # Jane Eyre
            'jane eyre': {'title': 'Jane Eyre', 'author': 'Charlotte Brontë'},
            'rochester': {'title': 'Jane Eyre', 'author': 'Charlotte Brontë'},
            'thornfield': {'title': 'Jane Eyre', 'author': 'Charlotte Brontë'},
            
            # Wuthering Heights
            'wuthering heights': {'title': 'Wuthering Heights', 'author': 'Emily Brontë'},
            'heathcliff': {'title': 'Wuthering Heights', 'author': 'Emily Brontë'},
            'catherine earnshaw': {'title': 'Wuthering Heights', 'author': 'Emily Brontë'},
            
            # Moby Dick
            'moby dick': {'title': 'Moby-Dick', 'author': 'Herman Melville'},
            'ishmael': {'title': 'Moby-Dick', 'author': 'Herman Melville'},
            'captain ahab': {'title': 'Moby-Dick', 'author': 'Herman Melville'},
            'pequod': {'title': 'Moby-Dick', 'author': 'Herman Melville'},
            
            # The Catcher in the Rye
            'catcher rye': {'title': 'The Catcher in the Rye', 'author': 'J.D. Salinger'},
            'holden caulfield': {'title': 'The Catcher in the Rye', 'author': 'J.D. Salinger'},
            'phoebe caulfield': {'title': 'The Catcher in the Rye', 'author': 'J.D. Salinger'},
            
            # The Lord of the Flies
            'lord flies': {'title': 'Lord of the Flies', 'author': 'William Golding'},
            'ralph': {'title': 'Lord of the Flies', 'author': 'William Golding'},
            'jack merridew': {'title': 'Lord of the Flies', 'author': 'William Golding'},
            'piggy': {'title': 'Lord of the Flies', 'author': 'William Golding'},
            'simon': {'title': 'Lord of the Flies', 'author': 'William Golding'},
            
            # Animal Farm
            'animal farm': {'title': 'Animal Farm', 'author': 'George Orwell'},
            'napoleon': {'title': 'Animal Farm', 'author': 'George Orwell'},
            'snowball': {'title': 'Animal Farm', 'author': 'George Orwell'},
            'boxer': {'title': 'Animal Farm', 'author': 'George Orwell'},
            'old major': {'title': 'Animal Farm', 'author': 'George Orwell'},
            
            # The Alchemist
            'alchemist': {'title': 'The Alchemist', 'author': 'Paulo Coelho'},
            'santiago': {'title': 'The Alchemist', 'author': 'Paulo Coelho'},
            'personal legend': {'title': 'The Alchemist', 'author': 'Paulo Coelho'},
            
            # The Little Prince
            'little prince': {'title': 'The Little Prince', 'author': 'Antoine de Saint-Exupéry'},
            'rose': {'title': 'The Little Prince', 'author': 'Antoine de Saint-Exupéry'},
            'baobab': {'title': 'The Little Prince', 'author': 'Antoine de Saint-Exupéry'},
            
            # The Kite Runner
            'kite runner': {'title': 'The Kite Runner', 'author': 'Khaled Hosseini'},
            'amir': {'title': 'The Kite Runner', 'author': 'Khaled Hosseini'},
            'hassan': {'title': 'The Kite Runner', 'author': 'Khaled Hosseini'},
            'kabul': {'title': 'The Kite Runner', 'author': 'Khaled Hosseini'},
            
            # A Thousand Splendid Suns
            'thousand splendid suns': {'title': 'A Thousand Splendid Suns', 'author': 'Khaled Hosseini'},
            'mariam': {'title': 'A Thousand Splendid Suns', 'author': 'Khaled Hosseini'},
            'laila': {'title': 'A Thousand Splendid Suns', 'author': 'Khaled Hosseini'},
            
            # The Book Thief
            'book thief': {'title': 'The Book Thief', 'author': 'Markus Zusak'},
            'liesel meminger': {'title': 'The Book Thief', 'author': 'Markus Zusak'},
            'death narrator': {'title': 'The Book Thief', 'author': 'Markus Zusak'},
            
            # The Fault in Our Stars
            'fault stars': {'title': 'The Fault in Our Stars', 'author': 'John Green'},
            'hazel grace': {'title': 'The Fault in Our Stars', 'author': 'John Green'},
            'augustus waters': {'title': 'The Fault in Our Stars', 'author': 'John Green'},
            
            # Looking for Alaska
            'looking alaska': {'title': 'Looking for Alaska', 'author': 'John Green'},
            'miles halter': {'title': 'Looking for Alaska', 'author': 'John Green'},
            'alaska young': {'title': 'Looking for Alaska', 'author': 'John Green'},
            
            # Paper Towns
            'paper towns': {'title': 'Paper Towns', 'author': 'John Green'},
            'quentin jacobsen': {'title': 'Paper Towns', 'author': 'John Green'},
            'margo roth spiegelman': {'title': 'Paper Towns', 'author': 'John Green'},
            
            # Divergent
            'divergent': {'title': 'Divergent', 'author': 'Veronica Roth'},
            'tris prior': {'title': 'Divergent', 'author': 'Veronica Roth'},
            'four': {'title': 'Divergent', 'author': 'Veronica Roth'},
            'dauntless': {'title': 'Divergent', 'author': 'Veronica Roth'},
            'abnegation': {'title': 'Divergent', 'author': 'Veronica Roth'},
            
            # The Maze Runner
            'maze runner': {'title': 'The Maze Runner', 'author': 'James Dashner'},
            'thomas': {'title': 'The Maze Runner', 'author': 'James Dashner'},
            'teresa': {'title': 'The Maze Runner', 'author': 'James Dashner'},
            'glade': {'title': 'The Maze Runner', 'author': 'James Dashner'},
            
            # The Giver
            'giver': {'title': 'The Giver', 'author': 'Lois Lowry'},
            'jonas': {'title': 'The Giver', 'author': 'Lois Lowry'},
            'giver': {'title': 'The Giver', 'author': 'Lois Lowry'},
            'community': {'title': 'The Giver', 'author': 'Lois Lowry'},
            
            # Ender's Game
            'ender game': {'title': 'Ender\'s Game', 'author': 'Orson Scott Card'},
            'ender wiggin': {'title': 'Ender\'s Game', 'author': 'Orson Scott Card'},
            'battle school': {'title': 'Ender\'s Game', 'author': 'Orson Scott Card'},
            'valentine wiggin': {'title': 'Ender\'s Game', 'author': 'Orson Scott Card'},
            'peter wiggin': {'title': 'Ender\'s Game', 'author': 'Orson Scott Card'},
            
            # Foundation
            'foundation': {'title': 'Foundation', 'author': 'Isaac Asimov'},
            'hari seldon': {'title': 'Foundation', 'author': 'Isaac Asimov'},
            'psychohistory': {'title': 'Foundation', 'author': 'Isaac Asimov'},
            'trantor': {'title': 'Foundation', 'author': 'Isaac Asimov'},
            
            # I, Robot
            'robot': {'title': 'I, Robot', 'author': 'Isaac Asimov'},
            'susan calvin': {'title': 'I, Robot', 'author': 'Isaac Asimov'},
            'three laws': {'title': 'I, Robot', 'author': 'Isaac Asimov'},
            
            # Neuromancer
            'neuromancer': {'title': 'Neuromancer', 'author': 'William Gibson'},
            'case': {'title': 'Neuromancer', 'author': 'William Gibson'},
            'molly': {'title': 'Neuromancer', 'author': 'William Gibson'},
            'matrix': {'title': 'Neuromancer', 'author': 'William Gibson'},
            
            # Snow Crash
            'snow crash': {'title': 'Snow Crash', 'author': 'Neal Stephenson'},
            'hiro protagonist': {'title': 'Snow Crash', 'author': 'Neal Stephenson'},
            'y.t.': {'title': 'Snow Crash', 'author': 'Neal Stephenson'},
            'metaverse': {'title': 'Snow Crash', 'author': 'Neal Stephenson'},
            
            # Ready Player One
            'ready player one': {'title': 'Ready Player One', 'author': 'Ernest Cline'},
            'wade watts': {'title': 'Ready Player One', 'author': 'Ernest Cline'},
            'parzival': {'title': 'Ready Player One', 'author': 'Ernest Cline'},
            'oasis': {'title': 'Ready Player One', 'author': 'Ernest Cline'},
            
            # The Martian
            'martian': {'title': 'The Martian', 'author': 'Andy Weir'},
            'mark watney': {'title': 'The Martian', 'author': 'Andy Weir'},
            'mars': {'title': 'The Martian', 'author': 'Andy Weir'},
            'ares': {'title': 'The Martian', 'author': 'Andy Weir'},
            
            # Project Hail Mary
            'project hail mary': {'title': 'Project Hail Mary', 'author': 'Andy Weir'},
            'ryland grace': {'title': 'Project Hail Mary', 'author': 'Andy Weir'},
            'rocky': {'title': 'Project Hail Mary', 'author': 'Andy Weir'},
            
            # The Three-Body Problem
            'three body problem': {'title': 'The Three-Body Problem', 'author': 'Liu Cixin'},
            'ye wenjie': {'title': 'The Three-Body Problem', 'author': 'Liu Cixin'},
            'wang miao': {'title': 'The Three-Body Problem', 'author': 'Liu Cixin'},
            'trisolaris': {'title': 'The Three-Body Problem', 'author': 'Liu Cixin'},
            
            # The Dark Forest
            'dark forest': {'title': 'The Dark Forest', 'author': 'Liu Cixin'},
            'luo ji': {'title': 'The Dark Forest', 'author': 'Liu Cixin'},
            'wallfacer': {'title': 'The Dark Forest', 'author': 'Liu Cixin'},
            
            # Death's End
            'death end': {'title': 'Death\'s End', 'author': 'Liu Cixin'},
            'cheng xin': {'title': 'Death\'s End', 'author': 'Liu Cixin'},
            'singer': {'title': 'Death\'s End', 'author': 'Liu Cixin'},
        }
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the input directory."""
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return []
        
        pdf_files = list(self.input_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.input_dir}")
        return pdf_files
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from the first few pages of the PDF."""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                # Extract text from first 5 pages (usually contains title/author info)
                text = ""
                pages_to_check = min(5, len(reader.pages))
                
                for i in range(pages_to_check):
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                logger.info(f"Extracted text from {pages_to_check} pages of {pdf_path.name}")
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize the extracted text."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def extract_title_and_author(self, text: str, filename: str) -> Dict[str, str]:
        """Extract title and author from the text using content analysis."""
        text_lower = text.lower()
        
        # Check for known books based on content
        for key, book_info in self.known_books.items():
            if key in text_lower:
                logger.info(f"Found known book: {book_info['title']} by {book_info['author']}")
                return {
                    'title': book_info['title'],
                    'author': book_info['author']
                }
        
        # If no known book found, try pattern matching
        title = self.extract_title_by_patterns(text, filename)
        author = self.extract_author_by_patterns(text)
        
        return {
            'title': title,
            'author': author
        }
    
    def extract_title_by_patterns(self, text: str, filename: str) -> str:
        """Extract title using pattern matching."""
        lines = text.split('\n')
        
        # Common patterns for titles
        title_patterns = [
            r'^[A-Z][A-Za-z\s\-\'\"]{3,80}$',  # Title case with reasonable length
            r'^[A-Z][A-Za-z\s\-\'\"]{3,80}\s*$',  # Title with trailing spaces
        ]
        
        # Look in first few lines for title
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if len(line) < 5:  # Skip very short lines
                continue
                
            # Check if line matches title patterns
            for pattern in title_patterns:
                if re.match(pattern, line):
                    # Additional checks for title-like content
                    if not any(word in line.lower() for word in ['page', 'chapter', 'table of contents', 'copyright', 'all rights reserved']):
                        logger.info(f"Found potential title: {line}")
                        return line
        
        # Fallback: use first substantial line
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 10 and not any(word in line.lower() for word in ['page', 'chapter', 'table of contents']):
                logger.info(f"Using fallback title: {line}")
                return line
        
        # If nothing found, use filename
        fallback_title = Path(filename).stem
        logger.warning(f"No title found, using filename: {fallback_title}")
        return fallback_title
    
    def extract_author_by_patterns(self, text: str) -> str:
        """Extract author using pattern matching."""
        # Common author indicators
        author_indicators = [
            r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'Author[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'Written by\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*©',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*All rights reserved',
        ]
        
        # Look for author patterns
        for pattern in author_indicators:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                logger.info(f"Found author: {author}")
                return author
        
        # Look for common name patterns in first few lines
        lines = text.split('\n')
        for line in lines[:20]:
            # Pattern for "First Last" or "First Middle Last" names
            name_match = re.search(r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$', line.strip())
            if name_match:
                potential_author = name_match.group(1)
                # Skip if it looks like a title or common words
                if len(potential_author.split()) >= 2 and not any(word in potential_author.lower() for word in ['page', 'chapter', 'table', 'contents', 'foreword', 'prologue']):
                    logger.info(f"Found potential author: {potential_author}")
                    return potential_author
        
        # Fallback: look for any capitalized phrase
        for line in lines[:15]:
            words = line.strip().split()
            if len(words) >= 2 and all(word[0].isupper() for word in words):
                potential_author = ' '.join(words)
                if not any(word in potential_author.lower() for word in ['page', 'chapter', 'table', 'contents']):
                    logger.info(f"Using fallback author: {potential_author}")
                    return potential_author
        
        # If nothing found, use default
        default_author = "Unknown Author"
        logger.warning(f"No author found, using default: {default_author}")
        return default_author
    
    def process_single_book(self, pdf_path: Path) -> Dict:
        """Process a single book and extract its metadata."""
        logger.info(f"Processing: {pdf_path.name}")
        
        # Extract text from PDF
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return {
                "title": "Unknown Title",
                "author": "Unknown Author",
                "filename": pdf_path.name,
                "extraction_method": "failed",
                "error": "No text extracted"
            }
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Extract title and author
        metadata = self.extract_title_and_author(cleaned_text, pdf_path.name)
        
        # Add additional metadata
        metadata.update({
            "filename": pdf_path.name,
            "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
            "extraction_method": "automated",
            "processing_date": datetime.now().isoformat()
        })
        
        logger.info(f"Extracted metadata for {pdf_path.name}: {metadata['title']} by {metadata['author']}")
        return metadata
    
    def process_all_books(self) -> List[Dict]:
        """Process all PDF books and extract metadata."""
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            logger.error("No PDF files found to process")
            return []
        
        all_metadata = []
        
        for pdf_file in pdf_files:
            try:
                metadata = self.process_single_book(pdf_file)
                all_metadata.append(metadata)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                # Add error entry
                all_metadata.append({
                    "title": "Error Processing",
                    "author": "Unknown Author",
                    "filename": pdf_file.name,
                    "extraction_method": "error",
                    "error": str(e),
                    "processing_date": datetime.now().isoformat()
                })
        
        logger.info(f"Successfully processed {len(all_metadata)} books")
        return all_metadata
    
    def save_metadata(self, metadata_list: List[Dict], output_filename: str = "all_books_metadata.json"):
        """Save all metadata to JSON file."""
        output_file = self.output_dir / output_filename
        
        # Create summary
        summary = {
            "total_books": len(metadata_list),
            "successful_extractions": len([m for m in metadata_list if m.get('extraction_method') == 'automated']),
            "failed_extractions": len([m for m in metadata_list if m.get('extraction_method') in ['failed', 'error']]),
            "processing_date": datetime.now().isoformat(),
            "books": metadata_list
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Metadata saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return None
    
    def generate_summary_report(self, metadata_list: List[Dict]):
        """Generate a summary report of the extraction process."""
        successful = [m for m in metadata_list if m.get('extraction_method') == 'automated']
        failed = [m for m in metadata_list if m.get('extraction_method') in ['failed', 'error']]
        
        print(f"\n{'='*60}")
        print(f"METADATA EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total books processed: {len(metadata_list)}")
        print(f"Successful extractions: {len(successful)}")
        print(f"Failed extractions: {len(failed)}")
        print(f"Success rate: {(len(successful)/len(metadata_list)*100):.1f}%")
        
        if successful:
            print(f"\nSUCCESSFULLY EXTRACTED BOOKS:")
            print(f"{'='*40}")
            for i, book in enumerate(successful, 1):
                print(f"{i:2d}. {book['title']} by {book['author']} ({book['filename']})")
        
        if failed:
            print(f"\nFAILED EXTRACTIONS:")
            print(f"{'='*25}")
            for i, book in enumerate(failed, 1):
                print(f"{i:2d}. {book['filename']} - {book.get('error', 'Unknown error')}")
        
        print(f"\n{'='*60}")

def main():
    """Main function to extract metadata from all PDF books."""
    print("📚 Multi-Book Metadata Extractor")
    print("=" * 50)
    
    # Create extractor
    extractor = MultiBookMetadataExtractor()
    
    # Process all books
    print("🔍 Processing all PDF books...")
    metadata_list = extractor.process_all_books()
    
    if not metadata_list:
        print("❌ No books were processed successfully.")
        return
    
    # Save metadata
    print("💾 Saving metadata...")
    output_file = extractor.save_metadata(metadata_list)
    
    if output_file:
        print(f"✅ Metadata saved to: {output_file}")
        
        # Generate summary report
        extractor.generate_summary_report(metadata_list)
        
        # Show sample of extracted data
        print(f"\n📄 Sample JSON Structure:")
        print(json.dumps(metadata_list[0], indent=2))
        
    else:
        print("❌ Failed to save metadata")

if __name__ == "__main__":
    main() 