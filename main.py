import pandas as pd
import spacy
import PyPDF2
import re
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model and increase max length
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

class BookAnalyzer:
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Enhanced stylometric features
        self.feature_names = [
            'sentence_count', 'word_count', 'avg_sentence_length', 'avg_word_length',
            'type_token_ratio', 'punctuation_ratio', 'capitalization_ratio',
            'function_word_ratio', 'content_word_ratio', 'syllable_count',
            'flesch_reading_ease', 'gunning_fog_index', 'smog_index'
        ]
    
    def extract_text(self, pdf_file_path: str) -> str:
        """Extract text from PDF with error handling."""
        try:
            with open(pdf_file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_file_path}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning."""
        # Remove page numbers and headers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()

    def segment_text_to_paragraphs(self, text: str, sentences_per_para: int = 5) -> List[str]:
        """Segment text into paragraphs with improved logic."""
        doc = nlp(text)
        paragraphs = []
        current_para = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) > 10:  # Filter out very short sentences
                current_para.append(sent_text)
                if len(current_para) >= sentences_per_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
        
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        return paragraphs

    def count_syllables(self, word: str) -> int:
        """Simple syllable counting."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        return max(count, 1)

    def calculate_readability_scores(self, text: str) -> Dict[str, float]:
        """Calculate various readability scores."""
        doc = nlp(text)
        sentences = list(doc.sents)
        words = [token.text.lower() for token in doc if not token.is_space and not token.is_punct]
        
        if not words or not sentences:
            return {'flesch_reading_ease': 0, 'gunning_fog_index': 0, 'smog_index': 0}
        
        # Count syllables
        syllables = sum(self.count_syllables(word) for word in words)
        
        # Flesch Reading Ease
        flesch = 206.835 - (1.015 * len(words) / len(sentences)) - (84.6 * syllables / len(words))
        
        # Gunning Fog Index
        complex_words = sum(1 for word in words if self.count_syllables(word) > 2)
        gunning_fog = 0.4 * ((len(words) / len(sentences)) + (100 * complex_words / len(words)))
        
        # SMOG Index
        smog = 1.043 * (complex_words * (30 / len(sentences)) ** 0.5) + 3.1291
        
        return {
            'flesch_reading_ease': max(0, min(100, flesch)),
            'gunning_fog_index': max(0, gunning_fog),
            'smog_index': max(0, smog)
        }

    def extract_stylometric_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive stylometric features."""
        doc = nlp(text)
        sentences = list(doc.sents)
        words = [token.text for token in doc if not token.is_space and not token.is_punct]
        
        if not words:
            return {feature: 0.0 for feature in self.feature_names}
        
        # Basic features
        sentence_count = len(sentences)
        word_count = len(words)
        avg_sentence_length = np.mean([len(sent.text.split()) for sent in sentences]) if sentences else 0
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        ttr = len(set(words)) / len(words) if words else 0
        
        # Punctuation and capitalization
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        punctuation_ratio = punctuation_count / len(text) if text else 0
        
        capitalization_count = sum(1 for char in text if char.isupper())
        capitalization_ratio = capitalization_count / len(text) if text else 0
        
        # Function vs content words
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        function_word_count = sum(1 for word in words if word.lower() in function_words)
        function_word_ratio = function_word_count / len(words) if words else 0
        content_word_ratio = 1 - function_word_ratio
        
        # Syllable count
        syllable_count = sum(self.count_syllables(word) for word in words)
        
        # Readability scores
        readability_scores = self.calculate_readability_scores(text)
        
        return {
            'sentence_count': sentence_count,
            'word_count': word_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'type_token_ratio': ttr,
            'punctuation_ratio': punctuation_ratio,
            'capitalization_ratio': capitalization_ratio,
            'function_word_ratio': function_word_ratio,
            'content_word_ratio': content_word_ratio,
            'syllable_count': syllable_count,
            **readability_scores
        }

    def infer_book_metadata(self, filename: str) -> Dict[str, str]:
        """Infer book metadata from filename or use defaults."""
        # You can enhance this with a database lookup or pattern matching
        base_name = Path(filename).stem
        
        # Default metadata - you should replace this with actual book information
        metadata = {
            'genre': 'Unknown',
            'author': 'Unknown Author',
            'book_title': base_name,
            'filename': filename,
            'processing_date': datetime.now().isoformat()
        }
        
        # Add pattern matching for common naming conventions
        if 'scf' in base_name.lower():
            metadata['genre'] = 'Science Fiction'
        elif 'mystery' in base_name.lower():
            metadata['genre'] = 'Mystery'
        elif 'romance' in base_name.lower():
            metadata['genre'] = 'Romance'
        elif 'fantasy' in base_name.lower():
            metadata['genre'] = 'Fantasy'
        
        return metadata

    def process_single_book(self, pdf_path: str) -> pd.DataFrame:
        """Process a single book and return DataFrame."""
        logger.info(f"Processing book: {pdf_path}")
        
        # Extract and clean text
        raw_text = self.extract_text(pdf_path)
        if not raw_text:
            logger.warning(f"No text extracted from {pdf_path}")
            return pd.DataFrame()
        
        cleaned_text = self.clean_text(raw_text)
        paragraphs = self.segment_text_to_paragraphs(cleaned_text)
        
        # Get metadata
        metadata = self.infer_book_metadata(Path(pdf_path).name)
        
        # Process each paragraph
        records = []
        for i, para in enumerate(paragraphs, start=1):
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue
                
            features = self.extract_stylometric_features(para)
            record = {
                **metadata,
                'paragraph_num': i,
                'text_content': para,
                'paragraph_length': len(para),
                **features
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Processed {len(records)} paragraphs from {pdf_path}")
        return df

    def process_all_books(self) -> pd.DataFrame:
        """Process all PDF books in the dataset directory."""
        pdf_files = list(self.dataset_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.dataset_path}")
            return pd.DataFrame()
        
        all_data = []
        
        for pdf_file in pdf_files:
            try:
                df = self.process_single_book(str(pdf_file))
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save combined data
            output_file = self.output_path / "all_processed_books.csv"
            combined_df.to_csv(output_file, index=False)
            
            # Save summary statistics
            self.save_summary_statistics(combined_df)
            
            logger.info(f"Successfully processed {len(pdf_files)} books with {len(combined_df)} total paragraphs")
            return combined_df
        else:
            logger.error("No books were successfully processed")
            return pd.DataFrame()

    def save_summary_statistics(self, df: pd.DataFrame):
        """Save summary statistics for the processed books."""
        summary = {
            'total_books': df['book_title'].nunique(),
            'total_paragraphs': len(df),
            'genres': df['genre'].value_counts().to_dict(),
            'authors': df['author'].value_counts().to_dict(),
            'processing_date': datetime.now().isoformat()
        }
        
        # Feature statistics
        feature_stats = {}
        for feature in self.feature_names:
            if feature in df.columns:
                feature_stats[feature] = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max())
                }
        
        summary['feature_statistics'] = feature_stats
        
        # Save to JSON
        summary_file = self.output_path / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary statistics saved to {summary_file}")

def main():
    """Main function to process all books."""
    dataset_path = "/home/orange/Documents/Projects/Copyy/Dataset_PDFs"
    output_path = "/home/orange/Documents/Projects/Copyy/Processed_Data"
    
    analyzer = BookAnalyzer(dataset_path, output_path)
    df = analyzer.process_all_books()
    
    if not df.empty:
        print(f"\nProcessing complete!")
        print(f"Total books processed: {df['book_title'].nunique()}")
        print(f"Total paragraphs: {len(df)}")
        print(f"Genres found: {list(df['genre'].unique())}")
        print(f"\nSample data:")
        print(df.head())
    else:
        print("No books were processed successfully.")

if __name__ == "__main__":
    main()
