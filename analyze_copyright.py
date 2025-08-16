#!/usr/bin/env python3
"""
Copyright Infringement Analysis System
Combines copy detection and stylometric analysis to identify potential copyright issues.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CopyrightAnalyzer:
    def __init__(self, data_path: str, output_path: str):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Load processed data
        self.df = None
        self.load_data()
        
        # Analysis results
        self.copy_detection_results = {}
        self.stylometric_results = {}
        self.combined_results = {}
    
    def load_data(self):
        """Load processed book data."""
        data_file = self.data_path / "all_processed_books.csv"
        if data_file.exists():
            self.df = pd.read_csv(data_file)
            logger.info(f"Loaded {len(self.df)} paragraphs from {self.df['book_title'].nunique()} books")
        else:
            logger.error(f"Data file not found: {data_file}")
            raise FileNotFoundError(f"Please run main.py first to process books")
    
    def copy_detection_analysis(self, similarity_threshold: float = 0.8) -> Dict:
        """
        Perform copy detection analysis using text similarity.
        
        Args:
            similarity_threshold: Minimum similarity score to flag as potential copy
            
        Returns:
            Dictionary with copy detection results
        """
        logger.info("Starting copy detection analysis...")
        
        # Extract text content
        texts = self.df['text_content'].fillna('').tolist()
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # 1-3 word n-grams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            stop_words='english'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            logger.info(f"Created TF-IDF matrix with {tfidf_matrix.shape[1]} features")
        except Exception as e:
            logger.error(f"Error creating TF-IDF matrix: {e}")
            return {}
        
        # Calculate cosine similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Find high similarity pairs
        high_similarity_pairs = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i][j] >= similarity_threshold:
                    high_similarity_pairs.append({
                        'paragraph_1_idx': i,
                        'paragraph_2_idx': j,
                        'similarity_score': similarities[i][j],
                        'book_1': self.df.iloc[i]['book_title'],
                        'book_2': self.df.iloc[j]['book_title'],
                        'author_1': self.df.iloc[i]['author'],
                        'author_2': self.df.iloc[j]['author'],
                        'text_1': self.df.iloc[i]['text_content'][:200] + "...",
                        'text_2': self.df.iloc[j]['text_content'][:200] + "..."
                    })
        
        # Sort by similarity score
        high_similarity_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Group by book pairs
        book_pair_similarities = {}
        for pair in high_similarity_pairs:
            book_pair = tuple(sorted([pair['book_1'], pair['book_2']]))
            if book_pair not in book_pair_similarities:
                book_pair_similarities[book_pair] = []
            book_pair_similarities[book_pair].append(pair)
        
        results = {
            'total_high_similarity_pairs': len(high_similarity_pairs),
            'similarity_threshold': similarity_threshold,
            'high_similarity_pairs': high_similarity_pairs[:50],  # Top 50
            'book_pair_summary': {},
            'overall_statistics': {
                'mean_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'similarity_std': np.std(similarities)
            }
        }
        
        # Create book pair summary
        for book_pair, pairs in book_pair_similarities.items():
            avg_similarity = np.mean([p['similarity_score'] for p in pairs])
            results['book_pair_summary'][f"{book_pair[0]} vs {book_pair[1]}"] = {
                'count': len(pairs),
                'avg_similarity': avg_similarity,
                'max_similarity': max([p['similarity_score'] for p in pairs])
            }
        
        self.copy_detection_results = results
        logger.info(f"Copy detection complete. Found {len(high_similarity_pairs)} high-similarity pairs")
        return results
    
    def stylometric_analysis(self) -> Dict:
        """
        Perform stylometric analysis to identify writing style similarities.
        
        Returns:
            Dictionary with stylometric analysis results
        """
        logger.info("Starting stylometric analysis...")
        
        # Select stylometric features
        feature_columns = [
            'avg_sentence_length', 'avg_word_length', 'type_token_ratio',
            'punctuation_ratio', 'capitalization_ratio', 'function_word_ratio',
            'content_word_ratio', 'flesch_reading_ease', 'gunning_fog_index'
        ]
        
        # Filter out rows with missing features
        df_features = self.df[feature_columns].dropna()
        
        if len(df_features) == 0:
            logger.error("No valid stylometric features found")
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_features)
        
        # Perform clustering to identify style groups
        n_clusters = min(10, len(df_features) // 10)  # Adaptive clustering
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to dataframe
        df_with_clusters = self.df.iloc[df_features.index].copy()
        df_with_clusters['style_cluster'] = cluster_labels
        
        # Analyze style clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['style_cluster'] == cluster_id]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'books': cluster_data['book_title'].unique().tolist(),
                'authors': cluster_data['author'].unique().tolist(),
                'genres': cluster_data['genre'].unique().tolist(),
                'avg_features': cluster_data[feature_columns].mean().to_dict()
            }
        
        # Calculate style similarity matrix
        style_similarity = cosine_similarity(features_scaled)
        
        # Find books with similar styles
        book_style_similarities = {}
        unique_books = df_with_clusters['book_title'].unique()
        
        for i, book1 in enumerate(unique_books):
            for j, book2 in enumerate(unique_books[i+1:], i+1):
                book1_indices = df_with_clusters[df_with_clusters['book_title'] == book1].index
                book2_indices = df_with_clusters[df_with_clusters['book_title'] == book2].index
                
                # Calculate average similarity between books
                similarities = []
                for idx1 in book1_indices:
                    for idx2 in book2_indices:
                        if idx1 < len(style_similarity) and idx2 < len(style_similarity):
                            similarities.append(style_similarity[idx1][idx2])
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    book_style_similarities[f"{book1} vs {book2}"] = avg_similarity
        
        # Sort by similarity
        sorted_style_similarities = sorted(
            book_style_similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        results = {
            'n_clusters': n_clusters,
            'cluster_analysis': cluster_analysis,
            'book_style_similarities': dict(sorted_style_similarities),
            'top_style_similar_pairs': sorted_style_similarities[:20],
            'feature_importance': dict(zip(feature_columns, kmeans.cluster_centers_.std(axis=0)))
        }
        
        self.stylometric_results = results
        logger.info(f"Stylometric analysis complete. Identified {n_clusters} style clusters")
        return results
    
    def sequence_matching_analysis(self, min_ratio: float = 0.7) -> Dict:
        """
        Perform sequence matching to find exact or near-exact text matches.
        
        Args:
            min_ratio: Minimum similarity ratio for sequence matching
            
        Returns:
            Dictionary with sequence matching results
        """
        logger.info("Starting sequence matching analysis...")
        
        texts = self.df['text_content'].fillna('').tolist()
        sequence_matches = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Use SequenceMatcher for detailed text comparison
                matcher = SequenceMatcher(None, texts[i], texts[j])
                ratio = matcher.ratio()
                
                if ratio >= min_ratio:
                    # Find the longest matching block
                    blocks = matcher.get_matching_blocks()
                    longest_match = max(blocks, key=lambda x: x.size)
                    
                    if longest_match.size > 50:  # Minimum match size
                        sequence_matches.append({
                            'paragraph_1_idx': i,
                            'paragraph_2_idx': j,
                            'similarity_ratio': ratio,
                            'match_size': longest_match.size,
                            'book_1': self.df.iloc[i]['book_title'],
                            'book_2': self.df.iloc[j]['book_title'],
                            'author_1': self.df.iloc[i]['author'],
                            'author_2': self.df.iloc[j]['author'],
                            'matched_text': texts[i][longest_match.a:longest_match.a + longest_match.size]
                        })
        
        # Sort by match size and ratio
        sequence_matches.sort(key=lambda x: (x['match_size'], x['similarity_ratio']), reverse=True)
        
        results = {
            'total_sequence_matches': len(sequence_matches),
            'min_ratio': min_ratio,
            'sequence_matches': sequence_matches[:50],  # Top 50
            'statistics': {
                'avg_match_size': np.mean([m['match_size'] for m in sequence_matches]) if sequence_matches else 0,
                'max_match_size': max([m['match_size'] for m in sequence_matches]) if sequence_matches else 0,
                'avg_similarity': np.mean([m['similarity_ratio'] for m in sequence_matches]) if sequence_matches else 0
            }
        }
        
        return results
    
    def combined_analysis(self) -> Dict:
        """
        Combine copy detection and stylometric analysis for comprehensive results.
        
        Returns:
            Dictionary with combined analysis results
        """
        logger.info("Performing combined analysis...")
        
        # Perform individual analyses
        copy_results = self.copy_detection_results
        style_results = self.stylometric_results
        sequence_results = self.sequence_matching_analysis()
        
        # Combine results
        combined_results = {
            'copy_detection': copy_results,
            'stylometric_analysis': style_results,
            'sequence_matching': sequence_results,
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Risk assessment
        risk_factors = []
        
        # High copy detection risk
        if copy_results.get('total_high_similarity_pairs', 0) > 10:
            risk_factors.append("High number of text similarities detected")
        
        # High style similarity risk
        top_style_pairs = style_results.get('top_style_similar_pairs', [])
        if any(similarity > 0.8 for _, similarity in top_style_pairs[:5]):
            risk_factors.append("Very similar writing styles detected")
        
        # Exact sequence matches
        if sequence_results.get('total_sequence_matches', 0) > 0:
            risk_factors.append("Exact or near-exact text sequences found")
        
        # Generate recommendations
        recommendations = []
        
        if risk_factors:
            recommendations.append("⚠️ HIGH RISK: Multiple indicators of potential copyright issues detected")
            recommendations.append("Review flagged text pairs for potential infringement")
            recommendations.append("Consider legal consultation for high-similarity content")
        else:
            recommendations.append("✅ LOW RISK: No significant copyright concerns detected")
        
        if copy_results.get('total_high_similarity_pairs', 0) > 0:
            recommendations.append("Review high-similarity text pairs for context and attribution")
        
        if style_results.get('n_clusters', 0) < 3:
            recommendations.append("Limited style diversity detected - consider expanding dataset")
        
        combined_results['risk_assessment'] = {
            'risk_level': 'HIGH' if risk_factors else 'LOW',
            'risk_factors': risk_factors,
            'total_risk_indicators': len(risk_factors)
        }
        
        combined_results['recommendations'] = recommendations
        
        self.combined_results = combined_results
        return combined_results
    
    def generate_report(self, output_file: str = "copyright_analysis_report.json"):
        """Generate comprehensive analysis report."""
        logger.info("Generating analysis report...")
        
        # Perform all analyses
        combined_results = self.combined_analysis()
        
        # Save detailed report
        report_path = self.output_path / output_file
        with open(report_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # Generate summary
        summary = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_books_analyzed': self.df['book_title'].nunique(),
            'total_paragraphs_analyzed': len(self.df),
            'risk_level': combined_results['risk_assessment']['risk_level'],
            'risk_factors': combined_results['risk_assessment']['risk_factors'],
            'key_findings': {
                'high_similarity_pairs': combined_results['copy_detection'].get('total_high_similarity_pairs', 0),
                'style_clusters': combined_results['stylometric_analysis'].get('n_clusters', 0),
                'sequence_matches': combined_results['sequence_matching'].get('total_sequence_matches', 0)
            },
            'recommendations': combined_results['recommendations']
        }
        
        # Save summary
        summary_path = self.output_path / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Analysis report saved to {report_path}")
        logger.info(f"Summary saved to {summary_path}")
        
        return summary
    
    def create_visualizations(self):
        """Create visualizations for the analysis results."""
        logger.info("Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Style cluster distribution
        if self.stylometric_results:
            cluster_sizes = [info['size'] for info in self.stylometric_results['cluster_analysis'].values()]
            cluster_names = list(self.stylometric_results['cluster_analysis'].keys())
            
            axes[0, 0].pie(cluster_sizes, labels=cluster_names, autopct='%1.1f%%')
            axes[0, 0].set_title('Writing Style Clusters Distribution')
        
        # 2. Similarity score distribution
        if self.copy_detection_results:
            similarities = [pair['similarity_score'] for pair in self.copy_detection_results.get('high_similarity_pairs', [])]
            if similarities:
                axes[0, 1].hist(similarities, bins=20, alpha=0.7, color='skyblue')
                axes[0, 1].set_xlabel('Similarity Score')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Text Similarity Distribution')
        
        # 3. Book similarity heatmap (top 10 books)
        if self.stylometric_results:
            top_books = list(self.stylometric_results['book_style_similarities'].keys())[:10]
            similarity_matrix = []
            book_names = []
            
            for book in top_books:
                if book not in book_names:
                    book_names.append(book.split(' vs ')[0])
                    book_names.append(book.split(' vs ')[1])
            
            book_names = list(set(book_names))[:10]  # Limit to 10 unique books
            
            for i, book1 in enumerate(book_names):
                row = []
                for j, book2 in enumerate(book_names):
                    if i == j:
                        row.append(1.0)
                    else:
                        pair_key = f"{book1} vs {book2}"
                        reverse_key = f"{book2} vs {book1}"
                        similarity = (self.stylometric_results['book_style_similarities'].get(pair_key, 0) + 
                                    self.stylometric_results['book_style_similarities'].get(reverse_key, 0)) / 2
                        row.append(similarity)
                similarity_matrix.append(row)
            
            if similarity_matrix:
                im = axes[1, 0].imshow(similarity_matrix, cmap='YlOrRd')
                axes[1, 0].set_xticks(range(len(book_names)))
                axes[1, 0].set_yticks(range(len(book_names)))
                axes[1, 0].set_xticklabels(book_names, rotation=45, ha='right')
                axes[1, 0].set_yticklabels(book_names)
                axes[1, 0].set_title('Book Style Similarity Heatmap')
                plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Risk assessment summary
        if self.combined_results:
            risk_data = {
                'Risk Level': [self.combined_results['risk_assessment']['risk_level']],
                'Risk Factors': [len(self.combined_results['risk_assessment']['risk_factors'])],
                'High Similarity Pairs': [self.combined_results['copy_detection'].get('total_high_similarity_pairs', 0)],
                'Sequence Matches': [self.combined_results['sequence_matching'].get('total_sequence_matches', 0)]
            }
            
            risk_df = pd.DataFrame(risk_data)
            axes[1, 1].text(0.1, 0.8, f"Risk Level: {risk_data['Risk Level'][0]}", fontsize=14, fontweight='bold')
            axes[1, 1].text(0.1, 0.6, f"Risk Factors: {risk_data['Risk Factors'][0]}", fontsize=12)
            axes[1, 1].text(0.1, 0.4, f"High Similarity Pairs: {risk_data['High Similarity Pairs'][0]}", fontsize=12)
            axes[1, 1].text(0.1, 0.2, f"Sequence Matches: {risk_data['Sequence Matches'][0]}", fontsize=12)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('Risk Assessment Summary')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_path / "copyright_analysis_visualizations.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {viz_path}")

def main():
    """Main function to run copyright analysis."""
    data_path = "/home/orange/Documents/Projects/Copyy/Processed_Data"
    output_path = "/home/orange/Documents/Projects/Copyy/Processed_Data"
    
    try:
        analyzer = CopyrightAnalyzer(data_path, output_path)
        
        # Generate comprehensive analysis
        summary = analyzer.generate_report()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Print summary
        print("\n=== Copyright Analysis Summary ===")
        print(f"Books analyzed: {summary['total_books_analyzed']}")
        print(f"Paragraphs analyzed: {summary['total_paragraphs_analyzed']}")
        print(f"Risk level: {summary['risk_level']}")
        print(f"Risk factors: {len(summary['risk_factors'])}")
        
        print("\nKey Findings:")
        for key, value in summary['key_findings'].items():
            print(f"  - {key.replace('_', ' ').title()}: {value}")
        
        print("\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
        
        print(f"\nDetailed report saved to: {output_path}/copyright_analysis_report.json")
        print(f"Visualizations saved to: {output_path}/copyright_analysis_visualizations.png")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 