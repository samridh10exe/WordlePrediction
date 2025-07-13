"""
Data preprocessing pipeline for Wordle prediction.
Handles missing values, outliers, and data quality issues.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re


class WordleDataPreprocessor:
    def __init__(self, data_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        
        # Quality control parameters
        self.min_frequency_threshold = 1e-10
        self.max_frequency_threshold = 1.0
        self.valid_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
    def clean_word_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate word data."""
        self.logger.info(f"Starting word data cleaning. Input: {len(df)} words")
        
        initial_count = len(df)
        
        # Ensure we have a word column
        if 'word' not in df.columns:
            self.logger.error("DataFrame must contain a 'word' column")
            raise ValueError("DataFrame must contain a 'word' column")
        
        # Remove invalid words (non-5-letter, non-alphabetic)
        df = df.copy()
        
        # Handle encoding issues and standardize case
        df['word'] = df['word'].astype(str).str.strip().str.upper()
        
        # Remove words that aren't exactly 5 letters
        df = df[df['word'].str.len() == 5]
        self.logger.info(f"After length filter: {len(df)} words ({initial_count - len(df)} removed)")
        
        # Remove words with non-alphabetic characters
        df = df[df['word'].str.isalpha()]
        self.logger.info(f"After alphabetic filter: {len(df)} words")
        
        # Remove words with invalid characters
        valid_mask = df['word'].apply(lambda x: all(c in self.valid_letters for c in x))
        df = df[valid_mask]
        self.logger.info(f"After valid character filter: {len(df)} words")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['word'])
        self.logger.info(f"After deduplication: {len(df)} words")
        
        # Additional quality checks
        df = self._additional_word_validation(df)
        
        self.logger.info(f"Word data cleaning complete. Final: {len(df)} words")
        return df.reset_index(drop=True)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with domain-specific strategies."""
        self.logger.info("Handling missing values...")
        
        df = df.copy()
        initial_shape = df.shape
        
        # Strategy for different column types
        for column in df.columns:
            if column == 'word':
                # Words cannot be missing - remove these rows
                df = df.dropna(subset=[column])
                continue
            
            missing_count = df[column].isna().sum()
            if missing_count > 0:
                self.logger.info(f"Column '{column}' has {missing_count} missing values")
                
                if column in ['frequency', 'log_frequency']:
                    # For frequency data, use minimum observed frequency
                    min_freq = df[column].min()
                    if pd.isna(min_freq):
                        min_freq = self.min_frequency_threshold
                    df[column] = df[column].fillna(min_freq)
                    self.logger.info(f"Filled missing {column} with {min_freq}")
                
                elif column in ['vowel_count', 'consonant_count', 'unique_letters']:
                    # For count data, calculate from word
                    df = self._calculate_missing_counts(df, column)
                
                elif column.startswith('pos_') and column.endswith('_letter'):
                    # For position letters, extract from word
                    df = self._calculate_missing_positions(df, column)
                
                elif df[column].dtype in ['int64', 'float64']:
                    # For other numeric data, use median
                    median_val = df[column].median()
                    df[column] = df[column].fillna(median_val)
                    self.logger.info(f"Filled missing {column} with median: {median_val}")
                
                elif df[column].dtype == 'bool':
                    # For boolean data, use mode (most common)
                    mode_val = df[column].mode()[0] if not df[column].mode().empty else False
                    df[column] = df[column].fillna(mode_val)
                    self.logger.info(f"Filled missing {column} with mode: {mode_val}")
                
                else:
                    # For categorical data, use mode or 'unknown'
                    if not df[column].mode().empty:
                        mode_val = df[column].mode()[0]
                        df[column] = df[column].fillna(mode_val)
                        self.logger.info(f"Filled missing {column} with mode: {mode_val}")
                    else:
                        df[column] = df[column].fillna('unknown')
                        self.logger.info(f"Filled missing {column} with 'unknown'")
        
        final_shape = df.shape
        self.logger.info(f"Missing value handling complete. Shape: {initial_shape} -> {final_shape}")
        return df
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in frequency data."""
        self.logger.info("Detecting and handling outliers...")
        
        df = df.copy()
        outlier_info = {}
        
        # Focus on frequency columns for outlier detection
        frequency_columns = [col for col in df.columns if 'frequency' in col.lower()]
        
        for column in frequency_columns:
            if df[column].dtype in ['int64', 'float64']:
                # Use IQR method for outlier detection
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                outlier_count = outliers_mask.sum()
                
                if outlier_count > 0:
                    self.logger.info(f"Found {outlier_count} outliers in {column}")
                    outlier_info[column] = {
                        'count': outlier_count,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    
                    # Handle outliers by capping
                    df.loc[df[column] < lower_bound, column] = lower_bound
                    df.loc[df[column] > upper_bound, column] = upper_bound
                    
                    self.logger.info(f"Capped outliers in {column} to bounds [{lower_bound:.6f}, {upper_bound:.6f}]")
        
        # Special handling for frequency values
        if 'frequency' in df.columns:
            # Ensure frequencies are within valid range
            df['frequency'] = df['frequency'].clip(lower=self.min_frequency_threshold, 
                                                 upper=self.max_frequency_threshold)
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Perform comprehensive data quality validation."""
        self.logger.info("Performing data quality validation...")
        
        quality_report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_words': 0,
            'invalid_words': [],
            'data_types': {},
            'value_ranges': {}
        }
        
        # Check for missing values
        for column in df.columns:
            missing_count = df[column].isna().sum()
            if missing_count > 0:
                quality_report['missing_values'][column] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(df)) * 100
                }
        
        # Check for duplicate words
        if 'word' in df.columns:
            quality_report['duplicate_words'] = df['word'].duplicated().sum()
        
        # Validate word format
        if 'word' in df.columns:
            invalid_words = df[~df['word'].str.match(r'^[A-Z]{5}$')]['word'].tolist()
            quality_report['invalid_words'] = invalid_words
        
        # Data type information
        for column in df.columns:
            quality_report['data_types'][column] = str(df[column].dtype)
        
        # Value ranges for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            quality_report['value_ranges'][column] = {
                'min': df[column].min(),
                'max': df[column].max(),
                'mean': df[column].mean(),
                'std': df[column].std()
            }
        
        self.logger.info(f"Data quality validation complete. Report generated for {len(df)} records")
        return quality_report
    
    def preprocess_pipeline(self, df: pd.DataFrame, save_results: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Run complete preprocessing pipeline."""
        self.logger.info("Starting complete preprocessing pipeline...")
        
        # Step 1: Clean word data
        df_clean = self.clean_word_data(df)
        
        # Step 2: Handle missing values
        df_complete = self.handle_missing_values(df_clean)
        
        # Step 3: Detect and handle outliers
        df_processed = self.detect_outliers(df_complete)
        
        # Step 4: Final validation
        quality_report = self.validate_data_quality(df_processed)
        
        # Step 5: Save results if requested
        if save_results:
            output_path = self.data_dir / "processed" / "preprocessed_data.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_processed.to_csv(output_path, index=False)
            self.logger.info(f"Saved preprocessed data to {output_path}")
            
            # Save quality report
            import json
            report_path = self.data_dir / "processed" / "quality_report.json"
            with open(report_path, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            self.logger.info(f"Saved quality report to {report_path}")
        
        self.logger.info("Preprocessing pipeline complete")
        return df_processed, quality_report
    
    def _additional_word_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additional word validation checks."""
        
        # Remove words with repeated substrings (likely encoding errors)
        pattern_mask = ~df['word'].str.match(r'^(.)\1{4}$')  # Not all same letter
        df = df[pattern_mask]
        
        # Remove words that are obviously not English (contain unusual letter patterns)
        # This is a basic check - could be enhanced with more sophisticated validation
        unusual_patterns = [
            r'[QX][^U]',  # Q or X not followed by U (rare in English)
            r'[BCDFGHJKLMNPQRSTVWXYZ]{4}',  # 4+ consecutive consonants
        ]
        
        for pattern in unusual_patterns:
            suspicious_mask = df['word'].str.contains(pattern, regex=True)
            if suspicious_mask.any():
                suspicious_words = df[suspicious_mask]['word'].tolist()
                self.logger.info(f"Found {len(suspicious_words)} words with pattern {pattern}: {suspicious_words[:5]}...")
        
        return df
    
    def _calculate_missing_counts(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Calculate missing count values from word data."""
        
        missing_mask = df[column].isna()
        
        for idx in df[missing_mask].index:
            word = df.loc[idx, 'word']
            if pd.notna(word):
                if column == 'vowel_count':
                    df.loc[idx, column] = sum(1 for c in word.lower() if c in 'aeiou')
                elif column == 'consonant_count':
                    df.loc[idx, column] = sum(1 for c in word.lower() if c.isalpha() and c not in 'aeiou')
                elif column == 'unique_letters':
                    df.loc[idx, column] = len(set(word.lower()))
        
        return df
    
    def _calculate_missing_positions(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Calculate missing position letter values from word data."""
        
        missing_mask = df[column].isna()
        
        # Extract position number from column name (e.g., 'pos_1_letter' -> 1)
        pos_match = re.search(r'pos_(\d+)_letter', column)
        if pos_match:
            position = int(pos_match.group(1)) - 1  # Convert to 0-based index
            
            for idx in df[missing_mask].index:
                word = df.loc[idx, 'word']
                if pd.notna(word) and len(word) > position:
                    df.loc[idx, column] = word[position].upper()
        
        return df