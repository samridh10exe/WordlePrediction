#!/usr/bin/env python3
"""
Data conversion script for historical Wordle data.

This script processes the raw historical Wordle data file and converts it into 
structured formats suitable for machine learning training.

Author: Wordle Prediction ML System
Created: July 13, 2025
"""

import re
import csv
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WordleDataConverter:
    """Converts raw historical Wordle data to structured formats."""
    
    def __init__(self, input_file: str, output_dir: str = None):
        """
        Initialize the converter.
        
        Args:
            input_file: Path to the raw historical data file
            output_dir: Directory to save converted data (default: same as input)
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir) if output_dir else self.input_file.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure output directory structure
        (self.output_dir / "processed").mkdir(exist_ok=True)
        
    def parse_raw_data(self) -> List[Dict[str, Any]]:
        """
        Parse the raw text file and extract structured data.
        
        Returns:
            List of dictionaries containing parsed Wordle data
        """
        logger.info(f"Reading raw data from {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into lines and filter out empty lines and headers
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Remove header lines
        lines = [line for line in lines if not line.startswith('Previous Wordle Solutions') 
                and line != 'Date' and line != '#' and line != 'Solution']
        
        parsed_data = []
        i = 0
        
        while i < len(lines):
            try:
                # Look for date pattern
                date_line = lines[i]
                if not self._is_date_line(date_line):
                    i += 1
                    continue
                
                # Next line should be puzzle number
                if i + 1 >= len(lines):
                    break
                    
                number_line = lines[i + 1]
                if not number_line.startswith('#'):
                    i += 1
                    continue
                
                # Next line should be solution
                if i + 2 >= len(lines):
                    break
                    
                solution_line = lines[i + 2]
                
                # Parse the data
                date_obj = self._parse_date(date_line)
                puzzle_number = int(number_line.replace('#', ''))
                solution = solution_line.upper()
                
                # Create entry
                entry = {
                    'date': date_obj.strftime('%Y-%m-%d'),
                    'puzzle_number': puzzle_number,
                    'solution': solution,
                    'word_length': len(solution),
                    'year': date_obj.year,
                    'month': date_obj.month,
                    'day': date_obj.day,
                    'day_of_week': date_obj.weekday(),  # 0=Monday, 6=Sunday
                    'day_of_year': date_obj.timetuple().tm_yday,
                    'week_of_year': date_obj.isocalendar()[1],
                    'quarter': (date_obj.month - 1) // 3 + 1,
                    'is_weekend': date_obj.weekday() >= 5,
                }
                
                # Add word features
                self._add_word_features(entry, solution)
                
                parsed_data.append(entry)
                logger.debug(f"Parsed: {date_line} #{puzzle_number} {solution}")
                
                i += 3  # Move to next group
                
            except Exception as e:
                logger.warning(f"Error parsing lines around index {i}: {e}")
                i += 1
                continue
        
        logger.info(f"Successfully parsed {len(parsed_data)} Wordle entries")
        return parsed_data
    
    def _is_date_line(self, line: str) -> bool:
        """Check if a line contains a date."""
        date_patterns = [
            r'^[A-Za-z]+ \d{1,2}, \d{4}$',  # January 1, 2022
            r'^\d{1,2}/\d{1,2}/\d{4}$',     # 1/1/2022
            r'^\d{4}-\d{2}-\d{2}$',         # 2022-01-01
        ]
        
        return any(re.match(pattern, line) for pattern in date_patterns)
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string into datetime object."""
        # Try different date formats
        formats = [
            '%B %d, %Y',    # January 1, 2022
            '%b %d, %Y',    # Jan 1, 2022
            '%m/%d/%Y',     # 1/1/2022
            '%Y-%m-%d',     # 2022-01-01
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse date: {date_str}")
    
    def _add_word_features(self, entry: Dict[str, Any], word: str) -> None:
        """Add linguistic features for the word."""
        word = word.upper()
        
        # Letter frequency features
        letter_counts = {}
        for letter in word:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
        
        # Basic word features
        entry.update({
            'unique_letters': len(set(word)),
            'repeated_letters': len(word) - len(set(word)),
            'has_duplicates': len(word) != len(set(word)),
            'vowel_count': sum(1 for c in word if c in 'AEIOU'),
            'consonant_count': sum(1 for c in word if c not in 'AEIOU'),
        })
        
        # Letter position features
        for i, letter in enumerate(word):
            entry[f'pos_{i+1}_letter'] = letter
        
        # Common letter patterns
        entry.update({
            'starts_with_vowel': word[0] in 'AEIOU',
            'ends_with_vowel': word[-1] in 'AEIOU',
            'contains_common_pairs': self._has_common_pairs(word),
            'contains_rare_letters': self._has_rare_letters(word),
        })
        
        # Double letter patterns
        double_letters = []
        for i in range(len(word) - 1):
            if word[i] == word[i + 1]:
                double_letters.append(word[i])
        
        entry.update({
            'has_double_letters': len(double_letters) > 0,
            'double_letters': ','.join(double_letters) if double_letters else '',
        })
    
    def _has_common_pairs(self, word: str) -> bool:
        """Check if word contains common letter pairs."""
        common_pairs = ['TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ED', 'ND', 'OU', 'EA', 'ST', 'EN', 'ON', 'AT', 'ES']
        
        for i in range(len(word) - 1):
            if word[i:i+2] in common_pairs:
                return True
        
        return False
    
    def _has_rare_letters(self, word: str) -> bool:
        """Check if word contains rare letters."""
        rare_letters = ['Q', 'X', 'Z', 'J']
        return any(letter in word for letter in rare_letters)
    
    def save_as_csv(self, data: List[Dict[str, Any]], filename: str = "wordle_historical_data.csv") -> Path:
        """Save data as CSV file."""
        output_file = self.output_dir / "processed" / filename
        
        if data:
            df = pd.DataFrame(data)
            # Sort by puzzle number
            df = df.sort_values('puzzle_number')
            df.to_csv(output_file, index=False)
            logger.info(f"Saved CSV data to {output_file}")
        
        return output_file
    
    def save_as_json(self, data: List[Dict[str, Any]], filename: str = "wordle_historical_data.json") -> Path:
        """Save data as JSON file."""
        output_file = self.output_dir / "processed" / filename
        
        # Sort by puzzle number
        sorted_data = sorted(data, key=lambda x: x['puzzle_number'])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved JSON data to {output_file}")
        return output_file
    
    def save_as_parquet(self, data: List[Dict[str, Any]], filename: str = "wordle_historical_data.parquet") -> Path:
        """Save data as Parquet file (efficient for ML)."""
        output_file = self.output_dir / "processed" / filename
        
        if data:
            df = pd.DataFrame(data)
            # Sort by puzzle number
            df = df.sort_values('puzzle_number')
            df.to_parquet(output_file, index=False)
            logger.info(f"Saved Parquet data to {output_file}")
        
        return output_file
    
    def generate_summary_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        summary = {
            'total_puzzles': len(data),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
            },
            'puzzle_number_range': {
                'min': df['puzzle_number'].min(),
                'max': df['puzzle_number'].max(),
            },
            'word_stats': {
                'total_unique_words': df['solution'].nunique(),
                'avg_word_length': df['word_length'].mean(),
                'avg_unique_letters': df['unique_letters'].mean(),
                'words_with_duplicates': df['has_duplicates'].sum(),
                'avg_vowel_count': df['vowel_count'].mean(),
            },
            'temporal_distribution': {
                'by_year': df['year'].value_counts().to_dict(),
                'by_month': df['month'].value_counts().to_dict(),
                'by_day_of_week': df['day_of_week'].value_counts().to_dict(),
                'weekend_count': df['is_weekend'].sum(),
            },
            'linguistic_patterns': {
                'starts_with_vowel': df['starts_with_vowel'].sum(),
                'ends_with_vowel': df['ends_with_vowel'].sum(),
                'has_common_pairs': df['contains_common_pairs'].sum(),
                'has_rare_letters': df['contains_rare_letters'].sum(),
                'has_double_letters': df['has_double_letters'].sum(),
            }
        }
        
        return summary
    
    def save_summary_report(self, data: List[Dict[str, Any]], filename: str = "data_summary_report.json") -> Path:
        """Save a comprehensive summary report."""
        output_file = self.output_dir / "processed" / filename
        
        summary = self.generate_summary_stats(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved summary report to {output_file}")
        return output_file
    
    def convert_all_formats(self) -> Dict[str, Path]:
        """Convert data to all supported formats and return file paths."""
        # Parse the raw data
        data = self.parse_raw_data()
        
        if not data:
            logger.error("No data was parsed from the input file")
            return {}
        
        # Save in all formats
        output_files = {}
        
        try:
            output_files['csv'] = self.save_as_csv(data)
            output_files['json'] = self.save_as_json(data)
            output_files['parquet'] = self.save_as_parquet(data)
            output_files['summary'] = self.save_summary_report(data)
            
            logger.info(f"Data conversion complete. Processed {len(data)} entries.")
            logger.info("Output files:")
            for format_name, file_path in output_files.items():
                logger.info(f"  {format_name.upper()}: {file_path}")
            
        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            raise
        
        return output_files


def main():
    """Main function to run the data conversion script."""
    parser = argparse.ArgumentParser(description='Convert historical Wordle data to structured formats')
    parser.add_argument('input_file', help='Path to the raw historical data file')
    parser.add_argument('--output-dir', help='Output directory for converted files')
    parser.add_argument('--format', choices=['csv', 'json', 'parquet', 'all'], default='all',
                       help='Output format (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize converter
    converter = WordleDataConverter(args.input_file, args.output_dir)
    
    try:
        if args.format == 'all':
            output_files = converter.convert_all_formats()
            print("\nConversion completed successfully!")
            print("Generated files:")
            for format_name, file_path in output_files.items():
                print(f"  {format_name.upper()}: {file_path}")
        else:
            # Parse data once
            data = converter.parse_raw_data()
            
            # Save in requested format
            if args.format == 'csv':
                output_file = converter.save_as_csv(data)
            elif args.format == 'json':
                output_file = converter.save_as_json(data)
            elif args.format == 'parquet':
                output_file = converter.save_as_parquet(data)
            
            print(f"\nConversion completed: {output_file}")
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())