"""
Feature engineering for Wordle prediction model.
Creates linguistic, temporal, and game-theory features.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import Counter
import itertools
from datetime import datetime, timedelta
import math


class WordleFeatureEngineer:
    def __init__(self, data_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        
        # English letter frequencies (from typical usage)
        self.english_letter_freq = {
            'E': 0.127, 'T': 0.091, 'A': 0.082, 'O': 0.075, 'I': 0.070,
            'N': 0.067, 'S': 0.063, 'H': 0.061, 'R': 0.060, 'D': 0.043,
            'L': 0.040, 'C': 0.028, 'U': 0.028, 'M': 0.024, 'W': 0.023,
            'F': 0.022, 'G': 0.020, 'Y': 0.020, 'P': 0.019, 'B': 0.013,
            'V': 0.010, 'K': 0.008, 'J': 0.001, 'X': 0.001, 'Q': 0.001, 'Z': 0.001
        }
        
        # Position-specific letter frequencies (Wordle-optimized)
        self.position_frequencies = self._initialize_position_frequencies()
        
        # Common letter combinations
        self.common_bigrams = ['TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ED', 'ND', 'ON', 'EN']
        self.common_trigrams = ['THE', 'AND', 'ING', 'HER', 'HAT', 'HIS', 'THA', 'ERE', 'FOR', 'ENT']
    
    def create_linguistic_features(self, words: List[str]) -> pd.DataFrame:
        """Create linguistic features for words."""
        self.logger.info(f"Creating linguistic features for {len(words)} words...")
        
        features_list = []
        
        for word in words:
            features = {'word': word.upper()}
            
            # Basic letter features
            features.update(self._letter_frequency_features(word))
            features.update(self._position_features(word))
            features.update(self._vowel_consonant_features(word))
            features.update(self._letter_combination_features(word))
            features.update(self._phonetic_features(word))
            features.update(self._complexity_features(word))
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        self.logger.info(f"Created {len(df.columns)-1} linguistic features")
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        self.logger.info("Creating temporal features...")
        
        df = df.copy()
        
        # If we have answer_id or date information, create temporal features
        if 'answer_id' in df.columns:
            # Simulate dates based on answer_id (Wordle started June 19, 2021)
            start_date = datetime(2021, 6, 19)
            df['estimated_date'] = df['answer_id'].apply(
                lambda x: start_date + timedelta(days=x-1) if pd.notna(x) else None
            )
            
            # Day of week patterns
            df['day_of_week'] = df['estimated_date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            df['is_monday'] = df['day_of_week'] == 0
            
            # Seasonal trends
            df['month'] = df['estimated_date'].dt.month
            df['season'] = df['month'].apply(self._get_season)
            df['is_holiday_season'] = df['month'].isin([11, 12, 1])  # Nov, Dec, Jan
            
            # Puzzle progression features
            df['early_puzzle'] = df['answer_id'] <= 100  # Early Wordle days
            df['puzzle_age_weeks'] = df['answer_id'] / 7
            df['puzzle_difficulty_trend'] = self._calculate_difficulty_trend(df)
        
        # Meta-game trend features
        df = self._add_metagame_features(df)
        
        self.logger.info("Temporal features created")
        return df
    
    def create_game_theory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create game-theory based features."""
        self.logger.info("Creating game theory features...")
        
        df = df.copy()
        
        # Information entropy features
        df['letter_entropy'] = df['word'].apply(self._calculate_letter_entropy)
        df['position_entropy'] = df['word'].apply(self._calculate_position_entropy)
        df['total_entropy'] = df['letter_entropy'] + df['position_entropy']
        
        # Letter elimination efficiency
        df['elimination_power'] = df['word'].apply(self._calculate_elimination_power)
        df['common_letter_score'] = df['word'].apply(self._calculate_common_letter_score)
        
        # Strategic difficulty scores
        df['guess_difficulty'] = df['word'].apply(self._calculate_guess_difficulty)
        df['solver_efficiency'] = df['word'].apply(self._calculate_solver_efficiency)
        
        # Pattern analysis
        df['pattern_commonality'] = df['word'].apply(self._calculate_pattern_commonality)
        df['letter_distribution_score'] = df['word'].apply(self._calculate_distribution_score)
        
        self.logger.info("Game theory features created")
        return df
    
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all feature types in one pipeline."""
        self.logger.info("Creating comprehensive feature set...")
        
        # Extract words if not already a list
        words = df['word'].tolist() if 'word' in df.columns else []
        
        if not words:
            self.logger.error("No words found in dataframe")
            return df
        
        # Create linguistic features
        linguistic_df = self.create_linguistic_features(words)
        
        # Merge with original data
        result_df = df.merge(linguistic_df, on='word', how='left')
        
        # Add temporal features
        result_df = self.create_temporal_features(result_df)
        
        # Add game theory features
        result_df = self.create_game_theory_features(result_df)
        
        # Save results
        output_path = self.data_dir / "processed" / "features_engineered.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Comprehensive features created. Shape: {result_df.shape}")
        self.logger.info(f"Saved to {output_path}")
        
        return result_df
    
    def _letter_frequency_features(self, word: str) -> Dict:
        """Calculate letter frequency-based features."""
        word = word.upper()
        features = {}
        
        # Overall frequency score
        letter_scores = [self.english_letter_freq.get(c, 0.001) for c in word]
        features['avg_letter_frequency'] = np.mean(letter_scores)
        features['total_letter_frequency'] = np.sum(letter_scores)
        features['min_letter_frequency'] = np.min(letter_scores)
        features['max_letter_frequency'] = np.max(letter_scores)
        features['letter_frequency_std'] = np.std(letter_scores)
        
        # High/low frequency letter counts
        features['high_freq_letters'] = sum(1 for score in letter_scores if score > 0.05)
        features['low_freq_letters'] = sum(1 for score in letter_scores if score < 0.02)
        
        return features
    
    def _position_features(self, word: str) -> Dict:
        """Calculate position-specific features."""
        word = word.upper()
        features = {}
        
        # Position-specific letter scores
        for i, letter in enumerate(word):
            pos_freq = self.position_frequencies[i].get(letter, 0.001)
            features[f'pos_{i+1}_frequency'] = pos_freq
        
        # Position frequency statistics
        pos_scores = [self.position_frequencies[i].get(letter, 0.001) 
                     for i, letter in enumerate(word)]
        features['avg_position_frequency'] = np.mean(pos_scores)
        features['position_frequency_std'] = np.std(pos_scores)
        
        return features
    
    def _vowel_consonant_features(self, word: str) -> Dict:
        """Calculate vowel/consonant ratio and pattern features."""
        word = word.upper()
        features = {}
        
        vowels = 'AEIOU'
        vowel_positions = [i for i, c in enumerate(word) if c in vowels]
        consonant_positions = [i for i, c in enumerate(word) if c not in vowels]
        
        features['vowel_count'] = len(vowel_positions)
        features['consonant_count'] = len(consonant_positions)
        features['vowel_ratio'] = len(vowel_positions) / len(word)
        
        # Vowel/consonant patterns
        features['vowel_clustering'] = self._calculate_clustering(vowel_positions)
        features['consonant_clustering'] = self._calculate_clustering(consonant_positions)
        
        # Specific vowel analysis
        for vowel in vowels:
            features[f'has_{vowel.lower()}'] = vowel in word
            features[f'count_{vowel.lower()}'] = word.count(vowel)
        
        return features
    
    def _letter_combination_features(self, word: str) -> Dict:
        """Calculate bigram, trigram, and pattern features."""
        word = word.upper()
        features = {}
        
        # Bigram analysis
        bigrams = [word[i:i+2] for i in range(len(word)-1)]
        features['common_bigrams'] = sum(1 for bg in bigrams if bg in self.common_bigrams)
        features['unique_bigrams'] = len(set(bigrams))
        
        # Trigram analysis
        trigrams = [word[i:i+3] for i in range(len(word)-2)]
        features['common_trigrams'] = sum(1 for tg in trigrams if tg in self.common_trigrams)
        features['unique_trigrams'] = len(set(trigrams))
        
        # Letter repetition patterns
        features['repeated_letters'] = len(word) - len(set(word))
        features['has_double_letters'] = any(word[i] == word[i+1] for i in range(len(word)-1))
        features['max_letter_repetition'] = max(word.count(c) for c in set(word))
        
        return features
    
    def _phonetic_features(self, word: str) -> Dict:
        """Calculate phonetic and pronunciation features."""
        word = word.upper()
        features = {}
        
        # Simple phonetic patterns (could be enhanced with CMU dict)
        features['starts_with_consonant_cluster'] = word[:2] in ['BL', 'BR', 'CL', 'CR', 'DR', 'FL', 'FR', 'GL', 'GR', 'PL', 'PR', 'SC', 'SK', 'SL', 'SM', 'SN', 'SP', 'ST', 'SW', 'TH', 'TR', 'TW']
        features['ends_with_consonant_cluster'] = word[-2:] in ['ND', 'NT', 'ST', 'LD', 'RD', 'CT', 'PT', 'FT', 'LT', 'MP', 'NK', 'SK', 'SP']
        
        # Rhyme and sound patterns
        features['rhyme_ending'] = word[-2:]
        features['common_ending'] = word[-2:] in ['ED', 'ER', 'LY', 'ES', 'AL', 'IC', 'LE', 'AR', 'OR', 'NG']
        
        return features
    
    def _complexity_features(self, word: str) -> Dict:
        """Calculate word complexity metrics."""
        word = word.upper()
        features = {}
        
        # Lexical diversity
        features['letter_diversity'] = len(set(word)) / len(word)
        
        # Pattern complexity
        letter_counts = Counter(word)
        features['pattern_entropy'] = -sum((count/len(word)) * math.log2(count/len(word)) 
                                         for count in letter_counts.values())
        
        # Spelling complexity
        features['irregular_patterns'] = self._count_irregular_patterns(word)
        features['readability_score'] = self._calculate_readability_score(word)
        
        return features
    
    def _calculate_letter_entropy(self, word: str) -> float:
        """Calculate information entropy based on letter frequencies."""
        word = word.upper()
        total_entropy = 0
        
        for letter in word:
            prob = self.english_letter_freq.get(letter, 0.001)
            total_entropy += -math.log2(prob)
        
        return total_entropy / len(word)
    
    def _calculate_position_entropy(self, word: str) -> float:
        """Calculate information entropy based on position-specific frequencies."""
        word = word.upper()
        total_entropy = 0
        
        for i, letter in enumerate(word):
            prob = self.position_frequencies[i].get(letter, 0.001)
            total_entropy += -math.log2(prob)
        
        return total_entropy / len(word)
    
    def _calculate_elimination_power(self, word: str) -> float:
        """Calculate how effectively this word eliminates possibilities."""
        word = word.upper()
        unique_letters = len(set(word))
        common_letters = sum(1 for c in word if self.english_letter_freq.get(c, 0) > 0.05)
        
        return (unique_letters * 0.7) + (common_letters * 0.3)
    
    def _calculate_common_letter_score(self, word: str) -> float:
        """Calculate score based on common letter usage."""
        word = word.upper()
        return sum(self.english_letter_freq.get(c, 0) for c in word)
    
    def _calculate_guess_difficulty(self, word: str) -> float:
        """Calculate how difficult this word would be to guess."""
        word = word.upper()
        
        # Factors that make a word harder to guess
        difficulty = 0
        
        # Uncommon letters increase difficulty
        for letter in word:
            if self.english_letter_freq.get(letter, 0) < 0.02:
                difficulty += 1
        
        # Repeated letters increase difficulty
        if len(set(word)) < len(word):
            difficulty += 0.5
        
        # Uncommon patterns increase difficulty
        common_patterns = ['ATION', 'OUGH', 'IGHT']
        if not any(pattern in word for pattern in common_patterns):
            difficulty += 0.3
        
        return difficulty
    
    def _calculate_solver_efficiency(self, word: str) -> float:
        """Calculate how efficient this word is as a Wordle guess."""
        word = word.upper()
        
        # Unique letters are good
        unique_score = len(set(word)) / len(word)
        
        # Common letters are good for information gathering
        common_score = sum(1 for c in word if self.english_letter_freq.get(c, 0) > 0.05) / len(word)
        
        # Vowel distribution is good
        vowel_score = min(sum(1 for c in word if c in 'AEIOU'), 2) / 2
        
        return (unique_score * 0.4) + (common_score * 0.4) + (vowel_score * 0.2)
    
    def _calculate_pattern_commonality(self, word: str) -> float:
        """Calculate how common the letter patterns are."""
        word = word.upper()
        
        # Check bigrams
        bigrams = [word[i:i+2] for i in range(len(word)-1)]
        common_bigram_score = sum(1 for bg in bigrams if bg in self.common_bigrams)
        
        return common_bigram_score / len(bigrams) if bigrams else 0
    
    def _calculate_distribution_score(self, word: str) -> float:
        """Calculate how well distributed the letters are."""
        word = word.upper()
        
        # Check for good vowel/consonant distribution
        vowels = 'AEIOU'
        vowel_positions = [i for i, c in enumerate(word) if c in vowels]
        
        if len(vowel_positions) == 0:
            return 0.1  # Very poor distribution
        
        # Ideal is vowels spread throughout the word
        distribution_score = 1.0
        for i in range(len(vowel_positions) - 1):
            gap = vowel_positions[i+1] - vowel_positions[i]
            if gap > 3:  # Too far apart
                distribution_score -= 0.2
            elif gap == 1:  # Too close together
                distribution_score -= 0.1
        
        return max(distribution_score, 0.1)
    
    def _initialize_position_frequencies(self) -> List[Dict]:
        """Initialize position-specific letter frequencies for Wordle."""
        # Simplified position frequencies - could be enhanced with real Wordle data
        positions = []
        
        for pos in range(5):
            if pos == 0:  # First position
                freq_dict = {'S': 0.15, 'C': 0.12, 'B': 0.10, 'T': 0.09, 'P': 0.08, 'A': 0.07, 'F': 0.06}
            elif pos == 4:  # Last position
                freq_dict = {'S': 0.18, 'E': 0.15, 'Y': 0.10, 'D': 0.08, 'T': 0.07, 'A': 0.05, 'R': 0.05}
            else:  # Middle positions
                freq_dict = {'A': 0.12, 'E': 0.11, 'I': 0.10, 'O': 0.09, 'U': 0.07, 'R': 0.08, 'N': 0.07, 'T': 0.06}
            
            # Fill in remaining letters with diminishing frequencies
            remaining_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - set(freq_dict.keys())
            base_freq = 0.04
            for letter in remaining_letters:
                freq_dict[letter] = base_freq
                base_freq *= 0.9
            
            positions.append(freq_dict)
        
        return positions
    
    def _get_season(self, month: int) -> str:
        """Get season from month number."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _calculate_difficulty_trend(self, df: pd.DataFrame) -> pd.Series:
        """Calculate difficulty trend over time."""
        # Simplified trend - could be enhanced with actual difficulty metrics
        return df['answer_id'].apply(lambda x: 1 + (x % 7) * 0.1)  # Weekly difficulty cycle
    
    def _add_metagame_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add meta-game trend features."""
        df = df.copy()
        
        # Simulate meta-game trends
        df['strategic_value'] = df['word'].apply(self._calculate_solver_efficiency)
        df['popularity_trend'] = df.get('frequency', 0) * 1000  # Convert to more readable scale
        
        return df
    
    def _calculate_clustering(self, positions: List[int]) -> float:
        """Calculate clustering score for positions."""
        if len(positions) <= 1:
            return 0
        
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_gap = np.mean(gaps)
        return 1 / (1 + avg_gap)  # Higher score for smaller gaps (more clustering)
    
    def _count_irregular_patterns(self, word: str) -> int:
        """Count irregular spelling patterns."""
        irregular_count = 0
        
        # Some basic irregular patterns
        irregular_patterns = ['GH', 'PH', 'CK', 'TCH', 'DGE']
        for pattern in irregular_patterns:
            if pattern in word:
                irregular_count += 1
        
        return irregular_count
    
    def _calculate_readability_score(self, word: str) -> float:
        """Calculate a simple readability score."""
        # Based on letter frequency and common patterns
        common_letters = sum(1 for c in word if self.english_letter_freq.get(c, 0) > 0.05)
        return common_letters / len(word)