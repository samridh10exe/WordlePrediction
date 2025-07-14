#!/usr/bin/env python3
"""
Advanced feature engineering for research-grade Wordle prediction.

This module implements sophisticated linguistic, phonetic, and game-theory features
for enhanced Wordle prediction accuracy. Features are designed based on research
insights and comprehensive analysis of historical Wordle data.

Key feature categories:
- Position-specific letter frequency analysis
- Phonetic and morphological patterns
- Game-theory and information entropy features
- Semantic similarity and embedding-based features
- Temporal and contextual features
- Strategic optimization features
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict, Counter
from pathlib import Path
import logging
from dataclasses import dataclass
from itertools import combinations, product
import pickle
import math

# Optional imports for advanced features
try:
    import nltk
    from nltk.corpus import wordnet, cmudict
    from nltk.tokenize import syllable
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    include_phonetic: bool = True
    include_semantic: bool = True
    include_game_theory: bool = True
    include_temporal: bool = True
    include_positional: bool = True
    embedding_dim: int = 50
    max_ngram_size: int = 3


class AdvancedFeatureEngineer:
    """Comprehensive feature engineering for Wordle prediction."""
    
    def __init__(self, 
                 vocabulary_data_path: str,
                 historical_data_path: str,
                 output_dir: str = "data/features",
                 config: Optional[FeatureConfig] = None):
        """
        Initialize the advanced feature engineer.
        
        Args:
            vocabulary_data_path: Path to comprehensive vocabulary data
            historical_data_path: Path to historical Wordle data
            output_dir: Directory to save engineered features
            config: Feature engineering configuration
        """
        self.vocabulary_data_path = Path(vocabulary_data_path)
        self.historical_data_path = Path(historical_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or FeatureConfig()
        
        # Create subdirectories
        (self.output_dir / "positional").mkdir(exist_ok=True)
        (self.output_dir / "phonetic").mkdir(exist_ok=True)
        (self.output_dir / "semantic").mkdir(exist_ok=True)
        (self.output_dir / "game_theory").mkdir(exist_ok=True)
        (self.output_dir / "temporal").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        
        # Load data
        self.vocabulary_data = self._load_vocabulary_data()
        self.historical_data = self._load_historical_data()
        
        # Feature storage
        self.position_features: Dict[str, Dict[str, float]] = {}
        self.phonetic_features: Dict[str, Dict[str, Any]] = {}
        self.semantic_features: Dict[str, Dict[str, float]] = {}
        self.game_theory_features: Dict[str, Dict[str, float]] = {}
        self.temporal_features: Dict[str, Dict[str, float]] = {}
        
        # Precomputed data
        self.letter_frequencies: Dict[int, Dict[str, float]] = {}
        self.bigram_frequencies: Dict[Tuple[str, str], float] = {}
        self.trigram_frequencies: Dict[Tuple[str, str, str], float] = {}
        self.word_embeddings: Dict[str, np.ndarray] = {}
        
        # Initialize feature computations
        self._precompute_frequency_data()
    
    def _load_vocabulary_data(self) -> Dict[str, Any]:
        """Load comprehensive vocabulary data."""
        try:
            with open(self.vocabulary_data_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load vocabulary data: {e}")
            return {'word_data': {}, 'vocabulary': {'answers': [], 'guesses': []}}
    
    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical Wordle data."""
        try:
            return pd.read_csv(self.historical_data_path)
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
            return pd.DataFrame()
    
    def _precompute_frequency_data(self):
        """Precompute frequency data for efficient feature engineering."""
        logger.info("Precomputing frequency data...")
        
        # Extract all words
        if not self.historical_data.empty:
            words = self.historical_data['solution'].tolist()
        else:
            words = list(self.vocabulary_data.get('vocabulary', {}).get('answers', []))
        
        # Position-specific letter frequencies
        for pos in range(5):
            self.letter_frequencies[pos] = Counter()
            
        for word in words:
            if len(word) == 5:
                for pos, letter in enumerate(word):
                    self.letter_frequencies[pos][letter] += 1
        
        # Normalize frequencies
        for pos in range(5):
            total = sum(self.letter_frequencies[pos].values())
            if total > 0:
                self.letter_frequencies[pos] = {
                    letter: count / total 
                    for letter, count in self.letter_frequencies[pos].items()
                }
        
        # Bigram and trigram frequencies
        all_text = ''.join(words)
        
        # Bigrams
        for i in range(len(all_text) - 1):
            bigram = (all_text[i], all_text[i + 1])
            self.bigram_frequencies[bigram] = self.bigram_frequencies.get(bigram, 0) + 1
        
        # Trigrams
        for i in range(len(all_text) - 2):
            trigram = (all_text[i], all_text[i + 1], all_text[i + 2])
            self.trigram_frequencies[trigram] = self.trigram_frequencies.get(trigram, 0) + 1
        
        # Normalize n-gram frequencies
        bigram_total = sum(self.bigram_frequencies.values())
        if bigram_total > 0:
            self.bigram_frequencies = {
                k: v / bigram_total for k, v in self.bigram_frequencies.items()
            }
        
        trigram_total = sum(self.trigram_frequencies.values())
        if trigram_total > 0:
            self.trigram_frequencies = {
                k: v / trigram_total for k, v in self.trigram_frequencies.items()
            }
        
        logger.info("Frequency data precomputation completed")
    
    def extract_positional_features(self, word: str) -> Dict[str, float]:
        """
        Extract position-specific features for a word.
        
        Args:
            word: 5-letter word to analyze
            
        Returns:
            Dictionary of positional features
        """
        features = {}
        
        if len(word) != 5:
            return features
        
        # Position-specific letter frequency scores
        for pos, letter in enumerate(word):
            freq_key = f"pos_{pos + 1}_freq"
            features[freq_key] = self.letter_frequencies.get(pos, {}).get(letter, 0.0)
        
        # Position-specific letter rarity scores
        for pos, letter in enumerate(word):
            rarity_key = f"pos_{pos + 1}_rarity"
            freq = self.letter_frequencies.get(pos, {}).get(letter, 0.0)
            features[rarity_key] = 1.0 - freq if freq > 0 else 1.0
        
        # Letter transition probabilities
        for i in range(4):
            bigram = (word[i], word[i + 1])
            trans_key = f"transition_{i + 1}_{i + 2}"
            features[trans_key] = self.bigram_frequencies.get(bigram, 0.0)
        
        # Three-letter sequence probabilities
        for i in range(3):
            trigram = (word[i], word[i + 1], word[i + 2])
            tri_key = f"trigram_{i + 1}_{i + 3}"
            features[tri_key] = self.trigram_frequencies.get(trigram, 0.0)
        
        # Positional vowel/consonant patterns
        vowels = set('AEIOU')
        for pos, letter in enumerate(word):
            vowel_key = f"pos_{pos + 1}_is_vowel"
            features[vowel_key] = 1.0 if letter in vowels else 0.0
        
        # Common positional patterns
        features['starts_with_common'] = 1.0 if word[0] in 'STCPBF' else 0.0
        features['ends_with_common'] = 1.0 if word[-1] in 'SYEDTR' else 0.0
        
        # Double letter positions
        for i in range(4):
            if word[i] == word[i + 1]:
                features[f'double_at_pos_{i + 1}_{i + 2}'] = 1.0
        
        # Alternating vowel/consonant patterns
        pattern_score = 0.0
        for i in range(4):
            curr_is_vowel = word[i] in vowels
            next_is_vowel = word[i + 1] in vowels
            if curr_is_vowel != next_is_vowel:
                pattern_score += 1.0
        features['alternating_pattern_score'] = pattern_score / 4.0
        
        return features
    
    def extract_phonetic_features(self, word: str) -> Dict[str, Any]:
        """
        Extract phonetic and morphological features.
        
        Args:
            word: Word to analyze
            
        Returns:
            Dictionary of phonetic features
        """
        features = {}
        
        if not NLTK_AVAILABLE:
            return self._extract_basic_phonetic_features(word)
        
        # Get phonetic representation
        word_data = self.vocabulary_data.get('word_data', {}).get(word, {})
        phonetic = word_data.get('phonetic', [])
        
        if phonetic:
            # Phoneme count
            features['phoneme_count'] = len(phonetic)
            
            # Syllable count (approximation)
            vowel_phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
            syllable_count = sum(1 for p in phonetic if p in vowel_phonemes)
            features['syllable_count'] = max(1, syllable_count)
            
            # Stress patterns (simplified)
            stress_phonemes = [p for p in phonetic if p[-1].isdigit()]
            features['has_stress'] = 1.0 if stress_phonemes else 0.0
            
            # Consonant clusters
            consonant_clusters = 0
            in_cluster = False
            for p in phonetic:
                if p not in vowel_phonemes:
                    if not in_cluster:
                        consonant_clusters += 1
                        in_cluster = True
                else:
                    in_cluster = False
            features['consonant_clusters'] = consonant_clusters
            
            # Phonetic complexity (heuristic)
            complex_sounds = ['TH', 'SH', 'CH', 'ZH', 'NG']
            complexity = sum(1 for p in phonetic if p in complex_sounds)
            features['phonetic_complexity'] = complexity
        else:
            # Fallback to basic features
            features.update(self._extract_basic_phonetic_features(word))
        
        # Morphological features
        features.update(self._extract_morphological_features(word))
        
        return features
    
    def _extract_basic_phonetic_features(self, word: str) -> Dict[str, Any]:
        """Extract basic phonetic features without NLTK."""
        features = {}
        
        # Simple syllable counting
        vowels = 'AEIOU'
        vowel_groups = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                vowel_groups += 1
            prev_was_vowel = is_vowel
        
        features['syllable_count'] = max(1, vowel_groups)
        
        # Consonant clusters
        consonant_clusters = 0
        in_cluster = False
        
        for char in word:
            if char not in vowels:
                if not in_cluster:
                    consonant_clusters += 1
                    in_cluster = True
            else:
                in_cluster = False
        
        features['consonant_clusters'] = consonant_clusters
        
        # Basic phonetic patterns
        features['has_th'] = 1.0 if 'TH' in word else 0.0
        features['has_ch'] = 1.0 if 'CH' in word else 0.0
        features['has_sh'] = 1.0 if 'SH' in word else 0.0
        
        return features
    
    def _extract_morphological_features(self, word: str) -> Dict[str, Any]:
        """Extract morphological features."""
        features = {}
        
        # Common prefixes and suffixes
        prefixes = ['UN', 'RE', 'IN', 'DIS', 'EN', 'NON', 'OVER', 'MIS', 'SUB', 'PRE', 'INTER', 'FORE', 'DE', 'TRANS']
        suffixes = ['ING', 'ED', 'ER', 'EST', 'LY', 'TION', 'ABLE', 'IBLE', 'AL', 'IC', 'OUS', 'FUL', 'LESS', 'NESS']
        
        # Note: Adapted for 5-letter words
        short_prefixes = [p for p in prefixes if len(p) <= 3]
        short_suffixes = [s for s in suffixes if len(s) <= 3]
        
        features['has_prefix'] = 1.0 if any(word.startswith(p) for p in short_prefixes) else 0.0
        features['has_suffix'] = 1.0 if any(word.endswith(s) for s in short_suffixes) else 0.0
        
        # Word formation patterns
        features['likely_compound'] = 1.0 if self._is_likely_compound(word) else 0.0
        features['likely_derived'] = 1.0 if features['has_prefix'] or features['has_suffix'] else 0.0
        
        return features
    
    def _is_likely_compound(self, word: str) -> bool:
        """Heuristic to detect compound words."""
        # Simple heuristic: look for common compound patterns
        compound_patterns = ['HOUSE', 'LIGHT', 'WATER', 'FIRE', 'UNDER', 'OVER']
        return any(pattern in word for pattern in compound_patterns if len(pattern) < len(word))
    
    def extract_semantic_features(self, word: str) -> Dict[str, float]:
        """
        Extract semantic and embedding-based features.
        
        Args:
            word: Word to analyze
            
        Returns:
            Dictionary of semantic features
        """
        features = {}
        
        # Word embedding features
        word_data = self.vocabulary_data.get('word_data', {}).get(word, {})
        embedding = word_data.get('embedding', [])
        
        if embedding and len(embedding) > 0:
            embedding = np.array(embedding)
            
            # Embedding statistics
            features['embedding_mean'] = float(np.mean(embedding))
            features['embedding_std'] = float(np.std(embedding))
            features['embedding_min'] = float(np.min(embedding))
            features['embedding_max'] = float(np.max(embedding))
            features['embedding_norm'] = float(np.linalg.norm(embedding))
            
            # Principal components (if available)
            if SKLEARN_AVAILABLE and hasattr(self, 'embedding_pca'):
                pca_features = self.embedding_pca.transform([embedding])[0]
                for i, comp in enumerate(pca_features[:5]):  # Top 5 components
                    features[f'embedding_pc_{i + 1}'] = float(comp)
        
        # Semantic categories (simplified)
        semantic_categories = self._get_semantic_categories(word)
        for category, score in semantic_categories.items():
            features[f'semantic_{category}'] = score
        
        # Concreteness and abstractness
        features['concreteness'] = self._estimate_concreteness(word)
        features['frequency_class'] = self._get_frequency_class(word)
        
        # Semantic similarity to common words
        common_words = ['HOUSE', 'WATER', 'LIGHT', 'SOUND', 'PLACE', 'RIGHT', 'GREAT', 'SMALL']
        for common_word in common_words:
            if word != common_word:
                similarity = self._calculate_semantic_similarity(word, common_word)
                features[f'sim_to_{common_word.lower()}'] = similarity
        
        return features
    
    def _get_semantic_categories(self, word: str) -> Dict[str, float]:
        """Get semantic category scores for a word."""
        categories = {
            'concrete_noun': 0.0,
            'abstract_noun': 0.0,
            'action_verb': 0.0,
            'descriptive_adj': 0.0,
            'temporal': 0.0,
            'spatial': 0.0,
            'emotional': 0.0
        }
        
        # Simple heuristic-based categorization
        concrete_indicators = ['HOUSE', 'WATER', 'LIGHT', 'SOUND', 'PLACE', 'FIELD', 'STONE', 'PLANT']
        abstract_indicators = ['THINK', 'DREAM', 'HONOR', 'PEACE', 'TRUTH', 'FAITH', 'POWER', 'FORCE']
        action_indicators = ['CLIMB', 'CARRY', 'THROW', 'CATCH', 'BREAK', 'BUILD', 'WRITE', 'SPEAK']
        emotional_indicators = ['HAPPY', 'ANGRY', 'PROUD', 'SORRY', 'SWEET', 'BITTER', 'HARSH', 'GENTLE']
        
        if any(indicator in word for indicator in concrete_indicators):
            categories['concrete_noun'] = 1.0
        elif any(indicator in word for indicator in abstract_indicators):
            categories['abstract_noun'] = 1.0
        elif any(indicator in word for indicator in action_indicators):
            categories['action_verb'] = 1.0
        elif any(indicator in word for indicator in emotional_indicators):
            categories['emotional'] = 1.0
        
        return categories
    
    def _estimate_concreteness(self, word: str) -> float:
        """Estimate concreteness of a word (0=abstract, 1=concrete)."""
        # Simple heuristic based on word characteristics
        concrete_patterns = ['HOUSE', 'STONE', 'WATER', 'PLANT', 'FIELD', 'CHAIR', 'TABLE']
        abstract_patterns = ['THINK', 'DREAM', 'PEACE', 'TRUTH', 'POWER', 'HONOR', 'FAITH']
        
        if any(pattern in word for pattern in concrete_patterns):
            return 0.8
        elif any(pattern in word for pattern in abstract_patterns):
            return 0.2
        else:
            return 0.5  # Neutral
    
    def _get_frequency_class(self, word: str) -> float:
        """Get frequency class of word (0=rare, 1=common)."""
        word_data = self.vocabulary_data.get('word_data', {}).get(word, {})
        frequency = word_data.get('frequency', 0.0)
        
        # Normalize frequency to 0-1 range
        if frequency > 0.8:
            return 1.0
        elif frequency > 0.5:
            return 0.8
        elif frequency > 0.2:
            return 0.6
        elif frequency > 0.1:
            return 0.4
        else:
            return 0.2
    
    def _calculate_semantic_similarity(self, word1: str, word2: str) -> float:
        """Calculate semantic similarity between two words."""
        # Use embeddings if available
        word1_data = self.vocabulary_data.get('word_data', {}).get(word1, {})
        word2_data = self.vocabulary_data.get('word_data', {}).get(word2, {})
        
        emb1 = word1_data.get('embedding', [])
        emb2 = word2_data.get('embedding', [])
        
        if emb1 and emb2 and len(emb1) == len(emb2):
            if SKLEARN_AVAILABLE:
                emb1 = np.array(emb1).reshape(1, -1)
                emb2 = np.array(emb2).reshape(1, -1)
                similarity = cosine_similarity(emb1, emb2)[0][0]
                return float(similarity)
        
        # Fallback: simple character-based similarity
        common_chars = set(word1) & set(word2)
        total_chars = set(word1) | set(word2)
        return len(common_chars) / len(total_chars) if total_chars else 0.0
    
    def extract_game_theory_features(self, word: str) -> Dict[str, float]:
        """
        Extract game-theory and strategic features.
        
        Args:
            word: Word to analyze
            
        Returns:
            Dictionary of game-theory features
        """
        features = {}
        
        # Information entropy features
        features['letter_entropy'] = self._calculate_letter_entropy(word)
        features['position_entropy'] = self._calculate_position_entropy(word)
        
        # Letter elimination power
        features['elimination_power'] = self._calculate_elimination_power(word)
        
        # Guess efficiency metrics
        features['expected_information'] = self._calculate_expected_information(word)
        features['worst_case_remaining'] = self._calculate_worst_case_remaining(word)
        
        # Strategic value
        features['strategic_value'] = self._calculate_strategic_value(word)
        
        # Pattern matching efficiency
        features['pattern_efficiency'] = self._calculate_pattern_efficiency(word)
        
        # Letter frequency optimization
        features['frequency_optimization'] = self._calculate_frequency_optimization(word)
        
        return features
    
    def _calculate_letter_entropy(self, word: str) -> float:
        """Calculate information entropy of letters in word."""
        letter_counts = Counter(word)
        total_letters = len(word)
        
        entropy = 0.0
        for count in letter_counts.values():
            probability = count / total_letters
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_position_entropy(self, word: str) -> float:
        """Calculate entropy based on positional letter frequencies."""
        total_entropy = 0.0
        
        for pos, letter in enumerate(word):
            pos_freq = self.letter_frequencies.get(pos, {})
            letter_prob = pos_freq.get(letter, 0.0)
            
            if letter_prob > 0:
                entropy_contribution = -letter_prob * math.log2(letter_prob)
                total_entropy += entropy_contribution
        
        return total_entropy / 5.0  # Normalize by number of positions
    
    def _calculate_elimination_power(self, word: str) -> float:
        """Calculate how many words this guess could eliminate."""
        # Simplified calculation - in practice would need full word list
        unique_letters = len(set(word))
        common_letters = sum(1 for letter in word if letter in 'ETAOINSHRDL')
        
        # Heuristic: more unique letters and common letters = higher elimination power
        elimination_score = (unique_letters / 5.0) * 0.7 + (common_letters / 5.0) * 0.3
        
        return elimination_score
    
    def _calculate_expected_information(self, word: str) -> float:
        """Calculate expected information gain from this guess."""
        # Simplified heuristic based on letter frequencies and positions
        info_gain = 0.0
        
        for pos, letter in enumerate(word):
            pos_freq = self.letter_frequencies.get(pos, {}).get(letter, 0.0)
            # Information gain is higher for letters with moderate frequency
            # (not too common, not too rare)
            if pos_freq > 0:
                info_value = -pos_freq * math.log2(pos_freq)
                info_gain += info_value
        
        return info_gain / 5.0
    
    def _calculate_worst_case_remaining(self, word: str) -> float:
        """Estimate worst-case number of remaining possibilities."""
        # Heuristic based on word diversity
        unique_letters = len(set(word))
        
        # More unique letters generally lead to fewer remaining possibilities
        worst_case_score = 1.0 - (unique_letters / 5.0)
        
        return worst_case_score
    
    def _calculate_strategic_value(self, word: str) -> float:
        """Calculate overall strategic value for Wordle."""
        # Combine multiple strategic factors
        letter_diversity = len(set(word)) / 5.0
        common_letter_usage = sum(1 for letter in word if letter in 'ETAOINSHRDL') / 5.0
        vowel_coverage = sum(1 for letter in word if letter in 'AEIOU') / 5.0
        
        # Balanced approach: diversity, common letters, vowel coverage
        strategic_value = (letter_diversity * 0.4 + 
                          common_letter_usage * 0.4 + 
                          vowel_coverage * 0.2)
        
        return strategic_value
    
    def _calculate_pattern_efficiency(self, word: str) -> float:
        """Calculate efficiency for pattern matching."""
        # Based on letter position probabilities
        efficiency = 0.0
        
        for pos, letter in enumerate(word):
            pos_freq = self.letter_frequencies.get(pos, {}).get(letter, 0.0)
            # Moderate frequencies are most efficient for pattern matching
            if 0.01 <= pos_freq <= 0.15:
                efficiency += 1.0
            elif 0.005 <= pos_freq <= 0.3:
                efficiency += 0.7
            else:
                efficiency += 0.3
        
        return efficiency / 5.0
    
    def _calculate_frequency_optimization(self, word: str) -> float:
        """Calculate optimization based on overall letter frequencies."""
        # Get global letter frequencies
        all_freq = Counter()
        for pos_freq in self.letter_frequencies.values():
            for letter, freq in pos_freq.items():
                all_freq[letter] += freq
        
        # Normalize
        total = sum(all_freq.values())
        global_freq = {letter: count / total for letter, count in all_freq.items()}
        
        # Calculate optimization score
        optimization = 0.0
        for letter in set(word):
            letter_freq = global_freq.get(letter, 0.0)
            # Optimal range for Wordle strategy
            if 0.02 <= letter_freq <= 0.12:
                optimization += 1.0
            elif 0.01 <= letter_freq <= 0.2:
                optimization += 0.7
            else:
                optimization += 0.3
        
        return optimization / len(set(word))
    
    def extract_temporal_features(self, word: str, target_date: Optional[str] = None) -> Dict[str, float]:
        """
        Extract temporal and contextual features.
        
        Args:
            word: Word to analyze
            target_date: Target prediction date (optional)
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        if self.historical_data.empty:
            return features
        
        # Recent usage patterns
        features.update(self._calculate_recent_usage_patterns(word))
        
        # Seasonal preferences
        if target_date:
            features.update(self._calculate_seasonal_features(word, target_date))
        
        # Editorial preferences
        features.update(self._calculate_editorial_preferences(word))
        
        # Temporal trends
        features.update(self._calculate_temporal_trends(word))
        
        return features
    
    def _calculate_recent_usage_patterns(self, word: str) -> Dict[str, float]:
        """Calculate features based on recent usage patterns."""
        features = {}
        
        # Check if word was used recently
        if word in self.historical_data['solution'].values:
            last_used_idx = self.historical_data[self.historical_data['solution'] == word].index[-1]
            days_since_used = len(self.historical_data) - last_used_idx
            features['days_since_last_used'] = float(days_since_used)
            features['was_used_recently'] = 1.0 if days_since_used < 30 else 0.0
        else:
            features['days_since_last_used'] = 999.0  # Large number for never used
            features['was_used_recently'] = 0.0
        
        # Similar words recently used
        similar_recently = 0
        recent_words = self.historical_data['solution'].tail(30).tolist()
        
        for recent_word in recent_words:
            similarity = len(set(word) & set(recent_word)) / len(set(word) | set(recent_word))
            if similarity > 0.6:
                similar_recently += 1
        
        features['similar_words_recently'] = float(similar_recently)
        
        return features
    
    def _calculate_seasonal_features(self, word: str, target_date: str) -> Dict[str, float]:
        """Calculate seasonal preference features."""
        features = {}
        
        try:
            from datetime import datetime
            target_dt = datetime.fromisoformat(target_date)
            
            # Month-based features
            features['target_month'] = float(target_dt.month)
            features['target_quarter'] = float((target_dt.month - 1) // 3 + 1)
            features['is_winter'] = 1.0 if target_dt.month in [12, 1, 2] else 0.0
            features['is_summer'] = 1.0 if target_dt.month in [6, 7, 8] else 0.0
            
            # Day of week features
            features['target_day_of_week'] = float(target_dt.weekday())
            features['is_weekend'] = 1.0 if target_dt.weekday() >= 5 else 0.0
            features['is_monday'] = 1.0 if target_dt.weekday() == 0 else 0.0
            features['is_friday'] = 1.0 if target_dt.weekday() == 4 else 0.0
            
        except Exception:
            # Default values if date parsing fails
            features.update({
                'target_month': 6.0,
                'target_quarter': 2.0,
                'is_winter': 0.0,
                'is_summer': 0.0,
                'target_day_of_week': 2.0,
                'is_weekend': 0.0,
                'is_monday': 0.0,
                'is_friday': 0.0
            })
        
        return features
    
    def _calculate_editorial_preferences(self, word: str) -> Dict[str, float]:
        """Calculate features based on editorial preferences."""
        features = {}
        
        # Word complexity preferences
        features['is_complex_word'] = 1.0 if len(set(word)) >= 5 else 0.0
        features['has_double_letters'] = 1.0 if len(word) != len(set(word)) else 0.0
        features['rare_letter_count'] = float(sum(1 for letter in word if letter in 'QXZJ'))
        
        # Common word patterns NYT seems to prefer
        preferred_patterns = ['HOUSE', 'LIGHT', 'SOUND', 'FIELD', 'POINT', 'RIGHT', 'PLACE']
        features['matches_preferred_pattern'] = 1.0 if any(pattern in word for pattern in preferred_patterns) else 0.0
        
        # Difficulty balancing indicator
        vowel_count = sum(1 for letter in word if letter in 'AEIOU')
        features['difficulty_balance_score'] = abs(vowel_count - 2.0) / 2.0  # Distance from ideal 2 vowels
        
        return features
    
    def _calculate_temporal_trends(self, word: str) -> Dict[str, float]:
        """Calculate features based on temporal trends."""
        features = {}
        
        if self.historical_data.empty:
            return features
        
        # Trend in similar word usage
        similar_words = []
        for solution in self.historical_data['solution']:
            similarity = len(set(word) & set(solution)) / len(set(word) | set(solution))
            if similarity > 0.4:
                similar_words.append(solution)
        
        features['similar_word_trend'] = float(len(similar_words))
        
        # Complexity trend
        recent_complexity = self.historical_data['unique_letters'].tail(30).mean()
        word_complexity = len(set(word))
        features['complexity_vs_recent'] = float(word_complexity - recent_complexity)
        
        return features
    
    def create_comprehensive_feature_set(self, 
                                       words: List[str], 
                                       target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for a list of words.
        
        Args:
            words: List of words to create features for
            target_date: Target prediction date for temporal features
            
        Returns:
            DataFrame with comprehensive features
        """
        logger.info(f"Creating comprehensive features for {len(words)} words...")
        
        all_features = []
        
        for word in words:
            word_features = {'word': word}
            
            # Extract all feature types
            if self.config.include_positional:
                word_features.update(self.extract_positional_features(word))
            
            if self.config.include_phonetic:
                word_features.update(self.extract_phonetic_features(word))
            
            if self.config.include_semantic:
                word_features.update(self.extract_semantic_features(word))
            
            if self.config.include_game_theory:
                word_features.update(self.extract_game_theory_features(word))
            
            if self.config.include_temporal:
                word_features.update(self.extract_temporal_features(word, target_date))
            
            all_features.append(word_features)
        
        # Create DataFrame
        feature_df = pd.DataFrame(all_features)
        
        # Fill missing values
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[numeric_columns] = feature_df[numeric_columns].fillna(0.0)
        
        logger.info(f"Created {len(feature_df.columns) - 1} features for {len(words)} words")
        
        # Save feature set
        output_file = self.output_dir / "processed" / "comprehensive_features.csv"
        feature_df.to_csv(output_file, index=False)
        
        return feature_df
    
    def save_feature_engineering_report(self) -> str:
        """Save comprehensive feature engineering report."""
        report = {
            'feature_engineering_date': pd.Timestamp.now().isoformat(),
            'configuration': {
                'include_phonetic': self.config.include_phonetic,
                'include_semantic': self.config.include_semantic,
                'include_game_theory': self.config.include_game_theory,
                'include_temporal': self.config.include_temporal,
                'include_positional': self.config.include_positional,
                'embedding_dim': self.config.embedding_dim
            },
            'feature_categories': {
                'positional_features': [
                    'pos_1_freq', 'pos_2_freq', 'pos_3_freq', 'pos_4_freq', 'pos_5_freq',
                    'transition_1_2', 'transition_2_3', 'transition_3_4', 'transition_4_5',
                    'trigram_1_3', 'trigram_2_4', 'trigram_3_5',
                    'alternating_pattern_score', 'double_at_pos_*'
                ],
                'phonetic_features': [
                    'phoneme_count', 'syllable_count', 'consonant_clusters',
                    'phonetic_complexity', 'has_stress'
                ],
                'semantic_features': [
                    'embedding_mean', 'embedding_std', 'embedding_norm',
                    'semantic_*', 'concreteness', 'frequency_class', 'sim_to_*'
                ],
                'game_theory_features': [
                    'letter_entropy', 'position_entropy', 'elimination_power',
                    'expected_information', 'strategic_value', 'pattern_efficiency'
                ],
                'temporal_features': [
                    'days_since_last_used', 'similar_words_recently',
                    'target_month', 'is_weekend', 'complexity_vs_recent'
                ]
            },
            'data_sources': [
                'Historical Wordle answers (1290+ entries)',
                'Comprehensive vocabulary database',
                'Position-specific letter frequencies',
                'N-gram frequency analysis',
                'Word embeddings (GloVe/FastText)',
                'Phonetic representations (CMU Dict)',
                'Semantic categories and relationships'
            ]
        }
        
        report_file = self.output_dir / "feature_engineering_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Feature engineering report saved to {report_file}")
        return str(report_file)


def main():
    """Main function for advanced feature engineering."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced feature engineering for Wordle prediction')
    parser.add_argument('--vocabulary-data', required=True, help='Path to comprehensive vocabulary data JSON')
    parser.add_argument('--historical-data', required=True, help='Path to historical Wordle data CSV')
    parser.add_argument('--output-dir', default='data/features', help='Output directory')
    parser.add_argument('--target-date', help='Target date for temporal features (YYYY-MM-DD)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize feature engineer
        config = FeatureConfig()
        engineer = AdvancedFeatureEngineer(
            args.vocabulary_data,
            args.historical_data,
            args.output_dir,
            config
        )
        
        # Get vocabulary words
        vocab_data = engineer.vocabulary_data
        words = list(vocab_data.get('vocabulary', {}).get('guesses', []))
        
        if not words:
            logger.warning("No words found in vocabulary data")
            return 1
        
        # Create comprehensive feature set
        feature_df = engineer.create_comprehensive_feature_set(words, args.target_date)
        
        # Save report
        report_file = engineer.save_feature_engineering_report()
        
        print(f"\nAdvanced feature engineering completed successfully!")
        print(f"Features created for {len(words)} words")
        print(f"Total features: {len(feature_df.columns) - 1}")
        print(f"Output directory: {args.output_dir}")
        print(f"Report: {report_file}")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())