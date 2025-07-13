"""
Data collection module for Wordle historical data and linguistic features.
Implements multiple data sources with robust error handling.
"""

import pandas as pd
import requests
from pathlib import Path
import logging
from typing import List, Dict, Optional
import time
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
import numpy as np


class WordleDataCollector:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('corpora/cmudict')
        except LookupError:
            nltk.download('cmudict')
    
    def collect_wordle_answers(self) -> pd.DataFrame:
        """Collect historical Wordle answers from GitHub sources."""
        self.logger.info("Collecting Wordle historical answers...")
        
        # GitHub source with historical answers
        github_urls = [
            "https://raw.githubusercontent.com/steve-kasica/wordle-words/main/wordle_words.csv",
            "https://gist.githubusercontent.com/DevilXD/6ad6cc1fe37872d069a795edd51233b2/raw/wordle_words.txt"
        ]
        
        answers_data = []
        
        for url in github_urls:
            try:
                self.logger.info(f"Fetching data from {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                if url.endswith('.csv'):
                    # Handle CSV format
                    with open(self.raw_dir / "wordle_answers_csv.csv", 'w') as f:
                        f.write(response.text)
                    df = pd.read_csv(self.raw_dir / "wordle_answers_csv.csv")
                    answers_data.append(df)
                
                elif url.endswith('.txt'):
                    # Handle text format
                    words = response.text.strip().split('\n')
                    words = [word.strip().upper() for word in words if len(word.strip()) == 5]
                    df = pd.DataFrame({'word': words})
                    answers_data.append(df)
                
                time.sleep(1)  # Be respectful to servers
                
            except Exception as e:
                self.logger.error(f"Error fetching from {url}: {e}")
                continue
        
        # Combine and deduplicate
        if answers_data:
            combined_df = pd.concat(answers_data, ignore_index=True)
            # Ensure word column exists and normalize
            if 'word' not in combined_df.columns:
                combined_df['word'] = combined_df.iloc[:, 0]
            
            combined_df['word'] = combined_df['word'].str.upper().str.strip()
            combined_df = combined_df[combined_df['word'].str.len() == 5]
            combined_df = combined_df[combined_df['word'].str.isalpha()]
            combined_df = combined_df.drop_duplicates(subset=['word'])
            
            # Add sequence numbers for historical tracking
            combined_df = combined_df.reset_index(drop=True)
            combined_df['answer_id'] = range(1, len(combined_df) + 1)
            
            # Save to processed directory
            output_path = self.processed_dir / "wordle_answers.csv"
            combined_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {len(combined_df)} Wordle answers to {output_path}")
            
            return combined_df
        else:
            self.logger.error("No Wordle answer data could be collected")
            return pd.DataFrame()
    
    def collect_word_frequencies(self) -> pd.DataFrame:
        """Collect word frequency data from multiple sources."""
        self.logger.info("Collecting word frequency data...")
        
        frequency_data = []
        
        # Use wordfreq library for general frequency data
        try:
            from wordfreq import word_frequency
            
            # Get a comprehensive word list (combine with Wordle answers)
            wordle_answers = self.collect_wordle_answers() if not hasattr(self, '_wordle_answers') else self._wordle_answers
            words = wordle_answers['word'].tolist() if not wordle_answers.empty else []
            
            # Add common 5-letter words
            common_5_letter = self._get_common_5_letter_words()
            words.extend(common_5_letter)
            words = list(set(words))  # Remove duplicates
            
            freq_data = []
            for word in tqdm(words, desc="Calculating word frequencies"):
                freq = word_frequency(word.lower(), 'en')
                freq_data.append({
                    'word': word.upper(),
                    'frequency': freq,
                    'log_frequency': np.log10(freq) if freq > 0 else -10
                })
            
            frequency_df = pd.DataFrame(freq_data)
            frequency_df = frequency_df.sort_values('frequency', ascending=False)
            
            # Save to processed directory
            output_path = self.processed_dir / "word_frequencies.csv"
            frequency_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved frequency data for {len(frequency_df)} words to {output_path}")
            
            return frequency_df
            
        except ImportError:
            self.logger.error("wordfreq library not available, using alternative method")
            return self._collect_frequency_fallback()
    
    def collect_linguistic_features(self) -> pd.DataFrame:
        """Collect linguistic features using NLTK and WordNet."""
        self.logger.info("Collecting linguistic features...")
        
        # Get word list
        word_freq_df = self.collect_word_frequencies()
        words = word_freq_df['word'].tolist() if not word_freq_df.empty else self._get_common_5_letter_words()
        
        linguistic_data = []
        
        for word in tqdm(words, desc="Extracting linguistic features"):
            features = self._extract_word_features(word)
            linguistic_data.append(features)
        
        linguistic_df = pd.DataFrame(linguistic_data)
        
        # Save to processed directory
        output_path = self.processed_dir / "linguistic_features.csv"
        linguistic_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved linguistic features for {len(linguistic_df)} words to {output_path}")
        
        return linguistic_df
    
    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect all data sources and return as dictionary."""
        self.logger.info("Starting comprehensive data collection...")
        
        data = {}
        
        # Collect each data type
        data['answers'] = self.collect_wordle_answers()
        data['frequencies'] = self.collect_word_frequencies()
        data['linguistic'] = self.collect_linguistic_features()
        
        # Create combined dataset
        if all(not df.empty for df in data.values()):
            combined = data['answers'].merge(
                data['frequencies'], on='word', how='left'
            ).merge(
                data['linguistic'], on='word', how='left'
            )
            
            output_path = self.processed_dir / "combined_dataset.csv"
            combined.to_csv(output_path, index=False)
            data['combined'] = combined
            self.logger.info(f"Saved combined dataset with {len(combined)} words to {output_path}")
        
        return data
    
    def _get_common_5_letter_words(self) -> List[str]:
        """Get a list of common 5-letter English words."""
        # Basic 5-letter word list for fallback
        common_words = [
            'ABOUT', 'ABOVE', 'ABUSE', 'ACTOR', 'ACUTE', 'ADMIT', 'ADOPT', 'ADULT', 'AFTER',
            'AGAIN', 'AGENT', 'AGREE', 'AHEAD', 'ALARM', 'ALBUM', 'ALERT', 'ALIEN', 'ALIGN',
            'ALIKE', 'ALIVE', 'ALLOW', 'ALONE', 'ALONG', 'ALTER', 'ANGER', 'ANGLE', 'ANGRY',
            'APART', 'APPLE', 'APPLY', 'ARENA', 'ARGUE', 'ARISE', 'ARRAY', 'ASIDE', 'ASSET',
            'AUDIO', 'AVOID', 'AWAKE', 'AWARD', 'AWARE', 'BADLY', 'BASIC', 'BEACH', 'BEGAN',
            'BEGIN', 'BEING', 'BELOW', 'BENCH', 'BILLY', 'BIRTH', 'BLACK', 'BLAME', 'BLANK',
            'BLIND', 'BLOCK', 'BLOOD', 'BOARD', 'BOOST', 'BOOTH', 'BOUND', 'BRAIN', 'BRAND',
            'BRASS', 'BRAVE', 'BREAD', 'BREAK', 'BREED', 'BRIEF', 'BRING', 'BROAD', 'BROKE',
            'BROWN', 'BUILD', 'BUILT', 'BURST', 'BUYER', 'CABLE', 'CALIF', 'CARRY', 'CATCH',
            'CAUSE', 'CHAIN', 'CHAIR', 'CHAOS', 'CHARM', 'CHART', 'CHASE', 'CHEAP', 'CHECK',
            'CHEST', 'CHIEF', 'CHILD', 'CHINA', 'CHOSE', 'CIVIL', 'CLAIM', 'CLASS', 'CLEAN',
            'CLEAR', 'CLICK', 'CLIMB', 'CLOCK', 'CLOSE', 'CLOUD', 'COACH', 'COAST', 'COULD',
            'COUNT', 'COURT', 'COVER', 'CRAFT', 'CRASH', 'CRAZY', 'CREAM', 'CRIME', 'CROSS',
            'CROWD', 'CROWN', 'CRUDE', 'CURVE', 'CYCLE', 'DAILY', 'DANCE', 'DATED', 'DEALT',
            'DEATH', 'DEBUT', 'DELAY', 'DEPTH', 'DOING', 'DOUBT', 'DOZEN', 'DRAFT', 'DRAMA',
            'DRANK', 'DREAM', 'DRESS', 'DRILL', 'DRINK', 'DRIVE', 'DROVE', 'DYING', 'EAGER',
            'EARLY', 'EARTH', 'EIGHT', 'ELITE', 'EMPTY', 'ENEMY', 'ENJOY', 'ENTER', 'ENTRY',
            'EQUAL', 'ERROR', 'EVENT', 'EVERY', 'EXACT', 'EXIST', 'EXTRA', 'FAITH', 'FALSE',
            'FAULT', 'FIBER', 'FIELD', 'FIFTH', 'FIFTY', 'FIGHT', 'FINAL', 'FIRST', 'FIXED',
            'FLASH', 'FLEET', 'FLOOR', 'FLUID', 'FOCUS', 'FORCE', 'FORTH', 'FORTY', 'FORUM',
            'FOUND', 'FRAME', 'FRANK', 'FRAUD', 'FRESH', 'FRONT', 'FRUIT', 'FULLY', 'FUNNY',
            'GIANT', 'GIVEN', 'GLASS', 'GLOBE', 'GOING', 'GRACE', 'GRADE', 'GRAND', 'GRANT',
            'GRASS', 'GRAVE', 'GREAT', 'GREEN', 'GROSS', 'GROUP', 'GROWN', 'GUARD', 'GUESS',
            'GUEST', 'GUIDE', 'HAPPY', 'HARSH', 'HATE', 'HEAD', 'HEART', 'HEAVY', 'HORSE'
        ]
        return common_words
    
    def _extract_word_features(self, word: str) -> Dict:
        """Extract linguistic features for a single word."""
        word_lower = word.lower()
        
        features = {
            'word': word.upper(),
            'length': len(word),
            'vowel_count': sum(1 for c in word_lower if c in 'aeiou'),
            'consonant_count': sum(1 for c in word_lower if c.isalpha() and c not in 'aeiou'),
            'unique_letters': len(set(word_lower)),
            'repeated_letters': len(word) - len(set(word_lower)),
            'double_letters': any(word_lower[i] == word_lower[i+1] for i in range(len(word)-1)),
        }
        
        # Letter position features
        for i, letter in enumerate(word_lower):
            features[f'pos_{i+1}_letter'] = letter.upper()
        
        # Common letter patterns
        features['starts_with_vowel'] = word_lower[0] in 'aeiou'
        features['ends_with_vowel'] = word_lower[-1] in 'aeiou'
        
        # WordNet features
        try:
            synsets = wordnet.synsets(word_lower)
            features['wordnet_synsets'] = len(synsets)
            features['has_wordnet_entry'] = len(synsets) > 0
        except:
            features['wordnet_synsets'] = 0
            features['has_wordnet_entry'] = False
        
        return features
    
    def _collect_frequency_fallback(self) -> pd.DataFrame:
        """Fallback method for collecting frequency data."""
        self.logger.info("Using fallback frequency collection method")
        
        # Create basic frequency estimates based on letter frequency
        words = self._get_common_5_letter_words()
        
        # English letter frequencies (approximate)
        letter_freq = {
            'e': 0.127, 't': 0.091, 'a': 0.082, 'o': 0.075, 'i': 0.070,
            'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060, 'd': 0.043,
            'l': 0.040, 'c': 0.028, 'u': 0.028, 'm': 0.024, 'w': 0.023,
            'f': 0.022, 'g': 0.020, 'y': 0.020, 'p': 0.019, 'b': 0.013,
            'v': 0.010, 'k': 0.008, 'j': 0.001, 'x': 0.001, 'q': 0.001, 'z': 0.001
        }
        
        freq_data = []
        for word in words:
            # Estimate frequency based on letter frequencies
            word_freq = np.mean([letter_freq.get(c.lower(), 0.001) for c in word])
            freq_data.append({
                'word': word.upper(),
                'frequency': word_freq,
                'log_frequency': np.log10(word_freq)
            })
        
        return pd.DataFrame(freq_data)