"""
Baseline models for Wordle prediction.
Implements frequency-based and heuristic approaches.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import math
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import joblib


class FrequencyBasedPredictor(BaseEstimator, ClassifierMixin):
    """Simple frequency-based word prediction."""
    
    def __init__(self, use_position_weights: bool = True, alpha: float = 1.0):
        self.use_position_weights = use_position_weights
        self.alpha = alpha  # Smoothing parameter
        self.word_frequencies = {}
        self.letter_frequencies = {}
        self.position_frequencies = [{} for _ in range(5)]
        self.vocabulary = []
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train frequency-based model."""
        self.logger.info("Training FrequencyBasedPredictor...")
        
        # Extract vocabulary
        if 'word' in X.columns:
            self.vocabulary = X['word'].unique().tolist()
        else:
            raise ValueError("X must contain 'word' column")
        
        # Calculate word frequencies
        if 'frequency' in X.columns:
            word_freq_series = X.set_index('word')['frequency']
            self.word_frequencies = word_freq_series.to_dict()
        else:
            # Uniform frequencies if not provided
            self.word_frequencies = {word: 1.0 for word in self.vocabulary}
        
        # Calculate letter frequencies
        all_letters = ''.join(self.vocabulary)
        letter_counts = Counter(all_letters)
        total_letters = sum(letter_counts.values())
        
        for letter, count in letter_counts.items():
            self.letter_frequencies[letter] = count / total_letters
        
        # Calculate position-specific frequencies
        if self.use_position_weights:
            for pos in range(5):
                pos_letters = [word[pos] for word in self.vocabulary if len(word) > pos]
                pos_counts = Counter(pos_letters)
                total_pos = sum(pos_counts.values())
                
                for letter, count in pos_counts.items():
                    self.position_frequencies[pos][letter] = count / total_pos
        
        self.is_fitted = True
        self.logger.info(f"Training complete. Vocabulary size: {len(self.vocabulary)}")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability distribution over words."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # For this baseline, we return the same probability distribution
        # regardless of input (could be enhanced with context)
        probabilities = []
        
        for word in self.vocabulary:
            # Base frequency
            word_prob = self.word_frequencies.get(word, 0.001)
            
            # Letter frequency component
            letter_prob = 1.0
            for letter in word:
                letter_prob *= self.letter_frequencies.get(letter, 0.001)
            
            # Position frequency component
            pos_prob = 1.0
            if self.use_position_weights:
                for i, letter in enumerate(word):
                    pos_prob *= self.position_frequencies[i].get(letter, 0.001)
            
            # Combine probabilities
            combined_prob = (word_prob ** 0.5) * (letter_prob ** 0.3) * (pos_prob ** 0.2)
            probabilities.append(combined_prob)
        
        # Normalize probabilities
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        # Return same distribution for all inputs
        return np.tile(probabilities, (len(X), 1))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return most likely word."""
        probs = self.predict_proba(X)
        return np.array([self.vocabulary[np.argmax(prob)] for prob in probs])
    
    def get_top_predictions(self, X: pd.DataFrame, k: int = 5) -> List[List[Tuple[str, float]]]:
        """Get top k predictions with probabilities."""
        probs = self.predict_proba(X)
        
        results = []
        for prob in probs:
            top_indices = np.argsort(prob)[-k:][::-1]
            top_predictions = [(self.vocabulary[i], prob[i]) for i in top_indices]
            results.append(top_predictions)
        
        return results


class InformationEntropyPredictor(BaseEstimator, ClassifierMixin):
    """Information theory-based word prediction."""
    
    def __init__(self, strategy: str = 'max_entropy'):
        self.strategy = strategy  # 'max_entropy', 'min_entropy', 'balanced'
        self.vocabulary = []
        self.word_entropies = {}
        self.letter_entropies = {}
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train entropy-based model."""
        self.logger.info(f"Training InformationEntropyPredictor with strategy: {self.strategy}")
        
        if 'word' in X.columns:
            self.vocabulary = X['word'].unique().tolist()
        else:
            raise ValueError("X must contain 'word' column")
        
        # Calculate entropy for each word
        for word in self.vocabulary:
            self.word_entropies[word] = self._calculate_word_entropy(word)
        
        # Calculate letter entropies
        all_letters = set(''.join(self.vocabulary))
        for letter in all_letters:
            self.letter_entropies[letter] = self._calculate_letter_entropy(letter)
        
        self.is_fitted = True
        self.logger.info(f"Training complete. Calculated entropies for {len(self.vocabulary)} words")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability distribution based on information gain."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        probabilities = []
        
        for word in self.vocabulary:
            entropy_score = self.word_entropies[word]
            
            if self.strategy == 'max_entropy':
                # Prefer words with higher entropy (more information)
                prob = entropy_score
            elif self.strategy == 'min_entropy':
                # Prefer words with lower entropy (more predictable)
                prob = 1.0 / (1.0 + entropy_score)
            else:  # balanced
                # Balance between entropy and frequency
                prob = math.sqrt(entropy_score)
            
            probabilities.append(prob)
        
        # Normalize probabilities
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        # Return same distribution for all inputs
        return np.tile(probabilities, (len(X), 1))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return most likely word based on information gain."""
        probs = self.predict_proba(X)
        return np.array([self.vocabulary[np.argmax(prob)] for prob in probs])
    
    def get_information_gain(self, word: str) -> float:
        """Calculate expected information gain for a word."""
        return self.word_entropies.get(word, 0.0)
    
    def _calculate_word_entropy(self, word: str) -> float:
        """Calculate information entropy for a word."""
        # Letter frequency entropy
        letter_counts = Counter(word)
        word_length = len(word)
        
        entropy = 0.0
        for count in letter_counts.values():
            prob = count / word_length
            entropy -= prob * math.log2(prob)
        
        # Position diversity bonus
        unique_letters = len(set(word))
        position_diversity = unique_letters / word_length
        
        # Common letter penalty (words with very common letters are less informative)
        common_letters = ['E', 'T', 'A', 'O', 'I', 'N', 'S', 'H', 'R']
        common_count = sum(1 for letter in word if letter in common_letters)
        common_penalty = 1.0 - (common_count / len(word))
        
        return entropy * position_diversity * (1.0 + common_penalty)
    
    def _calculate_letter_entropy(self, letter: str) -> float:
        """Calculate information entropy for a letter across all words."""
        # Count positions where this letter appears
        positions = [0] * 5
        total_occurrences = 0
        
        for word in self.vocabulary:
            for i, char in enumerate(word):
                if char == letter:
                    positions[i] += 1
                    total_occurrences += 1
        
        if total_occurrences == 0:
            return 0.0
        
        # Calculate entropy based on position distribution
        entropy = 0.0
        for count in positions:
            if count > 0:
                prob = count / total_occurrences
                entropy -= prob * math.log2(prob)
        
        return entropy


class HeuristicPredictor(BaseEstimator, ClassifierMixin):
    """Heuristic-based predictor using game theory principles."""
    
    def __init__(self, strategy: str = 'elimination'):
        self.strategy = strategy  # 'elimination', 'frequency', 'hybrid'
        self.vocabulary = []
        self.word_scores = {}
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
        
        # Common starting words for Wordle
        self.optimal_starters = ['AROSE', 'SLATE', 'CRATE', 'AUDIO', 'ORATE', 'ROATE', 'RAISE']
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train heuristic model."""
        self.logger.info(f"Training HeuristicPredictor with strategy: {self.strategy}")
        
        if 'word' in X.columns:
            self.vocabulary = X['word'].unique().tolist()
        else:
            raise ValueError("X must contain 'word' column")
        
        # Calculate scores for each word based on strategy
        for word in self.vocabulary:
            if self.strategy == 'elimination':
                self.word_scores[word] = self._calculate_elimination_score(word)
            elif self.strategy == 'frequency':
                self.word_scores[word] = self._calculate_frequency_score(word, X)
            else:  # hybrid
                elim_score = self._calculate_elimination_score(word)
                freq_score = self._calculate_frequency_score(word, X)
                self.word_scores[word] = 0.6 * elim_score + 0.4 * freq_score
        
        self.is_fitted = True
        self.logger.info("Training complete")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability distribution based on heuristic scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert scores to probabilities
        scores = np.array([self.word_scores[word] for word in self.vocabulary])
        
        # Apply softmax transformation
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probabilities = exp_scores / exp_scores.sum()
        
        # Return same distribution for all inputs
        return np.tile(probabilities, (len(X), 1))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return most likely word based on heuristic."""
        probs = self.predict_proba(X)
        return np.array([self.vocabulary[np.argmax(prob)] for prob in probs])
    
    def get_optimal_starter(self) -> str:
        """Return optimal starting word."""
        # Find the best starter from our vocabulary
        available_starters = [word for word in self.optimal_starters if word in self.vocabulary]
        
        if available_starters:
            return available_starters[0]
        else:
            # Fall back to highest scoring word
            best_word = max(self.word_scores.items(), key=lambda x: x[1])[0]
            return best_word
    
    def _calculate_elimination_score(self, word: str) -> float:
        """Calculate how well this word eliminates possibilities."""
        # Unique letters are valuable
        unique_letters = len(set(word))
        
        # Common letters are valuable for information gathering
        common_letters = ['E', 'A', 'R', 'I', 'O', 'T', 'N', 'S']
        common_count = sum(1 for letter in word if letter in common_letters)
        
        # Vowel distribution
        vowels = 'AEIOU'
        vowel_count = sum(1 for letter in word if letter in vowels)
        optimal_vowels = min(vowel_count, 2)  # 1-2 vowels is optimal
        
        # Position diversity (avoid repeated patterns)
        position_score = 1.0
        if word[0] == word[-1]:  # Same first and last letter
            position_score -= 0.2
        
        return (unique_letters * 0.4 + 
                common_count * 0.3 + 
                optimal_vowels * 0.2 + 
                position_score * 0.1)
    
    def _calculate_frequency_score(self, word: str, data: pd.DataFrame) -> float:
        """Calculate score based on word frequency."""
        if 'frequency' in data.columns:
            word_data = data[data['word'] == word]
            if not word_data.empty:
                frequency = word_data['frequency'].iloc[0]
                return math.log(frequency + 1e-10)  # Log transform for better distribution
        
        # Fallback: calculate based on letter frequencies
        letter_freq = {
            'E': 0.127, 'T': 0.091, 'A': 0.082, 'O': 0.075, 'I': 0.070,
            'N': 0.067, 'S': 0.063, 'H': 0.061, 'R': 0.060, 'D': 0.043,
        }
        
        score = 0.0
        for letter in word:
            score += letter_freq.get(letter, 0.01)
        
        return score / len(word)


class EnsembleBaseline(BaseEstimator, ClassifierMixin):
    """Ensemble of baseline models."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {'frequency': 0.4, 'entropy': 0.3, 'heuristic': 0.3}
        self.models = {}
        self.vocabulary = []
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train ensemble of baseline models."""
        self.logger.info("Training EnsembleBaseline...")
        
        # Initialize and train individual models
        self.models['frequency'] = FrequencyBasedPredictor()
        self.models['entropy'] = InformationEntropyPredictor()
        self.models['heuristic'] = HeuristicPredictor()
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            model.fit(X, y)
        
        # Extract vocabulary from first model
        self.vocabulary = self.models['frequency'].vocabulary
        self.is_fitted = True
        
        self.logger.info("Ensemble training complete")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return weighted ensemble probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from each model
        ensemble_probs = np.zeros((len(X), len(self.vocabulary)))
        
        for name, model in self.models.items():
            model_probs = model.predict_proba(X)
            weight = self.weights.get(name, 0.0)
            ensemble_probs += weight * model_probs
        
        # Normalize
        ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        
        return ensemble_probs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return most likely word from ensemble."""
        probs = self.predict_proba(X)
        return np.array([self.vocabulary[np.argmax(prob)] for prob in probs])
    
    def save_model(self, filepath: Path):
        """Save the trained ensemble model."""
        joblib.dump(self, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path):
        """Load a trained ensemble model."""
        return joblib.load(filepath)


def evaluate_baseline_models(X_test: pd.DataFrame, y_test: pd.Series, models: Dict) -> Dict:
    """Evaluate baseline models and return metrics."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating baseline models...")
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        
        # Get predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        # Top-k accuracy
        top_5_acc = top_k_accuracy_score(y_test, probabilities, k=5, 
                                       labels=model.vocabulary)
        
        results[name] = {
            'accuracy': accuracy,
            'top_5_accuracy': top_5_acc,
            'vocabulary_size': len(model.vocabulary)
        }
        
        logger.info(f"{name} - Accuracy: {accuracy:.4f}, Top-5: {top_5_acc:.4f}")
    
    return results