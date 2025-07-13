"""
Basic tests for Wordle prediction system.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_collection import WordleDataCollector
from data.preprocessing import WordleDataPreprocessor
from features.feature_engineering import WordleFeatureEngineer
from models.baseline_models import FrequencyBasedPredictor, InformationEntropyPredictor


class TestDataCollection:
    """Test data collection functionality."""
    
    def test_data_collector_init(self):
        """Test data collector initialization."""
        collector = WordleDataCollector(Path("test_data"))
        assert collector.data_dir == Path("test_data")
        assert collector.raw_dir == Path("test_data/raw")
        assert collector.processed_dir == Path("test_data/processed")
    
    def test_word_similarity_calculation(self):
        """Test word similarity calculation."""
        collector = WordleDataCollector(Path("test_data"))
        similarity = collector._calculate_word_similarity("AROSE", "AROSE")
        assert similarity['letter_overlap'] == 1.0
        assert similarity['position_matches'] == 1.0
        
        similarity = collector._calculate_word_similarity("AROSE", "SLATE")
        assert 0 <= similarity['letter_overlap'] <= 1
        assert 0 <= similarity['position_matches'] <= 1


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        preprocessor = WordleDataPreprocessor()
        assert preprocessor.min_frequency_threshold == 1e-10
        assert preprocessor.max_frequency_threshold == 1.0
    
    def test_clean_word_data(self):
        """Test word data cleaning."""
        preprocessor = WordleDataPreprocessor()
        
        # Create test data with various issues
        test_data = pd.DataFrame({
            'word': ['AROSE', 'arose', 'SLATE', 'ABC', 'TOOLONG', 'AR0SE', '  CRANE  ']
        })
        
        cleaned = preprocessor.clean_word_data(test_data)
        
        # Should keep only valid 5-letter alphabetic words
        assert len(cleaned) <= len(test_data)
        assert all(len(word) == 5 for word in cleaned['word'])
        assert all(word.isalpha() for word in cleaned['word'])
        assert all(word.isupper() for word in cleaned['word'])
    
    def test_missing_value_handling(self):
        """Test missing value handling."""
        preprocessor = WordleDataPreprocessor()
        
        test_data = pd.DataFrame({
            'word': ['AROSE', 'SLATE', 'CRANE'],
            'frequency': [0.1, None, 0.05],
            'vowel_count': [3, 2, None]
        })
        
        result = preprocessor.handle_missing_values(test_data)
        
        # No missing values should remain
        assert not result.isnull().any().any()


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_feature_engineer_init(self):
        """Test feature engineer initialization."""
        engineer = WordleFeatureEngineer()
        assert len(engineer.english_letter_freq) == 26
        assert 'E' in engineer.english_letter_freq
        assert engineer.english_letter_freq['E'] > 0
    
    def test_linguistic_features(self):
        """Test linguistic feature creation."""
        engineer = WordleFeatureEngineer()
        
        words = ['AROSE', 'SLATE', 'CRANE']
        features = engineer.create_linguistic_features(words)
        
        assert len(features) == len(words)
        assert 'word' in features.columns
        assert 'avg_letter_frequency' in features.columns
        assert 'vowel_count' in features.columns
        assert 'consonant_count' in features.columns
        
        # Check that features make sense
        arose_features = features[features['word'] == 'AROSE'].iloc[0]
        assert arose_features['vowel_count'] >= 2  # A, O, E
        assert arose_features['consonant_count'] >= 2  # R, S
    
    def test_letter_frequency_features(self):
        """Test letter frequency feature calculation."""
        engineer = WordleFeatureEngineer()
        
        features = engineer._letter_frequency_features('AROSE')
        
        assert 'avg_letter_frequency' in features
        assert 'total_letter_frequency' in features
        assert 'high_freq_letters' in features
        assert 'low_freq_letters' in features
        
        assert features['avg_letter_frequency'] > 0
        assert features['total_letter_frequency'] > 0


class TestBaselineModels:
    """Test baseline model functionality."""
    
    def test_frequency_based_predictor(self):
        """Test frequency-based predictor."""
        # Create test data
        test_data = pd.DataFrame({
            'word': ['AROSE', 'SLATE', 'CRANE', 'AUDIO', 'ORATE'],
            'frequency': [0.1, 0.08, 0.06, 0.05, 0.04]
        })
        
        model = FrequencyBasedPredictor()
        model.fit(test_data, test_data['word'])
        
        assert model.is_fitted
        assert len(model.vocabulary) == len(test_data)
        assert 'AROSE' in model.vocabulary
        
        # Test prediction
        predictions = model.predict(test_data.head(1))
        assert len(predictions) == 1
        assert predictions[0] in model.vocabulary
        
        # Test probabilities
        probabilities = model.predict_proba(test_data.head(1))
        assert probabilities.shape == (1, len(model.vocabulary))
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Should sum to 1
    
    def test_information_entropy_predictor(self):
        """Test information entropy predictor."""
        test_data = pd.DataFrame({
            'word': ['AROSE', 'SLATE', 'CRANE', 'AUDIO', 'ORATE']
        })
        
        model = InformationEntropyPredictor(strategy='max_entropy')
        model.fit(test_data, test_data['word'])
        
        assert model.is_fitted
        assert len(model.vocabulary) == len(test_data)
        assert len(model.word_entropies) == len(test_data)
        
        # Test that entropies are calculated
        for word in model.vocabulary:
            assert word in model.word_entropies
            assert model.word_entropies[word] > 0
        
        # Test prediction
        predictions = model.predict(test_data.head(1))
        assert len(predictions) == 1
        assert predictions[0] in model.vocabulary


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_end_to_end_basic_pipeline(self):
        """Test basic end-to-end pipeline."""
        # Create minimal test dataset
        words = ['AROSE', 'SLATE', 'CRANE', 'AUDIO', 'ORATE']
        test_data = pd.DataFrame({
            'word': words,
            'frequency': np.random.uniform(0.01, 0.1, len(words)),
            'answer_id': range(1, len(words) + 1)
        })
        
        # Preprocessing
        preprocessor = WordleDataPreprocessor()
        cleaned_data = preprocessor.clean_word_data(test_data)
        processed_data = preprocessor.handle_missing_values(cleaned_data)
        
        # Feature engineering
        engineer = WordleFeatureEngineer()
        featured_data = engineer.create_comprehensive_features(processed_data)
        
        # Model training
        model = FrequencyBasedPredictor()
        model.fit(featured_data, featured_data['word'])
        
        # Prediction
        predictions = model.predict(featured_data.head(2))
        
        # Assertions
        assert len(predictions) == 2
        assert all(pred in words for pred in predictions)
        assert model.is_fitted
    
    def test_model_serialization(self):
        """Test model saving and loading."""
        import joblib
        import tempfile
        
        # Create and train model
        test_data = pd.DataFrame({
            'word': ['AROSE', 'SLATE', 'CRANE'],
            'frequency': [0.1, 0.08, 0.06]
        })
        
        model = FrequencyBasedPredictor()
        model.fit(test_data, test_data['word'])
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            joblib.dump(model, f.name)
            loaded_model = joblib.load(f.name)
        
        # Test that loaded model works
        assert loaded_model.is_fitted
        assert loaded_model.vocabulary == model.vocabulary
        
        predictions_original = model.predict(test_data.head(1))
        predictions_loaded = loaded_model.predict(test_data.head(1))
        
        assert predictions_original[0] == predictions_loaded[0]


if __name__ == "__main__":
    pytest.main([__file__])