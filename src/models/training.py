"""
Training pipeline for Wordle prediction models.
Handles cross-validation, hyperparameter tuning, and model evaluation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .baseline_models import (FrequencyBasedPredictor, InformationEntropyPredictor, 
                             HeuristicPredictor, EnsembleBaseline)
from .advanced_models import EnsembleAdvancedModel


class ModelTrainer:
    def __init__(self, config: Dict, data_dir: Path = Path("data"), models_dir: Path = Path("models")):
        self.config = config
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Training results storage
        self.training_results = {}
        self.best_models = {}
        
    def _setup_logging(self):
        """Setup training-specific logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create training log file
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.info(f"Training logging setup complete. Log file: {log_file}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training and testing data."""
        self.logger.info("Preparing training data...")
        
        # Ensure we have required columns
        if 'word' not in df.columns:
            raise ValueError("DataFrame must contain 'word' column")
        
        # Create features and target
        X = df.copy()
        
        # If we have answer_id, use it for temporal split
        if 'answer_id' in df.columns:
            # Use first 80% for training, last 20% for testing (temporal split)
            split_point = int(len(df) * 0.8)
            train_df = df.iloc[:split_point].copy()
            test_df = df.iloc[split_point:].copy()
            
            X_train = train_df.drop('word', axis=1) if len(train_df.columns) > 1 else train_df
            y_train = train_df['word']
            X_test = test_df.drop('word', axis=1) if len(test_df.columns) > 1 else test_df
            y_test = test_df['word']
            
            self.logger.info(f"Temporal split: {len(X_train)} train, {len(X_test)} test samples")
        
        else:
            # Random split
            # For Wordle, we typically predict the next word, so we create a shifted target
            y = df['word'].copy()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            
            self.logger.info(f"Random split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Log data statistics
        self.logger.info(f"Training vocabulary size: {len(y_train.unique())}")
        self.logger.info(f"Test vocabulary size: {len(y_test.unique())}")
        self.logger.info(f"Feature columns: {list(X_train.columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train all baseline models."""
        self.logger.info("Training baseline models...")
        
        models = {
            'frequency_basic': FrequencyBasedPredictor(use_position_weights=False),
            'frequency_position': FrequencyBasedPredictor(use_position_weights=True),
            'entropy_max': InformationEntropyPredictor(strategy='max_entropy'),
            'entropy_balanced': InformationEntropyPredictor(strategy='balanced'),
            'heuristic_elimination': HeuristicPredictor(strategy='elimination'),
            'heuristic_hybrid': HeuristicPredictor(strategy='hybrid'),
            'ensemble_baseline': EnsembleBaseline()
        }
        
        trained_models = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            start_time = time.time()
            
            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                trained_models[name] = {
                    'model': model,
                    'training_time': training_time,
                    'vocab_size': len(model.vocabulary) if hasattr(model, 'vocabulary') else 0
                }
                
                self.logger.info(f"{name} training complete in {training_time:.2f}s")
                
                # Save model
                model_path = self.models_dir / f"{name}.pkl"
                joblib.dump(model, model_path)
                self.logger.info(f"Saved {name} to {model_path}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                continue
        
        self.training_results['baseline'] = trained_models
        return trained_models
    
    def train_advanced_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train advanced ML models."""
        self.logger.info("Training advanced models...")
        
        # Get vocabulary size
        vocab_size = len(y_train.unique())
        
        models = {
            'ensemble_advanced': EnsembleAdvancedModel(vocab_size=vocab_size)
        }
        
        trained_models = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            start_time = time.time()
            
            try:
                # Advanced models may need GPU and more time
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                trained_models[name] = {
                    'model': model,
                    'training_time': training_time,
                    'vocab_size': vocab_size
                }
                
                self.logger.info(f"{name} training complete in {training_time:.2f}s")
                
                # Save model
                model_path = self.models_dir / f"{name}.pth"
                model.save_model(model_path)
                self.logger.info(f"Saved {name} to {model_path}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                continue
        
        self.training_results['advanced'] = trained_models
        return trained_models
    
    def hyperparameter_search(self, model_class, param_grid: Dict, X: pd.DataFrame, y: pd.Series, 
                            search_type: str = 'grid', cv: int = 3, n_iter: int = 20) -> Dict:
        """Perform hyperparameter optimization."""
        self.logger.info(f"Hyperparameter search for {model_class.__name__}...")
        
        # Create base model
        model = model_class()
        
        # Setup search
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, param_grid, cv=cv, scoring='accuracy',
                n_iter=n_iter, n_jobs=-1, verbose=1, random_state=42
            )
        
        # Perform search
        start_time = time.time()
        search.fit(X, y)
        search_time = time.time() - start_time
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'search_time': search_time,
            'cv_results': search.cv_results_
        }
        
        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")
        self.logger.info(f"Search completed in {search_time:.2f}s")
        
        return results
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """Perform cross-validation evaluation."""
        self.logger.info(f"Cross-validating {model.__class__.__name__}...")
        
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = []
        cv_detailed = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            self.logger.info(f"Fold {fold + 1}/{cv}")
            
            # Split data
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            
            # Train and evaluate
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(accuracy)
            
            # Detailed metrics for this fold
            fold_results = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'n_train': len(X_fold_train),
                'n_val': len(X_fold_val)
            }
            cv_detailed.append(fold_results)
            
            self.logger.info(f"Fold {fold + 1} accuracy: {accuracy:.4f}")
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'detailed_results': cv_detailed
        }
        
        self.logger.info(f"CV Results - Mean: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        
        return results
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str = "Model") -> Dict:
        """Comprehensive model evaluation."""
        self.logger.info(f"Evaluating {model_name}...")
        
        start_time = time.time()
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        prediction_time = time.time() - start_time
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Top-k accuracy (if probabilities available)
        top_k_accuracies = {}
        if y_proba is not None and hasattr(model, 'vocabulary'):
            for k in [1, 3, 5, 10]:
                if k <= len(model.vocabulary):
                    top_k_acc = self._calculate_top_k_accuracy(y_test, y_proba, model.vocabulary, k)
                    top_k_accuracies[f'top_{k}'] = top_k_acc
        
        # Detailed classification report
        unique_labels = sorted(list(set(y_test) | set(y_pred)))
        class_report = classification_report(y_test, y_pred, labels=unique_labels, 
                                           output_dict=True, zero_division=0)
        
        # Confusion matrix (limited to top classes for readability)
        top_classes = sorted(list(set(y_test)), key=lambda x: list(y_test).count(x), reverse=True)[:20]
        cm = confusion_matrix(y_test, y_pred, labels=top_classes)
        
        results = {
            'accuracy': accuracy,
            'top_k_accuracies': top_k_accuracies,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': top_classes,
            'prediction_time': prediction_time,
            'predictions_per_second': len(X_test) / prediction_time,
            'n_test_samples': len(X_test),
            'n_classes': len(unique_labels)
        }
        
        # Log key metrics
        self.logger.info(f"{model_name} Evaluation Results:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        for k, acc in top_k_accuracies.items():
            self.logger.info(f"  {k} accuracy: {acc:.4f}")
        self.logger.info(f"  Prediction time: {prediction_time:.2f}s")
        self.logger.info(f"  Predictions/sec: {results['predictions_per_second']:.1f}")
        
        return results
    
    def compare_models(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Compare multiple models on the same test set."""
        self.logger.info("Comparing models...")
        
        comparison_results = {}
        
        for name, model_info in models.items():
            model = model_info['model']
            
            # Evaluate model
            results = self.evaluate_model(model, X_test, y_test, name)
            results['training_time'] = model_info.get('training_time', 0)
            
            comparison_results[name] = results
        
        # Create summary comparison
        summary = self._create_model_comparison_summary(comparison_results)
        
        # Save comparison results
        comparison_path = self.models_dir / "model_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        self.logger.info(f"Model comparison saved to {comparison_path}")
        
        return comparison_results
    
    def full_training_pipeline(self, df: pd.DataFrame) -> Dict:
        """Run complete training pipeline."""
        self.logger.info("Starting full training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Train baseline models
        baseline_models = self.train_baseline_models(X_train, y_train)
        
        # Train advanced models (if configured)
        if self.config.get('train_advanced', False):
            advanced_models = self.train_advanced_models(X_train, y_train)
            all_models = {**baseline_models, **advanced_models}
        else:
            all_models = baseline_models
        
        # Model comparison
        comparison_results = self.compare_models(all_models, X_test, y_test)
        
        # Identify best model
        best_model_name, best_model_results = self._identify_best_model(comparison_results)
        
        # Save best model separately
        if best_model_name in all_models:
            best_model = all_models[best_model_name]['model']
            best_model_path = self.models_dir / "best_model.pkl"
            joblib.dump(best_model, best_model_path)
            self.logger.info(f"Best model ({best_model_name}) saved to {best_model_path}")
        
        # Generate final report
        final_report = {
            'training_config': self.config,
            'data_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'vocabulary_size': len(y_train.unique()),
                'features': list(X_train.columns)
            },
            'model_results': comparison_results,
            'best_model': {
                'name': best_model_name,
                'results': best_model_results
            },
            'training_summary': self._create_training_summary(all_models, comparison_results)
        }
        
        # Save final report
        report_path = self.models_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        self.logger.info(f"Training pipeline complete. Report saved to {report_path}")
        self.logger.info(f"Best model: {best_model_name} (accuracy: {best_model_results['accuracy']:.4f})")
        
        return final_report
    
    def _calculate_top_k_accuracy(self, y_true: pd.Series, y_proba: np.ndarray, 
                                 vocabulary: List[str], k: int) -> float:
        """Calculate top-k accuracy."""
        correct = 0
        
        for i, true_word in enumerate(y_true):
            if i < len(y_proba):
                # Get top k predictions
                top_k_indices = np.argsort(y_proba[i])[-k:]
                top_k_words = [vocabulary[idx] for idx in top_k_indices]
                
                if true_word in top_k_words:
                    correct += 1
        
        return correct / len(y_true)
    
    def _create_model_comparison_summary(self, results: Dict) -> Dict:
        """Create summary table of model performance."""
        summary = {
            'accuracy': {},
            'top_5_accuracy': {},
            'training_time': {},
            'prediction_speed': {}
        }
        
        for name, result in results.items():
            summary['accuracy'][name] = result['accuracy']
            summary['top_5_accuracy'][name] = result['top_k_accuracies'].get('top_5', 0)
            summary['training_time'][name] = result.get('training_time', 0)
            summary['prediction_speed'][name] = result['predictions_per_second']
        
        return summary
    
    def _identify_best_model(self, results: Dict) -> Tuple[str, Dict]:
        """Identify best performing model."""
        best_name = None
        best_score = -1
        
        for name, result in results.items():
            # Use weighted score: 70% accuracy, 30% top-5 accuracy
            score = (0.7 * result['accuracy'] + 
                    0.3 * result['top_k_accuracies'].get('top_5', 0))
            
            if score > best_score:
                best_score = score
                best_name = name
        
        return best_name, results[best_name]
    
    def _create_training_summary(self, models: Dict, results: Dict) -> Dict:
        """Create training summary statistics."""
        return {
            'total_models_trained': len(models),
            'total_training_time': sum(m['training_time'] for m in models.values()),
            'best_accuracy': max(r['accuracy'] for r in results.values()),
            'average_accuracy': np.mean([r['accuracy'] for r in results.values()]),
            'model_count_by_type': {
                'baseline': len([k for k in models.keys() if 'ensemble_advanced' not in k]),
                'advanced': len([k for k in models.keys() if 'ensemble_advanced' in k])
            }
        }