"""
Comprehensive evaluation framework for Wordle prediction models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, classification_report)
from sklearn.metrics import top_k_accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings


class WordleEvaluator:
    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Performance benchmarks (based on research)
        self.benchmarks = {
            'excellent': {'avg_guesses': 3.5, 'success_rate': 0.95},
            'good': {'avg_guesses': 3.9, 'success_rate': 0.90},
            'average': {'avg_guesses': 4.2, 'success_rate': 0.85},
            'poor': {'avg_guesses': 5.0, 'success_rate': 0.70}
        }
        
        # MIT optimal benchmark
        self.optimal_benchmark = {'avg_guesses': 3.421, 'success_rate': 1.0}
        
        self.metrics = {}
    
    def calculate_rank_accuracy(self, y_true: List[str], y_pred_proba: np.ndarray, 
                               vocabulary: List[str], k: int = 5) -> float:
        """Calculate top-k rank accuracy."""
        self.logger.info(f"Calculating top-{k} rank accuracy...")
        
        if len(y_pred_proba.shape) != 2:
            raise ValueError("y_pred_proba must be 2D array (samples x classes)")
        
        correct_predictions = 0
        total_predictions = len(y_true)
        
        for i, true_word in enumerate(y_true):
            if i >= len(y_pred_proba):
                continue
                
            # Get top-k predictions
            top_k_indices = np.argsort(y_pred_proba[i])[-k:][::-1]
            top_k_words = [vocabulary[idx] for idx in top_k_indices if idx < len(vocabulary)]
            
            if true_word in top_k_words:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        self.logger.info(f"Top-{k} accuracy: {accuracy:.4f}")
        return accuracy
    
    def calculate_daily_hit_rate(self, predictions: List[str], actuals: List[str]) -> Dict[str, float]:
        """Calculate daily prediction hit rate."""
        self.logger.info("Calculating daily hit rate...")
        
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        # Overall hit rate
        hits = sum(1 for pred, actual in zip(predictions, actuals) if pred == actual)
        overall_hit_rate = hits / len(predictions)
        
        # Calculate by position in top predictions (if available)
        results = {
            'overall_hit_rate': overall_hit_rate,
            'total_predictions': len(predictions),
            'correct_predictions': hits
        }
        
        self.logger.info(f"Daily hit rate: {overall_hit_rate:.4f} ({hits}/{len(predictions)})")
        return results
    
    def simulate_wordle_game(self, model, target_words: List[str], 
                           max_guesses: int = 6) -> Dict[str, Any]:
        """Simulate Wordle gameplay to calculate game-specific metrics."""
        self.logger.info(f"Simulating Wordle games for {len(target_words)} target words...")
        
        game_results = []
        total_guesses = 0
        successful_games = 0
        
        for target_word in target_words:
            game_result = self._simulate_single_game(model, target_word, max_guesses)
            game_results.append(game_result)
            
            if game_result['solved']:
                successful_games += 1
                total_guesses += game_result['guesses_used']
            else:
                total_guesses += max_guesses  # Count as max guesses if failed
        
        # Calculate statistics
        avg_guesses = total_guesses / len(target_words)
        success_rate = successful_games / len(target_words)
        
        # Guess distribution
        guess_distribution = {}
        for i in range(1, max_guesses + 1):
            count = sum(1 for result in game_results 
                       if result['solved'] and result['guesses_used'] == i)
            guess_distribution[f'{i}_guess'] = count / len(target_words)
        
        # Failed games
        guess_distribution['failed'] = (len(target_words) - successful_games) / len(target_words)
        
        results = {
            'avg_guesses': avg_guesses,
            'success_rate': success_rate,
            'successful_games': successful_games,
            'total_games': len(target_words),
            'guess_distribution': guess_distribution,
            'detailed_results': game_results
        }
        
        self.logger.info(f"Game simulation complete:")
        self.logger.info(f"  Average guesses: {avg_guesses:.3f}")
        self.logger.info(f"  Success rate: {success_rate:.3f}")
        
        return results
    
    def generate_confusion_analysis(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Generate detailed confusion analysis."""
        self.logger.info("Generating confusion analysis...")
        
        # Get unique labels
        unique_labels = sorted(list(set(y_true + y_pred)))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_labels, average=None, zero_division=0
        )
        
        # Create detailed classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Identify most confused pairs
        confused_pairs = self._find_most_confused_pairs(cm, unique_labels)
        
        # Error analysis
        error_analysis = self._analyze_prediction_errors(y_true, y_pred)
        
        results = {
            'confusion_matrix': cm.tolist(),
            'labels': unique_labels,
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1_score': f1.tolist(),
                'support': support.tolist()
            },
            'classification_report': class_report,
            'most_confused_pairs': confused_pairs,
            'error_analysis': error_analysis
        }
        
        self.logger.info("Confusion analysis complete")
        return results
    
    def benchmark_against_baselines(self, model_results: Dict, baseline_results: Dict) -> Dict[str, Any]:
        """Compare model performance against baselines."""
        self.logger.info("Benchmarking against baselines...")
        
        comparison = {
            'model_performance': model_results,
            'baseline_performance': baseline_results,
            'improvements': {},
            'benchmark_classification': self._classify_performance(model_results)
        }
        
        # Calculate improvements over baselines
        for metric in ['avg_guesses', 'success_rate']:
            if metric in model_results and metric in baseline_results:
                baseline_value = baseline_results[metric]
                model_value = model_results[metric]
                
                if metric == 'avg_guesses':
                    # Lower is better for avg_guesses
                    improvement = (baseline_value - model_value) / baseline_value
                else:
                    # Higher is better for success_rate
                    improvement = (model_value - baseline_value) / baseline_value
                
                comparison['improvements'][metric] = {
                    'absolute': model_value - baseline_value,
                    'relative': improvement,
                    'percentage': improvement * 100
                }
        
        # Compare against MIT optimal
        comparison['vs_optimal'] = self._compare_to_optimal(model_results)
        
        self.logger.info("Benchmarking complete")
        return comparison
    
    def calculate_information_metrics(self, model, vocabulary: List[str], 
                                    sample_words: List[str]) -> Dict[str, float]:
        """Calculate information theory metrics."""
        self.logger.info("Calculating information metrics...")
        
        # Create sample data
        sample_df = pd.DataFrame({'word': sample_words})
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(sample_df)
        else:
            # Create uniform probabilities as fallback
            probabilities = np.ones((len(sample_words), len(vocabulary))) / len(vocabulary)
        
        # Calculate metrics
        metrics = {}
        
        # Entropy
        entropies = []
        for prob_dist in probabilities:
            # Add small epsilon to avoid log(0)
            prob_dist = prob_dist + 1e-10
            entropy = -np.sum(prob_dist * np.log2(prob_dist))
            entropies.append(entropy)
        
        metrics['mean_entropy'] = np.mean(entropies)
        metrics['std_entropy'] = np.std(entropies)
        
        # Perplexity
        perplexities = [2 ** entropy for entropy in entropies]
        metrics['mean_perplexity'] = np.mean(perplexities)
        
        # Confidence (max probability)
        confidences = [np.max(prob_dist) for prob_dist in probabilities]
        metrics['mean_confidence'] = np.mean(confidences)
        metrics['std_confidence'] = np.std(confidences)
        
        # Diversity (effective vocabulary size)
        effective_vocab_sizes = [1 / np.sum(prob_dist ** 2) for prob_dist in probabilities]
        metrics['mean_effective_vocab_size'] = np.mean(effective_vocab_sizes)
        
        self.logger.info(f"Information metrics calculated:")
        self.logger.info(f"  Mean entropy: {metrics['mean_entropy']:.3f}")
        self.logger.info(f"  Mean perplexity: {metrics['mean_perplexity']:.3f}")
        self.logger.info(f"  Mean confidence: {metrics['mean_confidence']:.3f}")
        
        return metrics
    
    def create_performance_visualizations(self, results: Dict[str, Any], 
                                        save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create comprehensive performance visualizations."""
        self.logger.info("Creating performance visualizations...")
        
        visualizations = {}
        
        # 1. Performance comparison chart
        if 'model_results' in results:
            fig_comparison = self._create_model_comparison_chart(results['model_results'])
            visualizations['model_comparison'] = fig_comparison
        
        # 2. Guess distribution
        if 'game_simulation' in results and 'guess_distribution' in results['game_simulation']:
            fig_distribution = self._create_guess_distribution_chart(
                results['game_simulation']['guess_distribution']
            )
            visualizations['guess_distribution'] = fig_distribution
        
        # 3. Confusion matrix heatmap
        if 'confusion_analysis' in results:
            fig_confusion = self._create_confusion_heatmap(results['confusion_analysis'])
            visualizations['confusion_matrix'] = fig_confusion
        
        # 4. Performance vs benchmarks
        if 'benchmark_comparison' in results:
            fig_benchmark = self._create_benchmark_comparison(results['benchmark_comparison'])
            visualizations['benchmark_comparison'] = fig_benchmark
        
        # Save visualizations if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)
            
            for name, fig in visualizations.items():
                if hasattr(fig, 'write_html'):  # Plotly figure
                    fig.write_html(save_path / f"{name}.html")
                elif hasattr(fig, 'savefig'):  # Matplotlib figure
                    fig.savefig(save_path / f"{name}.png", dpi=300, bbox_inches='tight')
        
        self.logger.info("Performance visualizations created")
        return visualizations
    
    def comprehensive_evaluation(self, model, X_test: pd.DataFrame, y_test: List[str],
                               vocabulary: List[str]) -> Dict[str, Any]:
        """Run comprehensive evaluation pipeline."""
        self.logger.info("Starting comprehensive evaluation...")
        
        results = {}
        
        # Basic predictions
        y_pred = model.predict(X_test)
        
        # Probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None
        
        # 1. Basic metrics
        results['basic_metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'total_samples': len(y_test)
        }
        
        # 2. Rank accuracy
        if y_pred_proba is not None:
            results['rank_accuracy'] = {}
            for k in [1, 3, 5, 10]:
                if k <= len(vocabulary):
                    results['rank_accuracy'][f'top_{k}'] = self.calculate_rank_accuracy(
                        y_test, y_pred_proba, vocabulary, k
                    )
        
        # 3. Daily hit rate
        results['daily_hit_rate'] = self.calculate_daily_hit_rate(y_pred, y_test)
        
        # 4. Game simulation
        unique_targets = list(set(y_test))[:100]  # Sample for efficiency
        results['game_simulation'] = self.simulate_wordle_game(model, unique_targets)
        
        # 5. Confusion analysis
        results['confusion_analysis'] = self.generate_confusion_analysis(y_test, y_pred)
        
        # 6. Information metrics
        results['information_metrics'] = self.calculate_information_metrics(
            model, vocabulary, unique_targets
        )
        
        # 7. Performance classification
        results['performance_classification'] = self._classify_performance(
            results['game_simulation']
        )
        
        # 8. Create visualizations
        viz_path = self.results_dir / "visualizations"
        results['visualizations'] = self.create_performance_visualizations(results, viz_path)
        
        # Save comprehensive results
        results_path = self.results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive evaluation complete. Results saved to {results_path}")
        return results
    
    def _simulate_single_game(self, model, target_word: str, max_guesses: int) -> Dict[str, Any]:
        """Simulate a single Wordle game."""
        guesses = []
        constraints = {}  # Track what we know about letter positions
        
        for guess_num in range(1, max_guesses + 1):
            # Create input for model (simplified - could be enhanced with constraints)
            sample_df = pd.DataFrame({'word': [target_word]})  # Simplified input
            
            # Get prediction
            if hasattr(model, 'get_top_predictions'):
                predictions = model.get_top_predictions(sample_df, k=1)[0]
                guess = predictions[0][0] if predictions else 'AROSE'  # Default guess
            else:
                guess = model.predict(sample_df)[0]
            
            guesses.append(guess)
            
            # Check if solved
            if guess == target_word:
                return {
                    'solved': True,
                    'guesses_used': guess_num,
                    'guesses': guesses,
                    'target': target_word
                }
            
            # Update constraints (simplified)
            # In a real implementation, would track letter positions and colors
        
        return {
            'solved': False,
            'guesses_used': max_guesses,
            'guesses': guesses,
            'target': target_word
        }
    
    def _find_most_confused_pairs(self, cm: np.ndarray, labels: List[str], top_k: int = 10) -> List[Dict]:
        """Find most confused word pairs."""
        confused_pairs = []
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append({
                        'true_label': labels[i],
                        'predicted_label': labels[j],
                        'count': int(cm[i][j]),
                        'confusion_rate': cm[i][j] / cm[i].sum() if cm[i].sum() > 0 else 0
                    })
        
        # Sort by count and return top k
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        return confused_pairs[:top_k]
    
    def _analyze_prediction_errors(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Analyze types of prediction errors."""
        errors = []
        
        for true_word, pred_word in zip(y_true, y_pred):
            if true_word != pred_word:
                # Calculate similarity metrics
                similarity = self._calculate_word_similarity(true_word, pred_word)
                errors.append({
                    'true_word': true_word,
                    'predicted_word': pred_word,
                    'letter_overlap': similarity['letter_overlap'],
                    'position_matches': similarity['position_matches']
                })
        
        # Analyze error patterns
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(y_true),
            'avg_letter_overlap': np.mean([e['letter_overlap'] for e in errors]) if errors else 0,
            'avg_position_matches': np.mean([e['position_matches'] for e in errors]) if errors else 0
        }
        
        return error_analysis
    
    def _calculate_word_similarity(self, word1: str, word2: str) -> Dict[str, float]:
        """Calculate similarity between two words."""
        # Letter overlap
        letters1 = set(word1.upper())
        letters2 = set(word2.upper())
        overlap = len(letters1 & letters2) / len(letters1 | letters2) if letters1 | letters2 else 0
        
        # Position matches
        position_matches = sum(1 for c1, c2 in zip(word1.upper(), word2.upper()) if c1 == c2)
        position_ratio = position_matches / max(len(word1), len(word2))
        
        return {
            'letter_overlap': overlap,
            'position_matches': position_ratio
        }
    
    def _classify_performance(self, game_results: Dict[str, float]) -> str:
        """Classify performance level based on benchmarks."""
        avg_guesses = game_results.get('avg_guesses', float('inf'))
        success_rate = game_results.get('success_rate', 0)
        
        if (avg_guesses <= self.benchmarks['excellent']['avg_guesses'] and 
            success_rate >= self.benchmarks['excellent']['success_rate']):
            return 'excellent'
        elif (avg_guesses <= self.benchmarks['good']['avg_guesses'] and 
              success_rate >= self.benchmarks['good']['success_rate']):
            return 'good'
        elif (avg_guesses <= self.benchmarks['average']['avg_guesses'] and 
              success_rate >= self.benchmarks['average']['success_rate']):
            return 'average'
        else:
            return 'poor'
    
    def _compare_to_optimal(self, model_results: Dict[str, float]) -> Dict[str, float]:
        """Compare performance to MIT optimal algorithm."""
        comparison = {}
        
        if 'avg_guesses' in model_results:
            optimal_guesses = self.optimal_benchmark['avg_guesses']
            model_guesses = model_results['avg_guesses']
            comparison['guess_efficiency'] = optimal_guesses / model_guesses
            comparison['excess_guesses'] = model_guesses - optimal_guesses
        
        if 'success_rate' in model_results:
            comparison['success_rate_gap'] = (self.optimal_benchmark['success_rate'] - 
                                            model_results['success_rate'])
        
        return comparison
    
    def _create_model_comparison_chart(self, model_results: Dict) -> go.Figure:
        """Create model comparison chart."""
        models = list(model_results.keys())
        accuracies = [result['accuracy'] for result in model_results.values()]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies, name='Accuracy')
        ])
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Accuracy',
            showlegend=False
        )
        
        return fig
    
    def _create_guess_distribution_chart(self, distribution: Dict[str, float]) -> go.Figure:
        """Create guess distribution chart."""
        categories = list(distribution.keys())
        values = list(distribution.values())
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, name='Distribution')
        ])
        
        fig.update_layout(
            title='Guess Distribution',
            xaxis_title='Number of Guesses',
            yaxis_title='Proportion',
            showlegend=False
        )
        
        return fig
    
    def _create_confusion_heatmap(self, confusion_data: Dict) -> go.Figure:
        """Create confusion matrix heatmap."""
        cm = np.array(confusion_data['confusion_matrix'])
        labels = confusion_data['labels']
        
        # Limit to top 20 classes for readability
        if len(labels) > 20:
            cm = cm[:20, :20]
            labels = labels[:20]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        
        return fig
    
    def _create_benchmark_comparison(self, benchmark_data: Dict) -> go.Figure:
        """Create benchmark comparison chart."""
        # This is a simplified version - would be enhanced with actual benchmark data
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=['Model', 'Excellent', 'Good', 'Average'],
            y=[3.8, 3.5, 3.9, 4.2],  # Example values
            mode='markers+lines',
            name='Average Guesses'
        ))
        
        fig.update_layout(
            title='Performance vs Benchmarks',
            xaxis_title='Category',
            yaxis_title='Average Guesses'
        )
        
        return fig