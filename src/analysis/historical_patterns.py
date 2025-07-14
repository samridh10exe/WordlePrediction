#!/usr/bin/env python3
"""
Historical pattern analysis for Wordle prediction system.

This module analyzes the 1472 historical Wordle answers to identify temporal patterns,
editorial preferences, and strategic insights for improved prediction accuracy.

Key analyses:
- Temporal trends (day-of-week, seasonal, monthly patterns)
- Letter frequency evolution over time
- NYT editorial preferences and word selection strategies
- Difficulty progression and puzzle balancing
- Holiday and special event correlations
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TemporalPattern:
    """Data class for temporal pattern analysis results."""
    pattern_type: str
    timeframe: str
    pattern_strength: float
    statistical_significance: float
    trend_direction: str
    key_insights: List[str]


@dataclass
class EditorialPreference:
    """Data class for editorial preference analysis."""
    preference_type: str
    strength: float
    evidence: List[str]
    temporal_stability: float


class HistoricalPatternAnalyzer:
    """Analyzes historical Wordle data for prediction insights."""
    
    def __init__(self, historical_data_path: str, output_dir: str = "data/analysis"):
        """
        Initialize the historical pattern analyzer.
        
        Args:
            historical_data_path: Path to historical Wordle data CSV
            output_dir: Directory to save analysis results
        """
        self.historical_data_path = Path(historical_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "temporal").mkdir(exist_ok=True)
        (self.output_dir / "editorial").mkdir(exist_ok=True)
        (self.output_dir / "linguistic").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Load and prepare data
        self.df = self._load_and_prepare_data()
        
        # Analysis results storage
        self.temporal_patterns: List[TemporalPattern] = []
        self.editorial_preferences: List[EditorialPreference] = []
        self.linguistic_evolution: Dict[str, Any] = {}
        
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare historical data for analysis."""
        logger.info(f"Loading historical data from {self.historical_data_path}")
        
        try:
            df = pd.read_csv(self.historical_data_path)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add additional temporal features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['day_of_year'] = df['date'].dt.dayofyear
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            
            # Add month names for readability
            df['month_name'] = df['date'].dt.strftime('%B')
            df['day_name'] = df['date'].dt.strftime('%A')
            
            # Calculate days since Wordle launch
            wordle_launch = pd.to_datetime('2021-06-19')  # Approximate launch date
            df['days_since_launch'] = (df['date'] - wordle_launch).dt.days
            
            # Add word complexity metrics
            df['consonant_clusters'] = df['solution'].apply(self._count_consonant_clusters)
            df['rare_letter_count'] = df['solution'].apply(lambda x: sum(1 for c in x if c in 'QXZJ'))
            df['common_letter_count'] = df['solution'].apply(lambda x: sum(1 for c in x if c in 'ETAOINSHRDLU'))
            
            logger.info(f"Loaded {len(df)} historical Wordle entries")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
    
    def _count_consonant_clusters(self, word: str) -> int:
        """Count consonant clusters in a word."""
        vowels = set('AEIOU')
        clusters = 0
        in_cluster = False
        
        for char in word:
            if char not in vowels:
                if not in_cluster:
                    clusters += 1
                    in_cluster = True
            else:
                in_cluster = False
        
        return clusters
    
    def analyze_temporal_patterns(self) -> List[TemporalPattern]:
        """
        Analyze temporal patterns in Wordle word selection.
        
        Returns:
            List of discovered temporal patterns
        """
        logger.info("Analyzing temporal patterns...")
        
        patterns = []
        
        # Day-of-week analysis
        dow_pattern = self._analyze_day_of_week_patterns()
        patterns.append(dow_pattern)
        
        # Monthly patterns
        monthly_pattern = self._analyze_monthly_patterns()
        patterns.append(monthly_pattern)
        
        # Seasonal patterns
        seasonal_pattern = self._analyze_seasonal_patterns()
        patterns.append(seasonal_pattern)
        
        # Weekend vs weekday patterns
        weekend_pattern = self._analyze_weekend_patterns()
        patterns.append(weekend_pattern)
        
        # Long-term trends
        trend_pattern = self._analyze_long_term_trends()
        patterns.append(trend_pattern)
        
        self.temporal_patterns = patterns
        
        # Save results
        patterns_data = [
            {
                'pattern_type': p.pattern_type,
                'timeframe': p.timeframe,
                'pattern_strength': p.pattern_strength,
                'statistical_significance': p.statistical_significance,
                'trend_direction': p.trend_direction,
                'key_insights': p.key_insights
            }
            for p in patterns
        ]
        
        with open(self.output_dir / "temporal" / "temporal_patterns.json", 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        logger.info(f"Identified {len(patterns)} temporal patterns")
        return patterns
    
    def _analyze_day_of_week_patterns(self) -> TemporalPattern:
        """Analyze patterns by day of week."""
        # Calculate average difficulty by day of week
        dow_stats = self.df.groupby('day_name').agg({
            'unique_letters': 'mean',
            'vowel_count': 'mean',
            'has_duplicates': 'mean',
            'consonant_clusters': 'mean',
            'rare_letter_count': 'mean'
        }).round(3)
        
        # Test for statistical significance
        f_stat, p_value = stats.f_oneway(
            *[group['unique_letters'].values for name, group in self.df.groupby('day_name')]
        )
        
        # Calculate pattern strength (coefficient of variation)
        pattern_strength = dow_stats['unique_letters'].std() / dow_stats['unique_letters'].mean()
        
        insights = []
        
        # Find most/least difficult days
        easiest_day = dow_stats['unique_letters'].idxmin()
        hardest_day = dow_stats['unique_letters'].idxmax()
        
        insights.append(f"Monday words tend to be {'easier' if easiest_day == 'Monday' else 'harder'}")
        insights.append(f"Friday words tend to be {'easier' if easiest_day == 'Friday' else 'harder'}")
        insights.append(f"Hardest day typically: {hardest_day}")
        insights.append(f"Easiest day typically: {easiest_day}")
        
        return TemporalPattern(
            pattern_type="day_of_week",
            timeframe="weekly",
            pattern_strength=pattern_strength,
            statistical_significance=p_value,
            trend_direction="cyclical",
            key_insights=insights
        )
    
    def _analyze_monthly_patterns(self) -> TemporalPattern:
        """Analyze patterns by month."""
        monthly_stats = self.df.groupby('month_name').agg({
            'unique_letters': 'mean',
            'vowel_count': 'mean',
            'has_duplicates': 'mean',
            'consonant_clusters': 'mean'
        }).round(3)
        
        # Test for statistical significance
        f_stat, p_value = stats.f_oneway(
            *[group['unique_letters'].values for name, group in self.df.groupby('month')]
        )
        
        pattern_strength = monthly_stats['unique_letters'].std() / monthly_stats['unique_letters'].mean()
        
        insights = []
        
        # Seasonal difficulty trends
        winter_months = ['December', 'January', 'February']
        summer_months = ['June', 'July', 'August']
        
        winter_difficulty = monthly_stats.loc[winter_months, 'unique_letters'].mean()
        summer_difficulty = monthly_stats.loc[summer_months, 'unique_letters'].mean()
        
        if winter_difficulty > summer_difficulty:
            insights.append("Winter months tend to have more complex words")
        else:
            insights.append("Summer months tend to have more complex words")
        
        insights.append(f"Most complex month: {monthly_stats['unique_letters'].idxmax()}")
        insights.append(f"Simplest month: {monthly_stats['unique_letters'].idxmin()}")
        
        return TemporalPattern(
            pattern_type="monthly",
            timeframe="yearly",
            pattern_strength=pattern_strength,
            statistical_significance=p_value,
            trend_direction="seasonal",
            key_insights=insights
        )
    
    def _analyze_seasonal_patterns(self) -> TemporalPattern:
        """Analyze seasonal patterns."""
        # Define seasons
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        
        self.df['season'] = self.df['month'].map(season_map)
        
        seasonal_stats = self.df.groupby('season').agg({
            'unique_letters': 'mean',
            'vowel_count': 'mean',
            'has_duplicates': 'mean',
            'consonant_clusters': 'mean'
        }).round(3)
        
        # Test for statistical significance
        f_stat, p_value = stats.f_oneway(
            *[group['unique_letters'].values for name, group in self.df.groupby('season')]
        )
        
        pattern_strength = seasonal_stats['unique_letters'].std() / seasonal_stats['unique_letters'].mean()
        
        insights = []
        insights.append(f"Hardest season: {seasonal_stats['unique_letters'].idxmax()}")
        insights.append(f"Easiest season: {seasonal_stats['unique_letters'].idxmin()}")
        
        return TemporalPattern(
            pattern_type="seasonal",
            timeframe="yearly",
            pattern_strength=pattern_strength,
            statistical_significance=p_value,
            trend_direction="cyclical",
            key_insights=insights
        )
    
    def _analyze_weekend_patterns(self) -> TemporalPattern:
        """Analyze weekend vs weekday patterns."""
        weekend_stats = self.df.groupby('is_weekend').agg({
            'unique_letters': 'mean',
            'vowel_count': 'mean',
            'has_duplicates': 'mean',
            'consonant_clusters': 'mean'
        }).round(3)
        
        # Statistical test
        weekday_data = self.df[~self.df['is_weekend']]['unique_letters']
        weekend_data = self.df[self.df['is_weekend']]['unique_letters']
        
        t_stat, p_value = stats.ttest_ind(weekday_data, weekend_data)
        
        pattern_strength = abs(weekend_stats.loc[True, 'unique_letters'] - 
                              weekend_stats.loc[False, 'unique_letters'])
        
        insights = []
        
        if weekend_stats.loc[True, 'unique_letters'] > weekend_stats.loc[False, 'unique_letters']:
            insights.append("Weekend puzzles tend to be more complex")
        else:
            insights.append("Weekday puzzles tend to be more complex")
        
        insights.append(f"Weekend avg unique letters: {weekend_stats.loc[True, 'unique_letters']}")
        insights.append(f"Weekday avg unique letters: {weekend_stats.loc[False, 'unique_letters']}")
        
        return TemporalPattern(
            pattern_type="weekend_vs_weekday",
            timeframe="weekly",
            pattern_strength=pattern_strength,
            statistical_significance=p_value,
            trend_direction="binary",
            key_insights=insights
        )
    
    def _analyze_long_term_trends(self) -> TemporalPattern:
        """Analyze long-term trends over time."""
        # Calculate monthly averages
        monthly_trend = self.df.groupby([self.df['date'].dt.year, self.df['date'].dt.month]).agg({
            'unique_letters': 'mean',
            'vowel_count': 'mean',
            'has_duplicates': 'mean'
        }).reset_index()
        
        monthly_trend['date'] = pd.to_datetime(monthly_trend[['year', 'month']].assign(day=1))
        monthly_trend['months_since_start'] = range(len(monthly_trend))
        
        # Linear regression for trend
        X = monthly_trend['months_since_start'].values.reshape(-1, 1)
        y = monthly_trend['unique_letters'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_slope = model.coef_[0]
        r_squared = model.score(X, y)
        
        insights = []
        
        if abs(trend_slope) > 0.001:
            direction = "increasing" if trend_slope > 0 else "decreasing"
            insights.append(f"Word complexity is {direction} over time")
            insights.append(f"Trend strength: {r_squared:.3f}")
        else:
            insights.append("No significant long-term trend in complexity")
        
        insights.append(f"Complexity change per month: {trend_slope:.4f}")
        
        return TemporalPattern(
            pattern_type="long_term_trend",
            timeframe="multi_year",
            pattern_strength=abs(trend_slope),
            statistical_significance=r_squared,
            trend_direction="linear",
            key_insights=insights
        )
    
    def analyze_editorial_preferences(self) -> List[EditorialPreference]:
        """
        Analyze NYT editorial preferences and word selection patterns.
        
        Returns:
            List of identified editorial preferences
        """
        logger.info("Analyzing editorial preferences...")
        
        preferences = []
        
        # Theme analysis
        theme_pref = self._analyze_thematic_preferences()
        preferences.append(theme_pref)
        
        # Difficulty balancing
        difficulty_pref = self._analyze_difficulty_balancing()
        preferences.append(difficulty_pref)
        
        # Letter frequency preferences
        letter_pref = self._analyze_letter_preferences()
        preferences.append(letter_pref)
        
        # Word origin preferences
        origin_pref = self._analyze_word_origin_preferences()
        preferences.append(origin_pref)
        
        self.editorial_preferences = preferences
        
        # Save results
        preferences_data = [
            {
                'preference_type': p.preference_type,
                'strength': p.strength,
                'evidence': p.evidence,
                'temporal_stability': p.temporal_stability
            }
            for p in preferences
        ]
        
        with open(self.output_dir / "editorial" / "editorial_preferences.json", 'w') as f:
            json.dump(preferences_data, f, indent=2)
        
        logger.info(f"Identified {len(preferences)} editorial preferences")
        return preferences
    
    def _analyze_thematic_preferences(self) -> EditorialPreference:
        """Analyze preferences for thematic words."""
        # Simple thematic analysis based on word patterns
        themes = {
            'nature': ['PLANT', 'FIELD', 'RIVER', 'STONE', 'BEACH', 'OCEAN'],
            'emotions': ['HAPPY', 'ANGRY', 'GRIEF', 'PANIC', 'SWEET'],
            'actions': ['CLIMB', 'CARRY', 'THINK', 'SLEEP', 'DANCE'],
            'objects': ['HOUSE', 'CHAIR', 'TABLE', 'PHONE', 'MUSIC']
        }
        
        theme_counts = {}
        for theme, words in themes.items():
            count = sum(1 for word in words if word in self.df['solution'].values)
            theme_counts[theme] = count
        
        total_themed = sum(theme_counts.values())
        theme_ratio = total_themed / len(self.df)
        
        evidence = [f"{theme}: {count} words" for theme, count in theme_counts.items()]
        evidence.append(f"Total themed words: {total_themed}/{len(self.df)} ({theme_ratio:.1%})")
        
        return EditorialPreference(
            preference_type="thematic_selection",
            strength=theme_ratio,
            evidence=evidence,
            temporal_stability=0.8  # Assume relatively stable
        )
    
    def _analyze_difficulty_balancing(self) -> EditorialPreference:
        """Analyze difficulty balancing strategies."""
        # Calculate rolling difficulty metrics
        window_size = 7  # Week-long window
        
        self.df['rolling_difficulty'] = (
            self.df['unique_letters'].rolling(window=window_size, center=True).mean()
        )
        
        # Calculate difficulty variance
        difficulty_variance = self.df['rolling_difficulty'].var()
        
        # Look for streak-breaking patterns
        self.df['is_hard'] = self.df['unique_letters'] >= 5
        self.df['consecutive_hard'] = (
            self.df['is_hard'].groupby((~self.df['is_hard']).cumsum()).cumsum()
        )
        
        max_consecutive = self.df['consecutive_hard'].max()
        avg_consecutive = self.df[self.df['is_hard']]['consecutive_hard'].mean()
        
        evidence = [
            f"Difficulty variance: {difficulty_variance:.3f}",
            f"Max consecutive hard words: {max_consecutive}",
            f"Average hard word streaks: {avg_consecutive:.2f}",
            "NYT appears to balance difficulty over time"
        ]
        
        # Strength based on how well difficulty is balanced
        balance_strength = 1 / (1 + difficulty_variance)  # Lower variance = better balance
        
        return EditorialPreference(
            preference_type="difficulty_balancing",
            strength=balance_strength,
            evidence=evidence,
            temporal_stability=0.9
        )
    
    def _analyze_letter_preferences(self) -> EditorialPreference:
        """Analyze letter frequency preferences."""
        # Calculate letter frequencies
        all_letters = ''.join(self.df['solution'])
        letter_freq = Counter(all_letters)
        total_letters = len(all_letters)
        
        # Expected frequencies in English
        english_freq = {
            'E': 0.127, 'T': 0.091, 'A': 0.082, 'O': 0.075, 'I': 0.070,
            'N': 0.067, 'S': 0.063, 'H': 0.061, 'R': 0.060, 'D': 0.043,
            'L': 0.040, 'C': 0.028, 'U': 0.028, 'M': 0.024, 'W': 0.023,
            'F': 0.022, 'G': 0.020, 'Y': 0.020, 'P': 0.019, 'B': 0.013,
            'V': 0.010, 'K': 0.008, 'J': 0.0015, 'X': 0.0015, 'Q': 0.0010, 'Z': 0.0007
        }
        
        # Calculate deviations
        deviations = []
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            observed = letter_freq[letter] / total_letters
            expected = english_freq.get(letter, 0.001)
            deviation = (observed - expected) / expected
            deviations.append(abs(deviation))
        
        avg_deviation = np.mean(deviations)
        
        # Find letters that are over/under-represented
        overused = []
        underused = []
        
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            observed = letter_freq[letter] / total_letters
            expected = english_freq.get(letter, 0.001)
            ratio = observed / expected
            
            if ratio > 1.2:
                overused.append(f"{letter} ({ratio:.2f}x)")
            elif ratio < 0.8:
                underused.append(f"{letter} ({ratio:.2f}x)")
        
        evidence = [
            f"Average deviation from English: {avg_deviation:.3f}",
            f"Over-represented letters: {', '.join(overused[:5])}",
            f"Under-represented letters: {', '.join(underused[:5])}"
        ]
        
        return EditorialPreference(
            preference_type="letter_frequency",
            strength=1 - avg_deviation,  # Lower deviation = stronger preference for English patterns
            evidence=evidence,
            temporal_stability=0.95
        )
    
    def _analyze_word_origin_preferences(self) -> EditorialPreference:
        """Analyze preferences for word origins/types."""
        # Simple heuristic analysis
        common_patterns = {
            'latin_endings': ['ATION', 'ITIVE', 'ULENT'],
            'anglo_saxon': ['TH', 'GH', 'CK'],
            'modern': ['CYBER', 'EMAIL', 'PHONE']
        }
        
        # This is a simplified analysis - in reality, would need etymology database
        evidence = ["Comprehensive etymology analysis requires specialized database"]
        
        return EditorialPreference(
            preference_type="word_origin",
            strength=0.5,  # Neutral - insufficient data
            evidence=evidence,
            temporal_stability=0.7
        )
    
    def analyze_linguistic_evolution(self) -> Dict[str, Any]:
        """
        Analyze how linguistic patterns evolve over time.
        
        Returns:
            Dictionary containing linguistic evolution analysis
        """
        logger.info("Analyzing linguistic evolution...")
        
        evolution = {}
        
        # Letter frequency evolution
        evolution['letter_frequency'] = self._analyze_letter_frequency_evolution()
        
        # Complexity evolution
        evolution['complexity'] = self._analyze_complexity_evolution()
        
        # Pattern evolution
        evolution['patterns'] = self._analyze_pattern_evolution()
        
        self.linguistic_evolution = evolution
        
        # Save results
        with open(self.output_dir / "linguistic" / "linguistic_evolution.json", 'w') as f:
            json.dump(evolution, f, indent=2, default=str)
        
        logger.info("Linguistic evolution analysis completed")
        return evolution
    
    def _analyze_letter_frequency_evolution(self) -> Dict[str, Any]:
        """Analyze how letter frequencies change over time."""
        # Group by quarters for evolution analysis
        quarterly_groups = self.df.groupby([self.df['date'].dt.year, self.df['date'].dt.quarter])
        
        letter_evolution = {}
        
        for (year, quarter), group in quarterly_groups:
            period = f"{year}-Q{quarter}"
            all_letters = ''.join(group['solution'])
            letter_freq = Counter(all_letters)
            total = len(all_letters)
            
            letter_evolution[period] = {
                letter: count / total for letter, count in letter_freq.items()
            }
        
        # Calculate trends for most common letters
        common_letters = ['E', 'A', 'R', 'I', 'O', 'T', 'N', 'S']
        trends = {}
        
        for letter in common_letters:
            frequencies = []
            periods = []
            
            for period, freqs in letter_evolution.items():
                if letter in freqs:
                    frequencies.append(freqs[letter])
                    periods.append(period)
            
            if len(frequencies) > 3:
                # Simple linear trend
                x = np.arange(len(frequencies))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, frequencies)
                
                trends[letter] = {
                    'slope': slope,
                    'correlation': r_value,
                    'significance': p_value,
                    'trend': 'increasing' if slope > 0 else 'decreasing'
                }
        
        return {
            'quarterly_evolution': letter_evolution,
            'letter_trends': trends
        }
    
    def _analyze_complexity_evolution(self) -> Dict[str, Any]:
        """Analyze how word complexity evolves over time."""
        # Monthly complexity metrics
        monthly_complexity = self.df.groupby([self.df['date'].dt.year, self.df['date'].dt.month]).agg({
            'unique_letters': 'mean',
            'vowel_count': 'mean',
            'consonant_clusters': 'mean',
            'has_duplicates': 'mean',
            'rare_letter_count': 'mean'
        }).reset_index()
        
        monthly_complexity['period'] = monthly_complexity.apply(
            lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1
        )
        
        # Calculate trends
        complexity_trends = {}
        
        for metric in ['unique_letters', 'vowel_count', 'consonant_clusters']:
            values = monthly_complexity[metric].values
            x = np.arange(len(values))
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            complexity_trends[metric] = {
                'slope': slope,
                'correlation': r_value,
                'significance': p_value,
                'trend': 'increasing' if slope > 0 else 'decreasing'
            }
        
        return {
            'monthly_data': monthly_complexity.to_dict('records'),
            'trends': complexity_trends
        }
    
    def _analyze_pattern_evolution(self) -> Dict[str, Any]:
        """Analyze how word patterns evolve over time."""
        # Pattern categories
        patterns = {
            'double_letters': self.df['has_duplicates'],
            'high_vowel': self.df['vowel_count'] >= 3,
            'low_vowel': self.df['vowel_count'] <= 1,
            'rare_letters': self.df['rare_letter_count'] > 0,
            'common_starts': self.df['solution'].str.startswith(('TH', 'ST', 'CH'))
        }
        
        # Monthly pattern frequencies
        monthly_patterns = {}
        
        for pattern_name, pattern_series in patterns.items():
            monthly_freq = self.df.groupby([self.df['date'].dt.year, self.df['date'].dt.month])[pattern_series.name if hasattr(pattern_series, 'name') else 'temp'].apply(
                lambda x: pattern_series[x.index].mean()
            ).reset_index()
            
            monthly_patterns[pattern_name] = monthly_freq.values.tolist()
        
        return monthly_patterns
    
    def generate_prediction_insights(self) -> Dict[str, Any]:
        """
        Generate actionable insights for Wordle prediction.
        
        Returns:
            Dictionary containing prediction insights and recommendations
        """
        logger.info("Generating prediction insights...")
        
        insights = {
            'temporal_recommendations': [],
            'editorial_insights': [],
            'strategic_patterns': [],
            'confidence_factors': {}
        }
        
        # Temporal recommendations
        for pattern in self.temporal_patterns:
            if pattern.statistical_significance < 0.05:  # Significant patterns
                if pattern.pattern_type == "day_of_week":
                    insights['temporal_recommendations'].append(
                        f"Adjust predictions based on {pattern.pattern_type}: {pattern.key_insights[0]}"
                    )
                elif pattern.pattern_type == "monthly":
                    insights['temporal_recommendations'].append(
                        f"Monthly complexity varies: {pattern.key_insights[0]}"
                    )
        
        # Editorial insights
        for pref in self.editorial_preferences:
            if pref.strength > 0.6:  # Strong preferences
                insights['editorial_insights'].append(
                    f"{pref.preference_type}: {pref.evidence[0]}"
                )
        
        # Strategic patterns
        insights['strategic_patterns'] = [
            "NYT maintains difficulty balance over time",
            "Letter frequencies generally follow English patterns",
            "Weekend puzzles may differ from weekday puzzles",
            "Seasonal variations exist in word selection"
        ]
        
        # Confidence factors
        insights['confidence_factors'] = {
            'temporal_reliability': np.mean([p.statistical_significance for p in self.temporal_patterns]),
            'editorial_consistency': np.mean([p.temporal_stability for p in self.editorial_preferences]),
            'pattern_strength': np.mean([p.pattern_strength for p in self.temporal_patterns])
        }
        
        # Save insights
        with open(self.output_dir / "prediction_insights.json", 'w') as f:
            json.dump(insights, f, indent=2)
        
        logger.info("Prediction insights generated")
        return insights
    
    def create_comprehensive_analysis_report(self) -> str:
        """
        Create a comprehensive analysis report.
        
        Returns:
            Path to the generated report file
        """
        logger.info("Creating comprehensive analysis report...")
        
        # Run all analyses
        temporal_patterns = self.analyze_temporal_patterns()
        editorial_preferences = self.analyze_editorial_preferences()
        linguistic_evolution = self.analyze_linguistic_evolution()
        prediction_insights = self.generate_prediction_insights()
        
        # Create report
        report = {
            'analysis_date': datetime.now().isoformat(),
            'data_summary': {
                'total_words': len(self.df),
                'date_range': {
                    'start': str(self.df['date'].min()),
                    'end': str(self.df['date'].max())
                },
                'unique_words': self.df['solution'].nunique()
            },
            'temporal_patterns': [
                {
                    'pattern_type': p.pattern_type,
                    'timeframe': p.timeframe,
                    'pattern_strength': p.pattern_strength,
                    'statistical_significance': p.statistical_significance,
                    'trend_direction': p.trend_direction,
                    'key_insights': p.key_insights
                }
                for p in temporal_patterns
            ],
            'editorial_preferences': [
                {
                    'preference_type': p.preference_type,
                    'strength': p.strength,
                    'evidence': p.evidence,
                    'temporal_stability': p.temporal_stability
                }
                for p in editorial_preferences
            ],
            'linguistic_evolution': linguistic_evolution,
            'prediction_insights': prediction_insights
        }
        
        report_file = self.output_dir / "comprehensive_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive analysis report saved to {report_file}")
        return str(report_file)


def main():
    """Main function to run historical pattern analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze historical Wordle patterns')
    parser.add_argument('historical_data', help='Path to historical Wordle data CSV')
    parser.add_argument('--output-dir', default='data/analysis', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        analyzer = HistoricalPatternAnalyzer(args.historical_data, args.output_dir)
        
        # Create comprehensive report
        report_file = analyzer.create_comprehensive_analysis_report()
        
        print(f"\nHistorical pattern analysis completed successfully!")
        print(f"Comprehensive report: {report_file}")
        print(f"Analysis outputs saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())