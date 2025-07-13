"""
Streamlit frontend for Wordle prediction visualization.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import json
from typing import Dict, List, Any
import time
import logging

# Configure page
st.set_page_config(
    page_title="Wordle Prediction Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'api_status' not in st.session_state:
    st.session_state.api_status = None


def check_api_health() -> bool:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = response.json()
            return True
        return False
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        st.session_state.api_status = None
        return False


def get_api_stats() -> Dict[str, Any]:
    """Get API statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        logger.error(f"Failed to get API stats: {e}")
        return {}


def get_prediction(target_date: str = None, num_predictions: int = 5) -> Dict[str, Any]:
    """Get prediction from API."""
    try:
        payload = {
            "date": target_date,
            "num_predictions": num_predictions,
            "context": {"source": "streamlit_app"}
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}
            
    except Exception as e:
        st.error(f"Failed to get prediction: {e}")
        return {}


def evaluate_model(model_name: str = None) -> Dict[str, Any]:
    """Evaluate model performance."""
    try:
        params = {"model_name": model_name} if model_name else {}
        response = requests.post(
            f"{API_BASE_URL}/evaluate",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Evaluation Error: {response.status_code}")
            return {}
            
    except Exception as e:
        st.error(f"Failed to evaluate model: {e}")
        return {}


def main():
    """Main application."""
    st.title("🎯 Wordle Prediction Engine")
    st.markdown("Predict tomorrow's Wordle answer using machine learning!")
    
    # Check API status
    api_healthy = check_api_health()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Status
        if api_healthy:
            st.success("✅ API Connected")
            if st.session_state.api_status:
                st.json(st.session_state.api_status)
        else:
            st.error("❌ API Disconnected")
            st.warning("Please ensure the FastAPI server is running on localhost:8000")
            return
        
        # Prediction settings
        st.subheader("Prediction Settings")
        prediction_date = st.date_input(
            "Target Date",
            value=date.today() + timedelta(days=1),
            help="Date for which to predict the Wordle answer"
        )
        
        num_predictions = st.slider(
            "Number of Predictions",
            min_value=1,
            max_value=10,
            value=5,
            help="How many top predictions to show"
        )
        
        # Model selection (if multiple models available)
        stats = get_api_stats()
        available_models = []
        if stats and 'models_available' in stats:
            available_models = [model['name'] for model in stats['models_available'] if model['loaded']]
        
        if available_models:
            selected_model = st.selectbox(
                "Model",
                options=['Auto'] + available_models,
                help="Select which model to use for prediction"
            )
        else:
            selected_model = 'Auto'
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔮 Generate Prediction")
        
        if st.button("🎯 Predict Wordle Answer", type="primary", use_container_width=True):
            with st.spinner("Generating predictions..."):
                prediction_result = get_prediction(
                    target_date=prediction_date.strftime("%Y-%m-%d"),
                    num_predictions=num_predictions
                )
                
                if prediction_result:
                    display_predictions(prediction_result)
                    
                    # Add to history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'date': prediction_date,
                        'result': prediction_result
                    })
        
        # Display latest prediction if available
        if st.session_state.prediction_history:
            st.subheader("📊 Latest Prediction")
            latest = st.session_state.prediction_history[-1]
            display_predictions(latest['result'])
    
    with col2:
        st.header("📈 Statistics")
        
        # API Statistics
        if stats:
            display_api_statistics(stats)
        
        # Performance metrics
        st.subheader("🏆 Model Performance")
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model performance..."):
                eval_result = evaluate_model()
                if eval_result:
                    display_evaluation_results(eval_result)
    
    # Additional sections
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "📚 Model Info", "🕒 History", "ℹ️ About"])
    
    with tab1:
        display_analytics_dashboard()
    
    with tab2:
        display_model_information()
    
    with tab3:
        display_prediction_history()
    
    with tab4:
        display_about_information()


def display_predictions(prediction_result: Dict[str, Any]):
    """Display prediction results."""
    if not prediction_result or 'predictions' not in prediction_result:
        st.error("No predictions available")
        return
    
    predictions = prediction_result['predictions']
    metadata = prediction_result.get('metadata', {})
    
    st.subheader("🎯 Top Predictions")
    
    # Create prediction display
    for i, pred in enumerate(predictions):
        col1, col2, col3 = st.columns([1, 3, 2])
        
        with col1:
            st.metric(f"#{i+1}", pred['word'])
        
        with col2:
            # Confidence bar
            confidence_pct = pred['confidence'] * 100
            st.progress(pred['confidence'])
            st.caption(f"Confidence: {confidence_pct:.1f}%")
        
        with col3:
            if pred.get('reasoning'):
                st.caption(pred['reasoning'])
    
    # Metadata
    with st.expander("🔍 Prediction Details"):
        st.json(metadata)
    
    # Visualization
    if len(predictions) > 1:
        create_prediction_chart(predictions)


def create_prediction_chart(predictions: List[Dict[str, Any]]):
    """Create visualization of predictions."""
    words = [p['word'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=words,
            y=confidences,
            text=[f"{c:.1%}" for c in confidences],
            textposition='auto',
            marker_color=px.colors.qualitative.Set3[:len(words)]
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence Distribution",
        xaxis_title="Predicted Words",
        yaxis_title="Confidence Score",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_api_statistics(stats: Dict[str, Any]):
    """Display API statistics."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Total Predictions",
            stats.get('total_predictions', 0)
        )
    
    with col2:
        st.metric(
            "Models Available",
            len(stats.get('models_available', []))
        )
    
    # Data statistics
    data_stats = stats.get('data_statistics', {})
    if data_stats:
        st.metric(
            "Vocabulary Size",
            data_stats.get('total_words', 0)
        )


def display_evaluation_results(eval_result: Dict[str, Any]):
    """Display model evaluation results."""
    st.subheader("🎯 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy = eval_result.get('accuracy', 0)
        st.metric(
            "Accuracy",
            f"{accuracy:.1%}",
            help="Percentage of correct predictions"
        )
    
    with col2:
        avg_guesses = eval_result.get('average_guesses', 0)
        st.metric(
            "Avg Guesses",
            f"{avg_guesses:.2f}",
            help="Average number of guesses to solve"
        )
    
    with col3:
        success_rate = eval_result.get('success_rate', 0)
        st.metric(
            "Success Rate",
            f"{success_rate:.1%}",
            help="Percentage of games solved"
        )
    
    # Performance classification
    if avg_guesses <= 3.5 and success_rate >= 0.95:
        performance_level = "🥇 Excellent"
        performance_color = "green"
    elif avg_guesses <= 3.9 and success_rate >= 0.90:
        performance_level = "🥈 Good"
        performance_color = "blue"
    elif avg_guesses <= 4.2 and success_rate >= 0.85:
        performance_level = "🥉 Average"
        performance_color = "orange"
    else:
        performance_level = "📈 Needs Improvement"
        performance_color = "red"
    
    st.markdown(f"**Performance Level:** :{performance_color}[{performance_level}]")


def display_analytics_dashboard():
    """Display analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.prediction_history:
        st.info("No prediction history available yet. Make some predictions to see analytics!")
        return
    
    # Prediction frequency over time
    history_df = pd.DataFrame([
        {
            'timestamp': pred['timestamp'],
            'date': pred['date'],
            'top_prediction': pred['result']['predictions'][0]['word'] if pred['result'].get('predictions') else 'N/A',
            'confidence': pred['result']['predictions'][0]['confidence'] if pred['result'].get('predictions') else 0
        }
        for pred in st.session_state.prediction_history
    ])
    
    if not history_df.empty:
        # Time series of predictions
        fig_timeline = px.line(
            history_df,
            x='timestamp',
            y='confidence',
            title="Prediction Confidence Over Time"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Most frequent predictions
        word_counts = history_df['top_prediction'].value_counts().head(10)
        if not word_counts.empty:
            fig_freq = px.bar(
                x=word_counts.index,
                y=word_counts.values,
                title="Most Frequently Predicted Words"
            )
            st.plotly_chart(fig_freq, use_container_width=True)


def display_model_information():
    """Display model information."""
    st.header("🤖 Model Information")
    
    # Get model information from API
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            models_info = response.json()
            
            if 'models' in models_info:
                for name, info in models_info['models'].items():
                    with st.expander(f"📋 {name}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Type:** {info['type']}")
                            st.write(f"**Loaded:** {'✅' if info['loaded'] else '❌'}")
                        
                        with col2:
                            st.write(f"**Vocabulary Size:** {info['vocabulary_size']:,}")
                            st.write(f"**Available:** {'✅' if info['available'] else '❌'}")
            
            # Default model
            if 'default_model' in models_info:
                st.info(f"**Default Model:** {models_info['default_model']}")
                
        else:
            st.error("Failed to load model information")
            
    except Exception as e:
        st.error(f"Error fetching model info: {e}")


def display_prediction_history():
    """Display prediction history."""
    st.header("🕒 Prediction History")
    
    if not st.session_state.prediction_history:
        st.info("No prediction history available.")
        return
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()
    
    # Display history
    for i, pred in enumerate(reversed(st.session_state.prediction_history)):
        with st.expander(f"Prediction {len(st.session_state.prediction_history) - i} - {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            st.write(f"**Target Date:** {pred['date']}")
            
            if pred['result'].get('predictions'):
                top_pred = pred['result']['predictions'][0]
                st.write(f"**Top Prediction:** {top_pred['word']} ({top_pred['confidence']:.1%} confidence)")
                
                # Show all predictions
                predictions_df = pd.DataFrame(pred['result']['predictions'])
                st.dataframe(predictions_df, use_container_width=True)


def display_about_information():
    """Display about information."""
    st.header("ℹ️ About Wordle Prediction Engine")
    
    st.markdown("""
    This application uses machine learning models to predict Wordle answers based on:
    
    - **Historical Patterns**: Analysis of past Wordle answers
    - **Linguistic Features**: Letter frequency, position patterns, and word structure
    - **Game Theory**: Information entropy and strategic word selection
    - **Multiple Models**: Ensemble of different prediction approaches
    
    ### 🎯 Model Types
    
    - **Frequency-Based**: Uses letter and word frequency analysis
    - **Information Entropy**: Maximizes information gain for word selection  
    - **Heuristic**: Game theory and strategic elimination approaches
    - **Ensemble**: Combines multiple models for better predictions
    
    ### 📊 Performance Benchmarks
    
    - **Excellent**: ≤3.5 avg guesses, ≥95% success rate
    - **Good**: ≤3.9 avg guesses, ≥90% success rate  
    - **Average**: ≤4.2 avg guesses, ≥85% success rate
    - **MIT Optimal**: 3.421 avg guesses (theoretical best)
    
    ### 🔧 Technical Stack
    
    - **Backend**: FastAPI with Python ML models
    - **Frontend**: Streamlit for interactive visualization
    - **Models**: scikit-learn, PyTorch, custom algorithms
    - **Data**: Historical Wordle answers and linguistic datasets
    """)
    
    # System information
    with st.expander("🔧 System Information"):
        st.json({
            "API Base URL": API_BASE_URL,
            "Streamlit Version": st.__version__,
            "API Status": "Connected" if check_api_health() else "Disconnected"
        })


if __name__ == "__main__":
    main()