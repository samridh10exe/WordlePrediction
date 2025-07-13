"""
FastAPI backend for Wordle prediction service.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, date
import asyncio
from contextlib import asynccontextmanager
import uvicorn
import os

# Import our models and utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.baseline_models import EnsembleBaseline
from models.evaluation import WordleEvaluator
from data.data_collection import WordleDataCollector


# Global variables for loaded models and data
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    await load_models_and_data()
    yield
    # Shutdown
    app_state.clear()


app = FastAPI(
    title="Wordle Prediction API",
    description="ML-powered Wordle answer prediction service",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class PredictionRequest(BaseModel):
    date: Optional[str] = Field(None, description="Target date for prediction (YYYY-MM-DD)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for prediction")
    num_predictions: int = Field(5, ge=1, le=20, description="Number of top predictions to return")


class WordPrediction(BaseModel):
    word: str = Field(..., description="Predicted word")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(None, description="Explanation for this prediction")


class PredictionResponse(BaseModel):
    predictions: List[WordPrediction] = Field(..., description="List of word predictions")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    data_loaded: bool = Field(..., description="Whether data is loaded")
    timestamp: str = Field(..., description="Health check timestamp")


class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    vocabulary_size: int = Field(..., description="Vocabulary size")
    loaded: bool = Field(..., description="Whether model is loaded")


class StatsResponse(BaseModel):
    total_predictions: int = Field(..., description="Total predictions made")
    models_available: List[ModelInfo] = Field(..., description="Available models")
    data_statistics: Dict[str, Any] = Field(..., description="Data statistics")


async def load_models_and_data():
    """Load models and data on startup."""
    logger.info("Loading models and data...")
    
    try:
        # Load models
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        app_state["models"] = {}
        app_state["model_metadata"] = {}
        
        # Try to load saved models
        model_files = {
            "ensemble_baseline": models_dir / "ensemble_baseline.pkl",
            "frequency_basic": models_dir / "frequency_basic.pkl",
            "frequency_position": models_dir / "frequency_position.pkl"
        }
        
        for model_name, model_path in model_files.items():
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    app_state["models"][model_name] = model
                    app_state["model_metadata"][model_name] = {
                        "type": type(model).__name__,
                        "vocabulary_size": len(getattr(model, 'vocabulary', [])),
                        "loaded": True
                    }
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
                    app_state["model_metadata"][model_name] = {
                        "type": "Unknown",
                        "vocabulary_size": 0,
                        "loaded": False
                    }
        
        # Load or create default data
        app_state["data"] = await load_default_data()
        
        # Initialize statistics
        app_state["stats"] = {
            "total_predictions": 0,
            "startup_time": datetime.now().isoformat()
        }
        
        logger.info("Models and data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        app_state["models"] = {}
        app_state["data"] = {}


async def load_default_data():
    """Load default dataset."""
    try:
        # Try to load processed data
        data_dir = Path("data/processed")
        combined_data_path = data_dir / "combined_dataset.csv"
        
        if combined_data_path.exists():
            df = pd.read_csv(combined_data_path)
            logger.info(f"Loaded existing dataset with {len(df)} words")
            return df
        
        # If no processed data, create minimal dataset
        logger.info("Creating minimal default dataset...")
        data_collector = WordleDataCollector(Path("data"))
        
        # Get basic word list
        basic_words = [
            'AROSE', 'SLATE', 'CRATE', 'AUDIO', 'ORATE', 'ROATE', 'RAISE', 'SOARE',
            'ABOUT', 'ABOVE', 'ABUSE', 'ACTOR', 'ACUTE', 'ADMIT', 'ADOPT', 'ADULT',
            'AFTER', 'AGAIN', 'AGENT', 'AGREE', 'AHEAD', 'ALARM', 'ALBUM', 'ALERT',
            'ALIEN', 'ALIGN', 'ALIKE', 'ALIVE', 'ALLOW', 'ALONE', 'ALONG', 'ALTER',
            'AMBER', 'AMEND', 'ANGER', 'ANGLE', 'ANGRY', 'APART', 'APPLE', 'APPLY',
            'ARENA', 'ARGUE', 'ARISE', 'ARRAY', 'ASIDE', 'ASSET', 'AVOID', 'AWAKE',
            'AWARD', 'AWARE', 'BADLY', 'BASIC', 'BATCH', 'BEACH', 'BEGAN', 'BEGIN',
            'BEING', 'BELOW', 'BENCH', 'BIRTH', 'BLACK', 'BLAME', 'BLANK', 'BLAST',
            'BLIND', 'BLOCK', 'BLOOD', 'BOARD', 'BOOST', 'BOOTH', 'BOUND', 'BRAIN'
        ]
        
        df = pd.DataFrame({
            'word': basic_words,
            'frequency': np.random.uniform(0.001, 0.1, len(basic_words)),
            'answer_id': range(1, len(basic_words) + 1)
        })
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame({'word': ['AROSE'], 'frequency': [0.1], 'answer_id': [1]})


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Wordle Prediction API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=len(app_state.get("models", {})) > 0,
        data_loaded=not app_state.get("data", pd.DataFrame()).empty,
        timestamp=datetime.now().isoformat()
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get service statistics."""
    models_info = []
    for name, metadata in app_state.get("model_metadata", {}).items():
        models_info.append(ModelInfo(
            name=name,
            type=metadata["type"],
            vocabulary_size=metadata["vocabulary_size"],
            loaded=metadata["loaded"]
        ))
    
    data_stats = {}
    if not app_state.get("data", pd.DataFrame()).empty:
        df = app_state["data"]
        data_stats = {
            "total_words": len(df),
            "unique_words": df['word'].nunique() if 'word' in df.columns else 0,
            "columns": list(df.columns)
        }
    
    return StatsResponse(
        total_predictions=app_state.get("stats", {}).get("total_predictions", 0),
        models_available=models_info,
        data_statistics=data_stats
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_wordle(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Predict next Wordle answer."""
    try:
        logger.info(f"Prediction request: {request}")
        
        # Validate models are loaded
        if not app_state.get("models"):
            raise HTTPException(status_code=503, detail="No models loaded")
        
        # Get best available model
        model_name, model = get_best_model()
        if not model:
            raise HTTPException(status_code=503, detail="No functional models available")
        
        # Prepare input data
        input_data = prepare_prediction_input(request)
        
        # Make prediction
        predictions = await make_prediction(model, input_data, request.num_predictions)
        
        # Create metadata
        metadata = {
            "model_used": model_name,
            "input_date": request.date,
            "context_provided": request.context is not None,
            "prediction_method": "ml_ensemble"
        }
        
        # Update statistics
        background_tasks.add_task(update_prediction_stats)
        
        return PredictionResponse(
            predictions=predictions,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List available models and their information."""
    models_info = {}
    
    for name, metadata in app_state.get("model_metadata", {}).items():
        models_info[name] = {
            "type": metadata["type"],
            "vocabulary_size": metadata["vocabulary_size"],
            "loaded": metadata["loaded"],
            "available": name in app_state.get("models", {})
        }
    
    return {
        "models": models_info,
        "default_model": get_best_model()[0] if app_state.get("models") else None,
        "total_models": len(models_info)
    }


@app.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_model_performance(model_name: Optional[str] = None):
    """Evaluate model performance on test data."""
    try:
        # Get model
        if model_name and model_name in app_state.get("models", {}):
            model = app_state["models"][model_name]
        else:
            model_name, model = get_best_model()
        
        if not model:
            raise HTTPException(status_code=404, detail="No models available for evaluation")
        
        # Prepare test data
        df = app_state.get("data", pd.DataFrame())
        if df.empty:
            raise HTTPException(status_code=503, detail="No data available for evaluation")
        
        # Create evaluator
        evaluator = WordleEvaluator()
        
        # Run basic evaluation
        test_words = df['word'].tolist()[:50]  # Sample for quick evaluation
        test_df = df.head(50)
        
        # Make predictions
        predictions = model.predict(test_df)
        
        # Calculate basic metrics
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(test_words, predictions)
        
        # Simple game simulation
        game_results = []
        for word in test_words[:10]:  # Small sample
            result = simulate_simple_game(model, word)
            game_results.append(result)
        
        avg_guesses = np.mean([r['guesses'] for r in game_results])
        success_rate = np.mean([r['solved'] for r in game_results])
        
        evaluation_results = {
            "model_name": model_name,
            "accuracy": accuracy,
            "average_guesses": avg_guesses,
            "success_rate": success_rate,
            "sample_size": len(test_words),
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


def get_best_model():
    """Get the best available model."""
    models = app_state.get("models", {})
    
    # Priority order for models
    priority = ["ensemble_baseline", "frequency_position", "frequency_basic"]
    
    for model_name in priority:
        if model_name in models:
            return model_name, models[model_name]
    
    # If no priority model, return first available
    if models:
        first_model = next(iter(models.items()))
        return first_model
    
    return None, None


def prepare_prediction_input(request: PredictionRequest) -> pd.DataFrame:
    """Prepare input data for prediction."""
    # Get data
    df = app_state.get("data", pd.DataFrame())
    
    if df.empty:
        # Create minimal input
        return pd.DataFrame({'word': ['AROSE']})
    
    # If date provided, try to find corresponding data
    if request.date:
        try:
            target_date = datetime.strptime(request.date, "%Y-%m-%d").date()
            # For simplicity, just return the dataset
            # In a real implementation, would filter by date
            return df.head(10)  # Sample
        except ValueError:
            logger.warning(f"Invalid date format: {request.date}")
    
    # Return sample of data
    return df.head(10)


async def make_prediction(model, input_data: pd.DataFrame, num_predictions: int) -> List[WordPrediction]:
    """Make prediction using the model."""
    try:
        # Get top predictions
        if hasattr(model, 'get_top_predictions'):
            top_predictions = model.get_top_predictions(input_data, k=num_predictions)
            if top_predictions:
                predictions = []
                for word, confidence in top_predictions[0]:
                    predictions.append(WordPrediction(
                        word=word,
                        confidence=float(confidence),
                        reasoning=f"ML prediction based on {type(model).__name__}"
                    ))
                return predictions
        
        # Fallback to basic prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)
            if len(probabilities) > 0 and hasattr(model, 'vocabulary'):
                # Get top k predictions
                top_indices = np.argsort(probabilities[0])[-num_predictions:][::-1]
                predictions = []
                for idx in top_indices:
                    if idx < len(model.vocabulary):
                        word = model.vocabulary[idx]
                        confidence = float(probabilities[0][idx])
                        predictions.append(WordPrediction(
                            word=word,
                            confidence=confidence,
                            reasoning="Probability-based prediction"
                        ))
                return predictions
        
        # Final fallback
        prediction = model.predict(input_data)
        if len(prediction) > 0:
            return [WordPrediction(
                word=str(prediction[0]),
                confidence=0.5,
                reasoning="Basic model prediction"
            )]
        
        # Last resort
        return [WordPrediction(
            word="AROSE",
            confidence=0.1,
            reasoning="Default prediction"
        )]
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return [WordPrediction(
            word="AROSE",
            confidence=0.1,
            reasoning=f"Error fallback: {str(e)}"
        )]


def simulate_simple_game(model, target_word: str, max_guesses: int = 6) -> Dict[str, Any]:
    """Simple game simulation for evaluation."""
    guesses = 0
    solved = False
    
    # Simple simulation - just check if model can predict the target
    try:
        sample_input = pd.DataFrame({'word': [target_word]})
        prediction = model.predict(sample_input)
        
        if len(prediction) > 0 and prediction[0] == target_word:
            guesses = 1
            solved = True
        else:
            guesses = max_guesses
            solved = False
            
    except Exception:
        guesses = max_guesses
        solved = False
    
    return {
        'target': target_word,
        'guesses': guesses,
        'solved': solved
    }


async def update_prediction_stats():
    """Update prediction statistics."""
    if "stats" in app_state:
        app_state["stats"]["total_predictions"] += 1


if __name__ == "__main__":
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )