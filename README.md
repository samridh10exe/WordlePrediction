# Wordle Prediction ML Project

A research-grade machine learning system that predicts Wordle answers using comprehensive data analysis, advanced ensemble methods, and production-ready optimization. Targets MIT optimal performance benchmarks with sophisticated feature engineering and evaluation frameworks.

## Features

- **Comprehensive Data Collection**: Official Wordle vocabulary (12,972+ words), historical patterns, linguistic features
- **Advanced Feature Engineering**: Position-specific analysis, phonetic patterns, game-theory optimization
- **Sophisticated ML Models**: Ensemble approaches, transformer architecture, reinforcement learning
- **Temporal Validation**: Proper time-series splits respecting chronological dependencies
- **Research Benchmarking**: MIT optimal performance targeting, statistical significance testing
- **Production Optimization**: Intelligent caching, performance monitoring, auto-scaling

## Performance Benchmarks

Research-grade performance targets:
- **Top-1 Accuracy**: ≥60% (direct word prediction)
- **Top-5 Accuracy**: ≥85% (word in top 5 predictions)
- **Average Guesses**: ≤3.8 (simulated game performance)
- **Success Rate**: ≥95% (solved within 6 guesses)
- **MIT Benchmark**: Target 3.421 avg guesses (optimal benchmark)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd WordlePrediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup data directories**
```bash
mkdir -p data/raw data/processed data/external
mkdir -p models logs results
```

### Running with Docker (Recommended)

1. **Build and run services**
```bash
docker-compose up --build
```

2. **Access the application**
- API: http://localhost:8000
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs

### Running Locally

1. **Start the API server**
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Start the Streamlit frontend** (in another terminal)
```bash
streamlit run src/app/streamlit_app.py --server.port 8501
```

## Usage Examples

### API Usage

```python
import requests

# Get prediction
response = requests.post("http://localhost:8000/predict", json={
    "date": "2024-01-15",
    "num_predictions": 5
})

predictions = response.json()["predictions"]
print(f"Top prediction: {predictions[0]['word']} ({predictions[0]['confidence']:.1%})")
```

### Training Models

```python
from src.models.training import ModelTrainer
from src.data.data_collection import WordleDataCollector
import pandas as pd

# Collect data
collector = WordleDataCollector(Path("data"))
data = collector.collect_all_data()

# Train models
config = {"train_advanced": False}
trainer = ModelTrainer(config)
results = trainer.full_training_pipeline(data["combined"])
```

### Model Evaluation

```python
from src.models.evaluation import WordleEvaluator
import joblib

# Load model
model = joblib.load("models/ensemble_baseline.pkl")

# Evaluate
evaluator = WordleEvaluator()
results = evaluator.comprehensive_evaluation(model, X_test, y_test, vocabulary)
print(f"Average guesses: {results['game_simulation']['avg_guesses']:.3f}")
```

## Project Structure

```
WordlePrediction/
├── src/
│   ├── data/                    # Comprehensive data collection
│   │   └── vocabulary_collector.py  # Multi-source vocabulary collection
│   ├── analysis/                # Historical pattern analysis
│   │   └── historical_patterns.py   # Temporal and editorial analysis
│   ├── features/                # Advanced feature engineering
│   │   └── advanced_feature_engineering.py  # 50+ linguistic features
│   ├── models/                  # Sophisticated ML models
│   │   └── ensemble_predictor.py    # Multi-model ensemble system
│   ├── training/                # Training and validation
│   │   └── validation_strategy.py   # Temporal cross-validation
│   ├── evaluation/              # Benchmarking system
│   │   └── benchmarking.py         # MIT performance comparison
│   └── production/              # Production optimization
│       └── optimization.py        # Caching and monitoring
├── data/                       # Multi-tier data storage
│   ├── vocabulary/             # Comprehensive word databases
│   ├── analysis/               # Historical pattern results
│   ├── features/               # Engineered feature sets
│   └── raw/                    # Source data files
├── models/                     # Trained model artifacts
├── validation_results/         # Validation outputs
├── benchmark_results/          # Benchmark evaluations
├── production_data/            # Production metrics
└── requirements.txt            # Dependencies
```

## System Architecture

### Phase 1: Comprehensive Data Collection
- **Official Wordle Vocabulary**: 2,315 answers + 12,972 valid guesses
- **Multi-Source Frequency Data**: Google N-grams, OpenSubtitles, common word lists
- **Linguistic Database**: WordNet features, CMU Pronouncing Dictionary
- **Word Embeddings**: GloVe/FastText pre-trained vectors

### Phase 2: Historical Pattern Analysis  
- **Temporal Analysis**: 1,290+ historical puzzles with statistical testing
- **Editorial Preferences**: NYT selection patterns and difficulty balancing
- **Linguistic Evolution**: Letter frequency and complexity trends over time

### Phase 3: Advanced Feature Engineering
- **Position-Specific Features**: Letter frequency by position, transition probabilities
- **Phonetic Features**: Syllable analysis, consonant clusters, stress patterns
- **Game-Theory Features**: Information entropy, elimination power, strategic value
- **Semantic Features**: Word embeddings, similarity measures, category scores

### Phase 4: Ensemble ML Models
- **Random Forest**: Optimized hyperparameters with OOB validation
- **Gradient Boosting**: Early stopping and regularization
- **XGBoost/LightGBM**: Advanced boosting with categorical features
- **Neural Networks**: Multi-layer perceptrons with adaptive learning
- **Meta-Learning**: Stacking ensemble with cross-validation

### Phase 5: Temporal Validation Strategy
- **Time-Series Splits**: Chronological validation preserving temporal order
- **Bayesian Optimization**: Hyperparameter tuning with acquisition functions
- **Statistical Testing**: Confidence intervals and significance analysis

### Phase 6: Research Benchmarking
- **MIT Optimal Comparison**: Target 3.421 average guesses benchmark
- **Statistical Significance**: Bootstrap confidence intervals, Mann-Whitney tests
- **Robustness Testing**: Performance across word categories and edge cases
- **Baseline Comparisons**: Human performance, frequency-based, random strategies

### Phase 7: Production Optimization
- **Intelligent Caching**: LRU memory cache with Redis distributed option
- **Performance Monitoring**: Real-time metrics, alerting, auto-scaling
- **Feedback Loops**: Continuous improvement with user satisfaction tracking

## Development

### Data Collection and Analysis
```bash
# Phase 1: Comprehensive vocabulary collection
python3 src/data/vocabulary_collector.py --verbose

# Phase 2: Historical pattern analysis
python3 src/analysis/historical_patterns.py data/raw/wordle_data.csv --verbose

# Phase 3: Advanced feature engineering
python3 src/features/advanced_feature_engineering.py --verbose
```

### Model Training and Validation
```bash
# Phase 4: Ensemble model training
python3 src/models/ensemble_predictor.py --verbose

# Phase 5: Temporal validation strategy
python3 src/training/validation_strategy.py --verbose

# Phase 6: Comprehensive benchmarking
python3 src/evaluation/benchmarking.py --verbose
```

### Production Deployment
```bash
# Phase 7: Production optimization
python3 src/production/optimization.py --verbose
```

### Testing
```bash
# Run tests
pytest tests/ -v --cov=src/

# Format code
black src/

# Type checking
mypy src/ --ignore-missing-imports
```

## Deployment

### Local Development
```bash
# API only
uvicorn src.app.main:app --reload

# Frontend only
streamlit run src/app/streamlit_app.py

# Both services
docker-compose up
```

### Production Deployment
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with scaling
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

### Cloud Deployment
The project includes configurations for:
- **Docker Hub**: Automated image builds
- **GitHub Actions**: CI/CD pipeline
- **Kubernetes**: Production orchestration (k8s/ directory)

## API Documentation

### Core Endpoints

- `GET /`: Service information
- `GET /health`: Health check
- `POST /predict`: Generate predictions
- `GET /stats`: Service statistics
- `GET /models`: Available models
- `POST /evaluate`: Model evaluation

### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Get prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-01-15", "num_predictions": 5}'

# Evaluate model
curl -X POST http://localhost:8000/evaluate
```

## Research Background

This project implements findings from multiple academic papers:

- **MIT Optimal Algorithm**: 3.421 average guesses using dynamic programming
- **A2C Reinforcement Learning**: Andrew Ho's implementation achieving ~99% win rates
- **Information Theory Approaches**: Entropy-based word selection strategies
- **Ensemble Methods**: Combining multiple approaches for robust predictions

### Key Papers
- Bertsimas & Paskov: "Optimal Wordle Solutions" (MIT)
- Ho: "Reinforcement Learning for Wordle" 
- Weng et al.: "ARIMAX and Neural Networks for Wordle Analysis"
- Xin et al.: "Time Series Classification of Wordle Words"

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive docstrings
- Write unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MIT Research**: Optimal Wordle algorithm benchmarks
- **Wordle Community**: Historical data and analysis
- **Open Source Libraries**: scikit-learn, PyTorch, FastAPI, Streamlit
- **Academic Papers**: Research methodologies and evaluation metrics

## Support

For questions, issues, or contributions:

- **Issues**: Create a GitHub issue
- **Documentation**: Check the `/docs` directory
- **Examples**: See `/notebooks` for Jupyter examples

---

**Built for the Wordle community and ML research**