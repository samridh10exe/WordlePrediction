# 🎯 Wordle Prediction ML Project

A comprehensive machine learning system that predicts Wordle answers using advanced NLP techniques, featuring multiple model architectures, real-time web interface, and production-ready deployment.

## 🌟 Features

- **Multiple ML Models**: Frequency-based, information entropy, heuristic, and ensemble approaches
- **Advanced Algorithms**: Transformer networks and reinforcement learning (A2C) agents
- **Web Interface**: Interactive Streamlit frontend with real-time predictions
- **REST API**: FastAPI backend with comprehensive endpoints
- **Comprehensive Evaluation**: Performance metrics aligned with research benchmarks
- **Production Ready**: Docker containerization and CI/CD pipeline

## 🏆 Performance Benchmarks

Based on academic research, our models target:
- **Excellent**: ≤3.5 avg guesses, ≥95% success rate
- **MIT Optimal**: 3.421 avg guesses (theoretical benchmark)
- **Human Average**: 3.9-4.0 avg guesses

## 🚀 Quick Start

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

## 📊 Usage Examples

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

## 🏗️ Project Structure

```
WordlePrediction/
├── src/
│   ├── data/                  # Data collection and preprocessing
│   │   ├── data_collection.py
│   │   └── preprocessing.py
│   ├── features/              # Feature engineering
│   │   └── feature_engineering.py
│   ├── models/                # ML models and training
│   │   ├── baseline_models.py
│   │   ├── advanced_models.py
│   │   ├── training.py
│   │   └── evaluation.py
│   ├── visualization/         # Data visualization utilities
│   └── app/                   # Web application
│       ├── main.py           # FastAPI backend
│       └── streamlit_app.py  # Streamlit frontend
├── data/                     # Data storage
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed datasets
│   └── external/            # External data sources
├── models/                  # Trained model storage
├── tests/                   # Test files
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks
├── config/                  # Configuration files
├── requirements.txt         # Python dependencies
├── Dockerfile              # API container
├── Dockerfile.streamlit    # Frontend container
├── docker-compose.yml      # Multi-service setup
└── .github/workflows/      # CI/CD pipeline
```

## 🧠 Model Architecture

### Baseline Models
- **FrequencyBasedPredictor**: Uses word and letter frequency analysis
- **InformationEntropyPredictor**: Maximizes information gain for predictions
- **HeuristicPredictor**: Game theory and strategic elimination approaches
- **EnsembleBaseline**: Combines multiple baseline approaches

### Advanced Models
- **WordleTransformer**: BERT-based transformer for context understanding
- **WordleRLAgent**: A2C reinforcement learning agent
- **EnsembleAdvanced**: Combines transformer and RL approaches

### Feature Engineering
- **Linguistic Features**: Letter frequency, position patterns, vowel/consonant ratios
- **Temporal Features**: Day-of-week patterns, seasonal trends, puzzle progression
- **Game Theory Features**: Information entropy, elimination power, strategic difficulty

## 📈 Performance Metrics

The system tracks multiple performance indicators:

- **Accuracy**: Direct prediction accuracy
- **Top-k Accuracy**: Correct answer in top k predictions
- **Average Guesses**: Simulated Wordle game performance
- **Success Rate**: Percentage of games solved within 6 guesses
- **Information Metrics**: Entropy, perplexity, confidence scores

## 🔧 Development

### Data Pipeline
```bash
# Collect data
python -m src.data.data_collection

# Preprocess data
python -m src.data.preprocessing

# Engineer features
python -m src.features.feature_engineering
```

### Model Training
```bash
# Train baseline models
python -m src.models.baseline_models

# Train advanced models
python -m src.models.advanced_models

# Run full training pipeline
python -m src.models.training
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

## 🚀 Deployment

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

## 🔍 API Documentation

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

## 📚 Research Background

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

## 🤝 Contributing

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MIT Research**: Optimal Wordle algorithm benchmarks
- **Wordle Community**: Historical data and analysis
- **Open Source Libraries**: scikit-learn, PyTorch, FastAPI, Streamlit
- **Academic Papers**: Research methodologies and evaluation metrics

## 📞 Support

For questions, issues, or contributions:

- **Issues**: Create a GitHub issue
- **Documentation**: Check the `/docs` directory
- **Examples**: See `/notebooks` for Jupyter examples

---

**Built with ❤️ for the Wordle community and ML research**