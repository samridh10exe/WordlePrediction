# Comprehensive Wordle ML Resources Directory

The landscape for building Wordle prediction ML projects is remarkably rich, with **over 50 high-quality datasets, APIs, and tools** available for academic research and open source development. **Most resources are freely available under permissive licenses**, making them ideal for research applications. The ecosystem spans from basic word frequency data to sophisticated reinforcement learning implementations, with **MIT's optimal algorithm achieving 3.421 average guesses** serving as the performance benchmark.

This comprehensive directory covers five essential categories of resources, each offering multiple high-quality options with detailed integration instructions. The combination of academic rigor, practical datasets, and production-ready tools creates an excellent foundation for developing state-of-the-art Wordle prediction systems.

## Wordle-specific datasets unlock game mechanics and historical patterns

### Official word lists and historical data

**GitHub - boompig/wordle-py** stands as the most comprehensive resource, providing the complete Wordle dataset with 2,315 answer words and 12,972 valid guesses in text format. The repository includes **pre-computed decision trees (90MB compressed)** and optimization algorithms achieving 100% solve rate with 3.58 average guesses. Access via `git clone https://github.com/boompig/wordle-py`, with data files in `data-raw/` and processed matrices in `data-parsed/`. Licensed under open source terms, this repository offers **immediate integration** with Python ML pipelines and includes solver validation frameworks.

**GitHub - steve-kasica/wordle-words** provides enhanced word lists with **Google Books N-gram frequency metadata**, linking each word to occurrence rates and historical usage patterns. The CSV format includes fields for word, occurrence probability, and answer dates. This dataset excels for **linguistic analysis and frequency-based ML features**, available at `https://github.com/steve-kasica/wordle-words` under open source licensing.

**Official Wordle dictionaries** extracted from the original JavaScript source code are available through GitHub Gists. The **DevilXD/6ad6cc1fe37872d069a795edd51233b2** gist contains the original 2,315 answer words (wordle_words.txt, ~30KB), while **scholtes/94f3c0303ba6a7768b47583aff36654d** provides structured dictionaries with 2,315 answer words (La) and 10,657 guess-only words (Ta). Both are **public domain** and ideal for baseline pre-NYT analysis.

### Community datasets and gameplay analysis

**Kaggle's Wordle ecosystem** offers multiple high-quality datasets. The **Wordle Games Dataset** (scarcalvetsis/wordle-games) contains real Twitter gameplay data in CSV format under CC0 license, perfect for **user behavior analysis**. The **Wordle Valid Words** dataset (bcruise/wordle-valid-words) provides comprehensive word lists with statistical analysis, while **Wordle 5 Letter Words** (cprosser3/wordle-5-letter-words) includes character and position probability analysis ideal for **probabilistic solving algorithms**.

**Historical answer archives** are maintained at multiple sites including YourDictionary, TechRadar, and Five Forks, offering **chronological and alphabetical** listings easily accessible via web scraping. These sources provide **regularly updated historical data** with high accuracy for timeline analysis.

### NYT transition and data evolution

The **NYT acquisition** reduced the answer list from 2,315 to 2,309 words, removing controversial terms like "slave," "lynch," and "wench." The original NYT API (`https://www.nytimes.com/svc/wordle/v2/{YYYY-MM-DD}.json`) is deprecated, but **browser developer tools** can still access daily solutions via network inspection. This transition creates **distinct datasets** for pre-NYT versus current analysis.

## Word frequency and linguistic datasets provide foundational language understanding

### Google Books N-gram datasets establish frequency baselines

**Google Books N-gram datasets** represent the gold standard for English word frequency analysis, containing **1.99 trillion words from 1800-2019** under Creative Commons Attribution 3.0 license. Access via `https://storage.googleapis.com/books/ngrams/books/datasetsv3.html` with files organized by n-gram length (1-5) and alphabetical segments. The **orgtre/google-books-ngram-frequency** GitHub repository provides cleaned, processed lists: 10,000 most frequent 1-grams, 5,000 2-grams, 3,000 3-grams, 1,000 4-grams, and 1,000 5-grams in CSV format.

**Integration approach**: Download specific n-gram files using `wget http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-1gram-20120701-[a-z].gz` and process using provided Python scripts. The **high-quality corpus coverage** makes this ideal for primary frequency analysis, though note some scientific literature bias.

### OpenSubtitles corpus captures conversational patterns

**OpenSubtitles2018** provides **22.10 billion tokens** across 62 languages, offering excellent coverage of spoken/conversational English patterns. The **orgtre/top-open-subtitles-sentences** GitHub repository contains cleaned frequency lists for 30,000 most common words in CSV format. This dataset **complements written text corpora** by providing popularity-weighted conversational frequency data, essential for understanding natural language usage patterns in games.

### CMU Pronouncing Dictionary enables phonetic analysis

**CMU Pronouncing Dictionary** contains **134,000+ North American English words** with ARPABET phoneme representations, available under completely unrestricted use. Access via `https://github.com/cmusphinx/cmudict` for raw data, or integrate through `pip install cmudict` or `nltk.download('cmudict')`. The **industry-standard phonetic data** includes stress markers and multiple pronunciations, making it essential for **phonetic pattern analysis** and rhyme detection in word games.

### Semantic embeddings capture word relationships

**GloVe vectors** provide pre-trained word embeddings from Stanford NLP, available in multiple dimensions (50d, 100d, 200d, 300d) and trained on various corpora. The **Wikipedia 2014 + Gigaword 5** model (`http://nlp.stanford.edu/data/glove.6B.zip`) offers excellent general-purpose embeddings, while **Common Crawl models** provide broader coverage. Licensed under Apache 2.0, these embeddings excel at **capturing semantic relationships** crucial for word similarity analysis.

**FastText models** (`https://fasttext.cc/docs/en/crawl-vectors.html`) provide **300-dimensional vectors** for 157 languages trained on Common Crawl and Wikipedia. The key advantage is **subword information** enabling handling of out-of-vocabulary words through character-level patterns. Integration via `import fasttext; ft = fasttext.load_model('cc.en.300.bin')` provides vectors for any word including those not in training data.

### WordNet provides linguistic structure

**WordNet** offers **155,327 words in 175,979 synsets** with comprehensive semantic relations including hypernymy, hyponymy, meronymy, and antonymy. Available through `https://wordnet.princeton.edu/` under unrestricted WordNet License, with NLTK integration via `nltk.download('wordnet')`. The **semantic relationship data** enables sophisticated word similarity analysis and sense disambiguation critical for advanced Wordle solving strategies.

## Academic research provides peer-reviewed methodologies and benchmarks

### ArXiv research establishes theoretical foundations

**"Puzzle game: Prediction and Classification of Wordle Solution Words"** (arXiv:2403.19433) by Haidong Xin et al. provides **time series analysis** of Wordle data with ARIMA modeling and XGBoost regression for word classification. The paper includes **raw gameplay data** with word attributes including usage frequency, information entropy, and repeated letter analysis. This **peer-reviewed mathematical modeling** offers rigorous validation methods for prediction accuracy.

**"Prediction Model For Wordle Game Results With High Robustness"** (arXiv:2309.14250) by Jiaqi Weng et al. implements **ARIMAX modeling** with backpropagation neural networks and K-means clustering for temporal gameplay analysis. The research includes **statistical validation** with ADF, ACF, and PACF tests, providing robust methodology for time series prediction approaches.

### Language modeling benchmarks enable standardized evaluation

**One Billion Word Benchmark** (`https://www.statmt.org/lm-benchmark/`) provides **1 billion words from WMT 2011 News Crawl** with baseline model scores and standardized evaluation protocols. The benchmark includes **preprocessing scripts** and held-out data available through `https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark`. Licensed for open research use, this benchmark enables **standardized comparison** of language modeling approaches for word prediction tasks.

**Papers with Code** (`https://paperswithcode.com/task/language-modelling`) aggregates **comprehensive language modeling benchmarks** including WikiText-103, Penn Treebank, and Text8. These datasets provide **standardized evaluation protocols** with performance leaderboards, enabling direct comparison of model architectures and training approaches.

### Vocabulary prediction research offers specialized methods

**ESL Vocabulary Knowledge Dataset** (`http://yoehara.com/vocabulary-prediction/`) contains **16 ESL learners** tested on 11,999 English words plus 1 pseudo word, providing **human vocabulary prediction data**. Published in COLING 2012, EMNLP 2014, and IJCAI 2016, this dataset enables **educational applications** and human-AI comparison studies under research-only licensing.

## Production-ready APIs and tools accelerate development

### Core NLP libraries provide essential processing

**NLTK** offers comprehensive NLP capabilities including WordNet integration, frequency analysis, and extensive corpora. Installation via `pip install nltk` with data downloads through `nltk.download()` commands. The **completely free and open-source** library provides excellent research and prototyping capabilities with extensive documentation and community support. Key features include `FreqDist` for frequency analysis and seamless integration with scikit-learn for ML pipelines.

**spaCy** delivers **production-ready NLP** with pre-trained models and statistical components. Installation via `pip install spacy` followed by model downloads (`python -m spacy download en_core_web_sm`) provides immediate access to **industrial-strength text processing**. The library excels in **performance optimization** and seamless integration with modern ML frameworks, supporting custom pipeline components for specialized applications.

### Vocabulary APIs enable real-time word analysis

**Datamuse API** (`https://api.datamuse.com/words`) provides **word-finding query engine** with semantic and phonetic search capabilities. The API offers **100,000 requests per day** free with no API key required, making it ideal for development and research. Example usage: `https://api.datamuse.com/words?sp=a????` for pattern matching or `https://api.datamuse.com/words?rel_rhy=wordle` for rhyme detection. The **RESTful interface** integrates easily into ML pipelines for real-time feature engineering.

**WordsAPI** (`https://wordsapiv1.p.mashape.com/words/`) through RapidAPI provides **comprehensive word metadata** including definitions, frequency scores, and relationships. Features include frequency analysis, rhyme detection, and pattern searching with `letterPattern=^a.{4}$` for 5-letter words starting with 'a'. The **rich metadata** includes pronunciation and semantic relationships ideal for comprehensive word analysis.

### Pre-trained language models offer advanced capabilities

**Hugging Face Transformers** provides **state-of-the-art pre-trained models** for text generation, classification, and analysis. Installation via `pip install transformers` gives access to GPT-2, BERT, and specialized models. Example usage: `from transformers import pipeline; generator = pipeline("text-generation", model="gpt2")` for word prediction tasks. The **extensive model zoo** with Apache 2.0 licensing enables **fine-tuning** for specific word game applications.

**Specialized word game models** include DistilBERT, ALBERT, and RoBERTa optimized for **efficient word prediction**. These models provide **excellent accuracy** for next-word prediction while maintaining reasonable computational requirements for real-time applications.

### MLOps tools enable production deployment

**MLflow** (`pip install mlflow`) provides **experiment tracking** and model lifecycle management. Example usage includes logging parameters and metrics with `mlflow.log_param()` and `mlflow.log_metric()`, plus model serving through `mlflow.pyfunc.serve_model()`. The **industry-standard platform** supports all major ML frameworks and enables **reproducible research** with version control for models and data.

**FastAPI** offers **high-performance API framework** for ML model serving. Installation via `pip install fastapi uvicorn` enables rapid development of **production-ready endpoints** with automatic documentation and validation. The framework excels in **low-latency model inference** crucial for real-time word prediction applications.

## Open source implementations provide reference architectures

### Rule-based and probability approaches establish baselines

**jason-chao/wordle-solver** (`https://github.com/jason-chao/wordle-solver`) implements **probability-based solving** with letter frequency analysis, achieving **4.6 average tries** and 90.6% success rate. The Python implementation includes **web interface** at solvewordle.games and CLI tool supporting variable word lengths. Integration via `pip install` with **clean documentation** makes it ideal for understanding fundamental approaches.

**deedy/wordle-solver** (`https://github.com/deedy/wordle-solver`) provides **comprehensive parameterized solver** with 5 modes (play, solve, show, save, eval) and **100% accuracy** achieving 3.65 average attempts. The implementation supports **hard mode** and custom dictionaries with optimal first guess "SOARE". The **extensively documented** command-line tool offers **excellent customization options** for research applications.

### Machine learning implementations demonstrate advanced techniques

**andrewkho/wordle-solver** implements **A2C reinforcement learning** using PyTorch Lightning, achieving **99% win rate** with less than 4 guesses average. The implementation includes **staged training** with custom neural network architecture and **Heroku deployment** for testing. The **well-documented approach** with accompanying blog post provides excellent learning resource for advanced AI applications.

**ericzbeard/wordle-solver** (`https://github.com/ericzbeard/wordle-solver`) creates **OpenAI Gym environment** for reinforcement learning experimentation. The implementation explores **DQN approaches** with custom Wordle environment making it **easy to experiment** with different RL algorithms. The **educational focus** with clear documentation makes it ideal for learning RL applications in game solving.

### Academic implementations provide optimal solutions

**MIT's optimal algorithm** (wordleopt.com) by Dimitris Bertsimas and Alex Paskov achieves **3.421 average guesses** using exact dynamic programming with proven optimal starting word "SALET". The **mathematically proven approach** provides **performance benchmark** for all other implementations and demonstrates **theoretical limits** of Wordle solving efficiency.

**3Blue1Brown implementation** (`https://github.com/woctezuma/3b1b-wordle-solver`) demonstrates **information theory approach** using Shannon entropy for information gain maximization. The **clean implementation** following academic methodology provides excellent **educational resource** for understanding information-theoretic approaches to game solving.

## Integration recommendations for production ML systems

### Recommended architecture combines multiple data sources

**Primary data pipeline**: Use **Google Books N-gram data** for frequency analysis, **FastText embeddings** for semantic features, **CMU Pronouncing Dictionary** for phonetic analysis, and **WordNet** for linguistic relationships. This combination provides **comprehensive word representation** covering frequency, semantics, phonetics, and structural relationships.

**Feature engineering pipeline**: Implement **positional letter frequency** analysis from 5-letter word subsets of Google Books data, **semantic similarity** calculations using pre-trained embeddings, and **phonetic pattern analysis** using CMU dictionary. Combine with **Wordle-specific features** like letter position constraints and word difficulty metrics from academic papers.

**Model training approach**: Use **Hugging Face Transformers** for pre-trained language model fine-tuning, **MLflow** for experiment tracking, and **FastAPI** for production serving. The **modular architecture** enables easy component replacement and **A/B testing** of different approaches.

### Quality assurance and validation protocols

**Multi-source validation**: Cross-reference frequency rankings across Google Books, OpenSubtitles, and academic corpora to **identify inconsistencies**. Validate semantic relationships using WordNet and embedding similarity to **ensure feature quality**. Monitor for **historical vs. contemporary usage biases** in different datasets.

**Performance benchmarking**: Compare against **MIT's optimal algorithm** (3.421 average guesses) and **deedy's implementation** (3.65 average) for accuracy validation. Use **academic papers' evaluation metrics** for standardized comparison and **reproducible research** protocols.

**Legal and ethical considerations**: All recommended datasets are **suitable for academic research** and non-commercial applications. For commercial use, verify **specific licensing terms** and respect intellectual property rights. The **open source ecosystem** generally supports research and educational applications under permissive licenses.

This comprehensive resource directory provides everything needed to build state-of-the-art Wordle prediction systems, from fundamental datasets to production-ready implementations. The combination of rigorous academic research, practical tools, and extensive open source implementations creates an exceptional foundation for both learning and advanced AI development in word game solving.
