# Stock Predictor

An end-to-end quantitative research project for short-horizon equity trend prediction using financial news sentiment analysis, natural language processing (NLP), and time-series modeling.

The project explores whether textual sentiment extracted from financial news can provide predictive signals for next-day stock-price direction.

---

## Overview

Financial markets react rapidly to new information. This project investigates the relationship between news sentiment and short-term stock-price movements by combining:

- Web crawling
- NLP-based sentiment extraction
- Time-series modeling
- Deep learning architectures
- Backtesting workflows

The system was developed as an experimental quantitative research framework during graduate research at the University of Delaware.

---

## Research Objective

The primary goal is not absolute price prediction, but evaluating whether textual sentiment contains incremental predictive information about short-term market direction.

The project focuses on:

- Signal generation from financial news
- Sentiment-feature engineering
- Sequence modeling for market prediction
- Backtested directional forecasting

---

## Pipeline Architecture

The framework consists of four major stages:

### 1. Data Collection
- Crawl financial news articles and headlines
- Aggregate recent market-related text data
- Preprocess and clean raw text

### 2. NLP Sentiment Extraction
- Tokenization and text normalization
- Sentiment scoring
- Feature construction from news sentiment

### 3. Time-Series Modeling
- LSTM-based sequential modeling
- Historical price integration
- Multi-feature temporal forecasting

### 4. Backtesting & Evaluation
- Directional accuracy evaluation
- Rolling-window testing
- Prediction-performance analysis

---

## Key Features

- End-to-end automated pipeline
- Financial-news web crawling
- NLP-based sentiment analysis
- LSTM time-series forecasting
- Feature engineering for market signals
- Historical backtesting framework
- Modular research workflow

---

## Model Performance

The system achieved approximately:

- **55% backtested directional accuracy** on next-day stock-price movement prediction

While modest, this level of performance is notable given the difficulty of short-horizon market forecasting and the noisy nature of financial text data.

---

## Tech Stack

### Languages
- Python
- Bash

### Libraries
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- BeautifulSoup
- NLP libraries

### Techniques
- Natural Language Processing (NLP)
- Sentiment Analysis
- LSTM Neural Networks
- Time-Series Forecasting
- Feature Engineering
- Backtesting

---

## Example Workflow

```bash
run.sh
```

---

## Research Notes

This project is intended as a quantitative research experiment rather than a production trading system.

Important limitations include:

- Market non-stationarity
- News-selection bias
- Limited predictive horizon
- Transaction costs not fully modeled
- Potential overfitting risks in financial ML

The repository is best viewed as a framework for exploring NLP-driven market signals and financial time-series modeling.

---

## Potential Future Improvements

Future extensions may include:

- Transformer-based language models (FinBERT, LLMs)
- Multi-asset prediction
- Intraday forecasting
- Alternative data integration
- Attention-based sequence models
- Reinforcement learning approaches
- Portfolio-level optimization
- Transaction-cost-aware backtesting
