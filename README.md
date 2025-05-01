# IFT6289 - Final Project

### Stock Price Prediction using LSTM + Sentiment Analysis

This project aims to predict stock prices using LSTM networks and sentiment analysis of news articles. The project is divided into several parts:
- data collection, data preprocessing, sentiment analysis, LSTM model training, and evaluation.
- Dora fine-tuning of Deepseek r1 models
- SLURM scripts (to adapt to your own cluster)
- Helper files for data preprocessing and model training


### Requirements
- Python 3.11.5 or higher

Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### Agents
Go to the folder `agents` to find the code for the agents. The folder contains the following files:
- Finbert
- deepseek

Two separate scripts for the models.

### How to use:


#### Path:

Be sure to be in ```stock_predictor_sentiment_analysis/Agents/```

Then run the following command:

##### For FinBERT:

First ensure the following two CSVs are in the directory 

1. AUGMENTED_entityMask_merged_news_stock(finbert).csv
2. news_socialmedia_merged_data(finbert).csv

Then run: 
```
python FinBERT_agent.py predict [TICKER] [MODEL-CHOICE]
```

For example, for Nvidia (NVDA), you can use one of the 2 models (sentiment from news only, or sentiment from news+reddit):
```
python FinBERT_agent.py predict NVDA news-only-finbert
```
OR
```
python FinBERT_agent.py predict NVDA news-socialmedia-finbert
```
---------------------
##### For DeepSeek:

```bash
python3 deepseek_agent.py
```

Make sure to adapt the paths in the python files.
