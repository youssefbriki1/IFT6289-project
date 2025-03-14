# data_loader.py
import pandas as pd

def load_financial_data(stock_path, macro_path):
    stock_data = pd.read_csv(stock_path, parse_dates=['Date'], index_col='Date')
    macro_data = pd.read_csv(macro_path, parse_dates=['Date'], index_col='Date')
    data = stock_data.join(macro_data, how='inner')
    return data

def load_text_data(news_path, social_path):
    news_data = pd.read_csv(news_path)
    social_data = pd.read_csv(social_path)
    text_data = pd.concat([news_data, social_data])
    return text_data['content'].tolist()