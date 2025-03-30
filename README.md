For acquiring financial news & earning calls transcript data along with their sentiment scores, simply run the notebook data_acquisition_news.ipynb. It should create a .json file containing:

For 50 of the most traded tickers/companies:
- news articles (earliest is 2019 if available, and onwards)
- earning call transcripts (Q1-4 2024),
- daily stock price (past 100 days)
News articles in specific topics:
 - ipo, earnings, mergers_and_acquisitions, economy_macro, economy_fiscal, economy_monetary, technology, finance
Economic indicators (i think its only most recent data):
- GDP, unemployment, inflation, interest rates
data is in raw json format, we can extract relevant features and reconstruct it as needed for the models

#### Rough Outline of json structure (not complete and may not be accurate) ####

Top Level Labels are: 
- stocks (news, earnings_call_trancript, stock_prices by ticker/company)
- topics (news by topics)
- economic_indicator (macroeconomic stuff)}:

```
"stocks":{
	"ticker": ... ,
	"news": {
		"feed": [{
			"title":...,
			"url":...,
			"time_published": ... ,
			"authors":...,
			"summary":...,
			"banner_image":...,
			"source":...,
			"category_within_source":...,
			"source_domain":...,
			"topics": [
			{
				"topic": ...,
				"relevance_score":...
			},....
			],
			"overall_sentiment_score": ...,
			"overall_sentiment_label":...,
			"ticker_sentiment": [
			{
				"ticker": ...,
				"relevance_score":..,
				"ticker_sentiment_score":...,
				"ticker_sentiment_label":...,
			},....
			]
		}]},
	"earning_call_transcript":[{
		"symbol": ...,
		"quarter":...,
		"transcript": [
		{
			"speaker":...,
			"title":...,
			"content":...,
			"sentiment":...,
		},...
		{...},
		]
		}],
	"prices": [{"date", "open", "high", "low", "close", "volume"}]
},
"topics": {
	"topic":...,
	"news":...,
},
"economic_indicator":
...
