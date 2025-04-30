from Agents.Sentiment_Analysis import Agent


if __name__ == "__main__":
    import logging
    import torch

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = None 
    company = "AAPL"
    agent = Agent(model, company)

    agent.analyze_sentiment()