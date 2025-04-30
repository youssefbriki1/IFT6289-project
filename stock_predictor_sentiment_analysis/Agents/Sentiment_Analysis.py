# To use langgraph 



# Check how many GPUs are available
# See if MPS could be applied
# Deepseek r1 1,5 - requires 10 gb of VRAM
# quantization 4bit & awq ???
# RAG over 

# Case 1: Multiple GPUs
# Case 2: Single GPU + CPU 
import torch
import logging
from socialMediaScraper.WebScraper import WebScraper
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Qwen2ForSequenceClassification
from peft import PeftModel


logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)
class Agent:
    def __init__(self, model, company=None, tools=None):
        self.model = model
        self.num_gpus = torch.cuda.device_count()
        self.company = company  
        
        if self.num_gpus == 0:
            logger.warning("No GPUs available. Using CPU. - Might take a while")
        elif self.num_gpus == 1:
            logger.warning("Using 1 GPU, using multiple GPUs might be better")
        else:
            logger.info(f"Using {self.num_gpus} GPUs")
        
        
        
        
    def __retrieve_social_media_data(self):
        """
        Retrieve data from social media using the WebScraper
        """
        web_scraper  = WebScraper()
        
        # Maybe to add case for CPU only
        if self.num_gpus <= 1:
            reddit_posts_limit = bluesky_posts_limit = 5
            reddit_comments_limit = 0
        elif 1 < self.num_gpus < 4:
            reddit_posts_limit = bluesky_posts_limit = 10
            reddit_comments_limit = 5
        elif self.num_gpus >= 4:
            reddit_posts_limit = bluesky_posts_limit = 50
            reddit_comments_limit = 10
        
        logging.info(f"Scraping social media data for {self.company}")
        logging.info(f"Reddit posts limit: {reddit_posts_limit}")
        logging.info(f"Bluesky posts limit: {bluesky_posts_limit}")
        logging.info(f"Reddit comments limit: {reddit_comments_limit}")
        
        web_scraper(company=self.company,
                              reddit_posts_limit=reddit_posts_limit, 
                              bluesky_posts_limit=bluesky_posts_limit, 
                              reddit_comments_limit=reddit_comments_limit)
        
        logging.info("Scraping completed")
        return web_scraper.date
            
            
    def analyze_sentiment(self):
        """
        Analyze the sentiment of the data using the model
        """
        date = self.__retrieve_social_media_data()
        data = {}
        for social_media in ["reddit", "bluesky"]:
            with open(f'data/{social_media}_{date}.json', 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    data.update(json.load(f))
        print(data)
        self.model
        
        
        
    
    def predict_stock_price(self, data):
        """
        Predict the stock price using the LSTM model
        """
        pass
    
    def __call__(self, data):
        """
        Call the agent with the data
        """


if __name__ == "__main__":
    # Example usage
    model = None  # Replace with your model
    company = "AAPL"
    tools = None  # Replace with your tools
    agent = Agent(model, tools)
    agent.analyze_sentiment() 