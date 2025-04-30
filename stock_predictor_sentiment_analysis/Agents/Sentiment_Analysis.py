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
from ../scraper.scraper import WebScraper



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class Agent:
    def __init__(self, model, company, tools):
        self.model = model
        self.num_gpus = torch.cuda.device_count()
        self.company = company  
        
        if self.num_gpus == 0:
            logger.warning("No GPUs available. Using CPU. - Might take a while")
        elif self.num_gpus == 1:
            logger.warning("Using 1 GPU, using multiple GPUs might be better")
        else:
            logger.info(f"Using {self.num_gpus} GPUs")
        
        
        
        
    def __retrieve_data_reddit(self):
        """
        Retrieve data from reddit using the reddit API (PRAW) about general market sentiment
        """
        pass
        
        
    def __retrieve_data_bluesky(self):
        """
        Retrieve data from bluesky using the bluesky API about company
        """
        pass 
    
    def analyze_sentiment(self, data):
        """
        Analyze the sentiment of the data using the model
        """
        pass
    
    def predict_stock_price(self, data):
        """
        Predict the stock price using the LSTM model
        """
        pass
    
    def __call__(self, data):
        """
        Call the agent with the data
        """
        pass 