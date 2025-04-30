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
import os 
from TextDataset import TextDataset
logging.basicConfig(level=logging.INFO)

################
os.environ["TRITON_CACHE_DIR"] = "/home/m/mehrad/brikiyou/scratch/triton_cache"
os.environ["TRITON_HOME"]      = "/home/m/mehrad/brikiyou/scratch/triton_home"
os.environ["CC"]  = "/scinet/balam/rocky9/software/2023a/opt/cuda-12.3.1/gcc/12.3.0/bin/gcc"
os.environ["CXX"] = "/scinet/balam/rocky9/software/2023a/opt/cuda-12.3.1/gcc/12.3.0/bin/g++"
os.environ["HF_HOME"]            = "/home/m/mehrad/brikiyou/scratch/huggingface_cache"
os.environ["HF_HUB_CACHE"]       = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "models")
os.environ["HF_DATASETS_CACHE"]  = os.path.join(os.environ["HF_HOME"], "datasets")
cache_dir = os.environ["HF_HOME"]
################


logger = logging.getLogger(__name__)
class Agent:
    def __init__(self, company=None, tools=None):
        self.base_model = Qwen2ForSequenceClassification.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    num_labels=3,
    trust_remote_code=True
)
        self.adapter_dir = "/home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/stock_predictor_sentiment_analysis/dora_fine-tuning/data/sentiment_data/model_1.5b"
        self.num_gpus = torch.cuda.device_count()
        self.company = company  
        self.tokenizer  = AutoTokenizer.from_pretrained(
    self.adapter_dir,
    use_fast=True,
    local_files_only=True
)
        self.model = PeftModel.from_pretrained(
    self.base_model,
    self.adapter_dir,
    local_files_only=True
)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.save_pretrained(cache_dir)
        self.model.save_pretrained(cache_dir)
    

        
        if self.num_gpus == 0:
            logger.warning("No GPUs available. Using CPU. - Might take a while")
        elif self.num_gpus == 1:
            logger.warning("Using 1 GPU, using multiple GPUs might be better")
        else:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_gpus)))
        self.model = self.model.cuda()
        self.model.eval()
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

        
        
        
        
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
            reddit_posts_limit = bluesky_posts_limit = 5
            reddit_comments_limit = 5
        
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
            


    def __analyze_sentiment(self, text: str) -> str:
        enc = self.tokenizer(
            text, return_tensors="pt",
            padding="max_length", truncation=True, max_length=512
        )
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        pred = logits.argmax(dim=-1).item()
        return self.id2label[pred]
    


    def analyze_social_media_sentiment(self):
        """
        Merges Reddit and Bluesky data into a single dictionary under separate keys.
        """
        date = self.__retrieve_social_media_data()
        merged_data = {"reddit": [], "bluesky": []}

        for platform in ["reddit", "bluesky"]:
            file_path = f"data/{platform}_{date}.json"
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    posts = json.load(f)
                    if isinstance(posts, list):
                        merged_data[platform] = posts
                    else:
                        logging.warning(f"Expected a list in {file_path}, got {type(posts)}")
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from {file_path}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error while reading {file_path}: {e}")


        for platform, posts in merged_data.items():
            if not posts:
                logging.warning(f"No posts found for {platform}")
                continue

            logging.info(f"Analyzing sentiment for {platform} data")
            for post in posts:            
                if platform == "reddit":
                    text = post['title'] + " " + post['description']
                    sentiment = self.__analyze_sentiment(text)
                    post['sentiment'] = sentiment
                elif platform == "bluesky":
                    text = post['content']
                    sentiment = self.__analyze_sentiment(text)
                    post['sentiment'] = sentiment
                    
            with open(f"data/{platform}_{date}_with_sentiment.json", "w") as out:
                json.dump(posts, out, ensure_ascii=False, indent=2)



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


if __name__ == "__main__":
    # Example usage
    model = None  # Replace with your model
    company = "AAPL"
    tools = None  # Replace with your tools
    agent = Agent( tools)
    agent.analyze_social_media_sentiment() 