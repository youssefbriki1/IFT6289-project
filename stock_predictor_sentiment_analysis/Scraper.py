import logging
import praw
import snscrape.modules.twitter as sntwitter


    
class WebScraper:    
    def __init__(self, date):
        self.date = date

    
    def scrap_reddit(self):
        logging.info("Scraping Reddit")
        pass
    
    def scrap_twitter(self):
        logging.info("Scraping Twitter")
        query = f"{topic} since:{} until:{}"
        tweets = sorted(sntwitter.TwitterSearchScraper(query).get_items(),key=lambda x:x.likeCount, reverse=True) # Sort them by likes 

        for i, tweet in enumerate(tweets[:5]):
            print(f"Likes: {tweet.likeCount}")
            print(f"Date: {tweet.date}")
            print(f"User: {tweet.username}")
            print(f"Tweet: {tweet.content}\n")
    
    def scrap_news(self):
        logging.info("Scraping News")
        pass
    
