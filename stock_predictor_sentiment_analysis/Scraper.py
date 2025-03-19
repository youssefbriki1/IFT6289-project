import logging
import praw
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
from texts import TOPICS, SUBREDDITS

# TODO:
# Store data by day in txt format
# Get daily top posts from subreddits (10-20)
# Get daily top tweets from twitter (10-20)
# Get daily top news articles from news websites (10-20)
# We must have 30-80 data points per day
# Atleast 10 pictures per day

def validator(date):
    try:
        datetime.strptime(date, "%Y-%m-%d")
        return True
    except ValueError:
        logging.error("Invalid date format. Please use 'YYYY-MM-DD'.")
        return False
class WebScraper:    
    def __init__(self, date):
        if validator(date):
            self.date = date
        else:
            self.date = datetime.now().strftime("%Y-%m-%d")
        #self.reddit = praw.Reddit(client_id='my_client_id', client_secret='my_client_secret', user_agent='my_user_agent')
        

    
    def scrap_reddit(self):
        logging.info("Scraping Reddit")
        pass
    
    def scrap_twitter(self):
        logging.info("Scraping Twitter")
        
        # Convert self.date into beginning and end dates.
        # Here we assume self.date is a string "YYYY-MM-DD"
        try:
            start_date = datetime.strptime(self.date, "%Y-%m-%d")
        except ValueError:
            logging.error("Invalid date format. Please use 'YYYY-MM-DD'.")
            return {}
        
        beginning_date = start_date.strftime("%Y-%m-%d")
        # Set end_date to the next day so that tweets from the entire day are included.
        end_date = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")
        
        all_tweets = {}
        for topic in TOPICS:
            query = f"{topic} since:{beginning_date} until:{end_date}"
            logging.info(f"Querying Twitter for topic: {topic} with query: {query}")
            
            try:
                # Get tweets using snscrape and sort them by likeCount (descending)
                tweets = list(sntwitter.TwitterSearchScraper(query).get_items())
                tweets = sorted(tweets, key=lambda x: x.likeCount, reverse=True)
            except Exception as e:
                logging.error(f"Error scraping tweets for topic '{topic}': {e}")
                continue
            
            top_tweets = tweets[:5]
            tweet_infos = []
            for tweet in top_tweets:
                tweet_info = {
                    "likes": tweet.likeCount,
                    "date": tweet.date,
                    "user": tweet.username,
                    "content": tweet.content
                }
                tweet_infos.append(tweet_info)
                # Output tweet details
                print(f"Likes: {tweet.likeCount}")
                print(f"Date: {tweet.date}")
                print(f"User: {tweet.username}")
                print(f"Tweet: {tweet.content}\n")
            all_tweets[topic] = tweet_infos
        
        return all_tweets
    
    
    
    def scrap_news(self):
        logging.info("Scraping News")
        pass
    
