import logging
import praw
import snscrape.modules.twitter as sntwitter
import json
from datetime import datetime, timedelta, date
from texts import TOPICS, SUBREDDITS
from reddit_scheme import RedditPost 
from pydantic import BaseModel, Field
from typing import List
from twitter_scheme import TwitterPost
import os
import certifi
import tweepy

# TODO:
# Store data by day in json format
# Get daily top posts from subreddits 15
# Get daily top tweets from twitter 15
# We must have 30-80 data points per day

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAD7S0AEAAAAAYKzkcZAwmKlWWkmpOY%2BEZHFgDo4%3D0KWroW4C7KIUMMV0nV8iTFGqoJgdjatfDxuU1gGCGslrZyqIPX"
def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

class WebScraper:
    def __init__(self, date_str=None):
        if date_str:
            try:
                self.date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                logging.error("Invalid date format. Please use 'YYYY-MM-DD'. Using today's date instead.")
                self.date = date.today()
        else:
            self.date = date.today()

        self.reddit = praw.Reddit(
            client_id="uKcCeuvtmq9fTXlksEmavQ",
            client_secret="L28blZHsJsv-AHU7gOlbXOSa4tCTAA",
            user_agent="stock_market_scrapper by semi-finalist2022"
        )
        
        
        self.twitter_client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            wait_on_rate_limit=True
        )

    def scrap_reddit(self):
        all_posts = []

        for subreddit_name in SUBREDDITS:
            logging.info(f"Scraping Reddit for subreddit: {subreddit_name}")
            subreddit = self.reddit.subreddit(subreddit_name)

            for post in subreddit.hot(limit=5):
                try:
                    post.comments.replace_more(limit=0)
                    comments = [comment.body for comment in post.comments.list()[:5]]  

                    images = []
                    if post.url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        images.append(post.url)

                    reddit_post = RedditPost(
                        id=post.id,
                        title=post.title,
                        description=post.selftext,
                        upvotes=post.ups,
                        downvotes=post.downs,
                        subreddit=post.subreddit.display_name,
                        comments=comments,
                        date=datetime.fromtimestamp(post.created_utc),
                        images=images
                    )
                    all_posts.append(reddit_post)
                except Exception as e:
                    logging.error(f"Error parsing post in subreddit {subreddit_name}: {e}")
                    continue

        output_filename = f"reddit_{self.date}.json"


        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump([post.model_dump() for post in all_posts],
                      f, indent=2, ensure_ascii=False, default=default_serializer)

        logging.info(f"Saved {len(all_posts)} Reddit posts to {output_filename}")


    def scrap_twitter(self):
        logging.info("Scraping Twitter using Tweepy v2")
        start_time = datetime.combine(self.date, datetime.min.time()).isoformat("T") + "Z"
        end_time = datetime.combine(self.date + timedelta(days=1), datetime.min.time()).isoformat("T") + "Z"

        all_tweets = {}

        for topic in TOPICS:
            logging.info(f"Querying Twitter for topic: {topic}")

            try:
                response = self.twitter_client.search_recent_tweets(
                    query=topic + " -is:retweet",
                    max_results=20,
                    tweet_fields=["created_at", "public_metrics", "text"],
                    expansions="author_id",
                    start_time=start_time,
                    end_time=end_time
                )

                tweets = response.data if response.data else []
                tweets = sorted(tweets, key=lambda x: x.public_metrics["like_count"], reverse=True)[:5]

                tweet_posts = []
                for tweet in tweets:
                    twitter_post = TwitterPost(
                        likes=tweet.public_metrics["like_count"],
                        date=tweet.created_at,
                        user=tweet.author_id,  # You can resolve this ID to a username if needed
                        content=tweet.text,
                        images=[]  # Twitter API v2 doesn’t give media URLs unless you request them separately
                    )
                    tweet_posts.append(twitter_post)

                all_tweets[topic] = tweet_posts

            except Exception as e:
                logging.error(f"Error fetching tweets for topic '{topic}': {e}")
                continue

        output_filename = f"twitter_{self.date}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(
                {topic: [post.model_dump() for post in posts] for topic, posts in all_tweets.items()},
                f, indent=2, ensure_ascii=False, default=default_serializer
            )
        
        
    def __call__(self):
        self.scrap_reddit()
        self.scrap_twitter()


if __name__ == "__main__":
    scraper = WebScraper()
    scraper()
