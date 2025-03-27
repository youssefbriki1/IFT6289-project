import logging
import praw
import json
from datetime import datetime, timedelta, date
from texts import TOPICS, SUBREDDITS
from reddit_scheme import RedditPost
from twitter_scheme import TwitterPost
from bluesky_scheme import BlueskyPost
from typing import List
import os
import certifi
from atproto import Client as BlueskyClient
from dotenv import load_dotenv

load_dotenv()
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")

if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
    raise ValueError("BLUESKY_HANDLE and BLUESKY_PASSWORD must be set in the .env file.")

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class WebScraper:
    def __init__(self, date_str=None):
        if date_str:
            try:
                self.date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                logging.error("Invalid date format. Using today's date instead.")
                self.date = date.today()
        else:
            self.date = date.today()

        self.reddit = praw.Reddit(
            client_id="uKcCeuvtmq9fTXlksEmavQ",
            client_secret="L28blZHsJsv-AHU7gOlbXOSa4tCTAA",
            user_agent="stock_market_scrapper by semi-finalist2022"
        )

        self.bluesky_client = BlueskyClient()
        self.bluesky_client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)

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

    def scrap_bluesky(self):
        logging.info("Scraping Bluesky")

        all_posts = []
        client = self.bluesky_client  # reuse the already logged-in client

        for topic in TOPICS:
            logging.info(f"Searching Bluesky for topic: {topic}")
            try:
                results = client.app.bsky.feed.search_posts({'q': topic, 'limit': 10})
                for post in results.posts:
                    try:
                        record = post.record  # use attribute access
                        author = post.author

                        # Handle embedded images if available.
                        images = []
                        if hasattr(record, 'embed') and getattr(record.embed, '$type', None) == 'app.bsky.embed.images':
                            images = [img.fullsize for img in record.embed.images]

                        # Get the creation timestamp.
                        # Try both 'createdAt' and 'created_at'.
                        created_at_str = getattr(record, 'createdAt', None) or getattr(record, 'created_at', None)
                        if not created_at_str:
                            raise ValueError("Missing creation timestamp")
                        # Replace 'Z' with '+00:00' so fromisoformat() can parse it.
                        created_at_str = created_at_str.replace("Z", "+00:00")
                        created_at = datetime.fromisoformat(created_at_str)

                        bluesky_post = BlueskyPost(
                            uri=post.uri,
                            cid=post.cid,
                            author_handle=author.handle,
                            author_did=author.did,
                            content=getattr(record, 'text', ''),
                            created_at=created_at,
                            images=images
                        )
                        all_posts.append(bluesky_post)
                    except Exception as e:
                        logging.error(f"Error parsing Bluesky post for topic '{topic}': {e}")
            except Exception as e:
                logging.error(f"Error searching Bluesky for topic '{topic}': {e}")

        output_filename = f"bluesky_{self.date}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump([post.model_dump() for post in all_posts],
                    f, indent=2, ensure_ascii=False, default=default_serializer)

        logging.info(f"Saved {len(all_posts)} Bluesky posts to {output_filename}")

    def __call__(self):
        self.scrap_reddit()
        self.scrap_bluesky()
        # self.scrap_twitter() 


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    
    scraper = WebScraper()

    with ThreadPoolExecutor(max_workers=100) as executor:
        future_reddit = executor.submit(scraper.scrap_reddit)
        future_bluesky = executor.submit(scraper.scrap_bluesky)

        future_reddit.result()
        future_bluesky.result()
