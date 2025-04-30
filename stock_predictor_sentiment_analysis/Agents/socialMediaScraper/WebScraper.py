import logging
import praw
import json
from datetime import datetime
from socialMediaScraper.texts import TOPICS, SUBREDDITS
from socialMediaScraper.reddit_scheme import RedditPost
from socialMediaScraper.bluesky_scheme import BlueskyPost
import os
import certifi
from atproto import Client as BlueskyClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")

if not BLUESKY_HANDLE or not BLUESKY_PASSWORD or not REDDIT_SECRET:
    logging.error("BLUESKY_HANDLE, BLUESKY_PASSWORD, and REDDIT_SECRET must be set in the .env file.")
    raise ValueError("BLUESKY_HANDLE and BLUESKY_PASSWORD and REDDIT_SECRET must be set in the .env file.")

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class WebScraper:
    def __init__(self):
            
        self.reddit = praw.Reddit(
            client_id="uKcCeuvtmq9fTXlksEmavQ",
            client_secret=REDDIT_SECRET,
            user_agent="stock_market_scrapper by semi-finalist2022"
        )
        self.date = datetime.today().date()
        self.bluesky_client = BlueskyClient()
        self.bluesky_client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)

    def __scrap_reddit(self, posts_limit=10, comments_limit=5):
        logging.info("Scraping Reddit")
        all_posts = []

        for subreddit_name in SUBREDDITS:
            logging.info(f"Scraping Reddit for subreddit: {subreddit_name}")
            subreddit = self.reddit.subreddit(subreddit_name)

            for post in subreddit.hot(limit=posts_limit):
                try:
                    post.comments.replace_more(limit=0)
                    comments = [comment.body for comment in post.comments.list()[:comments_limit]]

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

        output_filename = f"data/reddit_{self.date}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump([post.model_dump() for post in all_posts],
                      f, indent=2, ensure_ascii=False, default=default_serializer)

        logging.info(f"Saved {len(all_posts)} Reddit posts to {output_filename}")
        return output_filename
    
    
    def __scrap_bluesky(self, posts_limit=10, company=None):
        """
        Scrapes posts from Bluesky for a given company or a list of topics if no company is provided.

        Args:
            posts_limit (int, optional): _description_. Defaults to 10.
            company (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        logging.info("Scraping Bluesky")
        all_posts = []
        client = self.bluesky_client 
        for topic in (TOPICS if company is None else [company]):
            logging.info(f"Searching Bluesky for topic: {topic}")
            try:
                results = client.app.bsky.feed.search_posts({'q': topic, 'limit': posts_limit})
                for post in results.posts:
                    try:
                        record = post.record  
                        author = post.author

                        images = []
                        if hasattr(record, 'embed') and getattr(record.embed, '$type', None) == 'app.bsky.embed.images':
                            images = [img.fullsize for img in record.embed.images]

                        created_at_str = getattr(record, 'createdAt', None) or getattr(record, 'created_at', None)
                        if not created_at_str:
                            raise ValueError("Missing creation timestamp")

                        created_at= datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))

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

        output_filename = f"data/bluesky_{self.date}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump([post.model_dump() for post in all_posts],
                    f, indent=2, ensure_ascii=False, default=default_serializer)

        logging.info(f"Saved {len(all_posts)} Bluesky posts to {output_filename}")
        return output_filename

    def __call__(self, parallelize=True, **kwargs):
        """
        Calls the scraper to scrape Reddit and Bluesky posts.

        Args:
            parallelize (bool, optional): _description_. Defaults to True.
        """
        reddit_kwargs = {
            "posts_limit": kwargs.get("reddit_posts_limit", 10),
            "comments_limit": kwargs.get("reddit_comments_limit", 5)
        }

        bluesky_kwargs = {
            "posts_limit": kwargs.get("bluesky_posts_limit", 10),
            "company": kwargs.get("company", None)
        }

        if not parallelize:
            self.__scrap_reddit(**reddit_kwargs)
            self.__scrap_bluesky(**bluesky_kwargs)
        else:
            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 5)) as executor:
                future_reddit = executor.submit(self.__scrap_reddit, **reddit_kwargs)
                future_bluesky = executor.submit(self.__scrap_bluesky, **bluesky_kwargs)
                future_reddit.result()
                future_bluesky.result()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Reddit and Bluesky for posts.")
    
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help="Run scraping in parallel."
    )
    parser.add_argument(
        "--reddit_posts_limit",
        type=int,
        default=10,
        help="Number of Reddit posts to scrape per subreddit."
    )
    parser.add_argument(
        "--reddit_comments_limit",
        type=int,
        default=5,
        help="Number of comments to scrape per Reddit post."
    )
    parser.add_argument(
        "--bluesky_posts_limit",
        type=int,
        default=10,
        help="Number of Bluesky posts to scrape per topic."
    )

    args = parser.parse_args()

    scraper = WebScraper()
    scraper(
        parallelize=args.parallelize,
        reddit_posts_limit=args.reddit_posts_limit,
        reddit_comments_limit=args.reddit_comments_limit,
        bluesky_posts_limit=args.bluesky_posts_limit
    )
