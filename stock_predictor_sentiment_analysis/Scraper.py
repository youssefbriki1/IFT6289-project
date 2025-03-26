import logging
import praw
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta, date
from texts import TOPICS, SUBREDDITS
from reddit_schema import RedditPost
import json

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

    def scrap_reddit(self):
        all_posts = []

        for subreddit_name in SUBREDDITS:
            logging.info(f"Scraping Reddit for subreddit: {subreddit_name}")
            subreddit = self.reddit.subreddit(subreddit_name)

            for post in subreddit.hot(limit=5):
                try:
                    post.comments.replace_more(limit=0)
                    comments = [comment.body for comment in post.comments.list()[:5]]  # Top 5 comments

                    reddit_post = RedditPost(
                        id=post.id,
                        title=post.title,
                        description=post.selftext,
                        upvotes=post.ups,
                        downvotes=post.downs,
                        subreddit=post.subreddit.display_name,
                        comments=comments,
                        date=datetime.fromtimestamp(post.created_utc)
                    )

                    all_posts.append(reddit_post)
                except Exception as e:
                    logging.error(f"Error parsing post in subreddit {subreddit_name}: {e}")
                    continue

        # Save all posts to JSON
        output_filename = f"reddit_{self.date}.json"
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")


        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump([post.model_dump() for post in all_posts], f, indent=2, ensure_ascii=False, default=default_serializer)

        logging.info(f"Saved {len(all_posts)} Reddit posts to {output_filename}")

    """
    def scrap_twitter(self):
        logging.info("Scraping Twitter")
        beginning_date = self.date.strftime("%Y-%m-%d")
        end_date = (self.date + timedelta(days=1)).strftime("%Y-%m-%d")

        all_tweets = {}
        for topic in TOPICS:
            query = f"{topic} since:{beginning_date} until:{end_date}"
            logging.info(f"Querying Twitter for topic: {topic} with query: {query}")
            
            try:
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
    """


if __name__ == "__main__":
    scraper = WebScraper()
    scraper.scrap_reddit()
    # scraper.scrap_reddit()
    # scraper.scrap_news()