#  WebScraper - Reddit & Bluesky Post Collector

This script collects posts from Reddit and Bluesky based on predefined topics and subreddits. It supports multithreading and offers configurable scraping limits.

##  Requirements

- Python 3.11.5
- `.env` file with the following content:

```env
BLUESKY_HANDLE=your_bluesky_handle
BLUESKY_PASSWORD=your_bluesky_password
REDDIT_SECRET=your_reddit_client_secret

Then run the following command:

```bash
python3 Sentiment_Analysis.py [OPTIONS]

python3 Sentiment_Analysis.py --reddit_posts_limit 20 --bluesky_posts_limit 15

```