import praw

reddit = praw.Reddit(
    client_id="uKcCeuvtmq9fTXlksEmavQ",
    client_secret="L28blZHsJsv-AHU7gOlbXOSa4tCTAA",
    user_agent="stock_market_scrapper by semi-finalist2022"
)

# Works fine without login for public subreddits
subreddit = reddit.subreddit("stocks")

for post in subreddit.hot(limit=5):
    print(post.title)
