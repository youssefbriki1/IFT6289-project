from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class RedditPost(BaseModel):
    id: str = Field(..., example="1j0w73o")
    title: str = Field(..., example="GME to the moon")
    description: str = Field(..., example="GME is going to the moon. Buy now!")
    upvotes: int = Field(..., example=100)
    downvotes: int = Field(..., example=10)
    subreddit: str = Field(..., example="stocks")
    comments: List[str] = Field(..., example=["I agree", "I disagree"])
    date: datetime = Field(..., example="2022-01-01T12:34:56")
