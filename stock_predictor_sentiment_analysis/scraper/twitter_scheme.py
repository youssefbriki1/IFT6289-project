from pydantic import BaseModel, Field
from datetime import datetime

class TwitterPost(BaseModel):
    likes: int = Field(..., example=100)
    date: datetime = Field(..., example="2022-01-01T12:34:56")
    user: str = Field(..., example="elonmusk")
    content: str = Field(..., example="To the moon!")
