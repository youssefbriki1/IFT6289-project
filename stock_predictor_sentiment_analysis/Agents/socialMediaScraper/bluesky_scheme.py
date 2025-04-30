from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class BlueskyPost(BaseModel):
    uri: str = Field(..., description="The unique URI of the Bluesky post")
    cid: str = Field(..., description="Content ID of the post")
    author_handle: str = Field(..., description="Handle of the user who made the post")
    author_did: str = Field(..., description="DID of the user")
    content: str = Field(..., description="Text content of the post")
    created_at: datetime = Field(..., description="Post creation date")
    images: Optional[List[str]] = Field(default_factory=list, description="List of image URLs attached to the post")
