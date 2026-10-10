from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ScrapedArticleIn(BaseModel):
    Site: str
    Baslik: str
    Link: str
    Yazar: Optional[str] = None
    Tarih: Optional[str] = None
    TamMetin: str

class ArticleEnrichmentOut(BaseModel):
    macro_tag: str = Field(..., description="One of: Tech, AI, Hardware, Automotive, Gaming, Finance, Science")
    micro_tags: List[str] = Field(..., description="3 to 6 lowercase specific tags")
    tldr: str = Field(..., description="A 2-sentence summary")

class QueryIntentOut(BaseModel):
    macro_tag: Optional[str] = None
    micro_tags: List[str] = Field(default_factory=list)

class ChatTurn(BaseModel):
    role: str
    content: str

class ArticleChatRequest(BaseModel):
    history: List[ChatTurn] = []
    question: str

class CommentCreate(BaseModel):
    user_id: str
    selected_text: Optional[str] = None
    comment_text: str

class UserLoginRequest(BaseModel):
    username: Optional[str] = None
    name: Optional[str] = None

class SynthesizeRequest(BaseModel):
    article_ids: List[str]
