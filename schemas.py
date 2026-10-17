from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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

class ArticleTldrOut(BaseModel):
    tldr: str = Field(..., description="A 2-sentence empirical summary of the news.")

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

class UserPreferencesUpdate(BaseModel):
    preferred_macro_tags: List[str] = Field(..., description="List of chosen macro categories")
    preferred_micro_tags: List[str] = Field(default_factory=list, description="List of chosen micro tags")

class UserInteractionRequest(BaseModel):
    user_id: str
    interaction_type: str = Field(default="like", description="'like', 'dislike', 'skip', 'bookmark', or 'read'")

class UserProfileOut(BaseModel):
    id: str
    username: str
    avatar_url: Optional[str] = None
    onboarding_completed: bool = False
    preferred_macro_tags: List[str] = Field(default_factory=list)
    preferred_micro_tags: List[str] = Field(default_factory=list)

class DailyDigestOut(BaseModel):
    date: str
    macro_tag: str
    cluster_title: str
    cluster_id: Optional[str] = None
    article_ids: List[str]
    articles: List[Dict[str, Any]] = Field(default_factory=list)
    tavily_sources: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_card: Optional[Dict[str, Any]] = None
    synthesis_report: str
    is_cached: bool = False

class SynthesizeRequest(BaseModel):
    article_ids: List[str]

class DeepResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Investigative topic or claim to research via Tavily")
    model: str = Field(default="mini", description="'mini' or 'pro'")

class SiteWatchTriggerRequest(BaseModel):
    max_extract_per_site: int = Field(default=2, ge=1, le=5)

