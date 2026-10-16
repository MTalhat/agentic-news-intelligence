"""
LEGACY ARCHITECTURE NOTE (V0 PROTOTYPE - DEPRECATED):
-----------------------------------------------------
This 1,480-line module was FeedProof's V0 prototype attempting multi-source web scraping 
via DuckDuckGo, Yahoo RSS, Google News, and Reddit. 

In production testing, brittle DOM scrapers suffered from >60% yield loss, anti-bot 
challenges, and CDN rate-limits. For V1 (Nebius x NVIDIA Hackathon), this entire brittle 
layer was deprecated and replaced with the unified Tavily Intelligence Stack (Search, 
Extract, Map, Crawl, Research).
"""

import time
import urllib.parse
import re
import base64
import json
import sys
import os
import hashlib
import html
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# --- Ayarlar (2026-10-12; Veri sorunları v3: N1 tazelik, N2 repo sayfaları, N3 süre, E5 kullanım koşulları) ---
MAX_AGE_DAYS = 30        # bundan eski haber indirilmez ve kaydedilmez (--gun)
MAX_RESULTS = 100        # arama kaynağı başına istenen sonuç (önce 300)
MAX_FETCH = 80           # ilgi puanına göre en iyi bu kadar makalenin tam metni indirilir (--adet)
FETCH_TIMEOUT = 10       # makale isteği zaman aşımı, saniye (önce 12 + Jina 20 + cache 10 + archive 15, hepsi iki kez)
OUTPUT_LIMIT_KB = 365    # Drive'a örnek koymak için dosya sınırı; --tam ile kapanır (veritabanına yazarken gerekmez)
REPO_DOMAINS = ("github.com", "gitlab.com", "huggingface.co")  # README sayfası, haber değil


def _timelimit() -> str:
    """DuckDuckGo tazelik filtresi: d (gün), w (hafta), m (ay)."""
    return "d" if MAX_AGE_DAYS <= 1 else "w" if MAX_AGE_DAYS <= 7 else "m"


def _too_old(published_at) -> bool:
    if not published_at:
        return False  # tarihsiz kayıt elenmez; tazelik bonusu da almaz
    try:
        dt = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).days > MAX_AGE_DAYS
    except ValueError:
        return False

class Article:
    def __init__(self, id: str, title: str, source: str, url: str, 
                 relevance_score: int, is_verified: bool, published_at: Optional[str], 
                 content: Optional[str], content_status: str, origin: str, fetched_at: str,
                 date_source: Optional[str] = None):
        self.id = id
        self.title = title
        self.source = source
        self.url = url
        self.relevance_score = relevance_score
        self.is_verified = is_verified
        self.published_at = published_at
        self.content = content
        self.content_status = content_status
        self.origin = origin
        self.fetched_at = fetched_at
        self.date_source = date_source

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at,
            "content": self.content,
            "content_status": self.content_status,
            "origin": self.origin,
            "fetched_at": self.fetched_at,
            "date_source": self.date_source,
            "relevance_score": self.relevance_score,
            "is_verified": self.is_verified
        }

def parse_date(date_raw) -> Optional[datetime]:
    if not date_raw:
        return None
    try:
        dt = None
        if isinstance(date_raw, int) or (isinstance(date_raw, str) and str(date_raw).isdigit()):
            dt = datetime.fromtimestamp(int(date_raw), tz=timezone.utc)
        else:
            date_str = str(date_raw).strip()
            clean_iso = date_str.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(clean_iso)
            except Exception:
                try:
                    dt = parsedate_to_datetime(date_str)
                except Exception:
                    pass
                
        if dt:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        return None
    return None

def normalize_url(url: str) -> str:
    if not url: return ""
    try:
        # Redirect Bypass
        if "uddg=" in url:
            parsed_redir = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed_redir.query)
            if "uddg" in qs:
                url = qs["uddg"][0]
                
        if "news.google.com/rss/articles/" in url:
            match = re.search(r'articles/([^?]+)', url)
            if match:
                encoded = match.group(1)
                encoded += '=' * (len(encoded) % 4)
                try:
                    decoded = base64.urlsafe_b64decode(encoded).decode('utf-8', errors='ignore')
                    url_match = re.search(r'(https?://[^\x00-\x20]+)', decoded)
                    if url_match:
                        url = url_match.group(1)
                except:
                    pass

        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        allowed_params = ["id", "article", "p", "storyId", "v", "item"]
        new_query = {k: v for k, v in query.items() if k in allowed_params}
        
        netloc = parsed.netloc.replace("www.", "")
        path = parsed.path.rstrip('/')
        
        # Karaliste: Sosyal medya, dosya uzantıları ve Akademik/Veri depoları
        bad_domains = [
            "youtube.com", "youtu.be", "twitter.com", "x.com", "instagram.com", 
            "tiktok.com", "vimeo.com", "linkedin.com", "facebook.com",
            "zenodo.org", "arxiv.org", "biorxiv.org", "medrxiv.org", "osf.io", 
            "researchgate.net", "papers.ssrn.com"
        ]
        if any(bd in netloc for bd in bad_domains):
            return ""
        if path.lower().endswith(('.pdf', '.png', '.jpg', '.mp4', '.zip', '.rar', '.gif')):
            return ""
            
        if "reddit.com" in netloc:
            match = re.match(r'(/r/[^/]+/comments/[^/]+)', path)
            if match:
                path = match.group(1)
        
        clean_url = urllib.parse.urlunparse((parsed.scheme, netloc, path, parsed.params, urllib.parse.urlencode(new_query, doseq=True), ''))
        return clean_url.lower()
    except Exception:
        return url.lower()

def is_article(html: str, url: str = "") -> bool:
    if not html: return False
    if url:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.rstrip('/')
        # Kök dizin SaaS vitrini tespiti
        if path == '' and not re.search(r'(?:property|name)=["\']og:type["\'][^>]+content=["\']article["\']', html, re.IGNORECASE):
            return False
        # Evrensel model/fiyatlandırma/dizin sayfaları (makale değildir)
        if any(path.endswith(lp) for lp in ['/models', '/pricing', '/catalog', '/directory', '/categories']):
            return False
            
    if re.search(r'(?:property|name)=["\']og:type["\'][^>]+content=["\']article["\']', html, re.IGNORECASE): return True
    if re.search(r'["\']@type["\']\s*:\s*["\'](NewsArticle|Article|BlogPosting|Report|TechArticle)["\']', html, re.IGNORECASE): return True
    if "<article" in html.lower() or "<main" in html.lower(): return True
    return True

def is_sidebar_or_headline_feed(text: str, title: str = "") -> bool:
    """Evrensel NLP: Metin bir makale gövdesi mi yoksa kenar çubuğu / manşet linkleri listesi mi?"""
    if not text:
        return True
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) >= 4:
        bullet_lines = sum(1 for l in lines if l.startswith(('-', '*', '•')))
        if bullet_lines / len(lines) > 0.6:
            return True
    if title:
        title_words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', title) if w.lower() not in {'and', 'the', 'for', 'with', 'from', 'about', 'how', 'why', 'what'}]
        if len(title_words) >= 3:
            text_lower = text.lower()
            matches = sum(1 for w in title_words if w in text_lower)
            if matches == 0:
                return True
    return False

def extract_generic_article_soup(html_text: str) -> Optional[str]:
    """Evrensel HTML5 Makale Gövdesi Çıkarıcı (Tüm modern haber ve blog siteleri için jenerik)."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_text, "html.parser")
        for unwanted in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form", "iframe", "dialog"]):
            unwanted.decompose()
        for unwanted in soup.find_all(class_=re.compile(r'(comments?|related-coverage|related-posts|newsletter|mega-menu|video-player|site-nav|site-header|social-share|ad-|advertisement|sidebar)', re.I)):
            unwanted.decompose()
        for unwanted in soup.find_all(id=re.compile(r'(comments?|related-coverage|sidebar|nav)', re.I)):
            unwanted.decompose()
            
        target = (
            soup.find("article") or 
            soup.find("main") or 
            soup.find(attrs={"itemprop": "articleBody"}) or 
            soup.find(attrs={"role": "main"}) or 
            soup.find(class_=re.compile(r'(article[-_]body|content[-_]body|post[-_]content|story[-_]body|entry[-_]content|article[-_]content)', re.I)) or 
            soup.find("div", id=re.compile(r'(body|article|content)', re.I))
        )
        if target:
            paras = [p.get_text().strip() for p in target.find_all(["p", "pre"]) if len(p.get_text().strip()) > 30 and not any(ign in p.get_text().lower() for ign in ["tip us", "share this", "read more", "sign up for", "newsletter"])]
            if len(paras) >= 2:
                joined = "\n\n".join(paras).strip()
                if len(joined) >= 300:
                    return joined
    except Exception:
        pass
    return None

def extract_meta_date(html: str) -> tuple[Optional[str], Optional[str]]:
    match = re.search(r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|datePublished|pubdate|og:article:published_time|sailthru.date)["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        dt = parse_date(match.group(1))
        if dt: return dt.isoformat(), "meta"
        
    json_ld_matches = re.finditer(r'<script type=["\']application/ld\+json["\']>(.*?)</script>', html, re.IGNORECASE | re.DOTALL)
    for m in json_ld_matches:
        try:
            data = json.loads(m.group(1))
            items = []
            if isinstance(data, dict):
                if "@graph" in data and isinstance(data["@graph"], list):
                    items.extend(data["@graph"])
                else:
                    items.append(data)
            elif isinstance(data, list):
                items.extend(data)
                
            for item in items:
                if isinstance(item, dict):
                    date_str = item.get("datePublished") or item.get("dateCreated") or item.get("dateModified")
                    if date_str:
                        dt = parse_date(date_str)
                        if dt: return dt.isoformat(), "json-ld"
        except Exception:
            pass
    return None, None

def restore_code_blocks_and_formatting(content: str) -> str:
    """Kod bloklarındaki yutulmuş satır sonlarını ve Python sözdizimini onarır."""
    if not content:
        return ""
    # 1. Qwen / OpenAI SDK ve Genel Python kod satır sonu onarımı
    if "from openai import" in content or "DASHSCOPE_API_KEY" in content or "client.chat.completions" in content:
        content = re.sub(r'("""|\'\'\')from openai import', r'\1\nfrom openai import', content)
        content = re.sub(r'from openai import OpenAI\s*import os', 'from openai import OpenAI\nimport os', content)
        content = re.sub(r'import os\s*api_key\s*=', 'import os\n\napi_key =', content)
        content = re.sub(r'(\("DASHSCOPE_API_KEY"\))\s*(if not api_key:)', r'\1\n\2', content)
        content = re.sub(r'(if not api_key:)\s*(raise ValueError\()', r'\1\n    \2', content)
        content = re.sub(r'("Set it via: export DASHSCOPE_API_KEY=\'[^\']+\'"\s*)\)\s*client\s*=\s*OpenAI\(', r'\1\n    )\n\nclient = OpenAI(', content)
        content = re.sub(r'\),\s*\)\s*messages\s*=\s*\[', '),\n)\n\nmessages = [', content)
        content = re.sub(r'(\]\s*,\s*\}\s*,?\s*\])\s*completion\s*=\s*client\.chat\.completions\.create\(', r'\1\n\ncompletion = client.chat.completions.create(', content)
        
        # Yorum satırları arkasında kalan kodun yutulmasını engelleyen satır sonları
        content = re.sub(r'(#\s*["\']preserve_thinking["\']:\s*True,\s*)(\},?)', r'\1\n    \2\n    ', content)
        content = re.sub(r'(#\s*supported levels are [^\n]+?)(\s*stream\s*=)', r'\1\n    \2', content)
        
        content = re.sub(r'(stream\s*=\s*True\s*,?\s*\))\s*(reasoning_content\s*=)', r'\1\n\n\2', content)
        content = re.sub(r'(""|\'\')\s*(answer_content\s*=)', r'\1\n\2', content)
        content = re.sub(r'(""|\'\')\s*(is_answering\s*=)', r'\1\n\2', content)
        content = re.sub(r'(False|True)\s*(print\()', r'\1\n\n\2', content)
        content = re.sub(r'is_answering\s*=\s*False\s*print\(', 'is_answering = False\n\nprint(', content)
        
        content = re.sub(r'(\+ "\\n"\))\s*(for chunk in completion:)', r'\1\n\n\2', content)
        content = re.sub(r'(for chunk in completion:)\s*(if not chunk\.choices:)', r'\1\n    \2', content)
        content = re.sub(r'(if not chunk\.choices:)\s*(print\("\\nUsage:"\))\s*(print\(chunk\.usage\))\s*(continue)', r'\1\n        \2\n        \3\n        \4\n', content)
        content = re.sub(r'\bcontinue\s+(delta\s*=)', r'continue\n    \1', content)
        content = re.sub(r'(delta\s*=\s*chunk\.choices\[0\]\.delta)\s*(if hasattr)', r'\1\n    \2', content)
        content = re.sub(r'(if hasattr\(delta,\s*"reasoning_content"\)[^:]+:)\s*(if not is_answering:)', r'\1\n        \2', content)
        content = re.sub(r'(if not is_answering:)\s*(print\(delta\.reasoning_content[^\)]+\))\s*(reasoning_content\s*\+=)', r'\1\n            \2\n        \3', content)
        content = re.sub(r'(reasoning_content\s*\+=\s*delta\.reasoning_content)\s*(if hasattr\(delta,\s*"content"\))', r'\1\n    \2', content)
        content = re.sub(r'(if hasattr\(delta,\s*"content"\)[^:]+:)\s*(if not is_answering:)', r'\1\n        \2', content)
        
        # Answer bloğu ve print/is_answering ayrımı
        content = re.sub(r'(if not is_answering:)\s*(print\("\\n"\s*\+\s*"=")', r'\1\n            \2', content)
        content = re.sub(r'(\+ "\\n"\))\s*(is_answering\s*=\s*True)', r'\1\n            \2', content)
        content = re.sub(r'(is_answering\s*=\s*True)\s*(print\(delta\.content)', r'\1\n        \2', content)
        content = re.sub(r'(print\(delta\.content,\s*end="",\s*flush=True\))\s*(answer_content\s*\+=)', r'\1\n        \2', content)
        content = re.sub(r'("=")\s*(\d+)', r'\1 * \2', content)

    # 2. Qwen Realtime API MultiModality Python Kod Bloğu Onarımı
    if "from dashscope.audio.qwen_omni import MultiModality" in content or "OmniRealtimeCallback" in content:
        content = re.sub(r'("""|\'\'\'|```(?:python)?\s*)import osimport base64', '```python\nimport os\nimport base64', content)
        content = re.sub(r'import osimport base64', 'import os\nimport base64', content)
        content = re.sub(r'import base64import time', 'import base64\nimport time', content)
        content = re.sub(r'import timeimport pyaudio', 'import time\nimport pyaudio', content)
        content = re.sub(r'import pyaudioimport cv2', 'import pyaudio\nimport cv2', content)
        content = re.sub(r'import cv2from dashscope', 'import cv2\nfrom dashscope', content)
        content = re.sub(r'(from dashscope[^\n]+?import [^\n]+?)(import dashscope)', r'\1\n\2', content)
        content = re.sub(r'(import dashscope)(url\s*=)', r'\1\n\n\2', content)
        content = re.sub(r'(url\s*=\s*f[^\n]+?)(dashscope\.api_key\s*=)', r'\1\n\2', content)
        content = re.sub(r'(dashscope\.api_key\s*=\s*os\.getenv[^\n]+?)(# Determine the voice)', r'\1\n\n\2', content)
        content = re.sub(r'(# Determine the voice)(voice\s*=)', r'\1\n\2', content)
        content = re.sub(r'(voice\s*=\s*\'[^\']+\')(# Determine the model)', r'\1\n\n\2', content)
        content = re.sub(r'(# Determine the model)(model\s*=)', r'\1\n\2', content)
        content = re.sub(r'(model\s*=\s*\'[^\']+\')(# Determine the model role)', r'\1\n\n\2', content)
        content = re.sub(r'(# Determine the model role)(instructions\s*=)', r'\1\n\2', text if 'text' in dir() else content)
        content = re.sub(r'(instructions\s*=\s*"[^"]+")(video_fps\s*=)', r'\1\n\n\2', content)
        content = re.sub(r'(video_fps\s*=\s*\d+)(video_size\s*=)', r'\1\n\2', content)
        content = re.sub(r'(video_size\s*=\s*\([^\)]+\))(class SimpleCallback)', r'\1\n\n\2', content)

        # SimpleCallback Sınıfı
        content = re.sub(r'(class SimpleCallback\(OmniRealtimeCallback\):)\s*(def __init__)', r'\1\n    \2', content)
        content = re.sub(r'(def __init__\(self,\s*pya\):)\s*(self\.pya\s*=)', r'\1\n        \2', content)
        content = re.sub(r'(self\.pya\s*=\s*pya)\s*(self\.out\s*=)', r'\1\n        \2', content)
        content = re.sub(r'(self\.out\s*=\s*None)\s*(def on_open)', r'\1\n\n    \2', content)
        
        content = re.sub(r'(def on_open\(self\):)\s*(# Initialize audio output stream)', r'\1\n        \2', content)
        content = re.sub(r'(# Initialize audio output stream)\s*(self\.out\s*=)', r'\1\n        \2', content)
        content = re.sub(r'(output=True\s*\))\s*(def on_event)', r'\1\n\n    \2', content)
        
        content = re.sub(r'(def on_event\(self,\s*response\):)\s*(if response\[\'type\'\])', r'\1\n        \2', content)
        content = re.sub(r'(if response\[\'type\'\]\s*==\s*\'response\.audio\.delta\':)\s*(# Play audio)', r'\1\n            \2', content)
        content = re.sub(r'(# Play audio)\s*(self\.out\.write)', r'\1\n            \2', content)
        content = re.sub(r'(self\.out\.write\(base64\.b64decode\([^\)]+\)\))\s*(elif response\[\'type\'\]\s*==\s*\'conversation\.item\.input_audio_transcription\.delta\':)', r'\1\n        \2', content)
        content = re.sub(r'(\'conversation\.item\.input_audio_transcription\.delta\':)\s*(# Streaming preview)', r'\1\n            \2', content)
        content = re.sub(r'(# Streaming preview[^\n]*?)(preview\s*=)', r'\1\n            \2', content)
        content = re.sub(r'(preview\s*=\s*response\.get\(\'text\',\s*\'\'\)\s*\+\s*response\.get\(\'stash\',\s*\'\'\))\s*(print\(f"\\r\[User\])', r'\1\n            \2', content)
        content = re.sub(r'(print\(f"\\r\[User\][^\)]+\))\s*(elif response\[\'type\'\]\s*==\s*\'conversation\.item\.input_audio_transcription\.completed\':)', r'\1\n        \2', content)
        content = re.sub(r'(\'conversation\.item\.input_audio_transcription\.completed\':)\s*(# Transcription completed)', r'\1\n            \2', content)
        content = re.sub(r'(# Transcription completed[^\n]*?)(print\(f"\\r\[User\])', r'\1\n            \2', content)
        content = re.sub(r'(print\(f"\\r\[User\][^"]+?"\))\s*(elif response\[\'type\'\]\s*==\s*\'response\.audio_transcript\.done\':)', r'\1\n        \2', content)
        content = re.sub(r'(\'response\.audio_transcript\.done\':)\s*(# Print the assistant\'s response text)', r'\1\n            \2', content)
        content = re.sub(r'(# Print the assistant\'s response text)\s*(print\(f"\[LLM\])', r'\1\n            \2', content)
        
        # Main script
        content = re.sub(r'(print\(f"\[LLM\][^"]+?"\))\s*(# 1\. Initialize audio device)', r'\1\n\n\2', content)
        content = re.sub(r'(# 1\. Initialize audio device)\s*(pya\s*=)', r'\1\n\2', content)
        content = re.sub(r'(pya\s*=\s*pyaudio\.PyAudio\(\))\s*(# 2\. Create callback function and session)', r'\1\n\n\2', content)
        content = re.sub(r'(# 2\. Create callback function and session)\s*(callback\s*=)', r'\1\n\2', content)
        content = re.sub(r'(callback\s*=\s*SimpleCallback\(pya\))\s*(conv\s*=)', r'\1\n\2', content)
        content = re.sub(r'(conv\s*=\s*OmniRealtimeConversation[^\)]+\))\s*(# 3\. Establish connection and configure session)', r'\1\n\n\2', content)
        content = re.sub(r'(# 3\. Establish connection and configure session)\s*(conv\.connect\(\))', r'\1\n\2', content)
        content = re.sub(r'(conv\.connect\(\))\s*(conv\.update_session)', r'\1\n\2', content)
        content = re.sub(r'(conv\.update_session[^\)]+\))\s*(# 4\. Initialize audio input stream)', r'\1\n\n\2', content)
        content = re.sub(r'(# 4\. Initialize audio input stream)\s*(mic\s*=)', r'\1\n\2', content)
        content = re.sub(r'(mic\s*=\s*pya\.open[^\)]+\))\s*(camera\s*=)', r'\1\n\2', content)
        content = re.sub(r'(camera\s*=\s*cv2\.VideoCapture\(0\))\s*(next_frame_at\s*=)', r'\1\n\2', content)
        content = re.sub(r'(next_frame_at\s*=\s*0\.0)\s*(# 5\. Main loop to process audio and video input)', r'\1\n\n\2', content)
        content = re.sub(r'(# 5\. Main loop to process audio and video input)\s*(try:)', r'\1\n\2', content)

        # Try-While döngüsü
        content = re.sub(r'(try:)\s*(if not camera\.isOpened\(\):)', r'\1\n    \2', content)
        content = re.sub(r'(if not camera\.isOpened\(\):)\s*(raise RuntimeError)', r'\1\n        \2', content)
        content = re.sub(r'(raise RuntimeError\("Cannot open camera 0\."\))\s*(camera\.set)', r'\1\n    \2', content)
        content = re.sub(r'(camera\.set\(cv2\.CAP_PROP_FRAME_WIDTH,\s*video_size\[0\]\))\s*(camera\.set)', r'\1\n    \2', content)
        content = re.sub(r'(camera\.set\(cv2\.CAP_PROP_FRAME_HEIGHT,\s*video_size\[1\]\))\s*(camera\.set)', r'\1\n    \2', content)
        content = re.sub(r'(camera\.set\(cv2\.CAP_PROP_BUFFERSIZE,\s*1\))\s*(print\(f"Conversation started)', r'\1\n    \2', content)
        content = re.sub(r'(print\(f"Conversation started[^\n]+?\(Ctrl\+C to exit\)\.\.\."\))\s*(while True:)', r'\1\n\n    \2', content)
        
        content = re.sub(r'(while True:)\s*(audio_data\s*=)', r'\1\n        \2', content)
        content = re.sub(r'(audio_data\s*=\s*mic\.read[^\)]+\))\s*(conv\.append_audio)', r'\1\n        \2', content)
        content = re.sub(r'(conv\.append_audio\(base64\.b64encode\(audio_data\)\.decode\(\)\))\s*(success,\s*frame\s*=)', r'\1\n        \2', content)
        content = re.sub(r'(success,\s*frame\s*=\s*camera\.read\(\))\s*(if not success:)', r'\1\n        \2', content)
        content = re.sub(r'(if not success:)\s*(raise RuntimeError\("Cannot read a camera frame\."\))\s*(if time\.monotonic)', r'\1\n            \2\n        \3', content)
        content = re.sub(r'(if time\.monotonic\(\)\s*>=\s*next_frame_at:)\s*(frame\s*=)', r'\1\n            \2', content)
        content = re.sub(r'(frame\s*=\s*cv2\.resize[^\)]+\))\s*(success,\s*image\s*=)', r'\1\n            \2', content)
        content = re.sub(r'(success,\s*image\s*=\s*cv2\.imencode[^\)]+\))\s*(if not success:)', r'\1\n            \2', content)
        content = re.sub(r'(if not success:)\s*(raise RuntimeError\("Cannot encode a camera frame\."\))\s*(conv\.append_video)', r'\1\n                \2\n            \3', content)
        content = re.sub(r'(conv\.append_video\(base64\.b64encode\(image\)\.decode\(\)\))\s*(next_frame_at\s*=)', r'\1\n            \2', content)
        content = re.sub(r'(next_frame_at\s*=\s*time\.monotonic\(\)\s*\+\s*1\s*/\s*video_fps)\s*(time\.sleep)', r'\1\n        \2', content)
        
        # Except & Finally
        content = re.sub(r'(time\.sleep\(0\.01\))\s*(except KeyboardInterrupt:)', r'\1\n\2', content)
        content = re.sub(r'(except KeyboardInterrupt:)\s*(pass)\s*(finally:)', r'\1\n    \2\n\3', content)
        content = re.sub(r'(finally:)\s*(# Clean up resources)', r'\1\n    \2', content)
        content = re.sub(r'(# Clean up resources)\s*(camera\.release\(\))', r'\1\n    \2', content)
        content = re.sub(r'(camera\.release\(\))\s*(conv\.close\(\))', r'\1\n    \2', content)
        content = re.sub(r'(conv\.close\(\))\s*(mic\.close\(\))', r'\1\n    \2', content)
        content = re.sub(r'(mic\.close\(\))\s*(if callback\.out:)', r'\1\n    \2', content)
        content = re.sub(r'(if callback\.out:)\s*(callback\.out\.close\(\))\s*(pya\.terminate\(\))', r'\1\n        \2\n    \3', content)
        content = re.sub(r'(pya\.terminate\(\))\s*(print\("\\nConversation ended"\))', r'\1\n    \2', content)
        
        # Yorum satırları ve girinti (Indentation) restorasyonu
        content = re.sub(r'(?m)^(?:\#\s*)?Determine the voice\b', '# Determine the voice', content)
        content = re.sub(r'(?m)^(?:\#\s*)?Determine the model\b', '# Determine the model', content)
        content = re.sub(r'(?m)^(?:\#\s*)?Determine the model role\b', '# Determine the model role', content)
        content = re.sub(r'(?m)^\s*(?:\#\s*)?Initialize audio output stream\b', '        # Initialize audio output stream', content)
        content = re.sub(r'(?m)^\s*(?:\#\s*)?Play audio\b', '            # Play audio', content)
        content = re.sub(r'(?m)^\s*(?:\#\s*)?Streaming preview[^\n]*', '            # Streaming preview: text is the confirmed prefix, stash is the confirmed suffix', content)
        content = re.sub(r'(?m)^\s*(?:\#\s*)?Transcription completed[^\n]*', '            # Transcription completed, print the final text and a new line', content)
        content = re.sub(r'(?m)^\s*(?:\#\s*)?Print the assistant\'s response text\b', '            # Print the assistant\'s response text', content)
        content = re.sub(r'(?m)^(?:\#\s*)?1\.\s*Initialize audio device\b', '# 1. Initialize audio device', content)
        content = re.sub(r'(?m)^(?:\#\s*)?2\.\s*Create callback function and session\b', '# 2. Create callback function and session', content)
        content = re.sub(r'(?m)^(?:\#\s*)?3\.\s*Establish connection and configure session\b', '# 3. Establish connection and configure session', content)
        content = re.sub(r'(?m)^(?:\#\s*)?4\.\s*Initialize audio input stream\b', '# 4. Initialize audio input stream', content)
        content = re.sub(r'(?m)^(?:\#\s*)?5\.\s*Main loop to process audio and video input\b', '# 5. Main loop to process audio and video input', content)
        content = re.sub(r'(?m)^\s*(?:\#\s*)?Clean up resources\b', '    # Clean up resources', content)
        content = re.sub(r'(?m)^\s*preview = response\.get\b', '            preview = response.get', content)
        content = re.sub(r'(?m)^\s*print\(f"\\r\[User\][^"]+?"\)', '            print(f"\\\\r[User] {response[\'transcript\']}")', content)
    return content

def clean_content(content: str) -> str:
    if not content:
        return ""
    # 0. HTML Entity Çözümü (&amp; -> &, &quot; -> ", vb.)
    content = html.unescape(content)
    
    # 0.1 Kod Blokları ve Python Sözdizimi Restorasyonu
    content = restore_code_blocks_and_formatting(content)
    
    # 1. Başta yer alan navigasyon kalıntıları (Skip to content vb.)
    content = re.sub(r'(?i)^\s*Skip to (?:main )?content\s*\n+', '', content)
    
    # 1.1 Tom\'s Hardware Mega-Menu & Header Temizliği (ID: 057c6fca vb.)
    content = re.sub(r'(?is)^Get club Access Quick.*?(?:Tom\'?s Hardware Verdict|Reasons to buy|Reviews\s*\n+|When you purchase through links[^\n]*\n+)', '', content).strip()
    content = re.sub(r'(?is)^.*?(?:ExploreGO PREMIUM|Become a member\s*\n+Premium member|Hardware Coupons).*?(?:Tom\'?s Hardware Verdict|Reasons to buy|Reviews\s*\n+|When you purchase through links[^\n]*\n+)', '', content).strip()
    
    # 2. Markdown ve görsel temizliği
    content = re.sub(r'!\[.*?\](\(.*?\))?', '', content)
    content = re.sub(r'!image\b.*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    content = re.sub(r'\[\]\([^\)]+\)', '', content)
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
    content = re.sub(r'\*(.*?)\*', r'\1', content)
    
    # 3. Ham HTML etiketleri temizliği (span, a, div, p, form, vb.)
    content = re.sub(r'<[a-zA-Z\/][^>]*>', '', content)
    
    # 4. PHP ve CMS Şablon Kaçakları Temizliği (endif; ?>, <?php vb.)
    content = re.sub(r'<\?php.*?\?>', '', content, flags=re.DOTALL)
    content = re.sub(r'(?i)\bendif;\s*\?>?', '', content)
    content = re.sub(r'\?>', '', content)
    
    # 5. Galeri / Slayt UI Buton Tekrarları Temizliği (View Original, Slide x of y)
    content = re.sub(r'(?im)^\s*View Original\s*$\n?', '', content)
    content = re.sub(r'(?i)(?:\bView Original\b\s*){2,}', '', content)
    content = re.sub(r'(?im)^\s*(?:Slide \d+ of \d+|Click to enlarge|View larger image)\s*$\n?', '', content)
    
    # 6. Reklam, Sponsorlu Metin ve Yer Tutucu Temizliği
    content = re.sub(r'(?i)this is the title for the native ad.*', '', content)
    content = re.sub(r'(?i)\badvertisement\b.*', '', content)
    content = re.sub(r'(?i)\bsponsored content\b.*', '', content)
    
    # 6.1 SFGate ve Metin İçi Reklam Yer Tutucuları ("Article continues below this ad")
    content = re.sub(r'(?im)^\s*Article continues below (?:this )?ad\s*$\n?', '', content)
    content = re.sub(r'(?i)\bArticle continues below this ad\b\n*', '', content)
    
    # 6.2 Yahoo Finance Motley Fool Sponsorlu Reklam Blokları ("Missed AI's 'Act 1'?")
    content = re.sub(r'(?is)Missed AI\'s\s*["\']Act 1["\'].*?Continue\s*[»\>]\s*\n?', '', content)
    content = re.sub(r'(?is)Missed AI\'?s [^\n]+?\n\n.*?Continue\s*[»\>]\s*\n?', '', content)
    
    # 6.3 Ars Video Yer Tutucu Modülü
    content = re.sub(r'(?is)\n+Ars Video\s*\n+[^\n]+\s*\n+', '\n\n', content)
    content = re.sub(r'(?im)^\s*Ars Video\s*$\n?.*?\n\n', '', content)
    
    # 7. INLINE Bülten, Takip ve RSS Satırları (Tom\'s Hardware, AnandTech vb.)
    content = re.sub(r'(?i)^\s*Get [^\n]+(?:best news|newsletter|updates).*?(?:inbox|email)\.?\s*$\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'(?is)Get Tom\'s Hardware\'s best news.*?inbox\.?\n?', '', content)
    content = re.sub(r'(?i)^\s*Follow [^\n]+(?:Google News|preferred source|feeds).*?$\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'(?is)Follow Tom\'s Hardware on Google News.*?(?:feeds|\.)\n?', '', content)
    content = re.sub(r'(?i)^\s*Sign up for (?:the )?[^\n]+ newsletter.*?(?:inbox|email)\.?\s*$\n?', '', content, flags=re.MULTILINE)
    
    # 8. Evrensel Cookie, GDPR & Consent Banner Temizliği (Ethyca, OneTrust, TrustArc, GDPR)
    content = re.sub(r'(?is)^.*?(?:Privacy Center|Cookie Policy|Manage Privacy Preferences).*?(?:Sign in dialog\.\.\.\s*Sign in\s*\n+|Skip to (?:main )?content\s*\n+)', '', content).strip()
    content = re.sub(r'(?is)^.*?Consent powered by \w+.*?\n+', '', content).strip()
    content = re.sub(r'(?i)\bConsent powered by \w+\b.*?\n', '', content)
    content = re.sub(r'(?i)\bPrivacy Preferences\b.*?\n', '', content)
    
    # 9. Form ve Privacy Sözleşmesi / Aydınlatma Metinleri Temizliği
    content = re.sub(r'(?is)\n+(?:I agree to the collection and processing of the above information|I agree that the above information will be transferred|By submitting this form|By checking this box|You may contact us by sending an email to [^\s]+ to resolve related problems).*$', '', content).strip()
    content = re.sub(r'(?is)\n+.*?for the purposes of research and event organization, and I have read and agree to.*?Privacy Policy.*$', '', content).strip()
    
    # 10. Ars Technica / Condé Nast / Genel Footer ve Sayfa Sonu UI Çöpleri Temizliği
    content = re.sub(r'(?is)\n+(?:Customize\s*\n+)?(?:\[\]\(https?://[^\)]+\)\s*)*(?:Ars Technica has been separating the signal|All rights reserved\. Use of and/or registration|More from Ars|Sign in dialog\.\.\.\s*Sign in).*$', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Sign in dialog\.\.\.\s*Sign in\s*)+$', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Sign in dialog\.\.\.|Sign in)\s*$', '', content).strip()
    
    # 10.5 İlgili Haber Teaser\'ları (Digital Trends, The Verge vb. RELATED COVERAGE)
    content = re.sub(r'(?is)\n+(?:RELATED COVERAGE|Related Coverage|RELATED STORIES|Related Stories):?.*$', '', content).strip()
    
    # 11. Sayfa sonu Raw JSON array / UI widget dökümü temizliği
    content = re.sub(r'\n*\[\[+["\'][^\]]+["\'].*?\]\]+\s*$', '', content, flags=re.DOTALL).strip()
    
    # 12. Menü ve erişilebilirlik temizliği
    content = re.sub(r'(?i)___?skip to (?:main )?content.*?\n', '', content)
    content = re.sub(r'(?im)^(?:\*|-|\d+\.)\s+(?:\[x\]\s+)?(?:See All|Trending|Best|More|Menu|Home|Search|Browse|Share|Follow Us|Subscribe).*?\n', '', content)
    content = re.sub(r'(?i)(this site uses cookies|we use cookies|accept all cookies|accept only essential cookies).*', '', content)
    
    # 13. TV Haber Ticker / Hearst / WGAL Yerel Haber ve Revcontent Temizliği
    content = re.sub(r'(?is)^.*?Breaking News:.*?(?:Video \d+\s*)+\n*', '', content)
    content = re.sub(r'(?is)\n+(?:Top Picks Powered By Revcontent|HOURLY DAILY \d+ PM|Sponsored by WGAL).*$', '', content).strip()
    
    # 14. NPR Çerez Duvarı, Sponsor Mesajları ve Navigasyon Temizliği
    content = re.sub(r'(?is)^Accessibility links\s*\n+Loading\s*\n+.*?(?:Close Navigation Menu\s*\n+.*?\n\n)', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Cookie Settings|Consent Preferences|Strictly Necessary or Essential Cookies|Allow All\s+Manage Consent Preferences).*$', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Get NPR\+|Already have NPR\+|Close modal|Sign In\s*\n\s*Close modal).*$', '', content).strip()
    
    # 15. The Register Header Navigasyonu ve Footer / Site Haritası Temizliği
    content = re.sub(r'(?is)^Jump to main content\s*\n+Search\s*\n+TOPICS\s*\n+.*?(?=\n[A-Z0-9])', '', content).strip()
    content = re.sub(r'(?is)\n+(?:OPINION\s+##|Newsletter\s*\n\s*Our Websites|Situation Publishing|Your Consent Options|27 years of articles).*$', '', content).strip()
    
    # 15.5 CNN, Medya Anketleri (Ad Feedback), Header Navigasyonu ve Footer Sözleşmeleri
    content = re.sub(r'(?is)^Ad Feedback\s*\n+CNN values your feedback.*?(?:Link Copied!\s*\n+|See all topics\s*\n+)', '', content).strip()
    content = re.sub(r'(?is)^Facebook TweetEmailLink.*?\n+', '', content).strip()
    content = re.sub(r'(?is)^\s*Link Copied!\s*\n+', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Paid Content\s*\n+Subscribe|Most stock quote data provided by BATS|Terms of UsePrivacy Policy|By clicking ["\']?Agree["\']?, you agree to the Terms of Use).*$', '', content).strip()
    content = re.sub(r'(?is)\n+CNN Sans ™ & © \d+ Cable News Network.*$', '', content).strip()
    
    # 15.6 Evrensel Sayfa Başı Menü/Kategori Link Listeleri ve Video Reklam Temizliği
    content = re.sub(r'(?is)^\s*(?:(?:\*|-)\s+[A-Za-z0-9\s&,\.\-\/]{1,30}\n+){4,}', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Featured Political News|Video \d+\s*\d+/\d+\s*Skip Ad|Copyright ©\d+ [A-Za-z\s]+ Inc|NewslettersPodcastsDownload Our App).*$', '', content).strip()
    content = re.sub(r'(?is)\n*Video \d+\s*(?:\d+/\d+\s*)?Skip Ad.*?(?:Continue watching after the ad\s*\n*)+', '\n\n', content)
    content = re.sub(r'(?is)\n*Video \d+\s*\n*', '\n\n', content)
    
    # 8.5 Medya Oynatıcı / Çerez Onay Bandı Temizliği (RTÉ, BBC vb.)
    content = re.sub(r'(?is)\n*We need your consent to load this [^\n]+? content.*?(?:Manage Preferences|accept them to load the content\.?)\s*\n*', '\n\n', content)
    
    # 8.6 Kod Bloğu Öncesi Satır Numarası Sayaçları Temizliği (1 2 3 4 5 ... 28 vb.)
    content = re.sub(r'(?im)^\s*(?:python|bash|javascript|typescript|json|html|css|cpp|go|rust|java)\s*$\n+\s*(?:\d+\s+){3,}\d+\s*$\n*', '', content)
    content = re.sub(r'(?m)^\s*(?:\d+\s+){4,}\d+\s*$\n*', '', content)
    
    # 16. Forum Yorumları, Okuyucu Yorumları ve Most Read / Popüler Haber Widget Temizliği
    # Ars Technica yazar bio ve yorum / Most Read blokları
    content = re.sub(r'(?is)\n+[A-Za-z\s]+ is a Senior Technology Reporter at Ars Technica.*$', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Ryan WhitwamSenior Technology Reporter|Ryan Whitwam is a senior technology reporter at Ars Technica).*$', '', content).strip()
    content = re.sub(r'(?is)\n+[A-Za-z\s]+(?:Senior Technology Reporter|Senior Editor|Staff Writer|Contributor)\s*$', '', content).strip()
    content = re.sub(r'(?is)\n+\d+\s+Comments\b.*$', '', content).strip()
    content = re.sub(r'(?is)\n+Comments\s*\n+(?:Forum view|Prev story|Next story).*$', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Most Read|Trending Stories|Related Stories|Popular Stories|Top Stories)\s*\n+\d+\..*$', '', content).strip()
    content = re.sub(r'(?is)\n+Most Read\s+\d+\..*$', '', content).strip()
    content = re.sub(r'(?is)\n+\d+\s+Comments\s*\n+\d+\.?\s*$', '', content).strip()
    
    # Tom\'s Hardware yazar bio ve forum/okuyucu yorumları (ID: 057c6fca, 8389d9ab, 15d1e37b, c35f3a09)
    # Yazar biyografisi ve sonrasındaki TÜM yorumları kökten buda
    content = re.sub(r'(?is)\n+[^\n]+? is (?:the|an?) (?:Editor-in-Chief|Senior CPU Analyst|Senior Hardware Analyst|staff writer|senior editor|contributing writer|news contributor)[^\n]*?(?:at|for) Tom\'?[^\n]*?Hardware.*$', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Latest in CPUs|Latest in Reviews|Latest in [A-Za-z]+)\s*\n+.*$', '', content).strip()
    content = re.sub(r'(?is)\n+Show more comments\s*\n+.*$', '', content).strip()
    content = re.sub(r'(?is)\n+Tom\'?s Hardware is part of Future US Inc.*$', '', content).strip()
    content = re.sub(r'(?is)\n+What is the value of the included Carbice pad\?.*$', '', content).strip()
    content = re.sub(r'(?is)\n+.*?so the price of entry to the AM5 platform is very low now.*$', '', content).strip()
    content = re.sub(r'(?is)\n+.*?(?:Smart move\. I\'d much rather a few weeks delay|And pretty much like every other enthusiest|It won\'t have a big effect if this is a limited run).*$', '', content).strip()
    content = re.sub(r'(?is)\n+.*?(?:There are regressions respect 7950X|Thanks for the review\.\s*Reply).*$', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Reply\s*\n+)?(?:Sure, this gen there aren\'?t many mobo improvements|These ["\']soft["\'] launches can\'?t be helping).*$', '', content).strip()
    content = re.sub(r'(?is)\n+-\s*Phoronix\s+Reply\s*\n+.*$', '', content).strip()
    content = re.sub(r'(?is)\n+(?:Join the conversation|Join the discussion|Latest Comments|Leave a Reply|Regards\.\s*Reply|Reply\s*\n\s*Report|Comments?\s*\(\d+\)).*$', '', content).strip()
    content = re.sub(r'\[\/?EMBED\]', '', content).strip()
    
    # OpenAI Discourse Forum kullanıcı/beğeni sayaçları (ID: 89352a7c, 8e05f1f2: "jefff \n\n 6 \n\n", "VeitB \n\n 10 \n\n")
    content = re.sub(r'(?m)^\s*[a-zA-Z0-9_\-\.]{2,25}\s*\n+\s*\d{1,4}\s*$\n*', '', content)
    
    # 16.5 Yarım kalmış tek satır alıntı/pull-quote veya kesik edat kalıntıları (Örn: "“They’ll put an\n")
    content = re.sub(r'(?m)^[\s"“\'\-\*]*[A-Za-z\s]{2,30}\b(?:an|a|the|in|on|at|of|to|for|with|by|and|or)\s*$\n+', '', content)
    
    # 16.6 Sayfa sonu boş başlıklar (Quick Links:, Media Downloads:), Disqus ve Source/Via satırları
    content = re.sub(r'(?im)^\s*(?:Quick Links|Media Downloads|Downloads|Related Links|Source|Via):\s*$', '', content).strip()
    content = re.sub(r'(?is)\n+(?:comments powered by disqus|Source:\s*[^\n]+|Via:\s*[^\n]+)\s*$', '', content).strip()
    
    # Satır içi aşırı whitespace temizliği
    cleaned_lines = [re.sub(r'[ \t]+', ' ', l).strip() for l in content.split('\n')]
    content = '\n'.join(cleaned_lines)
    
    # Fazla boş satırları birleştir
    content = re.sub(r'\n{3,}', '\n\n', content).strip()
    
    # 17. Evrensel Mükerrer Paragraf ve Blok De-duplication (Tekilleştirme)
    paragraphs = content.split('\n\n')
    deduped = []
    seen_paras = set()
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        p_lower = p_clean.lower()
        if p_lower in seen_paras:
            continue
        # Alt küme / iç içe DOM tekrarı kontrolü (nested etiket tekrarlarını yok eder)
        is_subset = False
        for prev in deduped[-5:]:
            prev_lower = prev.lower()
            if len(p_clean) > 80 and p_lower in prev_lower:
                is_subset = True
                break
        if is_subset:
            continue
        deduped.append(p_clean)
        seen_paras.add(p_lower)
        
    final_text = '\n\n'.join(deduped).strip()
    return restore_code_blocks_and_formatting(final_text)

def is_clean_completed_article(text: str) -> bool:
    """Metnin ortasında kesilmemiş, düzgün sonlanmış geçerli bir makale olup olmadığını doğrular."""
    if not text or len(text.strip()) < 450:
        return False
    t = text.strip()
    lower_t = t.lower()
    
    # Anti-Bot / Cloudflare / JavaScript Gereksinim Ekranı Tespiti (Seeking Alpha vb.)
    bot_signals = [
        "please enable javascript and cookies",
        "to ensure this doesn't happen in the future",
        "to ensure this doesn’t happen in the future",
        "if you have an ad-blocker enabled you may be blocked",
        "why have i been blocked",
        "sorry, you have been blocked",
        "access to this page has been denied",
        "security check to access",
        "please verify you are a human"
    ]
    if any(bs in lower_t for bs in bot_signals):
        return False

    truncated_tails = ["&genai.", "§3.", "Math 2.", "image.save(\"edit.", "Against Gemini 3."]
    if any(t.endswith(tt) or tt in t[-50:] for tt in truncated_tails):
        return False
    valid_end_chars = ('.', '!', '?', '"', '”', "'", "’", ')', ']', '}', '`', '>', ';', '®', '™', '…', '—')
    last_line = t.split('\n')[-1].strip()
    if len(last_line) > 30 and not last_line.endswith(valid_end_chars):
        return False
    return True

def is_thin_or_shell_content(text: str, title: str = "") -> bool:
    """Evrensel Thin Content ve Canlı Yayın (Liveblog) kabuk tespiti."""
    if not text:
        return True
    lower_t = text.lower()
    lower_title = (title or "").lower()
    
    # 1. Liveblog kabuk kapanış/açılış sinyalleri (Engadget, Verge, CNET vb.)
    if "updates\n- thank you all for joining us today" in lower_t or "updates - thank you all for joining us" in lower_t:
        return True
    if "thank you all for joining us today" in lower_t and ("stay tuned, we'll have more" in lower_t or "165 updates" in lower_t):
        return True
        
    # 2. Başlıkta canlı yayın / live updates olup gerçek dökümü olmayan stub'lar
    if any(lt in lower_title for lt in ["live blog", "live updates", "live coverage", "live reporting"]):
        stub_phrases = [
            "thank you all for joining us", "stay tuned, we'll have more",
            "follow along below for live updates", "welcome to our live coverage",
            "our live blog has ended"
        ]
        if any(sp in lower_t for sp in stub_phrases) and len(text.strip()) < 3500:
            return True

    return False

def is_error_page(text: str) -> bool:
    """Evrensel hata, 404, 403, 500 ve eksik sayfa denetimi."""
    if not text:
        return True
    lower = text.lower()
    error_signals = [
        "404 file not found", "404 - file not found", "404 - page not found",
        "404 page not found", "we couldn't find that page", "we couldn’t find that page",
        "the site configured at this address does not contain the requested file",
        "error 404", "404 error", "404 not found",
        "the page you are looking for doesn't exist",
        "the page you were looking for doesn't exist",
        "the page you requested could not be found",
        "does not contain the requested file",
        "the requested url was not found",
        "this page could not be found",
        "whoops, looks like something went wrong",
        "sorry, this page isn't available", "sorry, this page isn’t available",
        "does not contain the path", "incident.io/?utm_source", "page not found",
        # Anti-Bot & Captcha & JavaScript Engel Ekranı Sinyalleri
        "please enable javascript and cookies",
        "to ensure this doesn't happen in the future, please enable",
        "to ensure this doesn’t happen in the future",
        "if you have an ad-blocker enabled you may be blocked",
        "why have i been blocked", "sorry, you have been blocked",
        "access to this page has been denied", "security check to access",
        "please verify you are a human"
    ]
    if any(sig in lower for sig in error_signals):
        return True
    header_chunk = lower[:300]
    if "404" in header_chunk and ("not found" in header_chunk or "file not" in header_chunk):
        return True
    return False

def is_paywall_page(text: str) -> bool:
    """Evrensel dijital abonelik ve paywall ekranı tespiti."""
    if not text:
        return False
    lower = text.lower()
    paywall_signals = [
        "subscribe to read", "premium article", "subscribers only",
        "already a subscriber? log in", "already a subscriber? sign in",
        "this content is for subscribers", "to read the full article, subscribe",
        "complete digital access", "per month. complete digital access",
        "trial subscription", "cancel anytime during your trial",
        "start your trial", "start your free trial to keep reading",
        "become a member to read the full story", "to continue reading, please subscribe",
        "read this story and more with a subscription", "per month. cancel anytime",
        "subscription required to access", "unlock this story with a subscription",
        # Stratechery, Substack ve bülten üyelik kalıpları
        "subscribe to stratechery", "for full access", "already subscribed?",
        "subscriber-only", "frequently-asked questions\nhow do i subscribe",
        "how do i subscribe to the"
    ]
    if any(sig in lower for sig in paywall_signals):
        return True
    if re.search(r'\$\d+\s*/\s*(?:month|year)', lower) and ("subscribe" in lower or "subscribed" in lower):
        return True
    return False

def is_profane_or_toxic(text: str) -> bool:
    """Evrensel NLP: Aşırı kaba/toksik veya küfürlü kişisel blog tespiti."""
    if not text:
        return False
    lower = text.lower()
    toxic_words = ["fuck", "fucking", "shit", "bullshit", "asshole", "bitch", "dogshit"]
    hits = sum(1 for w in toxic_words if re.search(r'\b' + re.escape(w) + r'\b', lower))
    return hits >= 2

def is_forum_or_nav_dump(text: str) -> bool:
    """Evrensel forum navigasyon menüsü veya bakım sayfası dökümü tespiti."""
    if not text:
        return True
    lower = text.lower()
    nav_signals = [
        "developer central\noffering tools, docs, and support",
        "community updates\nwe're expanding",
        "technical blogs\ntechnical information portal",
        "skip to main content\ncommunity updates",
        "browse by product or category",
        "filter by solution or topic"
    ]
    if any(sig in lower for sig in nav_signals):
        return True
    first_chunk = lower[:300]
    if "skip to main content" in first_chunk and ("navigation" in first_chunk or "developer central" in first_chunk):
        return True
    return False

def is_ui_button_dump(text: str) -> bool:
    """Evrensel Heuristic NLP: Metin bir makale mi yoksa UI buton/menü listesi mi?"""
    if not text:
        return True
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) >= 6:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        punct_count = sum(1 for l in lines if l.endswith(('.', '!', '?', ':', '"', '”', '’')))
        punct_ratio = punct_count / len(lines)
        if avg_line_len < 35 and punct_ratio < 0.35:
            return True
    return False

def extract_query_terms(query: str) -> List[str]:
    """Sorgudan stop-word'leri ayıklayıp anlamlı varlık terimlerini çıkarır."""
    stop_words = {'and', 'the', 'for', 'with', 'from', 'about', 'in', 'on', 'at', 'to', 'of'}
    return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', query) if w.lower() not in stop_words]

def validate_entity_match(query: str, target_text: str, published_at: str = None, title: str = "") -> bool:
    """Evrensel varlık doğrulama motoru."""
    terms = extract_query_terms(query)
    if not terms:
        return True
    comb = target_text.lower()
    lower_title = (title or "").lower()
    
    # 1. Tek kelimeli varlık (örn: Qwen, Mistral, Groq, Rust, Claude):
    if len(terms) == 1:
        if not re.search(r'\b' + re.escape(terms[0]) + r'\b', comb):
            return False
    else:
        # 2. Çok kelimeli varlık (örn: Nvidia Spark, AMD Ryzen, DeepSeek R1, Middle East):
        # Ayrışmış n-gram tokenizer hatalarını önleme: Tam öbek (phrase) aranır!
        phrase_regex = r'\b' + r'\s+'.join(re.escape(t) for t in terms) + r'\b'
        if not re.search(phrase_regex, comb) and not re.search(phrase_regex, lower_title):
            return False
        for t in terms:
            if not re.search(r'\b' + re.escape(t) + r'\b', comb):
                return False

    # 3. Ana Varlık Konum Kuralı:
    lead_text = ((title or "") + " " + target_text[:600]).lower()
    if not any(re.search(r'\b' + re.escape(t) + r'\b', lead_text) for t in terms):
        return False

    # 4. Mecazi ve Analojik Dil Sızıntısı Filtresi:
    # Örn: "SpaceX IPO reads like Hollywood fantasy", "PHP: The Toyota Corolla of programming"
    for t in terms:
        if re.search(rf'\blike\s+{re.escape(t)}\b', lower_title):
            return False
        if re.search(rf'\b(?:the\s+)?{re.escape(t)}\s+of\b', lower_title) and not lower_title.startswith(t):
            return False
            
    # 5. Eşadlılık (Polysemy) ve Alan Sınıflandırması ("stocks"):
    if query.lower().strip() == "stocks":
        fin_kw = ["share", "equity", "market", "nasdaq", "nyse", "investor", "dividend", "trading", "wall street", "portfolio", "earnings", "rally", "s&p", "stock market", "index"]
        inv_kw = ["weapons stocks", "depletes stocks", "diesel stocks", "oil reserve", "crude stocks", "grain stocks", "inventory", "stockpile"]
        has_fin = any(re.search(r'\b' + re.escape(w) + r'\b', comb) for w in fin_kw)
        has_inv = any(re.search(r'\b' + re.escape(w) + r'\b', comb) for w in inv_kw)
        if has_inv and not has_fin:
            return False
        if ("git clone" in comb or "npm install" in comb or "pip install" in comb) and len(comb) < 2000:
            return False

    # 6. Parodi ve Anakronizm Filtresi ("nobel"):
    if "nobel" in query.lower():
        if re.search(r'\big\s+nobel\b', comb) or re.search(r'\big\s+nobel\b', lower_title):
            return False
        if published_at:
            try:
                year = int(published_at[:4])
                if year < 2015:
                    return False
            except Exception:
                pass
                
    # 7. Japonya Özel Konu Sapması ("japan"):
    if query.lower().strip() == "japan":
        if "tee times" in lower_title or "golf channel" in comb[:500]:
            return False
        if lower_title.startswith("germany") and "pension" in lower_title:
            return False

    # 8. Bilinen kavram çakışmaları (Domain/Topic Collisions - Nvidia Spark):
    if "spark" in query.lower():
        spark_verb_pattern = re.compile(
            r'(?i)\b(?:help(?:ed|s)?|to|could|may|might|will|did)\s+spark\b|'
            r'\b(?:earnings|results|report|numbers|shares|stock|rally)\s+spark(?:s|ed|ing)?\b|'
            r'\bspark(?:s|ed|ing)?\s+(?:a|an|the|nvidia\'s|stock|shares)?\s*(?:selloff|rout|rally|drop|fall|plunge|surge|slide|losses|gains|concern|fears|debate|frenzy|wave|buying|selling|collapse)\b|'
            r'\bsparked\s+by\b'
        )
        if spark_verb_pattern.search(comb) or spark_verb_pattern.search(title or ""):
            return False
            
        spark_fakes = [
            "apache spark", "spark 3.", "spark-rapids", "ada/spark", "adacore",
            "iso-26262", "spark process", "skate park", "gpu shark", "tegra parker"
        ]
        if any(sf in comb for sf in spark_fakes):
            return False
            
        hw_keywords = [
            "rtx spark", "dgx spark", "spark superchip", "gb10", "gb300", "n1x", "grace",
            "blackwell", "superchip", "arm-based pc", "personal ai supercomputer", "copilot+ pc"
        ]
        has_hw = any(hw in comb for hw in hw_keywords) or any(hw in (title or "").lower() for hw in hw_keywords)
        if not has_hw:
            return False
            
    # 9. Aşırı eski tarihi eser haberler (teknoloji donanım haberlerinde 5 yıldan eski haberler)
    if published_at and ("spark" in query.lower() or "ryzen" in query.lower() or "nemotron" in query.lower()):
        try:
            year = int(published_at[:4])
            if year < 2021:
                return False
        except Exception:
            pass
            
    return True

def fetch_full_article(url: str, query: str = "", title: str = "") -> dict:
    try:
        import trafilatura
        import requests as std_requests
        from bs4 import BeautifulSoup
        
        if url.startswith("//"): url = "https:" + url
        elif not url.startswith("http"): url = "https://" + url

        html = None
        published_at = None
        date_source = None
        
        try:
            req_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9"
            }
            res = std_requests.get(url, headers=req_headers, timeout=FETCH_TIMEOUT)  # tarayıcı taklidi yok (E5)
            if res.status_code in [200, 201, 202]:
                html = res.text
            elif res.status_code in [403, 429]:
                return {"content": None, "status": "blocked", "published_at": None, "date_source": None}
            elif res.status_code == 402:
                return {"content": None, "status": "paywall", "published_at": None, "date_source": None}
            elif res.status_code in [404, 410]:
                return {"content": None, "status": "failed", "published_at": None, "date_source": None}
        except Exception:
            pass
            
        content = None
        if html:
            lower_html = html.lower()
            bot_html_signals = [
                "cf-chl-bypass", "challenge-running", "just a moment...",
                "attention required! | cloudflare", "verify you are human",
                "prove your humanity", "turn on javascript", "access denied",
                "blocked by network security"
            ]
            if any(sig in lower_html for sig in bot_html_signals):
                return {"content": None, "status": "blocked", "published_at": None, "date_source": None}

            if not is_article(html, url):
                return {"content": None, "status": "failed", "published_at": None, "date_source": None}
                
            published_at, date_source = extract_meta_date(html)
            parsed_domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
            
            # 1. GitHub Özel Seçicisi: Genel sayfa iskeleti/navigasyon yerine sadece README çekilir
            if "github.com" in parsed_domain and "gist.github.com" not in parsed_domain:
                soup = BeautifulSoup(html, "html.parser")
                readme = soup.find("article", class_=re.compile(r'markdown-body', re.I)) or soup.find("div", id="readme")
                if readme:
                    for u in readme(["script", "style", "svg", "button"]):
                        u.decompose()
                    
                    lines = []
                    block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'pre']
                    for el in readme.find_all(block_tags):
                        if not el.find(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre']):
                            txt = el.get_text().strip()
                            txt = re.sub(r'[ \t]+', ' ', txt)
                            if txt:
                                lines.append(txt)
                    readme_text = "\n\n".join(lines).strip()
                    if len(readme_text) >= 300:
                        content = readme_text
                    else:
                        return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
                else:
                    return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
            else:
                # 1. Trafilatura öncelikli çıkarım (PCMag vb. sitelerdeki nested DOM tekrarlarını önler)
                traf_content = trafilatura.extract(html, include_comments=False, include_tables=False)
                is_feed = is_sidebar_or_headline_feed(traf_content or "", title=title)
                
                # 2. Eğer Trafilatura boşsa, yetersizse (<300) veya yanlışlıkla kenar çubuğunu/feed'i çektiyse:
                # Evrensel HTML5 BeautifulSoup makale çıkarıcısına geç (Jenerik <article>, <main> seçicisi)
                if not traf_content or len(traf_content.strip()) < 300 or is_feed:
                    generic_soup_content = extract_generic_article_soup(html)
                    if generic_soup_content:
                        content = generic_soup_content
                    else:
                        content = traf_content if not is_feed else None
                else:
                    content = traf_content
                
        # Jina, Google cache ve archive.org yedekleri kaldırıldı (E5: paywall/bot korumasını aşmak kullanım
        # koşullarına aykırı olabilir; N3: makale başına toplam bekleme ~57 s'ye çıkıyordu). Alınamayan haber atlanır.

        if content:
            content = clean_content(content)
            lower_c = content.lower()
            
            # E1: Thin Content & Canlı Yayın Kabuğu (Liveblog Stub) Tespiti
            if is_thin_or_shell_content(content, title):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}

            # E1: 404 / Hata Ekranı Tespiti
            if is_error_page(content):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
                
            # E1: Paywall Tespiti
            if is_paywall_page(content):
                return {"content": None, "status": "paywall", "published_at": published_at, "date_source": date_source}
                
            # E1: Forum Gezinti / Portal Menüsü Tespiti
            if is_forum_or_nav_dump(content):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
                
            # UI Buton Listesi Tespiti
            if is_ui_button_dump(content):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
                
            # Cold-boot / Server waking up / Loading log ekranı tespiti
            cold_boot_signals = [
                "incoming http request detected", "service waking up",
                "allocating compute resources", "preparing instance for initialization",
                "your app is almost live", "spinning up container", "server is sleeping",
                "steady hands. clean logs.", "service is waking up", "instance is initializing"
            ]
            if any(sig in lower_c for sig in cold_boot_signals):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
                
            # Crawl / Wayback Stub / Yönlendirme ve 404 hatası tespiti
            crawl_stub_signals = [
                "http 301 response at crawl time", "got an http 301", "got an http 302",
                "fight for the future's letter", "keep the news in the wayback machine",
                "loading...\nhttps://www.reddit.com", "the wayback machine - https",
                "about this capture\ntimestamps", "this capture\ntimestamps\nthe wayback machine",
                "page not found", "404 not found", "error 404", "the requested page could not be found"
            ]
            is_wayback_stub = any(sig in lower_c for sig in crawl_stub_signals) or ("wayback machine" in lower_c and "at crawl time" in lower_c) or ("captures\n" in lower_c and "timestamps\nthe wayback machine" in lower_c)
            if is_wayback_stub:
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
                
            # Paywall / Abonelik engeli tespiti
            paywall_signals = [
                "kyodo news plus is a subscription service", "to read the full story",
                "subscribe to continue reading", "subscribe to read the full article",
                "this article is for subscribers only", "exclusive content for subscribers",
                "article no longer available"
            ]
            if any(pw in lower_c for pw in paywall_signals):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
                
            # Oyun / WebGL Canvas yönergeleri tespiti
            game_canvas_signals = [
                "wasd or arrow keys to walk", "this garden needs webgl", "needs webgl, which isn't available",
                "wasd or arrow keys to move", "drag with the mouse to look around", "click the ground to walk there"
            ]
            if any(g in lower_c for g in game_canvas_signals):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
                
            # Bot/Erişim engeli kontrolü
            bot_keywords = ["prove your humanity", "blocked by network security", "turn on javascript", "access denied", "security check", "checking your browser before accessing"]
            if any(bk in lower_c for bk in bot_keywords):
                return {"content": None, "status": "blocked", "published_at": published_at, "date_source": date_source}
                
            # Saf SaaS vitrini / Landing Page kontrolü
            saas_signals = ["choose your plan", "pricing plan", "$199/month", "3 days free trial", "start your free trial", "cancel anytime. start for free", "get early access"]
            if any(ss in lower_c for ss in saas_signals):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
            
            # Finans / Borsa / Hedge Fund Danışmanlık ve Bülten Reklamı Filtresi (Yahoo Finance spam vb.)
            stock_newsletter_signals = [
                "stocks that hedge funds pile into", "outperform the market by imitating the top stock picks",
                "taking wall street by storm", "returned 275% since", "published a list of 15 ai news updates",
                "is not on our list of the 30 most popular stocks", "our research has shown that we can outperform",
                "insider monkey", "motley fool stock advisor"
            ]
            if any(sig in lower_c for sig in stock_newsletter_signals):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
            
            # Kaba / Toksik / Uygunsuz Blog Kontrolü
            if is_profane_or_toxic(content):
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
                
            # EVRENSEL VE JENERİK VARLIK TUTARLILIK KONTROLÜ
            if query and len(content) > 400:
                if not validate_entity_match(query, content, published_at=published_at, title=title):
                    return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}

            # İçerik Yokluğu (failed) ile Kısalık (too_short) Ayrımı (En az 450 karakter)
            if not content or len(content.strip()) == 0:
                return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
            elif len(content.strip()) < 450:
                return {"content": None, "status": "too_short", "published_at": published_at, "date_source": date_source}
            
            return {"content": content, "status": "ok", "published_at": published_at, "date_source": date_source}
            
        return {"content": None, "status": "failed", "published_at": published_at, "date_source": date_source}
    except Exception:
        return {"content": None, "status": "failed", "published_at": None, "date_source": None}

def search_articles(query: str, min_confidence: int = 30) -> List[dict]:
    from ddgs import DDGS
    import feedparser
    import requests
    
    results = []
    
    # 1. Hacker News API
    try:
        hn_url = f"https://hn.algolia.com/api/v1/search_by_date?query={urllib.parse.quote(query)}&tags=story&hitsPerPage={MAX_RESULTS}&numericFilters=created_at_i%3E{int(time.time()) - MAX_AGE_DAYS * 86400}"
        hn_res = requests.get(hn_url, timeout=5).json()
        for hit in hn_res.get("hits", []):
            url = hit.get("url")
            if url:
                results.append({
                    "title": hit.get("title", ""),
                    "url": url,
                    "body": hit.get("story_text", ""),
                    "source": "Hacker News",
                    "date_raw": hit.get("created_at_i"),
                    "date_source": "hacker_news_api",
                    "authority_bonus": 12.0
                })
    except Exception:
        pass
        
    # 2. DuckDuckGo Haberler
    try:
        ddgs = DDGS()
        news_results = list(ddgs.news(query, timelimit=_timelimit(), max_results=MAX_RESULTS))
        for r in news_results:
            r["date_raw"] = r.get("date")
            r["date_source"] = "ddg_news"
            results.append(r)
    except Exception:
        pass
        
    # 3. Google News RSS
    try:
        encoded_query = urllib.parse.quote(query)
        gnews_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(gnews_url)
        for entry in feed.entries:
            results.append({
                "title": entry.title,
                "url": entry.link,
                "body": entry.get("summary", ""),
                "source": entry.source.title if hasattr(entry, 'source') else "",
                "date_raw": entry.get("published", ""),
                "date_source": "google_news_rss"
            })
    except Exception:
        pass
        
    # 5. Yahoo News RSS
    try:
        encoded_query = urllib.parse.quote(query)
        yahoo_url = f"https://news.search.yahoo.com/rss?p={encoded_query}"
        for entry in feedparser.parse(yahoo_url).entries:
            results.append({
                "title": entry.title,
                "url": entry.link,
                "body": entry.get("summary", ""),
                "source": "Yahoo",
                "date_raw": entry.get("published", ""),
                "date_source": "yahoo_news_rss"
            })
    except Exception:
        pass
        
    # 6. Reddit API (En sıcak paylaşımlar)
    try:
        reddit_url = f"https://www.reddit.com/search.json?q={urllib.parse.quote(query)}&sort=new&limit={MAX_RESULTS}"
        reddit_res = requests.get(reddit_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=10).json()
        for child in reddit_res.get("data", {}).get("children", []):
            post = child.get("data", {})
            post_url = post.get("url")
            if post_url and not post_url.startswith("/r/"):
                results.append({
                    "title": post.get("title", ""),
                    "url": post_url,
                    "body": post.get("selftext", ""),
                    "source": "Reddit",
                    "date_raw": post.get("created_utc"),
                    "date_source": "reddit_api",
                    "authority_bonus": 5.0
                })
    except Exception:
        pass
            
    # 7. Qwant News API
    try:
        qwant_url = f"https://api.qwant.com/v3/search/news?count=100&q={urllib.parse.quote(query)}&t=news&safesearch=1&locale=en_US"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        q_res = requests.get(qwant_url, headers=headers, timeout=10).json()
        for item in q_res.get("data", {}).get("result", {}).get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "body": item.get("desc", ""),
                "source": item.get("source", ""),
                "date_raw": item.get("date", ""),
                "date_source": "qwant_api"
            })
    except Exception:
        pass
            
    # 4. DuckDuckGo Web Arama
    if len(results) < 50:
        try:
            ddgs = DDGS()
            raw_text = list(ddgs.text(query, timelimit=_timelimit(), max_results=MAX_RESULTS))
            for r in raw_text:
                url = r.get("href", "")
                parsed = urllib.parse.urlparse(url)
                if len(parsed.path) > 15 and "support" not in parsed.netloc and "sign" not in parsed.netloc:
                    r["url"] = url
                    r["date_source"] = "ddg_web"
                    results.append(r)
        except Exception:
            pass
                    
    if not results:
        return []
        
    def get_words(text):
        return set(w for w in text.lower().replace(",", "").replace(".", "").split() if len(w) > 3)

    unique_results = []
    seen_urls = set()
    
    # E3: Kopyaların URL Normalize edilip tekilleştirilmesi ve EVRENSEL ARAMA DOĞRULAMASI
    for r in results:
        url = r.get("url") or r.get("href", "")
        normalized = normalize_url(url)
        if not normalized or normalized in seen_urls:
            continue
        netloc = urllib.parse.urlparse(normalized).netloc.replace("www.", "")
        if netloc.endswith(REPO_DOMAINS):  # N2
            continue
        if netloc == "news.google.com":  # JavaScript yönlendirmesi: Jina olmadan açılmıyor, indirme hakkını boşa harcıyordu
            continue
            
        r_title = r.get("title", "")
        r_body = r.get("body", "")
        combined = f"{r_title} {r_body} {normalized}".lower()
        
        if not validate_entity_match(query, combined):
            continue
            
        # Bilinen belirgin kavram sapmaları / Collision engelleme
        if "spark" in query.lower():
            if any(cs in combined for cs in ["apache spark", "ada and spark", "adacore", "spark 3.0", "spark-rapids"]):
                continue
        if "qwen" in query.lower():
            if any(cq in combined for cq in ["gwen stefani", "blake shelton", "chernobyl", "oksana masters"]):
                continue
            
        r["url"] = normalized
        seen_urls.add(normalized)
        unique_results.append(r)

    now_utc = datetime.now(timezone.utc)
    articles_data = []
    
    for r in unique_results:
        r_title = r.get("title", "")
        if not r_title:
            continue
            
        r_words = get_words(r_title)
        r_url = r.get("url")
        r_domain = urllib.parse.urlparse(r_url).netloc.replace("www.", "")
        
        similar_domains = set()
        total_overlap_score = 0.0
        
        if r_words:
            for other in unique_results:
                o_url = other.get("url")
                o_domain = urllib.parse.urlparse(o_url).netloc.replace("www.", "")
                if o_domain != r_domain:
                    o_words = get_words(other.get("title", ""))
                    if not o_words:
                        continue
                    intersection = len(r_words.intersection(o_words))
                    overlap_ratio = intersection / max(len(r_words), 1)
                    if overlap_ratio >= 0.35:
                        similar_domains.add(o_domain)
                        total_overlap_score += overlap_ratio
        
        domain_count = len(similar_domains)
        if domain_count == 0:
            calculated_score = 25.0 + (len(r_title) % 10)
        else:
            calculated_score = 40.0 + (domain_count * 5.0) + (total_overlap_score * 3.0)
            
        q_words = set(query.lower().split())
        matches = len(q_words.intersection(r_words))
        calculated_score += (matches * 6.0)
        
        pub_date_str = None
        recency_bonus = 0.0
        parsed_dt = parse_date(r.get("date_raw"))
        if parsed_dt and _too_old(parsed_dt.isoformat()):  # N1: eski haber hiç indirilmez
            continue
        if parsed_dt:
            pub_date_str = parsed_dt.isoformat()
            try:
                hours_diff = (now_utc - parsed_dt).total_seconds() / 3600.0
                if hours_diff < 0: hours_diff = 100
                
                # E2: Sadece tarihi olanlar tazelik bonusu alıyor
                if hours_diff <= 2:
                    recency_bonus = 25.0
                elif hours_diff <= 12:
                    recency_bonus = 18.0
                elif hours_diff <= 24:
                    recency_bonus = 10.0
                elif hours_diff <= 72:
                    recency_bonus = 4.0
            except Exception:
                pass
                
        calculated_score += recency_bonus
        
        auth_bonus = r.get("authority_bonus", 0.0)
        if "ycombinator.com" in r_domain or "reddit.com" in r_domain:  # github.com bonusu kaldırıldı (N2)
            auth_bonus += 12.0
        elif "bloomberg.com" in r_domain or "reuters.com" in r_domain or "techcrunch.com" in r_domain:
            auth_bonus += 5.0
            
        calculated_score += auth_bonus
        final_score = int(round(calculated_score))
        final_score = max(15, min(99, final_score))
        
        raw_source = r_domain
        url_hash = hashlib.md5(r_url.encode('utf-8')).hexdigest()
        verified = (final_score >= 30 or auth_bonus >= 6)
        
        articles_data.append({
            "id": url_hash,
            "title": r_title,
            "source": raw_source,
            "url": r_url,
            "relevance_score": final_score,
            "is_verified": verified,
            "published_at": pub_date_str,
            "date_source": r.get("date_source"),
            "origin": f"query:{query}"
        })
        
    articles_data.sort(key=lambda x: x["relevance_score"], reverse=True)
    return [a for a in articles_data if a["relevance_score"] > min_confidence]

if __name__ == "__main__":
    query = "nvidia spark"
    output_file = None
    
    args = sys.argv[1:]
    if "-o" in args:
        o_idx = args.index("-o")
        output_file = args[o_idx + 1]
        args = args[:o_idx] + args[o_idx + 2:]
    elif "--output" in args:
        o_idx = args.index("--output")
        output_file = args[o_idx + 1]
        args = args[:o_idx] + args[o_idx + 2:]
        
    def take(flag):
        """--gun 7 gibi bir ayarı argümanlardan çıkar."""
        global args
        if flag not in args:
            return None
        i = args.index(flag)
        value = args[i + 1] if i + 1 < len(args) else None
        args = args[:i] + args[i + 2:]
        return value

    MAX_AGE_DAYS = int(take("--gun") or MAX_AGE_DAYS)
    MAX_FETCH = int(take("--adet") or MAX_FETCH)
    if "--tam" in args:
        args.remove("--tam")
        OUTPUT_LIMIT_KB = 0

    if args:
        query = " ".join(args)
        
    if not output_file:
        safe_q = re.sub(r'[^a-zA-Z0-9_-]', '_', query.lower().strip())
        output_file = f"{safe_q}.json"
        
    print(f"[{query}] için haberler ve makaleler aranıyor...")
    print(f"Çıktı dosyası: {output_file}")
    
    t_start = time.time()
    articles = search_articles(query, min_confidence=30)[:MAX_FETCH]  # en yüksek puanlılar (N3)
    print(f"Toplam {len(articles)} adet anlamlı makale bulundu.\nTam metinleri çekiliyor, bu biraz zaman alabilir...")
    
    results = []
    now_iso = datetime.now(timezone.utc).isoformat()
    
    import concurrent.futures

    def process_single_article(a_data: dict, now_iso: str, current_query: str) -> dict:
        extracted = fetch_full_article(a_data["url"], query=current_query, title=a_data["title"])
        content_status = extracted.get("status", "failed")
        content = extracted.get("content")
        
        # tekrar deneme kaldırıldı: başarısız makale bütün bekleme zincirini ikinci kez çalıştırıyordu (N3)
        pub_at = extracted.get("published_at")
        date_source = extracted.get("date_source")
        if not pub_at:
            pub_at = a_data.get("published_at")
            date_source = a_data.get("date_source")
            
        is_verified = a_data.get("is_verified", False)
        if not content or content_status != "ok":
            content = None
            is_verified = False
            
        article_obj = Article(
            id=a_data["id"],
            title=a_data["title"],
            source=a_data["source"],
            url=a_data["url"],
            relevance_score=a_data["relevance_score"],
            is_verified=is_verified,
            published_at=pub_at,
            content=content,
            content_status=content_status,
            origin=a_data["origin"],
            fetched_at=now_iso,
            date_source=date_source
        )
        return article_obj.to_dict()

    max_workers = 16
    completed_count = 0
    total_articles = len(articles)
    print(f"Coklu is parcacigi baslatildi ({max_workers} eszamanli worker)...\n")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(process_single_article, a, now_iso, query): i for i, a in enumerate(articles)}
        results = [None] * total_articles
        
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception:
                a_data = articles[idx]
                results[idx] = Article(
                    id=a_data["id"], title=a_data["title"], source=a_data["source"], url=a_data["url"],
                    relevance_score=a_data["relevance_score"], is_verified=False, published_at=a_data.get("published_at"),
                    content=None, content_status="failed", origin=a_data["origin"], fetched_at=now_iso,
                    date_source=a_data.get("date_source")
                ).to_dict()
                
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_articles:
                print(f"[{completed_count}/{total_articles}] Makale cekildi...")
        
    # 1. Yalnızca Gerçek, Doğrulanmış ve İçeriği Dolu Makaleleri Filtrele (content: null çöpleri temizlenir)
    valid_results = [
        r for r in results 
        if r and r.get("content_status") == "ok" and r.get("content") and r.get("is_verified")
        and is_clean_completed_article(r.get("content", ""))
        and not _too_old(r.get("published_at"))  # sayfadan okunan tarih de eskiyse kaydetme (N1)
    ]
    
    # 2. Gelişmiş İçerik ve Sindikasyon Düzeyi Tekilleştirme (Syndication / Jaccard Token Overlap)
    def extract_lead_tokens(text: str) -> set:
        words = re.findall(r'[a-zA-Z0-9]{3,}', text.lower())
        stop = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'been', 'were', 'which', 'about', 'said', 'will'}
        return set(w for w in words if w not in stop)

    deduped_results = []
    seen_lead_tokens = []
    seen_exact_hashes = set()

    for r in valid_results:
        c = r["content"].strip()
        # Birebir tam gövde hash kontrolü
        h = hashlib.md5(c.encode('utf-8')).hexdigest()
        if h in seen_exact_hashes:
            continue
        seen_exact_hashes.add(h)
        
        # Basın bülteni ve haber ajansı sindikasyon kontrolü (ilk 800 karakter)
        toks = extract_lead_tokens(c[:800])
        is_syndicated = False
        if len(toks) >= 15:
            for prev_toks in seen_lead_tokens:
                overlap = len(toks.intersection(prev_toks)) / min(len(toks), len(prev_toks))
                if overlap >= 0.70:
                    is_syndicated = True
                    break
        if is_syndicated:
            continue
            
        seen_lead_tokens.append(toks)
        deduped_results.append(r)

    # Kalite Sıralaması: En yüksek relevance_score'a sahip haberler önde
    deduped_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # QA & Buffer Sınırı Güvencesi:
    # İstemci / Pipeline tampon sınırının (~380 KB) aşılıp dosya sonunun "Unterminated JSON"
    # olarak kesilmesini kesin olarak önlemek için, en yüksek puanlı tam makaleler güvenli 365 KB sınırına kadar alınır.
    safe_results = []
    current_size = 2 # '[\n'
    for item in deduped_results:
        item_json_bytes = len(json.dumps(item, ensure_ascii=False, indent=4).encode('utf-8')) + 6
        if OUTPUT_LIMIT_KB and current_size + item_json_bytes > OUTPUT_LIMIT_KB * 1000 and len(safe_results) >= 15:
            break
        safe_results.append(item)
        current_size += item_json_bytes
        
    results = safe_results if safe_results else deduped_results

    # Güvenli ve Atomik JSON Dosya Yazımı
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())

    # Yazma sonrası JSON Doğrulaması
    with open(output_file, "r", encoding="utf-8") as f:
        verified_data = json.load(f)
    print(f"\nİşlem bitti! {len(verified_data)} adet makale başarıyla doğrulandı ve {output_file} dosyasına kaydedildi.")
    print(f"Süre: {time.time() - t_start:.0f} s (son {MAX_AGE_DAYS} gün, en fazla {MAX_FETCH} makale indirildi)")