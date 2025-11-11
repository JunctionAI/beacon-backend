"""
ADCAST MVP BACKEND - Actually Works Edition
============================================
Focus: Generate REAL podcasts with REAL audio
- YouTube: Real video search + transcripts
- Articles: Web scraping
- Books: Claude's knowledge (skip downloads for now)
- Audio: ElevenLabs text-to-speech
"""

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
import anthropic
import os
import requests
from bs4 import BeautifulSoup
import json
from youtube_transcript_api import YouTubeTranscriptApi
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    print("⚠️  ElevenLabs not installed - audio generation disabled")
    ELEVENLABS_AVAILABLE = False

app = FastAPI(title="AdCast MVP API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "AIzaSyAeqykSy0cSKCzpGYmn7mEFmJGpLoxPNg8")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_2110086ea581e84ac1881e781281406530df9e033ca6443c")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
elevenlabs_client = None
if ELEVENLABS_AVAILABLE and ELEVENLABS_API_KEY:
    try:
        import httpx
        # Create httpx client with aggressive timeout
        http_client = httpx.Client(timeout=30.0)
        elevenlabs_client = ElevenLabs(
            api_key=ELEVENLABS_API_KEY,
            httpx_client=http_client
        )
    except Exception as e:
        print(f"⚠️  ElevenLabs init failed: {e}")


# =================== STORAGE CONFIGURATION ===================

from pathlib import Path

def get_storage_dir():
    """Get appropriate storage directory based on environment"""
    if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("PORT"):
        # Running on Railway - use /tmp directory
        storage_dir = Path("/tmp/podcasts")
    else:
        # Local development - use home directory
        storage_dir = Path.home()

    # Create directory if it doesn't exist
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir

def get_backend_url():
    """Get backend URL based on environment"""
    # Railway provides RAILWAY_PUBLIC_DOMAIN or we can use custom env var
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
    if railway_domain:
        return f"https://{railway_domain}"

    # Custom environment variable for Railway URL
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    return backend_url


# =================== DATABASE SETUP ===================

# SQLite database
DATABASE_URL = "sqlite:///./adcast.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# HTTP Bearer for token authentication
security = HTTPBearer()


# =================== DATABASE MODELS ===================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    podcasts = relationship("Podcast", back_populates="owner", cascade="all, delete-orphan")


class Podcast(Base):
    __tablename__ = "podcasts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    topic = Column(String, nullable=False)
    script = Column(Text, nullable=True)
    audio_url = Column(String, nullable=True)
    duration = Column(Integer, nullable=True)  # in minutes
    sources_used = Column(Text, nullable=True)  # JSON string
    is_favorite = Column(Boolean, default=False)

    # Personalization fields
    tone_level = Column(Integer, default=5)  # 1-10 scale (1=serious, 10=funny)
    format_type = Column(String, default="conversational")  # conversational/debate/interview/storytelling/educational
    technical_depth = Column(Integer, default=5)  # 1-10 (1=beginner, 10=expert)
    pacing = Column(Integer, default=5)  # 1-10 (1=slow detailed, 10=fast overview)
    custom_instructions = Column(Text, nullable=True)
    teaching_persona = Column(String, nullable=True)  # Feynman, Sagan, Rogan, etc.

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    owner = relationship("User", back_populates="podcasts")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    # Onboarding preferences
    learning_style = Column(String, nullable=True)  # Visual, Auditory, Reading, Hands-on
    teaching_preference = Column(String, nullable=True)  # Socratic, Direct, Story-based, Data-driven
    expertise_level = Column(String, nullable=True)  # Beginner, Intermediate, Advanced, Expert
    humor_preference = Column(String, nullable=True)  # All business, Light humor, Very casual
    favorite_educators = Column(Text, nullable=True)  # JSON array of educator names/styles

    # Learned patterns
    prompt_writing_patterns = Column(Text, nullable=True)  # JSON with complexity, tone, depth analysis
    avg_completion_rate = Column(Integer, default=0)  # Percentage 0-100
    preferred_duration = Column(Integer, default=30)  # Minutes
    preferred_voices = Column(Text, nullable=True)  # JSON array of voice IDs

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="profile")


class PodcastFeedback(Base):
    __tablename__ = "podcast_feedback"

    id = Column(Integer, primary_key=True, index=True)
    podcast_id = Column(Integer, ForeignKey("podcasts.id"), nullable=False)

    # Ratings
    rating = Column(Integer, nullable=False)  # 1-5 stars
    complexity_feedback = Column(String, nullable=True)  # "too_simple" / "just_right" / "too_complex"
    tone_feedback = Column(String, nullable=True)  # "too_serious" / "just_right" / "too_funny"

    # Free text feedback
    loved = Column(Text, nullable=True)
    improvements = Column(Text, nullable=True)

    # Analytics
    completion_rate = Column(Integer, default=0)  # Percentage listened

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    podcast = relationship("Podcast", backref="feedback")


# Create tables
Base.metadata.create_all(bind=engine)


# =================== AUTH UTILITIES ===================

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    return user


# =================== YOUTUBE REAL SEARCH ===================

def search_youtube_real(query: str, max_results: int = 5) -> List[Dict]:
    """Search YouTube using YouTube Data API v3 - REAL results, not hallucinated"""
    if not YOUTUBE_API_KEY:
        print("⚠️  No YouTube API key - falling back to web scraping")
        return search_youtube_fallback(query, max_results)

    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "maxResults": max_results,
            "type": "video",
            "videoCaption": "closedCaption"  # Only videos with captions
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "items" not in data:
            print(f"YouTube API error: {data.get('error', {}).get('message', 'Unknown error')}")
            return search_youtube_fallback(query, max_results)

        videos = []
        for item in data["items"]:
            videos.append({
                "videoId": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "creator": item["snippet"]["channelTitle"],
                "description": item["snippet"]["description"][:200]
            })

        print(f"✅ Found {len(videos)} YouTube videos via API")
        return videos

    except Exception as e:
        print(f"YouTube API error: {e}")
        return search_youtube_fallback(query, max_results)


def search_youtube_fallback(query: str, max_results: int = 5) -> List[Dict]:
    """Fallback: Scrape YouTube search results (no API key needed)"""
    try:
        from urllib.parse import quote
        search_url = f"https://www.youtube.com/results?search_query={quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        response = requests.get(search_url, headers=headers, timeout=10)

        # Extract video IDs from page
        video_ids = re.findall(r'"videoId":"([\w-]{11})"', response.text)
        video_titles = re.findall(r'"title":{"runs":\[{"text":"([^"]+)"}', response.text)

        videos = []
        for i in range(min(len(video_ids), max_results)):
            if i < len(video_titles):
                videos.append({
                    "videoId": video_ids[i],
                    "title": video_titles[i],
                    "creator": "Unknown",
                    "description": ""
                })

        print(f"✅ Found {len(videos)} YouTube videos via scraping")
        return videos

    except Exception as e:
        print(f"YouTube scraping error: {e}")
        return []


def get_youtube_transcript(video_id: str) -> str:
    """Get transcript from YouTube video"""
    try:
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id, languages=['en'])
        full_text = " ".join([entry['text'] for entry in transcript_data])

        print(f"✅ Extracted {len(full_text)} chars from video {video_id}")
        return full_text[:100000]  # First 100k chars

    except Exception as e:
        print(f"❌ Transcript error for {video_id}: {str(e)[:100]}")
        return ""


# =================== WEB SEARCH FOR ARTICLES ===================

def search_web_for_articles(query: str, num_results: int = 3) -> List[Dict]:
    """Search for articles using improved multi-method approach"""
    try:
        from urllib.parse import quote
        import re

        # Try DuckDuckGo Lite (simpler HTML, more reliable)
        search_url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []

        # DuckDuckGo Lite has simpler structure - look for result links
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')

            # Skip DuckDuckGo internal links
            if href.startswith('http') and 'duckduckgo.com' not in href:
                title = link.get_text(strip=True)

                if title and len(title) > 10:  # Avoid short/garbage links
                    domain = href.split('/')[2] if '/' in href else "Unknown"

                    results.append({
                        "title": title[:200],  # Limit title length
                        "url": href,
                        "source": domain
                    })

                    if len(results) >= num_results:
                        break

        print(f"✅ Found {len(results)} articles via DuckDuckGo Lite")
        return results

    except Exception as e:
        print(f"⚠️ Web search error: {e}")

        # FALLBACK: Return some popular general sources if all else fails
        print("⚠️ Returning fallback sources...")
        return [
            {
                "title": f"Search results for: {query}",
                "url": f"https://www.google.com/search?q={quote(query)}",
                "source": "Google Search"
            }
        ]


def search_news(query: str, max_results: int = 10) -> List[Dict]:
    """Search for recent NEWS articles using multiple approaches

    This implements a Perplexity-style multi-source search:
    1. Try Google News RSS (free, reliable)
    2. Try DuckDuckGo Instant Answer API (free)
    3. Fallback to web scraping with better selectors
    """
    news_articles = []

    # APPROACH 1: Google News RSS (Most Reliable)
    try:
        from urllib.parse import quote
        import xml.etree.ElementTree as ET

        # Google News RSS feed
        rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        response = requests.get(rss_url, headers=headers, timeout=10)

        if response.status_code == 200:
            root = ET.fromstring(response.content)

            for item in root.findall('.//item')[:max_results]:
                title_elem = item.find('title')
                link_elem = item.find('link')
                pub_date_elem = item.find('pubDate')
                source_elem = item.find('source')

                if title_elem is not None and link_elem is not None:
                    url = link_elem.text
                    # Extract actual article URL from Google News redirect
                    if 'google.com' in url:
                        # Try to extract the real URL from Google News URL
                        import re
                        url_match = re.search(r'url=([^&]+)', url)
                        if url_match:
                            from urllib.parse import unquote
                            url = unquote(url_match.group(1))

                    domain = url.split('/')[2] if '/' in url else "Unknown"

                    # Check if it's a reputable news source
                    is_news = any(news_domain in domain.lower() for news_domain in [
                        'bbc', 'cnn', 'reuters', 'ap.org', 'nytimes', 'wsj', 'theguardian',
                        'bloomberg', 'forbes', 'axios', 'politico', 'npr.org', 'abcnews',
                        'nbcnews', 'cbsnews', 'time.com', 'washingtonpost', 'economist',
                        'aljazeera', 'france24', 'dw.com'
                    ])

                    news_articles.append({
                        "title": title_elem.text,
                        "url": url,
                        "source": source_elem.text if source_elem is not None else domain,
                        "is_news_source": is_news,
                        "date": pub_date_elem.text if pub_date_elem is not None else "Recent"
                    })

            print(f"✅ Google News RSS: Found {len(news_articles)} articles")

            if len(news_articles) > 0:
                # Sort to prioritize reputable news sources
                news_articles.sort(key=lambda x: x.get("is_news_source", False), reverse=True)
                return news_articles

    except Exception as e:
        print(f"⚠️ Google News RSS failed: {e}")

    # APPROACH 2: DuckDuckGo JSON API (if RSS fails)
    try:
        from urllib.parse import quote

        # DuckDuckGo instant answer API (returns JSON)
        api_url = f"https://api.duckduckgo.com/?q={quote(query + ' news')}&format=json"
        response = requests.get(api_url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Check for related topics/news
            for result in data.get('RelatedTopics', [])[:max_results]:
                if isinstance(result, dict) and 'FirstURL' in result:
                    url = result.get('FirstURL', '')
                    text = result.get('Text', 'News Article')

                    if url:
                        domain = url.split('/')[2] if '/' in url else "Unknown"
                        is_news = any(nd in domain.lower() for nd in ['bbc', 'cnn', 'reuters', 'news'])

                        news_articles.append({
                            "title": text[:200],
                            "url": url,
                            "source": domain,
                            "is_news_source": is_news,
                            "date": "Recent"
                        })

            print(f"✅ DuckDuckGo API: Found {len(news_articles)} articles")

            if len(news_articles) > 0:
                news_articles.sort(key=lambda x: x.get("is_news_source", False), reverse=True)
                return news_articles

    except Exception as e:
        print(f"⚠️ DuckDuckGo API failed: {e}")

    # APPROACH 3: Fallback to web scraping (last resort)
    print("⚠️ Falling back to web scraping...")
    return search_web_for_articles(query + " news breaking today", max_results)


def perplexity_deep_research(query: str, use_deep_research: bool = False) -> Dict:
    """Use Perplexity API for DEEP RESEARCH with 15-100+ sources

    This replaces simple news scraping with Perplexity's sonar-pro model:
    - Real-time web search with HIGH context size
    - 15-25+ sources from sonar-pro (default)
    - 50-100+ sources from sonar-deep-research (if use_deep_research=True)
    - Deep synthesis and cross-referencing
    - Verified citations with rich metadata

    Args:
        query: The research query
        use_deep_research: If True, uses sonar-deep-research model ($0.10/query, slower, 50-100+ sources)
                          If False, uses sonar-pro with high context (default, 15-25 sources)

    Returns:
        {
            "answer": "Comprehensive synthesized answer",
            "sources": [{"title": "...", "url": "...", "source": "...", "date": "..."}],
            "citations": ["1", "2", "3"],
            "metadata": {
                "num_search_queries": 3,
                "search_context_size": "high",
                "model": "sonar-pro"
            }
        }
    """
    if not PERPLEXITY_API_KEY:
        print("⚠️ No Perplexity API key - falling back to Google News RSS")
        articles = search_news(query, max_results=15)
        return {
            "answer": f"Found {len(articles)} news articles about: {query}",
            "sources": articles,
            "citations": [],
            "metadata": {"fallback": True}
        }

    try:
        print(f"🔍 Perplexity Deep Research: '{query}'")

        # Choose model based on depth requirement
        model = "sonar-deep-research" if use_deep_research else "sonar-pro"
        print(f"📊 Using model: {model}")

        # Build API request payload
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant. Provide comprehensive, well-researched answers with citations from multiple authoritative sources."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "return_citations": True,
            "return_images": False
        }

        # Add model-specific parameters
        if model == "sonar-deep-research":
            payload["reasoning_effort"] = "high"
        else:
            # sonar-pro specific parameters
            payload["search_recency_filter"] = "day"

        # Call Perplexity API using raw requests
        http_response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=25  # Reduced from 60 to 25 seconds for faster failure
        )

        if http_response.status_code != 200:
            raise Exception(f"HTTP {http_response.status_code}: {http_response.text}")

        data = http_response.json()

        # Extract comprehensive data from response
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Extract search_results array (richer than citations)
        search_results = data.get("search_results", [])

        # Fallback to citations if search_results not available
        citations = data.get("citations", [])

        # Parse sources from search_results (preferred) or citations (fallback)
        sources = []
        from urllib.parse import urlparse

        if search_results:
            # Use rich search_results data
            print(f"📚 Extracting from search_results: {len(search_results)} items")
            for i, result in enumerate(search_results, 1):
                title = result.get("title", f"Source {i}")
                url = result.get("url", "")
                date = result.get("date", "Recent")

                # Extract domain for source name
                domain = "Unknown"
                if url:
                    try:
                        parsed = urlparse(url)
                        domain = parsed.netloc.replace('www.', '')
                    except:
                        pass

                sources.append({
                    "title": title,
                    "url": url,
                    "source": domain,
                    "date": date,
                    "is_news_source": True
                })
        elif citations:
            # Fallback to citations (less rich data)
            print(f"📚 Extracting from citations: {len(citations)} items")
            for i, citation in enumerate(citations, 1):
                if isinstance(citation, str):
                    try:
                        parsed = urlparse(citation)
                        domain = parsed.netloc.replace('www.', '')
                        sources.append({
                            "title": f"Source {i}: {domain}",
                            "url": citation,
                            "source": domain,
                            "is_news_source": True,
                            "date": "Recent"
                        })
                    except:
                        sources.append({
                            "title": f"Source {i}",
                            "url": citation,
                            "source": "Unknown",
                            "is_news_source": False,
                            "date": "Recent"
                        })

        # Extract metadata from usage
        usage = data.get("usage", {})
        metadata = {
            "model": model,
            "search_context_size": usage.get("search_context_size", "unknown"),
            "num_search_queries": usage.get("num_search_queries", 1),
            "total_tokens": usage.get("total_tokens", 0)
        }

        if "reasoning_tokens" in usage:
            metadata["reasoning_tokens"] = usage["reasoning_tokens"]

        print(f"✅ Perplexity: {len(sources)} sources found")
        print(f"📊 Metadata: {metadata}")
        print(f"📝 Answer: {answer[:200]}...")

        return {
            "answer": answer,
            "sources": sources,
            "citations": [str(i) for i in range(1, len(sources) + 1)],
            "metadata": metadata
        }

    except Exception as e:
        print(f"❌ Perplexity API error: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to Google News RSS
        articles = search_news(query, max_results=15)
        return {
            "answer": f"Found {len(articles)} news articles about: {query}",
            "sources": articles,
            "citations": [],
            "metadata": {"fallback": True, "error": str(e)}
        }


# ==========================
# X/TWITTER THOUGHT LEADER DISCOVERY
# ==========================

def discover_x_thought_leaders(curiosity: str, max_accounts: int = 10) -> list[Dict]:
    """
    Discover X/Twitter thought leaders and their recent posts about a topic.

    Uses Perplexity to:
    1. Identify leading voices and experts on the topic
    2. Find their X/Twitter handles
    3. Discover specific relevant posts they've made

    Args:
        curiosity: The user's topic of interest
        max_accounts: Maximum number of thought leaders to return

    Returns:
        List of thought leader accounts with post content:
        [
            {
                "id": "1",
                "name": "Person Name",
                "handle": "username",
                "why": "Why they're relevant",
                "url": "https://x.com/username/status/123",  # Link to specific post
                "postContent": "Content of their relevant post",
                "postDate": "2025-01-10"
            }
        ]
    """
    print(f"\n🐦 Discovering X/Twitter thought leaders for: {curiosity}")

    if not PERPLEXITY_API_KEY:
        print("⚠️  No Perplexity API key - skipping X discovery")
        return []

    try:
        # Step 1: Find thought leaders and their handles
        thought_leader_query = f"""Who are the top thought leaders, experts, and influential voices on X/Twitter discussing {curiosity}?

Provide their X/Twitter handles and explain why they're important voices on this topic.
Focus on accounts that actively post insightful content about {curiosity}."""

        print(f"  🔍 Query 1: Finding thought leaders...")
        result = perplexity_deep_research(thought_leader_query, use_deep_research=False)
        answer_text = result.get("answer", "")
        sources = result.get("sources", [])

        # Extract X handles from the answer and sources
        import re
        from urllib.parse import urlparse

        thought_leaders = []
        seen_handles = set()

        # Pattern to find @handles in text
        handle_pattern = r'@([A-Za-z0-9_]{1,15})'

        # Extract handles from answer text
        handles_in_answer = re.findall(handle_pattern, answer_text)

        # Also look for x.com or twitter.com URLs in sources
        for source in sources:
            url = source.get("url", "")
            if "x.com" in url or "twitter.com" in url:
                # Extract handle from URL
                match = re.search(r'(?:x\.com|twitter\.com)/([A-Za-z0-9_]+)', url)
                if match:
                    handles_in_answer.append(match.group(1))

        # Look for mentions in answer text (e.g., "John Doe (@handle)")
        name_handle_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(@?([A-Za-z0-9_]{1,15})\)'
        name_handle_matches = re.findall(name_handle_pattern, answer_text)

        # Build list of potential thought leaders
        potential_leaders = []
        for name, handle in name_handle_matches:
            if handle.lower() not in seen_handles:
                seen_handles.add(handle.lower())
                potential_leaders.append((name, handle))

        # Add handles without names
        for handle in handles_in_answer:
            if handle.lower() not in seen_handles and len(handle) > 2:
                seen_handles.add(handle.lower())
                potential_leaders.append((handle, handle))

        print(f"  ✅ Found {len(potential_leaders)} potential thought leaders")

        # Step 2: For each thought leader, find their recent relevant posts
        for name, handle in potential_leaders[:max_accounts]:
            print(f"  🔍 Finding posts from @{handle}...")

            # Query for specific posts by this person
            posts_query = f"""Find recent insightful posts by @{handle} on X/Twitter about {curiosity}.

What specific posts or threads have they shared about this topic? Include the post content and what makes it valuable."""

            try:
                posts_result = perplexity_deep_research(posts_query, use_deep_research=False)
                posts_answer = posts_result.get("answer", "")
                posts_sources = posts_result.get("sources", [])

                # Extract post URL from sources
                post_url = f"https://x.com/{handle}"  # Default to profile
                post_content = None
                post_date = None

                # Look for specific post URLs in sources
                for source in posts_sources:
                    url = source.get("url", "")
                    if handle.lower() in url.lower() and ("/status/" in url or "/post/" in url):
                        post_url = url.replace("twitter.com", "x.com")
                        post_date = source.get("date", "Recent")
                        break

                # Extract post content from answer (usually Perplexity quotes the post)
                # Look for quoted text in the answer
                quote_patterns = [
                    r'"([^"]{20,280})"',  # Double quotes (tweets are max 280 chars, look for 20+)
                    r'"([^"]{20,280})"',  # Curly quotes
                ]

                for pattern in quote_patterns:
                    matches = re.findall(pattern, posts_answer)
                    if matches:
                        # Take the longest quoted text as the post content
                        post_content = max(matches, key=len)
                        break

                # If no quoted content, take first sentence from answer
                if not post_content and len(posts_answer) > 50:
                    sentences = posts_answer.split('.')[:2]
                    post_content = '.'.join(sentences).strip()[:280]

                # Extract why they're relevant from original answer
                why_pattern = rf'@?{re.escape(handle)}[^.!?]*[.!?]'
                why_match = re.search(why_pattern, answer_text, re.IGNORECASE)
                why = why_match.group(0) if why_match else f"Expert voice on {curiosity}"

                thought_leaders.append({
                    "id": f"x_{len(thought_leaders) + 1}",
                    "name": name if name != handle else f"@{handle}",
                    "handle": handle,
                    "why": why[:200],  # Limit to 200 chars
                    "url": post_url,
                    "postContent": post_content[:280] if post_content else None,
                    "postDate": post_date
                })

                print(f"    ✅ Added @{handle} with {len(thought_leaders[-1].get('postContent', '') or '')} char post")

            except Exception as e:
                print(f"    ⚠️  Could not find posts for @{handle}: {e}")
                # Add them anyway, just without post content
                thought_leaders.append({
                    "id": f"x_{len(thought_leaders) + 1}",
                    "name": name if name != handle else f"@{handle}",
                    "handle": handle,
                    "why": f"Influential voice on {curiosity}",
                    "url": f"https://x.com/{handle}"
                })

        print(f"\n✅ X Discovery complete: {len(thought_leaders)} thought leaders")
        return thought_leaders[:max_accounts]

    except Exception as e:
        print(f"❌ X thought leader discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ==========================
# DEEP RESEARCH SYNTHESIS SYSTEM
# ==========================

def discover_comprehensive_sources(curiosity: str, themes: list[str]) -> Dict:
    """
    Multi-query deep research strategy to discover 100-150+ sources

    Runs 5-8 targeted Perplexity searches to comprehensively cover:
    - Books and book reviews/summaries
    - Recent articles and analysis
    - Expert commentary and interviews
    - Research papers
    - Videos and media discussions

    Args:
        curiosity: The user's curiosity query
        themes: List of key themes extracted from the curiosity

    Returns:
        {
            "all_sources": [...],  # 100-150+ total sources
            "books_identified": [...],  # Books that need meta-content discovery
            "total_count": 150,
            "queries_run": 8
        }
    """
    print(f"\n🔍 DEEP RESEARCH: Discovering comprehensive sources...")
    print(f"📋 Curiosity: {curiosity}")
    print(f"🎯 Themes: {', '.join(themes[:5])}")

    all_sources = []
    books_identified = []
    queries_run = []

    # Build targeted search queries
    search_queries = []

    # Query 1: Best books on the topic
    search_queries.append(f"{curiosity} best books recommended by experts")

    # Query 2: Recent analysis and articles
    search_queries.append(f"{curiosity} latest analysis expert opinions")

    # Query 3: Foundational sources and key concepts
    if themes:
        search_queries.append(f"{themes[0]} comprehensive guide resources")

    # Query 4: Critical perspectives and debates
    search_queries.append(f"{curiosity} criticism debate different perspectives")

    # Query 5: Research and academic sources
    search_queries.append(f"{curiosity} research papers academic studies")

    # Query 6-10: Theme-specific deep dives (up to 5 themes)
    for theme in themes[:5]:
        search_queries.append(f"{theme} explained expert resources")

    # Additional queries for depth
    search_queries.append(f"{curiosity} case studies real world examples")
    search_queries.append(f"{curiosity} key thinkers influential voices")

    # Execute all searches (max 12 queries for comprehensive coverage)
    max_queries = min(12, len(search_queries))
    for i, query in enumerate(search_queries[:max_queries], 1):
        print(f"\n📊 Query {i}/{max_queries}: {query}")
        try:
            result = perplexity_deep_research(query, use_deep_research=False)  # Use sonar-pro for speed
            sources = result.get("sources", [])
            answer_text = result.get("answer", "")
            print(f"✅ Found {len(sources)} sources")

            # Add query context to each source
            for source in sources:
                source["discovery_query"] = query
                source["discovery_query_index"] = i

            all_sources.extend(sources)
            queries_run.append(query)

            # Extract book mentions from Perplexity's answer text
            if "book" in query.lower() and answer_text:
                import re
                # Look for book titles in quotes or after "by"
                book_patterns = [
                    r'"([^"]{10,100})"',  # Quoted titles
                    r"'([^']{10,100})'",  # Single quoted
                    r'_([^_]{10,100})_',  # Italicized
                    r'\*([^\*]{10,100})\*',  # Asterisk emphasis
                ]

                for pattern in book_patterns:
                    matches = re.findall(pattern, answer_text)
                    for match in matches:
                        # Check if this looks like a book title (not a sentence)
                        if match and 10 < len(match) < 100 and not match.endswith('.'):
                            # Extract author if mentioned
                            author_pattern = rf'{re.escape(match)}.*?by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
                            author_match = re.search(author_pattern, answer_text)
                            author = author_match.group(1) if author_match else ""

                            # Add to books if not duplicate
                            if not any(b.get("title", "").lower() == match.lower() for b in books_identified):
                                books_identified.append({
                                    "title": match,
                                    "author": author,
                                    "why": f"Recommended in expert analysis about {curiosity}",
                                    "needs_meta_discovery": True
                                })

        except Exception as e:
            print(f"❌ Query {i} failed: {e}")
            continue

    # Identify books from sources
    print(f"\n📚 Identifying books from {len(all_sources)} total sources...")

    for source in all_sources:
        title = source.get("title", "").lower()
        url = source.get("url", "").lower()

        # Heuristics to identify book-related sources
        is_book_source = any([
            "book" in title,
            "author" in title,
            "review" in title and "book" in url,
            "goodreads.com" in url,
            "amazon.com/dp" in url,
            "barnesandnoble.com" in url,
            "publishers" in url
        ])

        if is_book_source:
            # Try to extract book info
            book_title_match = re.search(r'"([^"]+)"|\'([^\']+)\'', title)
            if book_title_match:
                potential_title = book_title_match.group(1) or book_title_match.group(2)
                # Check if we already have this book
                if not any(b.get("title", "").lower() == potential_title.lower() for b in books_identified):
                    books_identified.append({
                        "title": potential_title,
                        "source": source,
                        "needs_meta_discovery": True
                    })

    print(f"\n✅ Deep Research Complete:")
    print(f"   📊 Total sources: {len(all_sources)}")
    print(f"   📚 Books identified: {len(books_identified)}")
    print(f"   🔍 Queries run: {len(queries_run)}")

    return {
        "all_sources": all_sources,
        "books_identified": books_identified,
        "total_count": len(all_sources),
        "queries_run": len(queries_run),
        "query_list": queries_run
    }


def discover_book_meta_content(book_title: str, author: str = "") -> list[Dict]:
    """
    Find 10-15 meta-sources ABOUT a book (reviews, summaries, analyses, interviews)

    This is the KEY function that enables book-based podcasts WITHOUT downloading books.
    Instead of reading the full book, we gather expert summaries, reviews, and analyses.

    Args:
        book_title: Title of the book
        author: Author name (optional, helps with accuracy)

    Returns:
        List of 10-15 sources with rich metadata about the book
    """
    print(f"\n📖 Discovering meta-content for: '{book_title}' by {author}")

    all_meta_sources = []

    # Build targeted search queries for book meta-content
    book_query = f'"{book_title}"'
    if author:
        book_query += f' {author}'

    queries = [
        f'{book_query} summary key arguments main ideas',
        f'{book_query} book review analysis',
        f'{book_query} chapter summaries table of contents',
        f'{author} interview about "{book_title}"',
        f'{book_query} quotes key passages explained',
        f'{book_query} criticism counterarguments response'
    ]

    for i, query in enumerate(queries, 1):
        print(f"  🔍 Meta-query {i}/{len(queries)}: {query[:60]}...")
        try:
            result = perplexity_deep_research(query, use_deep_research=False)
            sources = result.get("sources", [])

            # Classify each source
            for source in sources:
                source["book_title"] = book_title
                source["meta_type"] = classify_book_meta_source(source)
                source["meta_query_index"] = i

            all_meta_sources.extend(sources)
            print(f"  ✅ Found {len(sources)} meta-sources")

        except Exception as e:
            print(f"  ❌ Meta-query {i} failed: {e}")
            continue

    # Deduplicate and rank
    unique_sources = deduplicate_sources(all_meta_sources)
    ranked_sources = rank_sources_by_quality(unique_sources, "book_meta")

    # Return top 15
    top_sources = ranked_sources[:15]

    print(f"  📊 Meta-content discovery complete: {len(top_sources)} high-quality sources")
    return top_sources


def classify_book_meta_source(source: Dict) -> str:
    """
    Classify what type of book meta-content this is

    Returns: 'official' | 'professional_summary' | 'review' | 'interview' | 'academic' | 'discussion'
    """
    title = source.get("title", "").lower()
    url = source.get("url", "").lower()
    domain = source.get("source", "").lower()

    # Official sources (publisher, author website)
    if any(x in domain for x in ["publisher", "author", "official"]):
        return "official"

    # Professional summaries
    if any(x in domain for x in ["blinkist", "sobrief", "getabstract", "shortform", "instaread"]):
        return "professional_summary"

    # Author interviews/commentary
    if "interview" in title or "transcript" in title or "talk" in title:
        return "author_commentary"

    # Academic analysis
    if any(x in domain for x in [".edu", "scholar", "jstor", "academia.edu", "researchgate"]):
        return "academic_analysis"

    # Book reviews
    if "review" in title or "critique" in title:
        return "critical_review"

    # General discussion
    return "general_discussion"


def rank_sources_by_quality(sources: list[Dict], source_context: str = "general") -> list[Dict]:
    """
    Rank sources by quality using multiple signals

    Priority scoring:
    - Official sources: 10
    - Academic sources: 9
    - Professional summaries: 9
    - Author commentary: 9
    - Critical reviews: 7
    - General discussion: 5

    Also considers:
    - Domain authority
    - Recency (if date available)
    - Title relevance
    """
    for source in sources:
        score = 50  # Base score

        # Type-based scoring
        meta_type = source.get("meta_type", "general")
        if meta_type == "official":
            score += 40
        elif meta_type in ["professional_summary", "author_commentary", "academic_analysis"]:
            score += 35
        elif meta_type == "critical_review":
            score += 25
        else:
            score += 10

        # Domain authority
        domain = source.get("source", "").lower()
        high_authority_domains = [
            "nytimes.com", "wsj.com", "economist.com", "ft.com",
            "harvard.edu", "mit.edu", "stanford.edu",
            "nature.com", "science.org", "jstor.org"
        ]
        if any(d in domain for d in high_authority_domains):
            score += 20

        # Recency bonus (if date available and recent)
        date_str = source.get("date", "")
        if date_str and "2024" in date_str or "2025" in date_str:
            score += 10
        elif date_str and "2023" in date_str:
            score += 5

        source["quality_score"] = score

    # Sort by quality score
    return sorted(sources, key=lambda x: x.get("quality_score", 0), reverse=True)


def deduplicate_sources(sources: list[Dict]) -> list[Dict]:
    """Remove duplicate sources based on URL"""
    seen_urls = set()
    unique = []

    for source in sources:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(source)

    return unique


def extract_content_from_url(url: str, source_title: str = "") -> str:
    """
    Fetch and extract main content from a URL

    Uses BeautifulSoup to:
    1. Fetch the page
    2. Remove navigation, ads, scripts
    3. Extract main content
    4. Use Claude to extract only relevant passages

    Returns: Clean text content (up to 3000 words)
    """
    try:
        print(f"  📥 Fetching: {url[:60]}...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"  ❌ HTTP {response.status_code}")
            return ""

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove junk
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            tag.decompose()

        # Try to find main content
        main_content = (
            soup.find('article') or
            soup.find('main') or
            soup.find(class_=re.compile('content|article|post|entry', re.I)) or
            soup.body
        )

        if not main_content:
            return ""

        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Clean whitespace

        # Limit to first 8000 characters for processing
        text = text[:8000]

        print(f"  ✅ Extracted {len(text)} characters")

        # Use Claude to extract only relevant content
        if len(text) > 500:  # Only use Claude if we have substantial content
            relevant_text = extract_relevant_passages_with_claude(text, source_title)
            return relevant_text

        return text

    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return ""


def extract_relevant_passages_with_claude(content: str, source_title: str) -> str:
    """
    Use Claude to extract only the relevant parts of a web page
    (removes ads, navigation, unrelated content)
    """
    try:
        prompt = f"""Extract the main content from this webpage about: {source_title}

Source content:
{content[:6000]}

Extract ONLY the parts that are relevant to the topic. Remove:
- Navigation menus
- Advertisements
- Comment sections
- Unrelated articles
- Site boilerplate

Return clean, relevant text only (max 2000 words). Focus on:
- Main arguments and key points
- Important quotes
- Specific examples
- Critical analysis
"""

        response = claude_client.messages.create(
            model="claude-3-haiku-20240307",  # Use Haiku for speed
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    except Exception as e:
        print(f"  ⚠️  Claude extraction failed: {e}, returning raw text")
        return content[:3000]  # Fallback to truncated raw text


def scrape_article(url: str) -> str:
    """Scrape article content"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove junk
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        # Get main content
        article_body = soup.find('article') or soup.find('main') or soup.body

        if article_body:
            text = article_body.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean whitespace

            print(f"✅ Scraped {len(text)} chars from article")
            return text[:50000]  # First 50k chars

    except Exception as e:
        print(f"❌ Article scraping error: {e}")

    return ""


# =================== AUDIO GENERATION ===================

def generate_podcast_audio(script: str, voice_ids: List[str] = None) -> bytes:
    """Generate audio from podcast script using ElevenLabs - paragraph-based speaker alternation"""
    if not elevenlabs_client:
        print("⚠️  No ElevenLabs API key - skipping audio generation")
        return b""

    try:
        # Voice ID mapping: friendly names -> real ElevenLabs IDs
        VOICE_MAPPING = {
            "standard_male": "pNInz6obpgDQGcFmaJgB",      # Adam (deep male voice)
            "standard_female": "21m00Tcm4TlvDq8ikWAM",    # Rachel (female, American)
            "morgan_freeman": "pNInz6obpgDQGcFmaJgB",     # Use Adam for now (similar deep voice)
        }

        # Default voices (ElevenLabs built-in)
        if not voice_ids or len(voice_ids) < 2:
            voice_ids = [
                "pNInz6obpgDQGcFmaJgB",  # Adam (deep male voice)
                "21m00Tcm4TlvDq8ikWAM"   # Rachel (female, American)
            ]
        else:
            # Map friendly voice IDs to real ElevenLabs IDs
            mapped_voices = []
            for voice_id in voice_ids:
                if voice_id in VOICE_MAPPING:
                    mapped_id = VOICE_MAPPING[voice_id]
                    print(f"🎤 Mapped voice '{voice_id}' -> '{mapped_id}'")
                    mapped_voices.append(mapped_id)
                else:
                    # Already a real voice ID or unknown - use as-is
                    print(f"🎤 Using voice ID: '{voice_id}'")
                    mapped_voices.append(voice_id)
            voice_ids = mapped_voices

        print(f"🎙️ Using {len(voice_ids)} voice(s) for audio generation")

        # Split script by paragraph breaks (double newlines)
        # Each paragraph = one speaker's turn
        paragraphs = script.split('\n\n')

        # Clean and filter paragraphs
        segments = []
        speaker_idx = 0  # Start with first speaker

        for para in paragraphs:
            # Clean the paragraph
            para = para.strip()

            # Skip empty paragraphs
            if not para:
                continue

            # Remove any remaining line breaks within the paragraph
            para = ' '.join(line.strip() for line in para.split('\n') if line.strip())

            # Skip very short segments (likely formatting artifacts)
            if len(para) < 10:
                continue

            # Add segment with alternating speaker
            segments.append((speaker_idx, para))

            # Alternate speakers
            speaker_idx = (speaker_idx + 1) % len(voice_ids)

        print(f"📝 Split script into {len(segments)} speaking segments across {len(voice_ids)} voices")

        # Generate audio for each segment with timeout protection
        audio_chunks = []
        for idx, (speaker_idx, text) in enumerate(segments):
            voice_id = voice_ids[speaker_idx]

            print(f"🎙️ Segment {idx + 1}/{len(segments)} - Voice {speaker_idx + 1}: {text[:60]}...")
            import sys
            sys.stdout.flush()  # Force immediate output on Railway

            try:
                audio = elevenlabs_client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_monolingual_v1",
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75
                    )
                )

                # Collect audio bytes
                audio_bytes = b''
                for chunk in audio:
                    audio_bytes += chunk

                audio_chunks.append(audio_bytes)

            except Exception as e:
                print(f"⚠️ Segment {idx + 1} failed: {e} - skipping")
                sys.stdout.flush()
                continue  # Skip failed segment, continue with rest

        # Combine all audio chunks
        full_audio = b''.join(audio_chunks)

        print(f"✅ Generated {len(full_audio) / (1024*1024):.2f} MB of audio from {len(segments)} segments")
        return full_audio

    except Exception as e:
        print(f"❌ Audio generation error: {e}")
        import traceback
        print(traceback.format_exc())
        return b""


# =================== API ENDPOINTS ===================

class CuriosityRequest(BaseModel):
    curiosity: str


class GeneratePodcastRequest(BaseModel):
    curiosity: str
    selectedBooks: List[Dict] = []
    selectedVideos: List[Dict] = []
    selectedTwitterAccounts: List[Dict] = []
    selectedArticles: List[Dict] = []
    duration: int = 45
    voice_ids: Optional[List[str]] = None  # ElevenLabs voice IDs for speakers

    # Personalization parameters
    tone_level: int = 5  # 1-10 scale (1=serious, 10=funny)
    format_type: str = "conversational"  # conversational/debate/interview/storytelling/educational
    technical_depth: int = 5  # 1-10 (1=beginner, 10=expert)
    pacing: int = 5  # 1-10 (1=slow detailed, 10=fast overview)
    custom_instructions: str = ""
    teaching_persona: Optional[str] = None  # Feynman, Sagan, Rogan, etc.


# =================== AUTH REQUEST/RESPONSE MODELS ===================

class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class PodcastResponse(BaseModel):
    id: int
    user_id: int
    topic: str
    script: Optional[str]
    audio_url: Optional[str]
    duration: Optional[int]
    sources_used: Optional[str]
    is_favorite: bool
    created_at: datetime

    class Config:
        from_attributes = True


class SavePodcastRequest(BaseModel):
    topic: str
    script: Optional[str] = None
    audio_url: Optional[str] = None
    duration: Optional[int] = None
    sources_used: Optional[str] = None  # JSON string


# =================== USER PROFILE MODELS ===================

class UserProfileRequest(BaseModel):
    learning_style: Optional[str] = None
    teaching_preference: Optional[str] = None
    expertise_level: Optional[str] = None
    humor_preference: Optional[str] = None
    favorite_educators: Optional[str] = None  # JSON string


class UserProfileResponse(BaseModel):
    id: int
    user_id: int
    learning_style: Optional[str]
    teaching_preference: Optional[str]
    expertise_level: Optional[str]
    humor_preference: Optional[str]
    favorite_educators: Optional[str]
    prompt_writing_patterns: Optional[str]
    avg_completion_rate: int
    preferred_duration: int
    preferred_voices: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# =================== FEEDBACK MODELS ===================

class PodcastFeedbackRequest(BaseModel):
    podcast_id: int
    rating: int  # 1-5 stars
    complexity_feedback: Optional[str] = None
    tone_feedback: Optional[str] = None
    loved: Optional[str] = None
    improvements: Optional[str] = None
    completion_rate: int = 0


class PodcastFeedbackResponse(BaseModel):
    id: int
    podcast_id: int
    rating: int
    complexity_feedback: Optional[str]
    tone_feedback: Optional[str]
    loved: Optional[str]
    improvements: Optional[str]
    completion_rate: int
    created_at: datetime

    class Config:
        from_attributes = True


@app.get("/")
async def root():
    return {
        "service": "AdCast MVP API",
        "version": "1.0.0",
        "status": "running",
        "features": ["Real YouTube search", "Web scraping", "ElevenLabs audio"]
    }


@app.get("/api/voices")
async def get_voices():
    """Get all available ElevenLabs voices for podcast generation"""
    if not elevenlabs_client:
        return {
            "voices": [],
            "error": "ElevenLabs not configured"
        }

    try:
        # Fetch all voices from ElevenLabs API
        voices_response = elevenlabs_client.voices.get_all()

        # Define standard voices (curated high-quality selection)
        STANDARD_VOICES = {
            "Eric", "Matilda", "George", "Alice", "Harry", "Sarah",
            "Clyde", "Roger", "Laura", "Jessica", "Brian", "Chris"
        }

        # Define premium/celebrity voices
        PREMIUM_VOICES = {
            "Ray Dalio", "Charlie Munger", "100% Pure NZ Voice",
            "Kim - friendly and descriptive", "Charlotte", "Kylee_M", "Jess"
        }

        # Categorization function
        def categorize_voice(voice_name: str) -> str:
            if voice_name in STANDARD_VOICES:
                return "standard"
            elif voice_name in PREMIUM_VOICES or "Dalio" in voice_name or "Munger" in voice_name:
                return "premium"
            else:
                # Default to standard for general voices, premium for specialty
                return "standard"

        # Transform to frontend-friendly format
        voices = []
        for voice in voices_response.voices:
            voice_data = {
                "id": voice.voice_id,
                "name": voice.name,
                "gender": voice.labels.get("gender", "unknown") if voice.labels else "unknown",
                "accent": voice.labels.get("accent", "unknown") if voice.labels else "unknown",
                "age": voice.labels.get("age", "unknown") if voice.labels else "unknown",
                "description": voice.labels.get("description", "") if voice.labels else "",
                "use_case": voice.labels.get("use case", "general") if voice.labels else "general",
                "preview_url": voice.preview_url if hasattr(voice, 'preview_url') else None,
                "tier": categorize_voice(voice.name)
            }
            voices.append(voice_data)

        print(f"✅ Fetched {len(voices)} voices from ElevenLabs")
        print(f"   - Standard: {sum(1 for v in voices if v['tier'] == 'standard')}")
        print(f"   - Premium: {sum(1 for v in voices if v['tier'] == 'premium')}")

        return {
            "voices": voices,
            "count": len(voices)
        }

    except Exception as e:
        print(f"❌ Error fetching voices: {e}")
        return {
            "voices": [],
            "error": str(e)
        }


@app.get("/api/analyze-curiosity-stream")
async def analyze_curiosity_stream(curiosity: str):
    """Stream research progress in real-time using Server-Sent Events"""

    async def event_generator():
        try:
            print(f"🔄 SSE: Starting research for '{curiosity}'")
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting research...', 'progress': 0})}\n\n"
            await asyncio.sleep(0.1)

            # Step 1: Use Claude to get source recommendations
            print("🔄 SSE: Calling Claude for source recommendations...")
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing your curiosity...', 'progress': 5})}\n\n"

            # Get source recommendations from Claude (run in thread to not block)
            def get_claude_recommendations():
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return claude_client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=2000,
                            messages=[{
                                "role": "user",
                                "content": f"""Analyze this curiosity and recommend sources:

CURIOSITY: "{curiosity}"

Provide:
1. Top 4 foundational books (if applicable) - must be real, well-known books
2. Keywords for YouTube videos (specific, relevant searches)
3. Keywords for articles/news

Respond in JSON:
{{
  "books": [
    {{"id": "1", "title": "Book Title", "author": "Author Name", "why": "Why essential"}}
  ],
  "youtubeSearchTerms": ["search term 1", "search term 2"],
  "articleSearchTerms": ["article search 1"]
}}"""
                            }]
                        )
                    except Exception as e:
                        if "overloaded" in str(e).lower() and attempt < max_retries - 1:
                            import time
                            wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                            print(f"⚠️  Claude API overloaded, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                            time.sleep(wait_time)
                            continue
                        raise

            message = await asyncio.to_thread(get_claude_recommendations)
            print("✅ SSE: Claude recommendations received")

            response_text = message.content[0].text
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            claude_data = json.loads(response_text)
            books = claude_data.get("books", [])
            youtube_terms = claude_data.get("youtubeSearchTerms", [curiosity])
            print(f"✅ SSE: Got {len(books)} books, {len(youtube_terms)} YouTube terms")

            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching for articles...', 'progress': 15})}\n\n"

            # Step 2: Search articles with Perplexity (with error handling)
            articles = []
            try:
                print("🔄 SSE: Searching articles with Perplexity...")
                yield f"data: {json.dumps({'type': 'searching', 'query': curiosity, 'category': 'articles'})}\n\n"
                await asyncio.sleep(0.2)

                # Run Perplexity search with aggressive timeout protection
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(perplexity_deep_research, curiosity, False),
                        timeout=30.0  # 30 second max wait
                    )
                    articles = result.get("sources", [])
                    print(f"✅ SSE: Got {len(articles)} articles from Perplexity")
                except asyncio.TimeoutError:
                    print(f"⚠️ SSE: Perplexity timed out after 30s, using empty fallback")
                    articles = []
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Article search timed out, continuing with other sources...', 'progress': 35})}\n\n"

                # Stream each article
                for i, article in enumerate(articles):
                    yield f"data: {json.dumps({'type': 'source_found', 'data': article, 'category': 'article', 'index': i})}\n\n"
                    await asyncio.sleep(0.1)

                yield f"data: {json.dumps({'type': 'status', 'message': f'Found {len(articles)} articles', 'progress': 35})}\n\n"
            except Exception as e:
                print(f"❌ Article search failed: {e}")
                yield f"data: {json.dumps({'type': 'status', 'message': 'Article search had issues, continuing...', 'progress': 35})}\n\n"

            # Step 3: Search YouTube channels (with error handling)
            youtube_channels = []
            try:
                print("🔄 SSE: Searching YouTube...")
                yield f"data: {json.dumps({'type': 'status', 'message': 'Finding YouTube channels...', 'progress': 45})}\n\n"
                yield f"data: {json.dumps({'type': 'searching', 'query': youtube_terms[0] if youtube_terms else curiosity, 'category': 'youtube'})}\n\n"

                for term in youtube_terms[:2]:  # Search top 2 terms
                    videos = await asyncio.to_thread(search_youtube_real, term, 3)
                    for video in videos:
                        # Skip videos without proper data
                        if not video.get("title") or not video.get("video_id"):
                            continue

                        # Use video data - treat as individual videos, not channels
                        video_data = {
                            "id": video.get("video_id"),
                            "title": video.get("title"),  # Video title is primary
                            "name": video.get("channel_name") or video.get("title"),  # Fallback to title
                            "author": video.get("channel_name"),  # Store channel as author
                            "url": f"https://youtube.com/watch?v={video.get('video_id')}"
                        }
                        youtube_channels.append(video_data)
                        yield f"data: {json.dumps({'type': 'source_found', 'data': video_data, 'category': 'youtube', 'index': len(youtube_channels)-1})}\n\n"
                        await asyncio.sleep(0.1)

                print(f"✅ SSE: Got {len(youtube_channels)} YouTube channels")
                yield f"data: {json.dumps({'type': 'status', 'message': f'Found {len(youtube_channels)} YouTube videos', 'progress': 60})}\n\n"
            except Exception as e:
                print(f"❌ YouTube search failed: {e}")
                yield f"data: {json.dumps({'type': 'status', 'message': 'YouTube search had issues, continuing...', 'progress': 60})}\n\n"

            # Step 4: Discover X/Twitter thought leaders (with error handling)
            x_accounts = []
            try:
                print("🔄 SSE: Discovering X/Twitter thought leaders...")
                yield f"data: {json.dumps({'type': 'status', 'message': 'Finding X thought leaders...', 'progress': 70})}\n\n"
                yield f"data: {json.dumps({'type': 'searching', 'query': curiosity, 'category': 'twitter'})}\n\n"
                await asyncio.sleep(0.2)

                # Run X/Twitter discovery with timeout protection
                try:
                    x_accounts = await asyncio.wait_for(
                        asyncio.to_thread(discover_x_thought_leaders, curiosity, 10),
                        timeout=30.0  # 30 second max wait
                    )
                    print(f"✅ SSE: Got {len(x_accounts)} X accounts")
                except asyncio.TimeoutError:
                    print(f"⚠️ SSE: X/Twitter discovery timed out after 30s")
                    x_accounts = []
                    yield f"data: {json.dumps({'type': 'status', 'message': 'X search timed out, continuing...', 'progress': 90})}\n\n"

                for i, account in enumerate(x_accounts):
                    yield f"data: {json.dumps({'type': 'source_found', 'data': account, 'category': 'twitter', 'index': i})}\n\n"
                    await asyncio.sleep(0.1)

                yield f"data: {json.dumps({'type': 'status', 'message': f'Found {len(x_accounts)} X accounts', 'progress': 90})}\n\n"
            except Exception as e:
                print(f"❌ X/Twitter search failed: {e}")
                yield f"data: {json.dumps({'type': 'status', 'message': 'X search had issues, continuing...', 'progress': 90})}\n\n"

            # Step 5: Stream books
            print(f"🔄 SSE: Streaming {len(books)} books...")
            yield f"data: {json.dumps({'type': 'status', 'message': 'Finding books...', 'progress': 95})}\n\n"
            for i, book in enumerate(books):
                yield f"data: {json.dumps({'type': 'source_found', 'data': book, 'category': 'book', 'index': i})}\n\n"
                await asyncio.sleep(0.1)

            # Final summary
            total_sources = len(articles) + len(youtube_channels) + len(x_accounts) + len(books)
            print(f"✅ SSE: Complete! Total sources: {total_sources}")
            yield f"data: {json.dumps({'type': 'complete', 'totalSources': total_sources, 'books': books, 'channels': youtube_channels, 'articles': articles, 'twitter': x_accounts, 'progress': 100})}\n\n"

        except Exception as e:
            print(f"❌ Streaming error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/api/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        # Check if OpenAI API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        # Read file contents
        audio_bytes = await file.read()
        print(f"🎤 Transcribing audio file: {file.filename} ({len(audio_bytes)} bytes)...")

        # Use OpenAI Whisper API for transcription
        import io
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)

        # Create a file-like object from bytes
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file.filename or "audio.m4a"  # Whisper needs a filename

        # Call Whisper API
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"  # Can be auto-detected by removing this
        )

        transcribed_text = transcription.text
        print(f"✅ Transcription complete: {transcribed_text[:100]}...")

        return {
            "text": transcribed_text,
            "success": True
        }

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/analyze-curiosity")
async def analyze_curiosity(request: CuriosityRequest):
    """Analyze curiosity with INTELLIGENT source routing - Perplexity-style"""
    try:
        # STEP 1: Understand the INTENT and TYPE of curiosity
        print(f"\n🧠 Analyzing curiosity intent: '{request.curiosity}'")

        intent_analysis = claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Analyze this curiosity and determine the best source strategy:

CURIOSITY: "{request.curiosity}"

Classify this into ONE of these categories:
1. "current_events" - asks about what's happening now, today, recently, news, current affairs
2. "evergreen_topic" - timeless topic that benefits from books, foundational knowledge
3. "technical_deep_dive" - specific technical topic needing expert explanations
4. "personal_curiosity" - exploratory question about a subject

Examples:
- "what happened in the world today" → current_events
- "explain quantum physics" → evergreen_topic
- "how does kubernetes work" → technical_deep_dive
- "why are people obsessed with Taylor Swift" → personal_curiosity

Respond in JSON:
{{
  "category": "current_events|evergreen_topic|technical_deep_dive|personal_curiosity",
  "reasoning": "Why this classification",
  "needs_news": true/false,
  "needs_books": true/false,
  "needs_videos": true/false,
  "needs_articles": true/false
}}"""
            }]
        )

        intent_text = intent_analysis.content[0].text
        if "```json" in intent_text:
            json_start = intent_text.find("```json") + 7
            json_end = intent_text.find("```", json_start)
            intent_text = intent_text[json_start:json_end].strip()

        intent_data = json.loads(intent_text)
        print(f"📊 Category: {intent_data.get('category')}")
        print(f"   Needs: Books={intent_data.get('needs_books')}, News={intent_data.get('needs_news')}, Videos={intent_data.get('needs_videos')}")

        # STEP 2: Use Claude to suggest appropriate sources based on intent
        message = claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": f"""Analyze this curiosity and recommend the best sources:

CURIOSITY: "{request.curiosity}"
CATEGORY: {intent_data.get('category')}

Based on the category, provide RELEVANT sources:
- If current_events: Focus on news search terms, recent articles, NO old books
- If evergreen_topic: Include foundational books, educational videos
- If technical: Focus on expert explanations, tutorials, technical articles
- If personal_curiosity: Mix of contemporary sources

Provide:
1. Top 3 foundational books (ONLY if needs_books=true, otherwise empty array)
2. Keywords for finding YouTube videos (if needs_videos=true)
3. Keywords for finding recent articles/news (if needs_articles=true or needs_news=true)
4. Suggested X/Twitter thought leaders (if applicable)

IMPORTANT: For current events queries, DO NOT suggest books. Use news-focused search terms.

Respond in JSON:
{{
  "understanding": "What they're curious about",
  "topics": ["topic1", "topic2", "topic3"],
  "books": [
    {{"id": "1", "title": "Book Title", "author": "Author", "why": "Why essential"}}
  ],
  "youtubeSearchTerms": ["search term 1", "search term 2"],
  "articleSearchTerms": ["article search 1"],
  "twitterAccounts": [
    {{"id": "1", "name": "Person", "handle": "username", "why": "Why follow"}}
  ]
}}"""
            }]
        )

        response_text = message.content[0].text
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"📄 Malformed JSON (first 500 chars):\n{response_text[:500]}")
            # Try to fix common JSON issues
            import re
            # Remove trailing commas before closing braces/brackets
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
            try:
                data = json.loads(fixed_text)
                print("✅ Fixed JSON by removing trailing commas")
            except json.JSONDecodeError:
                print("❌ Could not fix JSON automatically - using fallback")
                # Fallback: use simple structure
                data = {
                    "understanding": f"Exploring {request.curiosity}",
                    "topics": [request.curiosity],
                    "books": [],
                    "twitterAccounts": [],
                    "youtubeSearchTerms": [request.curiosity],
                    "articleSearchTerms": [request.curiosity]
                }

        # NOW SEARCH FOR REAL VIDEOS (10-15 results for scoring)
        youtube_search_terms = data.get("youtubeSearchTerms", [request.curiosity])
        real_videos = []
        for search_term in youtube_search_terms[:3]:  # Search top 3 terms
            videos = search_youtube_real(search_term, max_results=5)
            real_videos.extend(videos)

        # SEARCH FOR REAL ARTICLES/NEWS (10-15 results for scoring)
        article_search_terms = data.get("articleSearchTerms", [request.curiosity])
        real_articles = []
        perplexity_research = None  # Store Perplexity's synthesized answer

        # Detect if this is an evergreen topic needing deep research
        use_deep_research = (
            intent_data.get("needs_books") or
            intent_data.get("category") in ["evergreen_topic", "technical_deep_dive"]
        )

        # USE PERPLEXITY DEEP RESEARCH for current events (50-100+ sources!)
        if intent_data.get("needs_news") or intent_data.get("category") == "current_events":
            print(f"🔍 Using PERPLEXITY DEEP RESEARCH for current events query...")
            # Use Perplexity for comprehensive research
            perplexity_research = perplexity_deep_research(request.curiosity)
            real_articles = perplexity_research.get("sources", [])
            print(f"✅ Perplexity returned {len(real_articles)} sources with synthesis")

        # USE COMPREHENSIVE DEEP RESEARCH for evergreen topics (100-150+ sources!)
        elif use_deep_research:
            print(f"🔍 Using COMPREHENSIVE DEEP RESEARCH for evergreen topic...")
            themes = data.get("topics", [request.curiosity])

            # Run multi-query deep research to discover comprehensive sources
            deep_research_result = discover_comprehensive_sources(request.curiosity, themes)
            real_articles = deep_research_result.get("all_sources", [])

            print(f"📊 Discovered {len(real_articles)} total sources from {deep_research_result.get('queries_run')} queries")
            print(f"📚 Identified {len(deep_research_result.get('books_identified', []))} books for meta-content discovery")

            # For each identified book (top 3), discover meta-content
            books_with_meta = []
            for book_info in deep_research_result.get("books_identified", [])[:3]:
                book_title = book_info.get("title", "")
                book_author = book_info.get("author", "")

                print(f"📖 Finding meta-content for: {book_title}")
                meta_sources = discover_book_meta_content(book_title, book_author)

                # Add meta-sources to articles
                real_articles.extend(meta_sources)

                # Enrich book with meta-content info
                books_with_meta.append({
                    "title": book_title,
                    "author": book_author,
                    "why": book_info.get("why", ""),
                    "meta_source_count": len(meta_sources),
                    "meta_sources_preview": meta_sources[:3]  # First 3 for preview
                })

            # Replace Claude's book recommendations with enriched books
            if books_with_meta:
                data["books"] = books_with_meta
                print(f"✅ Enriched {len(books_with_meta)} books with meta-content")

            print(f"🎯 TOTAL SOURCES: {len(real_articles)} (target: 100-150+)")

        else:
            print(f"📄 Using general article search...")
            for search_term in article_search_terms[:2]:  # Search top 2 terms
                articles = search_web_for_articles(search_term, num_results=7)
                real_articles.extend(articles)

        # SCORE SOURCES based on relevance, recency, authority
        def score_source(source, source_type, curiosity_text):
            """Score a source from 0-100 based on multiple factors"""
            score = 50  # Base score

            # Relevance scoring (up to +30 points)
            title = source.get("title", "").lower()
            curiosity_lower = curiosity_text.lower()
            curiosity_keywords = set(curiosity_lower.split())
            title_keywords = set(title.split())

            # Count keyword matches
            keyword_matches = len(curiosity_keywords & title_keywords)
            score += min(keyword_matches * 5, 30)

            # Authority scoring (up to +15 points)
            if source_type == "video":
                # Prefer established creators (rough heuristic)
                if source.get("creator", "") and len(source.get("creator", "")) > 0:
                    score += 10
                # View count if available (future enhancement)

            elif source_type == "article":
                # Prefer known domains
                url = source.get("url", "").lower()
                if any(domain in url for domain in ["edu", "gov", "wikipedia", "stanford", "mit"]):
                    score += 15
                elif any(domain in url for domain in ["medium", "substack", "arxiv"]):
                    score += 10

            # Recency bonus (up to +5 points) - future enhancement
            # For now, newer search results get slight preference

            return min(score, 100)

        # Score all sources
        for video in real_videos:
            video["relevance_score"] = score_source(video, "video", request.curiosity)
            video["category"] = "Recent Content"

        for article in real_articles:
            article["relevance_score"] = score_source(article, "article", request.curiosity)
            article["category"] = "Expert Analysis"

        for book in data.get("books", []):
            book["relevance_score"] = 95  # Books are pre-vetted by Claude
            book["category"] = "Foundational"

        # Sort by relevance score
        real_videos.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        real_articles.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # DISCOVER X/TWITTER THOUGHT LEADERS WITH THEIR POSTS
        print(f"\n🐦 Discovering X/Twitter thought leaders...")
        x_thought_leaders = discover_x_thought_leaders(request.curiosity, max_accounts=10)
        print(f"✅ Found {len(x_thought_leaders)} X thought leaders")

        return {
            "understanding": data.get("understanding", ""),
            "topics": data.get("topics", []),
            "suggestedBooks": data.get("books", []),
            "suggestedChannels": real_videos[:10],  # Top 10 scored videos
            "suggestedTwitterAccounts": x_thought_leaders,  # Real X/Twitter accounts with posts
            "suggestedArticles": real_articles[:10],  # Top 10 scored articles
            "podcastDuration": 45,
            "totalSourcesScanned": {
                "videos": len(real_videos),
                "articles": len(real_articles),
                "books": len(data.get("books", [])),
                "x_accounts": len(x_thought_leaders)
            }
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# =================== DYNAMIC PROMPT SYSTEM ===================

def build_tone_instructions(tone_level: int) -> str:
    """Generate tone-specific instructions based on 1-10 scale"""
    if tone_level <= 3:  # Serious
        return """
**TONE: Serious & Professional**
- Use measured, thoughtful language
- Focus on facts and evidence
- Avoid casual jokes or humor
- Maintain gravitas and authority
- Examples: "Interestingly," "The data suggests," "From a rigorous perspective"
"""
    elif tone_level >= 8:  # Funny/Casual
        return """
**TONE: Light & Engaging**
- Include relevant jokes and witty observations
- Use pop culture analogies when appropriate
- React with genuine excitement and humor
- Don't be afraid to laugh at the absurdity of things
- Examples: "Wait, that's wild!", "No way!", "This is like if [funny analogy]"
"""
    else:  # Balanced (4-7)
        return """
**TONE: Conversational & Warm**
- Balance professionalism with approachability
- Use light humor when natural
- Show enthusiasm without overdoing it
- Friendly but informative
- Examples: "That's fascinating," "Here's the interesting part," "Let me break this down"
"""


def build_format_instructions(format_type: str, duration: int) -> str:
    """Generate format-specific instructions"""
    formats = {
        "conversational": f"""
**FORMAT: NotebookLM-Style Conversational**
Two enthusiastic hosts (Host A and Host B) having a natural dialogue.
- Natural interruptions and building on each other's ideas
- Show hosts processing information together
- Progressive understanding from broad to deep
Structure ({duration} min):
- Opening (30s): Warm welcome, topic intro
- Main Content (80%): Natural back-and-forth exploration
- Closing (30s): Summary and key takeaways
""",
        "debate": f"""
**FORMAT: Friendly Debate**
Three speakers with different perspectives (Host A, Host B, Host C):
- Host A: Moderator/facilitator
- Host B: Presents one perspective
- Host C: Presents alternative viewpoint
- Respectful disagreement with evidence-based arguments
- Find common ground and synthesize insights
Structure ({duration} min):
- Setup (1 min): Moderator introduces the debate topic
- Arguments (70%): Back-and-forth with evidence
- Synthesis (20%): Find consensus and key insights
- Closing (1 min): Moderator wraps up key points
""",
        "interview": f"""
**FORMAT: Expert Interview**
Two speakers (Interviewer and Expert):
- Interviewer: Asks insightful questions, represents audience curiosity
- Expert: Provides deep knowledge and explanations
- Use Socratic method when appropriate
- Build understanding through questioning
Structure ({duration} min):
- Introduction (1 min): Interviewer introduces expert and topic
- Deep Dive (80%): Q&A with follow-up questions
- Rapid Fire (10%): Quick interesting questions
- Closing (1 min): Final thoughts and takeaways
""",
        "storytelling": f"""
**FORMAT: Narrative Story**
One or two narrators telling a compelling story:
- Use narrative arc (setup, conflict, resolution)
- Include vivid details and scenes
- Weave in facts within the narrative
- Create emotional connection
- Use "show don't tell" when possible
Structure ({duration} min):
- Hook (30s): Grab attention with compelling opening
- Story (85%): Unfold the narrative with key insights
- Reflection (10%): What we learn from this story
- Closing (30s): Memorable conclusion
""",
        "educational": f"""
**FORMAT: Educational Explainer**
Two teachers breaking down complex topics:
- Start simple, build complexity gradually
- Use analogies and real-world examples extensively
- Check for understanding throughout
- Recap key concepts periodically
- Very structured and organized
Structure ({duration} min):
- Overview (1 min): What we'll learn today
- Lesson (80%): Systematic explanation with examples
- Review (15%): Recap main concepts
- Application (5%): How to use this knowledge
"""
    }
    return formats.get(format_type, formats["conversational"])


def build_depth_instructions(technical_depth: int) -> str:
    """Generate depth-specific instructions based on 1-10 scale"""
    if technical_depth <= 3:  # Beginner
        return """
**DEPTH: Beginner-Friendly**
- Define ALL jargon and technical terms
- Use simple analogies from everyday life
- Break complex ideas into small digestible chunks
- Repeat key concepts in different ways
- Examples: "Think of it like [simple analogy]," "In plain English, that means..."
"""
    elif technical_depth >= 8:  # Expert
        return """
**DEPTH: Expert-Level**
- Use technical terminology without over-explaining
- Discuss nuances and edge cases
- Reference academic research and specialized concepts
- Assume strong foundational knowledge
- Examples: "The stochastic nature of," "From a theoretical framework," "The empirical evidence suggests"
"""
    else:  # Intermediate (4-7)
        return """
**DEPTH: Intermediate**
- Explain technical terms when first introduced
- Balance accessibility with intellectual depth
- Use analogies for complex concepts, but don't oversimplify
- Assume some general knowledge
- Examples: "You might be familiar with," "Building on that concept," "To put it another way"
"""


def build_pacing_instructions(pacing: int) -> str:
    """Generate pacing instructions based on 1-10 scale"""
    if pacing <= 3:  # Slow, detailed
        return """
**PACING: Slow & Thorough**
- Take time to explore ideas deeply
- Include detailed examples and case studies
- Circle back to reinforce concepts
- Don't rush through complex topics
- Verbal pauses showing thinking
"""
    elif pacing >= 8:  # Fast overview
        return """
**PACING: Fast & Dynamic**
- Move quickly between ideas
- Hit highlights without dwelling
- Energetic, rapid-fire exchanges
- Cover breadth over depth
- Keep momentum high
"""
    else:  # Moderate (4-7)
        return """
**PACING: Balanced**
- Spend time on key concepts
- Move efficiently through supporting details
- Natural flow without feeling rushed
- Mix of depth and breadth
"""


def build_persona_instructions(persona: Optional[str]) -> str:
    """Generate teaching persona instructions"""
    if not persona:
        return ""

    personas = {
        "Feynman": """
**TEACHING STYLE: Richard Feynman**
- Break everything down to first principles
- Use simple, vivid analogies
- Show genuine curiosity and joy in discovery
- "The trick is to not fool yourself"
- Challenge assumptions constantly
""",
        "Sagan": """
**TEACHING STYLE: Carl Sagan**
- Weave in wonder and cosmic perspective
- Poetic and inspiring language
- Connect ideas to the human experience
- "We are made of star stuff"
- Humble yet authoritative
""",
        "Rogan": """
**TEACHING STYLE: Joe Rogan**
- Ask lots of clarifying questions
- Express genuine surprise and interest
- Conversational and accessible
- "Wait, that's crazy"
- Connect to practical applications
""",
        "Huberman": """
**TEACHING STYLE: Andrew Huberman**
- Science-based and protocol-oriented
- Explain mechanisms thoroughly
- Practical takeaways and action items
- References to neuroscience and biology
- "The data suggests"
"""
    }
    return personas.get(persona, "")


@app.post("/api/generate-smart-podcast")
async def generate_smart_podcast(request: GeneratePodcastRequest):
    """Generate podcast with REAL sources and REAL audio"""
    try:
        sources_text = f"# CURIOSITY: {request.curiosity}\n\n"
        sources_used = {
            "books_referenced": [],
            "videos_extracted": [],
            "articles_extracted": []
        }

        # 1. Books (use Claude's knowledge)
        print(f"\n📚 Referencing {len(request.selectedBooks)} books (Claude's knowledge)...")
        for book in request.selectedBooks:
            title = book.get("title", "")
            author = book.get("author", "")
            sources_text += f"\n\n=== BOOK: {title} by {author} ===\n(Use your knowledge of this book)\n\n"
            sources_used["books_referenced"].append(f"{title} by {author}")

        # 2. YouTube videos (REAL transcripts)
        print(f"\n🎥 Processing {len(request.selectedVideos)} videos...")
        for video in request.selectedVideos:
            video_id = video.get("videoId")
            title = video.get("title", "Unknown Video")

            if video_id:
                transcript = get_youtube_transcript(video_id)
                if transcript:
                    sources_text += f"\n\n=== VIDEO: {title} ===\n\n{transcript}\n\n"
                    sources_used["videos_extracted"].append(title)

        # 3. Articles (REAL scraping)
        print(f"\n📄 Processing {len(request.selectedArticles)} articles...")
        for article in request.selectedArticles:
            url = article.get("url", "")
            title = article.get("title", "Article")

            if url:
                article_text = scrape_article(url)
                if article_text:
                    sources_text += f"\n\n=== ARTICLE: {title} ===\n\n{article_text}\n\n"
                    sources_used["articles_extracted"].append(title)

        # 4. Build dynamic prompt based on personalization parameters
        # Set defaults if not provided
        tone_level = request.tone_level if request.tone_level is not None else 5
        technical_depth = request.technical_depth if request.technical_depth is not None else 5
        format_type = request.format_type or "conversational"
        pacing = request.pacing if request.pacing is not None else 5

        tone_instructions = build_tone_instructions(tone_level)
        format_instructions = build_format_instructions(format_type, request.duration)
        depth_instructions = build_depth_instructions(technical_depth)
        pacing_instructions = build_pacing_instructions(pacing)
        persona_instructions = build_persona_instructions(request.teaching_persona)
        custom_instructions = f"\n**CUSTOM INSTRUCTIONS:**\n{request.custom_instructions}\n" if request.custom_instructions else ""

        # 4. Generate script
        print(f"\n🎙️ Generating {request.duration}-minute podcast (tone={tone_level}, format={format_type}, depth={technical_depth})...")

        # Calculate target word count (150 words per minute for natural speech)
        target_word_count = request.duration * 150
        min_word_count = int(target_word_count * 0.9)  # 90% of target
        max_word_count = int(target_word_count * 1.1)  # 110% of target

        print(f"📊 Target: {target_word_count} words ({min_word_count}-{max_word_count} range)")

        message = claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",  # Claude 4.5 Sonnet - most capable model
            max_tokens=16000,  # Much higher for longer podcasts
            messages=[{
                "role": "user",
                "content": f"""You are creating a {request.duration}-minute podcast with the following specifications:

**PODCAST HOSTS:**
The two hosts are named James (male voice) and Sophia (female voice). They are knowledgeable, engaging conversationalists who work well together. Note: You will NOT write their names in the script - the script uses pure dialogue without labels.

**LISTENER'S CURIOSITY:**
The listener asked: "{request.curiosity}"

This is a thoughtful question that reveals genuine intellectual curiosity. Your role as podcast hosts James and Sophia is to be supportive guides helping the listener explore this topic together. Throughout the conversation:
- Acknowledge what makes this question interesting and worth exploring
- Frame the discussion as a collaborative journey of discovery with the listener
- Be encouraging of curiosity in a genuine, non-effusive way
- Use subtle phrases like "This is a fascinating area to explore..." or "What makes this particularly interesting is..." or "Let's dive into this together..."
- Position yourselves as knowledgeable friends sharing insights, not distant experts lecturing

**CRITICAL LENGTH REQUIREMENT:**
- Target word count: {target_word_count} words (EXACTLY)
- Minimum acceptable: {min_word_count} words
- Maximum acceptable: {max_word_count} words
- Average speaking rate: 150 words per minute
- This MUST produce a {request.duration}-minute podcast when read aloud

SOURCE MATERIAL TO ANALYZE:
{sources_text}

{format_instructions}

{tone_instructions}

{depth_instructions}

{pacing_instructions}

{persona_instructions}

{custom_instructions}

**GENERAL CONVERSATION DYNAMICS:**
1. **Natural Interruptions & Building**: Hosts should interrupt each other naturally, finish each other's thoughts, and build on ideas
2. **Genuine Reactions**: Include authentic emotional responses appropriate to the tone
3. **Source Attribution**: Reference sources naturally and FREQUENTLY in conversation
   - Use phrases like "According to [Source Name]...", "The research shows...", "In [Author]'s work..."
   - Mention specific data points, statistics, and quotes from the sources
   - Reference multiple sources throughout, not just at the beginning
4. **Progressive Understanding**: Start broad, go deeper, circle back to connect ideas
5. **Verbal Thinking**: Show hosts processing information live (when appropriate for pacing)
6. **Listener-Centric Framing**: Occasionally reference "you" the listener, acknowledging their curiosity

**FORMATTING RULES - CRITICAL FOR AUDIO GENERATION:**
- Write PURE DIALOGUE with NO speaker labels, names, or identifiers
- Use paragraph breaks (double line breaks) to separate each speaker's turn
- DO NOT write "Alex:", "Host A:", "Speaker 1:", or ANY labels
- Each paragraph = one speaker's complete thought/statement
- Speakers will alternate automatically based on paragraph breaks
- Write ONLY what should be spoken aloud
- DO NOT use brackets, asterisks, or stage directions
- DO NOT use placeholder text like [laughs] or *pauses*
- If a host laughs, write it as dialogue: "Ha! That's wild!"
- If there's a pause, use "..." within the dialogue

**EXAMPLES OF CORRECT FORMAT:**

✅ CORRECT (No labels, just dialogue with paragraph breaks):
```
Welcome back! Today we're exploring something really fascinating - the economic state of Nicaragua. This is such a timely question, and there's a lot to unpack here.

You know what's interesting about this? Nicaragua's economy has gone through such dramatic shifts over the past few decades. Let's start with where things stand today.

Absolutely. So according to the latest IMF data, Nicaragua's GDP has been...
```

❌ WRONG (Has labels):
```
Alex: Welcome back! Today we're exploring...
Jordan: You know what's interesting about this?
```

❌ WRONG (Has stage directions):
```
*excited* Welcome back! *laughs* Today we're exploring...
```

**CRITICAL:**
- This is AUDIO ONLY - listeners cannot see labels or stage directions
- The audio generation system will automatically assign voices to alternating paragraphs
- Make it sound natural and authentic
- Use contractions and conversational language
- Express reactions through WORDS, not descriptions
- Create a podcast that matches ALL the specified parameters

Generate the complete {request.duration}-minute podcast script now (pure dialogue, no labels, paragraph breaks between speakers):"""
            }]
        )

        script = message.content[0].text

        # Save script to file for review
        storage_dir = get_storage_dir()
        script_filename = f"podcast_script_{hash(request.curiosity) % 100000}.txt"
        script_path = storage_dir / script_filename
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(f"TOPIC: {request.curiosity}\n")
            f.write(f"DURATION: {request.duration} minutes\n")
            f.write(f"TONE: {tone_level}/10, DEPTH: {technical_depth}/10\n")
            f.write(f"HOSTS: James (male) and Sophia (female)\n")
            f.write("="*80 + "\n\n")
            f.write(script)
        print(f"📝 Saved script to {script_path}")

        # 5. Generate AUDIO with overall timeout protection
        print(f"\n🔊 Generating audio with ElevenLabs...")
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        try:
            # Run with 10-minute max timeout for entire audio generation
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_podcast_audio, script, request.voice_ids)
                audio_bytes = future.result(timeout=600)  # 10 minutes max
        except FuturesTimeoutError:
            print("⚠️ Audio generation timed out after 10 minutes - returning without audio")
            audio_bytes = b""
        except Exception as e:
            print(f"⚠️ Audio generation failed: {e}")
            audio_bytes = b""

        # Save audio to file
        audio_filename = f"podcast_{hash(request.curiosity) % 100000}.mp3"
        audio_path = storage_dir / audio_filename

        if audio_bytes:
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"✅ Saved audio to {audio_path}")

        backend_url = get_backend_url()
        return {
            "status": "completed",
            "script": script,
            "audio_url": f"{backend_url}/audio/{audio_filename}" if audio_bytes else None,
            "audio_available": bool(audio_bytes),
            "sources_used": sources_used
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated podcast audio files"""
    import os

    # Security: Only allow .mp3 files and prevent directory traversal
    if not filename.endswith('.mp3') or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    storage_dir = get_storage_dir()
    audio_path = storage_dir / filename

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        str(audio_path),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Accept-Ranges": "bytes",  # Enable streaming
            "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
        }
    )


# =================== AUTH ENDPOINTS ===================

@app.post("/api/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        hashed_password=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Generate access token
    access_token = create_access_token(data={"sub": new_user.id})

    return TokenResponse(access_token=access_token)


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(credentials: UserLoginRequest, db: Session = Depends(get_db)):
    """Login with email and password"""
    # Find user by email
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate access token
    access_token = create_access_token(data={"sub": user.id})

    return TokenResponse(access_token=access_token)


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user


# =================== PODCAST CRUD ENDPOINTS ===================

@app.get("/api/podcasts", response_model=List[PodcastResponse])
async def get_podcasts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all podcasts for the current user"""
    podcasts = db.query(Podcast).filter(Podcast.user_id == current_user.id).order_by(Podcast.created_at.desc()).all()
    return podcasts


@app.get("/api/podcasts/{podcast_id}", response_model=PodcastResponse)
async def get_podcast(
    podcast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific podcast by ID"""
    podcast = db.query(Podcast).filter(
        Podcast.id == podcast_id,
        Podcast.user_id == current_user.id
    ).first()

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    return podcast


@app.post("/api/podcasts", response_model=PodcastResponse, status_code=status.HTTP_201_CREATED)
async def save_podcast(
    podcast_data: SavePodcastRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save a new podcast to user's library"""
    new_podcast = Podcast(
        user_id=current_user.id,
        topic=podcast_data.topic,
        script=podcast_data.script,
        audio_url=podcast_data.audio_url,
        duration=podcast_data.duration,
        sources_used=podcast_data.sources_used
    )

    db.add(new_podcast)
    db.commit()
    db.refresh(new_podcast)

    return new_podcast


@app.patch("/api/podcasts/{podcast_id}", response_model=PodcastResponse)
async def update_podcast(
    podcast_id: int,
    is_favorite: Optional[bool] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update podcast (toggle favorite, etc)"""
    podcast = db.query(Podcast).filter(
        Podcast.id == podcast_id,
        Podcast.user_id == current_user.id
    ).first()

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    if is_favorite is not None:
        podcast.is_favorite = is_favorite

    db.commit()
    db.refresh(podcast)

    return podcast


@app.delete("/api/podcasts/{podcast_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_podcast(
    podcast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a podcast from user's library"""
    podcast = db.query(Podcast).filter(
        Podcast.id == podcast_id,
        Podcast.user_id == current_user.id
    ).first()

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    db.delete(podcast)
    db.commit()

    return None


# =================== FEEDBACK ENDPOINTS ===================

@app.post("/api/feedback", response_model=PodcastFeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback_data: PodcastFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit feedback for a podcast"""
    # Verify the podcast belongs to the user
    podcast = db.query(Podcast).filter(
        Podcast.id == feedback_data.podcast_id,
        Podcast.user_id == current_user.id
    ).first()

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    # Create feedback entry
    new_feedback = PodcastFeedback(
        podcast_id=feedback_data.podcast_id,
        rating=feedback_data.rating,
        complexity_feedback=feedback_data.complexity_feedback,
        tone_feedback=feedback_data.tone_feedback,
        loved=feedback_data.loved,
        improvements=feedback_data.improvements,
        completion_rate=feedback_data.completion_rate
    )

    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)

    # TODO: Update user profile based on feedback patterns
    # This will be implemented in behavioral tracking phase

    return new_feedback


@app.get("/api/feedback/{podcast_id}", response_model=List[PodcastFeedbackResponse])
async def get_podcast_feedback(
    podcast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all feedback for a specific podcast"""
    # Verify the podcast belongs to the user
    podcast = db.query(Podcast).filter(
        Podcast.id == podcast_id,
        Podcast.user_id == current_user.id
    ).first()

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    feedback = db.query(PodcastFeedback).filter(
        PodcastFeedback.podcast_id == podcast_id
    ).order_by(PodcastFeedback.created_at.desc()).all()

    return feedback


# =================== USER PROFILE ENDPOINTS ===================

@app.get("/api/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user profile and learning preferences"""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()

    if not profile:
        # Create default profile if doesn't exist
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
        db.commit()
        db.refresh(profile)

    return profile


@app.post("/api/profile", response_model=UserProfileResponse)
@app.put("/api/profile", response_model=UserProfileResponse)
async def update_user_profile(
    profile_data: UserProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create or update user profile and preferences"""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()

    if not profile:
        # Create new profile
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)

    # Update fields
    if profile_data.learning_style is not None:
        profile.learning_style = profile_data.learning_style
    if profile_data.teaching_preference is not None:
        profile.teaching_preference = profile_data.teaching_preference
    if profile_data.expertise_level is not None:
        profile.expertise_level = profile_data.expertise_level
    if profile_data.humor_preference is not None:
        profile.humor_preference = profile_data.humor_preference
    if profile_data.favorite_educators is not None:
        profile.favorite_educators = profile_data.favorite_educators
    if profile_data.preferred_duration is not None:
        profile.preferred_duration = profile_data.preferred_duration
    if profile_data.preferred_voices is not None:
        profile.preferred_voices = profile_data.preferred_voices

    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return profile


if __name__ == "__main__":
    import uvicorn
    # Get port from environment (Railway) or default to 8000
    port = int(os.getenv("PORT", 8000))

    print("\n" + "="*70)
    print("🎙️ ADCAST MVP API - Actually Works Edition")
    print("="*70)
    print("✅ Real YouTube search (API + fallback scraping)")
    print("✅ Real web search for articles (DuckDuckGo)")
    print("✅ ElevenLabs audio generation")
    print("✅ Claude's knowledge for books")
    print("="*70)
    print(f"\n🚀 Running on http://0.0.0.0:{port}\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
