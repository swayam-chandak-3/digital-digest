# AI-Powered Intelligence Digest System

A unified pipeline that scrapes multiple sources, evaluates relevance, deduplicates, summarizes, and delivers a daily AI/product digest to Telegram.

Bot name: `@DailyDigestEpamBot`

## Tech stack
- Python 3.x
- BeautifulSoup + Readability-LXML for content extraction
- SQLite for storage (`items`, `evaluations`)
- FAISS for deduplication
- Ollama + Gemma3 for evaluation
- Telegram Bot API for delivery

## Pipeline flow
1. **Scrape sources**
   - Hacker News: list pages → article content extraction
   - Indie Hackers: tech/AI listings → detail pages (likes/comments)
   - Reddit: subreddit listings → recent posts
2. **Normalize + Store**
   - Normalize into a shared schema
   - Insert into `items` with `digest_type` and engagement metrics
3. **Prefilter**
   - Time window + engagement thresholds
   - Category/keyword filtering (LLM/AI, Programming/Software, Product/Startup)
4. **Evaluate**
   - LLM evaluation (Gemma3) writes to `evaluations`
   - Decisions are saved as `KEEP` or `DROP`
5. **Deduplicate**
   - FAISS embeddings to cluster near-duplicates
   - Canonical item chosen per cluster
6. **Summarize + Deliver**
   - Summaries + formatting
   - Telegram delivery via `@DailyDigestEpamBot`

## Current features
- Multi-source scraping (HN, Indie Hackers, Reddit)
- Engagement-based prefiltering
- Full-content extraction with metadata
- LLM evaluation with configurable thresholds
- SQLite persistence for items and evaluations
- Deduplication pipeline (FAISS)
- Telegram delivery

## Notes
- Telegram recipient ID are currently **hardcoded** because a UI for configuration has not been implemented yet.

## Setup
Install dependencies:
```
pip install -r requirements.txt
```
