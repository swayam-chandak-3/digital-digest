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

### Step 1: Run `pipeline.py`
Scrapes and ingests articles from multiple sources:
- **Scrape sources**
  - Hacker News: list pages → article content extraction
  - Indie Hackers: tech/AI listings → detail pages (likes/comments)
  - Reddit: subreddit listings → recent posts
- **Normalize + Store**
  - Normalize into a shared schema
  - Insert into `items` with `digest_type` and engagement metrics
- **Prefilter**
  - Time window + engagement thresholds
  - Category/keyword filtering (LLM/AI, Programming/Software, Product/Startup)

### Step 2: Run `evaluation.py`
LLM-based evaluation and filtering:
- **Evaluate**
  - LLM evaluation (Gemma3) writes to `evaluations` table
  - Decisions are saved as `KEEP` or `DROP`
  - Assigns relevance scores and target audience classification
  - Filters articles based on relevance thresholds

### Step 3: Run deduplication
Removes duplicate and near-duplicate articles:
- **Deduplicate**
  - FAISS embeddings to cluster near-duplicates
  - Canonical item chosen per cluster

### Step 4: Run delivery
Summarizes and delivers the curated digest:
- **Summarize + Deliver**
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

## Digest Article Features
Each article in the digest includes:

### Core Information
- **Title**: Article headline
- **Link**: Direct URL to the source article
- **Summary**: AI-generated concise overview of the article content
- **Engagement Metrics**: Number of likes and comments from the source platform

### Evaluation Metadata
- **Best Suited For**: Target audience classification
  - `Developer` - For software engineers and technical practitioners
  - `Architect` - For system designers and technical leads
  - `Manager` - For team leads and project managers
- **Why It Matters**: AI-generated explanation of the article's relevance and impact

### Intelligent Filtering
- **Gemma3 LLM Evaluation**: Articles are evaluated using Gemma3 AI model to:
  - Determine relevance to the target audience
  - Classify articles by topic and category
  - Assign relevance scores
  - Flag the most valuable articles for inclusion in the daily digest
- Only articles that pass the evaluation threshold are included in the final digest

## Notes
- Telegram recipient ID are currently **hardcoded** because a UI for configuration has not been implemented yet.

## Setup
Install dependencies:
```
pip install -r requirements.txt
```
