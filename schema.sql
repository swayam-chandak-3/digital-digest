CREATE TABLE items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Source metadata
    source TEXT NOT NULL,                    -- hackernews, reddit, rss, etc
    source_item_id TEXT,                     -- HN ID, Reddit ID, etc
    source_url TEXT,

    -- Content
    title TEXT NOT NULL,
    description TEXT,                        -- short summary if available
    content TEXT,                            -- full text if fetched
    url TEXT,

    -- Timing & engagement
    published_at DATETIME,
    engagement_score REAL,                   -- points, upvotes, comments-derived
    raw_metadata JSON,                       -- source-specific payload

    -- Pipeline control
    ingestion_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'INGESTED'            -- INGESTED | PREFILTERED | EVALUATED | DEDUPED | SUMMARIZED | DISCARDED
);


CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,

    persona TEXT NOT NULL,                   -- GENAI_NEWS | PRODUCT_IDEAS

    -- Shared intelligence
    decision TEXT CHECK(decision IN ('KEEP', 'DROP')),
    relevance_score REAL,

    -- GENAI_NEWS fields
    topic TEXT,
    why_it_matters TEXT,
    target_audience TEXT,                    -- developer | architect | manager

    -- PRODUCT_IDEAS fields
    idea_type TEXT,
    problem_statement TEXT,
    solution_summary TEXT,
    maturity_level TEXT,                     -- idea | mvp | early-traction
    reusability_score REAL,

    -- Metadata
    llm_model TEXT,
    evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (item_id) REFERENCES items(id),
    UNIQUE(item_id, persona)
);

CREATE TABLE embeddings (
    item_id INTEGER PRIMARY KEY,
    persona TEXT NOT NULL,
    vector BLOB NOT NULL,                    -- serialized float32 array
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (item_id) REFERENCES items(id),
    UNIQUE(item_id, persona)
);

