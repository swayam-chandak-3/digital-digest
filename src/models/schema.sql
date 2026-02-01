-- CREATE TABLE preference (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     news_type TEXT NOT NULL
-- );

-- CREATE TABLE users (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     name TEXT,
--     emailid TEXT,
--     telegramid TEXT,
--     topic_preference INTEGER NOT NULL,
--     FOREIGN KEY (topic_preference) REFERENCES preference(id)
-- );

-- CREATE TABLE sources (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     source TEXT NOT NULL UNIQUE              -- hackernews, reddit, rss, etc
-- );

-- INSERT INTO sources (source) VALUES ('hackernews');
-- INSERT INTO sources (source) VALUES ('reddit');

-- CREATE TABLE items (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,

--     -- Source metadata
--     source TEXT NOT NULL,                    -- hackernews, reddit, rss, etc
--     source_id INTEGER,                       -- FK to sources(id)
--     source_url TEXT,

--     -- Content
--     digest_type TEXT NOT NULL,                
--     title TEXT NOT NULL,
--     description TEXT,                        -- short summary if available
--     summary TEXT,                            -- generated or stored summary
--     content TEXT,                            -- full text if fetched
--     url TEXT,

--     -- Timing & engagement
--     published_at DATETIME,
--     engagement_score REAL,                   -- points, upvotes, comments-derived
--     likes INTEGER DEFAULT 0,                 -- number of likes
--     comments INTEGER DEFAULT 0,              -- number of comments
--     views INTEGER DEFAULT 0,                 -- number of views
--     raw_metadata JSON,                       -- source-specific payload

--     -- Pipeline control
--     ingestion_time DATETIME DEFAULT CURRENT_TIMESTAMP,
--     status TEXT DEFAULT 'INGESTED'            -- INGESTED | PREFILTERED | EVALUATED | DEDUPED | SUMMARIZED | DISCARDED

--     ,FOREIGN KEY (source_id) REFERENCES sources(id)
-- );




-- CREATE TABLE evaluations (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     item_id INTEGER NOT NULL,

--     persona TEXT NOT NULL,                   -- GENAI_NEWS | PRODUCT_IDEAS

--     -- Shared intelligence
--     decision TEXT CHECK(decision IN ('KEEP', 'DROP')),
--     relevance_score REAL,

--     -- GENAI_NEWS fields
--     topic TEXT,
--     why_it_matters TEXT,
--     target_audience TEXT,                    -- developer | architect | manager

--     -- PRODUCT_IDEAS fields
--     idea_type TEXT,
--     problem_statement TEXT,
--     solution_summary TEXT,
--     maturity_level TEXT,                     -- idea | mvp | early-traction
--     reusability_score REAL,

--     -- Metadata
--     llm_model TEXT,
--     evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
--     evaluation_type TEXT,                    -- FULL | TFIDF | TEXTRANK

--     FOREIGN KEY (item_id) REFERENCES items(id),
--     UNIQUE(item_id, persona)
-- );

-- CREATE TABLE embeddings (
--     item_id INTEGER PRIMARY KEY,
--     persona TEXT NOT NULL,
--     vector BLOB NOT NULL,                    -- serialized float32 array
--     created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

--     FOREIGN KEY (item_id) REFERENCES items(id),
--     UNIQUE(item_id, persona)
-- );

-- -- Deduplication (FAISS + topic similarity): one canonical item per cluster
-- CREATE TABLE dedup_clusters (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     canonical_item_id INTEGER NOT NULL,
--     created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
--     FOREIGN KEY (canonical_item_id) REFERENCES items(id)
-- );
-- CREATE TABLE dedup_item_cluster (
--     item_id INTEGER NOT NULL,
--     cluster_id INTEGER NOT NULL,
--     is_canonical INTEGER NOT NULL DEFAULT 0,
--     PRIMARY KEY (item_id),
--     FOREIGN KEY (item_id) REFERENCES items(id),
--     FOREIGN KEY (cluster_id) REFERENCES dedup_clusters(id)
-- );

-- -- Summarization output: 3-5 line technical summary, why_it_matters, target_audience (source: LLM or TEXTRANK)
-- CREATE TABLE item_summaries (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     item_id INTEGER NOT NULL,
--     technical_summary TEXT NOT NULL,
--     why_it_matters TEXT,
--     target_audience TEXT,
--     source TEXT NOT NULL,
--     created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
--     FOREIGN KEY (item_id) REFERENCES items(id),
--     UNIQUE(item_id, source)
-- );

-- -- delete from item_summaries;
-- -- delete from evaluations;
-- -- delete from embeddings;
-- -- delete from dedup_item_cluster;
-- -- delete from dedup_clusters;
-- -- delete from items;

UPDATE users
SET telegramid = 1676531263
where id=2;

-- -- Add summary column to items if it does not exist (run once on existing DBs):
-- -- ALTER TABLE items ADD COLUMN summary TEXT;

-- -- Add evaluation_type column to evaluations (run once on existing DBs):
-- -- ALTER TABLE evaluations ADD COLUMN evaluation_type TEXT;

-- -- DELETE FROM evaluations;
-- -- delete from items;

-- -- alter table user add digest_type TEXT DEFAULT "";
-- -- alter table items add column likes INTEGER DEFAULT 0;
-- --     alter table items add column comments INTEGER DEFAULT 0;
-- --     alter table items add column views INTEGER DEFAULT 0;

-- INSERT INTO users (
--     name,
--     emailid,
--     telegramid,
--     topic_preference
-- )
-- VALUES (
--     'Swayam',
--     '',
--     '7669364505',
--     2
-- );


-- INSERT INTO users (
--     name,
--     emailid,
--     telegramid,
--     topic_preference
-- )
-- VALUES (
--     'Jayesg',
--     '',
--     '1728415003',
--     1
-- );

-- INSERT INTO preference (news_type) VALUES ('PRODUCT');
-- INSERT INTO preference (news_type) VALUES ('GENAI');
