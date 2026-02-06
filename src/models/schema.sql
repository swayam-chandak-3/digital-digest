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
-- delete from evaluations;
-- -- delete from embeddings;
-- -- delete from dedup_item_cluster;
-- -- delete from dedup_clusters;
-- delete from items;

-- UPDATE users
-- SET telegramid = 1676531263
-- where id=2;

-- update items 
-- set status='INGESTED'
-- where digest_type='GENAI';

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

-- =============================================================================
-- TOPIC-BASED PREFERENCE SYSTEM
-- =============================================================================

-- Master topics table - predefined topics for article classification
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,              -- internal name (e.g., 'llm_research')
    display_name TEXT NOT NULL,             -- human-readable name
    category TEXT NOT NULL CHECK(category IN ('GENAI_NEWS', 'PRODUCT_IDEAS')),
    description TEXT,                       -- brief description of the topic
    keywords TEXT                           -- comma-separated keywords for fallback matching
);

-- Item-to-topic mapping (many-to-many relationship)
-- Presence of a row means the topic IS assigned (boolean true)
-- No row means the topic is NOT assigned (boolean false)
CREATE TABLE IF NOT EXISTS item_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    assigned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE,
    UNIQUE(item_id, topic_id)
);

-- User topic preferences (affinity scores that learn from user interactions)
CREATE TABLE IF NOT EXISTS user_topic_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    score INTEGER DEFAULT 0,                -- increments on positive feedback, decrements on negative
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE,
    UNIQUE(user_id, topic_id)
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_item_topics_item_id ON item_topics(item_id);
CREATE INDEX IF NOT EXISTS idx_item_topics_topic_id ON item_topics(topic_id);
CREATE INDEX IF NOT EXISTS idx_user_topic_preferences_user_id ON user_topic_preferences(user_id);

-- =============================================================================
-- SEED DATA: Predefined Topics (10 total: 5 GENAI_NEWS + 5 PRODUCT_IDEAS)
-- =============================================================================

-- GENAI_NEWS Topics (5)
INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('llm_research', 'LLM Research', 'GENAI_NEWS', 
 'Large Language Model research, papers, and breakthroughs', 
 'llm,gpt,transformer,attention,fine-tuning,training,bert,language model,neural network,deep learning');

INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('ai_tools', 'AI Tools', 'GENAI_NEWS', 
 'AI-powered tools, applications, and assistants', 
 'ai tool,chatbot,assistant,copilot,agent,ai app,ai assistant,claude,gemini,chatgpt');

INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('ml_infrastructure', 'ML Infrastructure', 'GENAI_NEWS', 
 'MLOps, model deployment, and AI infrastructure', 
 'mlops,deployment,inference,gpu,kubernetes,docker,serving,pipeline,training infrastructure,vllm');

INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('prompt_engineering', 'Prompt Engineering', 'GENAI_NEWS', 
 'Prompt design, techniques, and optimization', 
 'prompt,chain of thought,few-shot,zero-shot,instruction,prompt engineering,rag,retrieval');

INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('ai_ethics', 'AI Ethics', 'GENAI_NEWS', 
 'AI safety, alignment, ethics, and responsible AI', 
 'alignment,safety,bias,ethics,responsible ai,hallucination,jailbreak,red team,guardrails');

-- PRODUCT_IDEAS Topics (5)
INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('saas_products', 'SaaS Products', 'PRODUCT_IDEAS', 
 'SaaS, subscription products, and web applications', 
 'saas,subscription,mrr,arr,b2b,b2c,startup,launch,pricing,freemium');

INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('developer_tools', 'Developer Tools', 'PRODUCT_IDEAS', 
 'Tools, APIs, and SDKs for developers', 
 'api,sdk,cli,developer tool,devtool,ide,extension,plugin,library,framework');

INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('automation', 'Automation', 'PRODUCT_IDEAS', 
 'Workflow automation and no-code/low-code solutions', 
 'automation,workflow,no-code,low-code,zapier,integration,automate,bot,script');

INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('analytics', 'Analytics', 'PRODUCT_IDEAS', 
 'Data analytics, dashboards, and reporting tools', 
 'analytics,dashboard,metrics,reporting,insights,data,visualization,tracking,monitoring');

INSERT OR IGNORE INTO topics (name, display_name, category, description, keywords) VALUES
('open_source', 'Open Source', 'PRODUCT_IDEAS', 
 'Open source projects and community-driven tools', 
 'open source,github,oss,community,free,self-hosted,contributor,repo,fork');
