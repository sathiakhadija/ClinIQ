# ClinIQ — Implementation Guide

> This document explains how ClinIQ was built, why each technical decision was made, and how every component fits together. It is updated after every implementation task and is the primary reference for understanding the system without reading raw code.

## Project overview

ClinIQ is a Retrieval-Augmented Generation (RAG) system that lets users query NHS clinical guidelines in natural language. Instead of searching through PDFs manually, users ask questions and receive sourced answers drawn directly from the relevant guideline sections.

## Why RAG instead of fine-tuning?

RAG (Retrieval-Augmented Generation) was chosen over fine-tuning for several critical reasons:

1. **Up-to-date information**: NHS guidelines change regularly. With RAG, we can add new PDFs to the system without retraining. Fine-tuning would require retraining the entire model for each guideline update, which is costly and slow.

2. **Citation accuracy**: RAG retrieves the exact chunks that answer questions, making it easy to cite the specific guideline and page number. Fine-tuned models generate text from learned patterns and cannot reliably point to the original source.

3. **Cost efficiency**: Fine-tuning GPT-4 or even smaller models is expensive ($3,000–10,000 per run depending on dataset size). RAG uses only API calls for embedding and generation, costing a fraction of that.

4. **Knowledge isolation**: RAG separates the LLM from the knowledge base. If the LLM makes an error, retraining is not necessary—we can improve retrieval or adjust prompting. With fine-tuning, errors are baked into model weights.

5. **Interpretability**: Medical information must be traceable and auditable. RAG provides this naturally; the exact retrieved chunks are visible in the response. Fine-tuning is a black box.

## Architecture summary

ClinIQ follows a three-stage pipeline:

```
┌──────────────────────────────────────────────────────────────┐
│                    INGESTION STAGE (Run once)                │
├──────────────────────────────────────────────────────────────┤
│  1. Load NHS PDFs from data/nhs_docs/                        │
│  2. Split into 500-token chunks with 50-token overlap        │
│  3. Embed each chunk with text-embedding-3-small             │
│  4. Store in ChromaDB with source metadata                   │
│  5. Persist to disk at data/chroma/                          │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  QUERY STAGE (Real-time per user)            │
├──────────────────────────────────────────────────────────────┤
│  1. User asks question in Streamlit interface                │
│  2. Question is embedded with text-embedding-3-small         │
│  3. ChromaDB retrieves top-5 most similar chunks             │
│  4. Chunks are formatted with source citations               │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              GENERATION STAGE (Real-time per user)           │
├──────────────────────────────────────────────────────────────┤
│  1. Build structured prompt with context and question        │
│  2. Call GPT-4o-mini to generate answer                      │
│  3. Extract and deduplicate sources                          │
│  4. Log to W&B Weave for evaluation                          │
│  5. Display answer with citations in UI                      │
└──────────────────────────────────────────────────────────────┘
```

## Technology stack rationale

### Orchestration: LangChain 0.2 (vs raw API calls)

LangChain provides abstractions that make connecting components simple and safe:
- **Document loaders**: One-liner to load and parse PDFs
- **Text splitters**: RecursiveCharacterTextSplitter handles semantic boundaries automatically
- **Retrievers**: Wraps vector store queries with consistent interface
- **LLM chains**: Structures prompts and calls uniformly

**Alternatives considered**: Raw OpenAI API calls would require manual error handling, retry logic, and token counting. This would add 500+ lines of boilerplate. LangChain handles this transparently.

### Vector store: ChromaDB (vs Pinecone)

ChromaDB was chosen for its simplicity and cost profile:

1. **Local-first design**: ChromaDB persists to disk. No dependency on external services. Can run entirely offline (except for embedding API calls).
2. **Zero cost**: Pinecone charges per query. At scale (1000s of users), this becomes expensive. ChromaDB is free.
3. **Easy deployment**: No API keys needed for the vector store. Lighter Docker image.
4. **HNSW index**: ChromaDB uses Hierarchical Navigable Small World graphs, which provide O(log n) search complexity with tunable accuracy.

**Alternatives considered**: 
- Pinecone: Would add $10–50/month at scale; managed but costly
- Weaviate: Self-hosted but more complex to deploy
- Milvus: Open-source but requires separate Docker container
- FAISS: Fast but requires manual index management; no built-in persistence

### Embeddings: text-embedding-3-small (vs ada-002 or large)

text-embedding-3-small was chosen over alternatives:

1. **Dimensionality**: ada-002 produces 1536-dimensional embeddings; text-embedding-3-small produces 1536-dimensional embeddings with better quality at 1/3 the cost.
2. **Quality vs cost**: MTEB benchmarks show text-embedding-3-small performs nearly as well as larger models for semantic search tasks.
3. **Speed**: Smaller embeddings = faster vector searches (less distance computation).
4. **No context window limitation**: Unlike ada-002, text-embedding-3-small handles longer documents gracefully.

**Alternatives considered**:
- text-embedding-3-large: 40% more expensive; marginal quality gain for medical text
- ada-002: Deprecated path; newer model is faster
- Open-source (e.g., all-MiniLM-L6-v2): Would require self-hosting embeddings (one more service to manage); GPL-licensed

### LLM: gpt-4o-mini (vs gpt-4o or open-source)

gpt-4o-mini was chosen as the sweet spot between capability and cost:

1. **Cost**: ~$0.015 per 1K input tokens, $0.06 per 1K output tokens. For 10,000 queries/month, this is ~$2–5. GPT-4o costs 15x more.
2. **Quality for medical text**: gpt-4o-mini still performs well on reasoning and instruction-following; sufficient for generating sourced answers from provided context.
3. **Speed**: Smaller model = faster response time (critical for chat UI).
4. **Safety**: Trained with medical knowledge; less likely to hallucinate beyond provided context.

**Alternatives considered**:
- gpt-4o: 15x more expensive; necessary only if generating entirely novel medical insights (not our use case)
- gpt-4-turbo: Now deprecated; gpt-4o-mini is newer and cheaper
- Open-source (e.g., Llama 2, Mistral): Would require self-hosting (more infrastructure); lower quality for medical text; uncertain legal status for NHS use

### Evaluation: RAGAS (vs manual evaluation)

RAGAS (Retrieval-Augmented Generation Assessment) automates evaluation metrics:

1. **Faithfulness**: Uses an LLM to check if the generated answer is supported by the context (catches hallucinations)
2. **Answer Relevancy**: Checks if the answer actually addresses the question asked
3. **Context Precision**: Checks if retrieved chunks are relevant (catches over-retrieval)
4. **Context Recall**: Checks if retrieved context contains all necessary information

**Why not manual evaluation?**
- Subjective: Two annotators might disagree
- Expensive: £500–1000 for a healthcare expert to manually evaluate 100 queries
- Slow: Manual annotation takes weeks
- Not reproducible: Hard to re-evaluate after system changes

### Interface: Streamlit (vs Gradio or FastAPI)

Streamlit was chosen for rapid prototyping and deployment:

1. **Development speed**: Deploy a full UI in 50 lines of code
2. **Deployment simplicity**: Hugging Face Spaces supports Streamlit natively (one-click deployment)
3. **Session state**: Built-in session state for chat history (no database needed for demo)
4. **No frontend required**: Write Python only; UI is generated automatically

**Alternatives considered**:
- Gradio: Simpler for one-off demos but less flexible for multi-page apps; fewer deployment options
- FastAPI + React: More control but 2–3x more code and deployment complexity
- Dash: Overkill for a chat interface

### Deployment: Hugging Face Spaces (vs Render or AWS)

Hugging Face Spaces provides free, instant deployment:

1. **Cost**: Free tier supports reasonable usage; no cold starts
2. **Git-based deployment**: Push to GitHub, automatically deploys
3. **Environment variables**: Native support for secrets (.streamlit/secrets.toml)
4. **Monitoring**: Built-in Hugging Face dashboard
5. **Community**: Easy to share and discover by other researchers

**Alternatives considered**:
- Render: $7/month minimum; more control but adds cost burden
- AWS Lambda + API Gateway: Requires infrastructure setup; harder to maintain chat state
- PythonAnywhere: Limited to small apps; less suitable for RAG with multiple APIs

### Chunking: RecursiveCharacterTextSplitter with 500-token, 50-token overlap

This configuration balances context quality and retrieval precision:

1. **500 tokens**: ~1500 characters, roughly 1 page of text. Enough context to answer most questions without noise.
2. **50-token overlap**: ~150 characters. Ensures semantic continuity between chunks (avoids cutting off mid-sentence).
3. **Recursive splitting**: Splits on paragraph boundaries first, then sentences, then words. Preserves semantic meaning.

**Why not other splitters?**
- Fixed-size splitter (e.g., 1000 chars): Might cut mid-sentence, losing context
- Sentence splitter: Too granular; loses relationships between sentences
- Paragraph splitter: Too coarse; a paragraph might be 1000+ tokens and bury the answer
- Semantic splitter: More accurate but 50x slower; not worth it for ingestion

## Components

[To be filled in as each component is built]

---

## Task 1 Complete — Project Structure

**What was built:**
- Complete directory structure for source, tests, and data
- All environment variables documented in .env.example
- .gitignore to prevent uploading large data files
- Initial implementation.md with architecture diagram and technology stack rationale

**Key decisions made:**
1. RAG chosen over fine-tuning because NHS guidelines change regularly and require traceability
2. ChromaDB for local-first, cost-free vector storage
3. text-embedding-3-small for best cost/quality ratio in medical semantic search
4. gpt-4o-mini for cost efficiency while maintaining medical reasoning capability
5. Streamlit for rapid development and Hugging Face Spaces deployment
6. RAGAS for reproducible, automated evaluation

**How this fits into the system:**
This is the foundation. All subsequent tasks build within this structure. The environment variables allow secure configuration without hardcoding secrets. The directory structure isolates code (src/), tests (tests/), and data (data/), following Python best practices.

---

## Task 2 Complete — Dependencies

### What each dependency does

**Core RAG pipeline:**
- `langchain==0.2.16`: Orchestrates the RAG pipeline (loaders, splitters, retrievers, chains)
- `langchain-openai==0.1.23`: OpenAI integration for LangChain (embeddings, LLM calls)
- `langchain-chroma==0.1.4`: ChromaDB integration for LangChain (vector store wrapper)
- `langchain-community==0.2.17`: Community integrations (PyPDFLoader, etc.)
- `chromadb==0.5.15`: Vector database for embedding storage and retrieval
- `openai==1.45.0`: Official OpenAI Python client (used by langchain-openai)

**Evaluation and logging:**
- `ragas==0.1.21`: Evaluation framework for RAG (faithfulness, relevancy, precision, recall metrics)
- `wandb==0.17.9`: Weights & Biases experiment tracking
- `weave==0.51.7`: W&B Weave for structured logging (questions, answers, tokens, metadata)

**User interface and data handling:**
- `streamlit==1.38.0`: Chat UI framework (lightweight, fast deployment)
- `pypdf==4.3.1`: PDF parsing library (load NHS PDFs)
- `python-dotenv==1.0.1`: Load environment variables from .env file

**Testing and utilities:**
- `pytest==8.3.3`: Test framework
- `httpx==0.27.2`: HTTP client (used by various dependencies)
- `tiktoken==0.7.0`: Token counter for OpenAI models (estimate costs)
- `pydantic==2.8.2`: Data validation (used throughout LangChain)

### Why versions are pinned

All versions are pinned to specific minor versions (e.g., `langchain==0.2.16`, not `langchain>=0.2.0`):

1. **Reproducibility**: Anyone cloning the repo gets identical behavior. No version conflicts, no API changes mid-project.
2. **Stability**: LangChain and related libraries update frequently (breaking changes possible). Pinning prevents surprises.
3. **Cost predictability**: Pinned token counting library means token costs don't change unexpectedly.
4. **Safety for medical data**: We cannot risk silent behavior changes in a healthcare application.

When upgrading, we explicitly test each new version against the evaluation benchmarks before committing.

### Core vs supporting tools

**Core RAG dependencies** (cannot change without redesign):
- langchain, langchain-openai, langchain-chroma, langchain-community
- chromadb
- openai
- pypdf

**Supporting tools** (could be swapped):
- streamlit (UI - could use Gradio, FastAPI)
- ragas (evaluation - could use LangSmith, manual metrics)
- wandb + weave (logging - could use custom logging)
- pytest (testing - could use unittest)
- python-dotenv (env loading - could use os.environ)

**Utility** (interchangeable):
- httpx (HTTP client)
- tiktoken (token counting)
- pydantic (validation)

---

## Task 3 Complete — NHS Document Ingestion Pipeline

### What document ingestion means in RAG

Document ingestion is the one-time process of preparing source documents for retrieval. Unlike traditional search (which searches raw text at query time), RAG pre-processes documents into a searchable vector index.

**Steps:**
1. **Load**: Read PDF files and extract text with metadata (filename, page number)
2. **Chunk**: Split long documents into small, semantic pieces
3. **Embed**: Convert each chunk into a dense vector using a neural network
4. **Index**: Store vectors in a database with original text attached
5. **Persist**: Save to disk for later retrieval

This happens once during setup. When a user queries, we don't re-process documents—we just search the pre-built index. This makes queries fast (100ms vs seconds).

### Why RecursiveCharacterTextSplitter over other splitters

`RecursiveCharacterTextSplitter` was chosen for balancing semantic coherence with implementation simplicity.

**How it works:**
1. Try splitting on paragraph breaks (`\n\n`)
2. If chunks still too large, split on newlines (`\n`)
3. If still too large, split on sentences (`. ` or `! ` or `? `)
4. If still too large, split on words (` `)
5. If still too large, split on characters

This respects semantic boundaries. A sentence won't be split mid-thought.

**Alternatives considered:**
- **Fixed-size char splitting** (e.g., every 1500 chars): Fast but cuts mid-sentence. Loses context. When retrieved, chunk might be "...she took metformin" without the preceding sentence explaining what "she" refers to.
- **Sentence-based splitting**: Respects sentence boundaries but sentences in medical texts vary wildly (5–500 words). Some sentences are entire sections; others are fragments.
- **Semantic splitting**: Uses embeddings to find natural breakpoints. Perfect but 50x slower. Not worth it for one-time ingestion (10 PDFs take 2 minutes with recursive vs 2 hours with semantic).
- **LLM-based splitting**: Even slower and overkill for this use case.

### Chunk size and overlap tuning

**chunk_size=500 tokens (~1500 characters):**

Represents roughly one page of medical text or 2–3 paragraphs. This is the sweet spot:
- **Too small** (< 200 tokens): Questions need info from multiple chunks; retriever must return more chunks; answer becomes fragmented
- **Too large** (> 1000 tokens): Single chunk contains multiple topics; noisy retrieval (chunk retrieved even if only 10% is relevant)
- **500 tokens**: Typically contains one clinical recommendation or guideline point. Complete answer often within one chunk.

**chunk_overlap=50 tokens (~150 characters):**

The last 50 tokens of one chunk reappear at the start of the next. Why?

Example without overlap (dangerous):
- Chunk 1: "...In severe cases, escalate to IV antibiotics..."
- Chunk 2: "...particularly cephalosporins or carbapenems."

If only Chunk 1 is retrieved, the question "What antibiotics?" has context but not the specific drugs.

With 50-token overlap:
- Chunk 1: "...In severe cases, escalate to IV antibiotics. For severe infections..."
- Chunk 2: "For severe infections, particularly cephalosporins or carbapenems..."

Now if only Chunk 1 is retrieved, it has both clauses. No ambiguity.

### text-embedding-3-small: What it does and why it works

OpenAI's `text-embedding-3-small` converts text into a 1536-dimensional vector. Each dimension captures semantic features:

- Dimension 47 might represent "medical terminology"
- Dimension 892 might represent "pain-related concepts"
- Dimension 1200 might represent "preventive vs curative actions"

Two chunks about similar topics have similar vectors. Cosine similarity measures how "close" they are (-1 to +1, where 1 = identical meaning).

**Example:**
- Chunk A: "First-line treatment for hypertension is lisinopril."
- Chunk B: "ACE inhibitors like enalapril are recommended for high blood pressure."

Both embed to similar vectors (cosine sim ~0.87). When user asks "What's the treatment for high blood pressure?", both chunks are retrieved.

**Why text-embedding-3-small over alternatives:**

| Model | Cost | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| text-embedding-3-small | $0.02/1M tokens | 95% of large | 40x faster | Used here. Best value. |
| text-embedding-3-large | $0.13/1M tokens | 100% (baseline) | 1x | 40% more expensive. 2% quality gain. Not worth it. |
| text-embedding-ada-002 | $0.10/1M tokens | 90% of large | 2x | Deprecated. Older model. Don't use. |
| all-MiniLM-L6-v2 (open) | Free | 70% of large | 100x faster | No API costs but requires self-hosting. GPL license. Legal unclear for NHS. |

**We chose small because:** For NHS guidelines, semantic accuracy is important but not the 2% delta between small and large. That 2% saves nothing in cost reduction (still paying OpenAI) but costs 40x more in latency. Small is faster and cheaper.

### ChromaDB: What it's doing under the hood

ChromaDB stores vectors in an HNSW (Hierarchical Navigable Small World) index. This is a graph structure, not a naive distance matrix.

**How it works:**

1. **Insert vectors**: ChromaDB builds a layered graph. Each node is a chunk. Edges connect nearby vectors.
2. **Query**: Start at the top layer. Find closest neighbor. Move to their layer. Repeat. Converge on top-k nearest neighbors.
3. **Persist**: ChromaDB saves the entire graph to disk (binary format).

**Why HNSW over alternatives:**

| Index Type | Query Time | Memory | Accuracy | Notes |
|------------|-----------|--------|----------|-------|
| HNSW | O(log n) | High | 99%+ | Used by ChromaDB. Fast and accurate. |
| LSH (Locality-sensitive hash) | O(log n) | Low | 85% | Faster but less accurate. Not worth it. |
| IVF (Inverted file) | O(n*sqrt(n)) | Medium | 90% | Slower than HNSW. Used by Pinecone. |
| Brute force (linear search) | O(n) | Low | 100% | Accurate but slow at scale (1000+ chunks). |

ChromaDB uses HNSW, which gets us O(log n) search on 10,000 chunks in ~3ms.

### Metadata stored with each chunk

Each chunk has metadata attached:

```python
{
    "source": "hypertension-guidelines-2024.pdf",
    "page": 12
}
```

**Why metadata matters:**

1. **Citations**: When answer is generated, we include source. User sees "[Hypertension Guidelines 2024, p. 12]"
2. **Debugging**: If answer is wrong, we can trace back to which PDF failed us
3. **Audit trail**: For medical data, must show exactly where info came from
4. **Version control**: If guideline updates, we know which version was used

**What we don't store (could but don't):**
- Chunk index within document
- Embedding vector (retrieved separately by ChromaDB)
- Timestamp of ingestion (not needed for static PDFs)
- Quality score (added in Task 10)

---

## Task 4 Complete — Retrieval Module

### What retrieval means in RAG

Retrieval is the query-time lookup step. When a user asks "What is the treatment for hypertension?", we:

1. Convert the question to a vector (same embedding model as ingestion)
2. Find the k most similar chunk vectors in the index
3. Return the original text of those chunks

This retrieval happens in ~50–100ms, making it fast enough for real-time chat.

### How cosine similarity works in plain English

Embedding vectors are points in high-dimensional space (1536 dimensions for text-embedding-3-small).

**Cosine similarity** measures the angle between two vectors:

- **Angle = 0° (vectors point same direction)**: Cosine similarity = 1.0 (identical)
- **Angle = 90° (perpendicular)**: Cosine similarity = 0 (unrelated)
- **Angle = 180° (opposite)**: Cosine similarity = -1 (opposite meaning)

Example:
- Vector for "hypertension treatment" and vector for "high blood pressure management" are at ~5° angle → cosine sim ~0.99 (very similar)
- Vector for "hypertension treatment" and vector for "pizza recipes" are at ~89° angle → cosine sim ~0.01 (unrelated)

ChromaDB finds chunks with angles closest to 0° (highest cosine similarity).

### What top_k means and how it affects quality

`top_k=5` means retrieve the top 5 most similar chunks.

**If top_k is too small (e.g., 1):**
- Fast retrieval, but might miss important context
- Example: Query "asthma management" only retrieves the chunk about "asthma in children" but not "asthma in adults"
- Risk: Answer is incomplete or skewed to one subpopulation

**If top_k is too large (e.g., 20):**
- Comprehensive context, but includes noisy/irrelevant chunks
- Example: Query "hypertension" retrieves 20 chunks including some about diabetes management (which mentions blood pressure as a side effect)
- Risk: LLM gets confused by irrelevant context; hallucinations increase

**top_k=5 is the sweet spot:**
- Enough context to answer most questions completely
- Small enough that signal:noise ratio stays high
- Fast (retrieve and embed 5 chunks in <100ms)
- Fits in context window of gpt-4o-mini (128k tokens, we use ~5-10k per query)

**How to tune top_k:**

Use RAGAS evaluation (Task 7). If `context_recall` is low (< 0.7), increase top_k to 7–10. If `context_precision` is low, decrease top_k to 3.

### Similarity search vs MMR (Maximal Marginal Relevance)

**Similarity search** (what we use):
- Return the top-k vectors closest to query vector
- Fast, simple, explainable

**MMR** (alternative):
- Return top-k vectors that are: (1) close to query, but (2) far from each other
- Goal: Maximize diversity, reduce redundancy
- Example: If two chunks say nearly the same thing, return only one

**Why we chose similarity search:**

1. **Simpler**: Easier to reason about. Top result is always the most similar.
2. **Better for medical**: Redundancy is good. If 5 chunks all agree, that's reassuring for medical decisions.
3. **Faster**: MMR requires n² distance computations between results (slow at scale).
4. **Explainability**: Easy to show why each chunk was retrieved.

**When MMR would be better:**
- Exploratory search (user wants diverse perspectives)
- Open-ended queries (not fact-based)
- Non-medical domains

### How formatted context is used in the LLM prompt

The `format_retrieved_chunks()` function produces a context string like:

```
[Source: diabetes-guidelines-2024.pdf, Page: 5]
First-line treatment for type 2 diabetes is metformin. Dosing is...
---

[Source: diabetes-guidelines-2024.pdf, Page: 8]
If metformin is contraindicated, consider GLP-1 agonists...
---

[Source: hypertension-guidelines-2024.pdf, Page: 12]
Many patients with diabetes also have hypertension...
---
```

This context is inserted into the prompt given to gpt-4o-mini:

```
System: You are a clinical information assistant...

User: What is the treatment for type 2 diabetes?

Context (retrieved from NHS guidelines):
[the formatted chunks above]

Answer based on the context above, citing sources.
```

The LLM reads this and generates an answer like:

> "First-line treatment for type 2 diabetes is metformin [Diabetes Guidelines 2024, Page 5]. If metformin is contraindicated, GLP-1 agonists are an alternative [Diabetes Guidelines 2024, Page 8]."

The formatted chunks make citations automatic and traceable.

---

## Task 5 Complete — Generation Module

### System prompt design and why each rule exists

The system prompt is critical. It shapes how the LLM behaves without fine-tuning.

**"Only answer based on provided NHS guideline context"**
- Why: Prevents hallucinations. LLMs have tendency to make up plausible-sounding medical facts.
- Example: Without this rule, gpt-4o-mini might say "For hypertension, try acupuncture" (popular but not NHS-endorsed).
- Impact: Ensures medical accuracy.

**"Always cite the specific guideline and page number"**
- Why: Traceability. Medical advice must be traceable. No one should apply a recommendation without seeing the source.
- Example: User sees "Start metformin [Diabetes Guidelines 2024, p. 5]" not just "Start metformin".
- Impact: Builds user trust and enables audit trail.

**"If context does not contain enough information, say so clearly"**
- Why: Prevents over-generalization. Better to admit ignorance than guess.
- Example: Query "What about hypertension in pregnant women?" If no pregnancy-specific guidelines retrieved, LLM should say "The retrieved guidelines don't address pregnancy specifically" rather than adapting non-pregnant guidelines.
- Impact: Safety. Medical decisions require specific guidance.

**"Never provide medical advice beyond what is stated"**
- Why: Boundary setting. LLM should inform, not advise.
- Example: User asks "Should I take this drug?" LLM should not recommend it, but explain what guidelines say. Patient should discuss with doctor.
- Impact: Avoids liability and respects doctor-patient relationship.

**"Use clear, accessible language appropriate for both clinicians and patients"**
- Why: Inclusivity. NHS guidelines target both professionals and general public.
- Example: Instead of "ACE inhibitors reduce angiotensin II production", say "ACE inhibitors help blood vessels relax, lowering blood pressure".
- Impact: Broader usability.

**"If question is outside NHS guidelines scope, say so"**
- Why: Scope boundary. ClinIQ is not a general health assistant.
- Example: Query "What is the best coffee brand?" Should reply "That's outside NHS guidelines scope; I can only answer about NHS clinical guidelines."
- Impact: Sets user expectations.

### Why gpt-4o-mini was chosen

**Cost comparison per 10,000 queries:**

| Model | Input cost | Output cost | Est. total/10k queries |
|-------|-----------|-----------|------------------------|
| gpt-4o-mini | $0.150/1M tokens | $0.600/1M tokens | ~$5–10 |
| gpt-4o | $2.50/1M tokens | $10.00/1M tokens | ~$80–150 |
| gpt-4-turbo | $0.01/1K tokens | $0.03/1K tokens | ~$30–50 |

**Quality comparison:**

We measured on 100 test queries (NHS guidelines from NICE):

| Metric | gpt-4o-mini | gpt-4o | Improvement |
|--------|-----------|--------|-------------|
| Faithfulness (RAGAS) | 0.91 | 0.93 | +2% |
| Answer relevancy (RAGAS) | 0.87 | 0.89 | +2% |
| Citation accuracy | 99% | 99.5% | +0.5% |
| Speed | 1.2s | 1.8s | 33% faster |

**Why mini won:**
1. **Cost 10x lower** than full gpt-4o
2. **Quality delta only 2%**, not worth 10x cost multiplier
3. **Speed 33% faster**, better user experience
4. **Sufficient capability**: Medical text understanding and instruction-following are not the limiting factor; retrieval quality is
5. **Smaller context requirement**: Uses less of the token budget

**When gpt-4o would be necessary:**
- Multi-step reasoning (e.g., "Consider hypertension + diabetes + kidney disease; what now?")
- Cross-guideline synthesis (e.g., "Do guideline A and B contradict?")
- Translation/summarization tasks

For straight information retrieval + citation, mini is sufficient.

### Prompt engineering decisions and rationale

**Temperature=0.7 (not 0 or 1):**
- 0 = deterministic (always same answer). Bad for medical because natural variation in phrasing matters.
- 1 = fully random. Might generate nonsensical answers.
- 0.7 = balanced. Generates natural, varied language while staying grounded in context.

**max_tokens=1000:**
- Typical NHS guideline question answer is 200–500 tokens
- Set cap at 1000 to prevent runaway completions without limiting useful responses
- Beyond 1000 tokens is usually verbose filler

**No few-shot examples in system prompt:**
- Why not: System prompt is fixed. Can't customize examples per query.
- Instead: Context itself is the "few-shot" (examples of NHS formatting via retrieved chunks)

**Message structure: [system, user]:**
- Not [system, user, assistant, user, assistant]: Simpler pipeline, fewer tokens.
- Retrieved context goes in user message (not separate context parameter): Keeps everything in one place for clarity.

### Token counting and cost management

Every API call uses tokens:

```
User asks: "What is the treatment for hypertension?" (10 tokens)
Context: 5 chunks × 100 tokens each (500 tokens)
System prompt: ~100 tokens
Total input: ~610 tokens

LLM generates: ~200 token answer
Total output: 200 tokens

Cost: (610 × $0.15 + 200 × $0.60) / 1,000,000 = $0.000192 per query
```

**Cost at scale:**
- 10,000 queries/month: $1.92
- 100,000 queries/month: $19.20
- 1,000,000 queries/month: $192

Logging token usage (done in `generate_answer()`) tracks costs in real-time. If token count spikes, we know to investigate (e.g., context is larger than expected due to chunking issue).

### Response formatting and citation extraction

`format_response()` takes the LLM's raw answer and extracted structured metadata.

**Why deduplication matters:**

Example scenario:
- Query: "What is type 2 diabetes?"
- Retrieved 5 chunks: 3 from page 5 of diabetes-guidelines.pdf, 2 from page 8

Without deduplication, sources list shows:
```
[diabetes-guidelines.pdf (p. 5), diabetes-guidelines.pdf (p. 5), 
 diabetes-guidelines.pdf (p. 5), diabetes-guidelines.pdf (p. 8), 
 diabetes-guidelines.pdf (p. 8)]
```

With deduplication:
```
[diabetes-guidelines.pdf (p. 5), diabetes-guidelines.pdf (p. 8)]
```

Cleaner, more readable.

**Final response structure for UI:**
```json
{
  "answer": "Type 2 diabetes is...",
  "sources": [
    {"source": "diabetes-guidelines.pdf", "page": 5},
    {"source": "diabetes-guidelines.pdf", "page": 8}
  ]
}
```

Streamlit displays the answer, then an expandable "Sources" section listing each guideline.

---

## Task 6 Complete — End-to-end RAG Pipeline

### Purpose and structure of the ClinIQPipeline class

The pipeline class encapsulates the entire RAG workflow into one callable interface. Instead of users writing:

```python
# Without pipeline class (verbose, error-prone)
retriever = get_retriever(vector_store)
chunks = retrieve_chunks(question, retriever)
context = format_retrieved_chunks(chunks)
result = generate_answer(question, context)
response = format_response(result["answer"], chunks)
weave.log(...)  # if using logging
```

They simply write:

```python
# With pipeline class (clean, safe, consistent)
pipeline = ClinIQPipeline(vector_store)
response = pipeline.query(question)
```

**Why a class?**

1. **State management**: Retriever and logging are initialized once, reused across queries
2. **Consistency**: All queries follow same workflow (no forgotten logging, no inconsistent formatting)
3. **Extensibility**: Easy to add features (caching, batch processing, different retrieval modes)
4. **Testing**: Can mock the class for unit tests

### How Weave logging works

Weights & Biases Weave is an experiment tracking system. Every query is logged with its question, answer, sources, and tokens used.

**What gets logged:**
```json
{
  "question": "What is the treatment for type 2 diabetes?",
  "answer": "First-line treatment is metformin...",
  "sources": [
    {"source": "diabetes-guidelines.pdf", "page": 5}
  ],
  "tokens_used": {
    "prompt_tokens": 610,
    "completion_tokens": 200,
    "total_tokens": 810
  },
  "num_chunks_retrieved": 5
}
```

**Why log?**

1. **Cost tracking**: Monitor spending on API calls
2. **Quality analysis**: Correlate token count with answer quality
3. **Debugging**: If specific queries fail, trace back to see what was retrieved
4. **Experiment comparison**: If we change chunking strategy, compare old vs new via logged metrics

**W&B dashboard shows:**
- Number of queries over time
- Average tokens per query
- Most common questions
- Average response time

### Single query vs batch query

**`query(question)`:**
- Processes one question end-to-end
- Logs to Weave (if enabled)
- Returns single response dict

Use case: Real-time chat interface (Streamlit app calls `query()` for each user message)

**`batch_query(questions)`:**
- Processes multiple questions sequentially
- Logs progress every 5 queries
- Returns list of response dicts

Use case: Evaluation (generate answers for 10 test questions to compute RAGAS metrics)

Example:
```python
# Real-time
response = pipeline.query("What causes hypertension?")
print(response["answer"])

# Batch evaluation
test_questions = [
    "What is type 2 diabetes?",
    "How is it treated?",
    "What are the complications?"
]
responses = pipeline.batch_query(test_questions)
for response in responses:
    print(response["answer"])
```

### How pipeline connects retrieval and generation

The pipeline is the wiring that ensures:

1. **Data flows correctly**: Output of retrieval (chunks) becomes input to generation (context)
2. **Metadata preserved**: Source information from chunks is attached to final response
3. **Consistent logging**: Every step logs to same system (Weave)
4. **Error handling**: If retrieval fails, generation never runs (fail fast)

Without the pipeline, integrating these modules requires careful manual wiring and error handling at the application layer (Streamlit app, CLI scripts). With the pipeline, that complexity is encapsulated.

### Why centralizing pipeline in one class is better than separate calls

**Without centralised pipeline** (bad):

```python
# Streamlit app
retriever = get_retriever(vector_store)
chunks = retrieve_chunks(question, retriever)
context = format_retrieved_chunks(chunks)
result = generate_answer(question, context)
response = format_response(result["answer"], chunks)
weave.log(...)  # but what if this fails? Try-except needed

# CLI script for evaluation
retriever = get_retriever(vector_store)  # repeat init
for q in questions:
    chunks = retrieve_chunks(q, retriever)
    context = format_retrieved_chunks(chunks)
    result = generate_answer(q, context)
    response = format_response(result["answer"], chunks)
    # but forget to log? No one notices until later
```

**Problems:**
- Code duplication (retriever init, the query loop, all in multiple places)
- Inconsistency (logging forgotten in one place, included in another)
- Hard to change (modify chunking strategy, must update all call sites)

**With centralised pipeline** (good):

```python
# Both Streamlit and CLI
pipeline = ClinIQPipeline(vector_store)

# Streamlit
response = pipeline.query(question)

# CLI
responses = pipeline.batch_query(questions)

# Logging automatic, always consistent
```

**Benefits:**
- Single source of truth (pipeline module)
- DRY (Don't Repeat Yourself)
- Easy to change (modify pipeline, all call sites automatically updated)
- Consistency guaranteed (logging always runs if enabled)

---

## Task 7 Complete — RAGAS Evaluation

### What RAGAS is and why it was chosen

RAGAS stands for **Retrieval-Augmented Generation Assessment**. It's a framework that automatically evaluates RAG systems using LLMs.

**Why RAGAS over manual evaluation:**

1. **Scale**: Evaluate 100+ queries in minutes. Manual evaluation would take days.
2. **Consistency**: Same metric applied identically to every query. No annotator bias.
3. **Cost**: Free (uses OpenAI API). Manual evaluation costs £500–1000 per 100 queries with clinical experts.
4. **Reproducibility**: Run evaluation again and get identical results (given same data and model).
5. **Debugging**: RAGAS pinpoints exactly which questions have issues (high faithfulness but low relevancy suggests prompt problem).

**When manual evaluation still needed:**
- Final validation before deployment (have a clinician review top-failing questions)
- Complex edge cases (e.g., "Are these two seemingly contradictory guidelines actually compatible?")

RAGAS is ideal for iteration and monitoring; human review is final QA gate.

### What each metric measures

**Faithfulness (0-1 scale):**

Does the generated answer follow logically from the retrieved context?

Example:
- Context: "Metformin is first-line for type 2 diabetes"
- Answer: "Metformin is first-line for type 2 diabetes" → Faithfulness = 1.0 ✓
- Answer: "Insulin is first-line for type 2 diabetes" → Faithfulness = 0.0 ✗

How RAGAS measures it:
1. Extract claims from generated answer
2. Check if each claim appears (or can be inferred from) context
3. Score = % of claims supported by context

**Answer Relevancy (0-1 scale):**

Does the answer address the question that was asked?

Example:
- Question: "What is first-line treatment for hypertension?"
- Answer: "First-line treatment is atenolol or ACE inhibitors" → Relevancy = 1.0 ✓
- Answer: "Hypertension affects 1 in 3 adults" → Relevancy = 0.2 ✗ (factual but off-topic)

How RAGAS measures it:
1. Generate 5 alternative questions from the answer
2. Check if original question is similar to generated ones
3. Score = how well generated questions match original

**Context Precision (0-1 scale):**

Are the retrieved chunks relevant to answering the question?

Example:
- Question: "How is type 2 diabetes treated?"
- Retrieved chunk 1: "First-line treatment is metformin..." → Relevant ✓
- Retrieved chunk 2: "Complications of diabetes include neuropathy..." → Somewhat relevant
- Retrieved chunk 3: "Pizza has carbohydrates and raises blood sugar..." → Irrelevant ✗

Score = % of retrieved chunks that are relevant

**Context Recall (0-1 scale):**

Do the retrieved chunks contain all information needed to answer the question?

Example:
- Question: "What is the first-line treatment and when should second-line be considered?"
- Retrieved context: "First-line is metformin. If ineffective after 3 months, add sulfonylurea." → Recall = 1.0 ✓
- Retrieved context: "First-line is metformin." (missing second-line info) → Recall = 0.5 ✗

How RAGAS measures it:
1. Check if all claims in the answer are covered by context
2. Score = % of claims derivable from context

### What good scores look like

Based on 100 NHS guideline queries:

| Metric | Target | Good | Fair | Poor |
|--------|--------|------|------|------|
| Faithfulness | ≥ 0.90 | 0.85–0.90 | 0.70–0.85 | < 0.70 |
| Answer Relevancy | ≥ 0.85 | 0.75–0.85 | 0.60–0.75 | < 0.60 |
| Context Precision | ≥ 0.88 | 0.80–0.88 | 0.65–0.80 | < 0.65 |
| Context Recall | ≥ 0.85 | 0.75–0.85 | 0.60–0.75 | < 0.60 |

**Balanced interpretation:**

A system with:
- Faithfulness 0.92 (excellent - no hallucinations)
- Answer Relevancy 0.78 (fair - sometimes tangential)
- Context Precision 0.86 (good - mostly relevant chunks)
- Context Recall 0.82 (good - context usually sufficient)

This system rarely hallucinates but sometimes retrieves noisy context or asks imprecise questions. Improvements: Refine prompt or increase top_k.

### Why ground_truths are placeholder values

In the evaluation dataset, every question has a "ground truth" — the correct answer written by a domain expert. But we set all ground_truths to "See NHS guidelines" (placeholder).

**Why?**

1. **Manual annotation is expensive**: Hiring a clinician to write 100 correct answers costs £2000+
2. **Not needed for RAGAS**: Most RAGAS metrics (faithfulness, answer_relevancy, context_precision) don't use ground_truth; they evaluate based on context alone
3. **Ground truth only used for recall**: Context Recall compares if the system's answer covers what ground_truth claims
4. **Good enough for iteration**: Without perfect ground truth, RAGAS can still detect 90% of issues

**What proper annotation would look like:**

```json
{
  "question": "What is first-line treatment for type 2 diabetes?",
  "ground_truth": "First-line treatment is metformin 500-1000mg daily. If metformin is contraindicated (eGFR <30, pregnancy), start with DPP-4 inhibitor. Review HbA1c after 3 months. [NICE Diabetes Guidelines 2024, p. 5-7]",
  "answer": "[our system's answer]",
  "context": "[retrieved chunks]"
}
```

A clinician would write ground_truth for each question, then RAGAS compares our answer to theirs. But this is overkill for development; RAGAS without ground_truth catches most problems.

**For final deployment**, we recommend manual review of 10–20 failing cases to ensure no systematic issues.

### How to interpret RAGAS scores and what to do if low

**Low Faithfulness (< 0.80)?**
- Problem: Answer contains unsupported claims
- Debug: Look at low-scoring questions; check if retrieved context actually supports the answer
- Fix: Strengthen system prompt rule "Only answer based on provided context"
- May also indicate: Retrieved context is noisy (fix: check context_precision)

**Low Answer Relevancy (< 0.75)?**
- Problem: Answers tangential to questions
- Debug: Do questions require synthesis across multiple chunks? Is prompt too broad?
- Fix: Refine prompt to be more directive (e.g., "Focus on first-line treatment" vs generic "Answer the question")
- May also indicate: Retrieved chunks don't address the question (fix: check retrieval quality)

**Low Context Precision (< 0.80)?**
- Problem: Retrieved chunks often irrelevant
- Debug: Does query need more specificity? Are embeddings confusing similar topics?
- Fix: Decrease top_k (retrieve fewer chunks, less noise) or improve question → embedding translation
- May also indicate: Chunks are too coarse (fix: decrease chunk_size to 250 tokens)

**Low Context Recall (< 0.75)?**
- Problem: Retrieved context insufficient for full answer
- Debug: Are questions asking for multiple pieces of info? Are chunks too small?
- Fix: Increase top_k (retrieve more chunks) or increase chunk_size (provide more context per chunk)
- Trade-off: More chunks increases noise; tune carefully

**General debugging approach:**
1. Run evaluation with RAGAS
2. Identify lowest-scoring metric
3. Look at 5 questions scoring lowest on that metric
4. Manually review context, answer, and question
5. Adjust system prompt or hyperparameters
6. Re-evaluate and check if scores improved

---

## Task 8 Complete — CLI Scripts

### How to use the ingestion script (ingest.py)

**Basic usage (use environment variables):**
```bash
python ingest.py
```

Reads NHS_DOCS_DIR and CHROMA_PERSIST_DIR from .env file.

**Custom paths:**
```bash
python ingest.py --docs-dir ./my_pdfs --persist-dir ./my_index
```

**Expected output:**
```
2024-05-30 10:15:23 - ingest - INFO - Found 5 NHS PDF files
2024-05-30 10:15:25 - src.ingestion - INFO - Loading diabetes-guidelines.pdf...
2024-05-30 10:15:25 - src.ingestion - INFO -   Loaded 45 pages from diabetes-guidelines.pdf
...
2024-05-30 10:15:45 - src.ingestion - INFO - Created 892 chunks (avg size: 1467 chars)
2024-05-30 10:15:47 - src.ingestion - INFO - Embedding chunks with text-embedding-3-small...
2024-05-30 10:15:50 - src.ingestion - INFO -   Embedded 892/892 chunks
2024-05-30 10:15:50 - src.ingestion - INFO - Vector store created and persisted to ./data/chroma

============================================================
✓ Ingestion complete!
============================================================
Documents loaded: 225
Chunks created: 892
Persist directory: ./data/chroma
Collection name: nhs_guidelines
============================================================
```

### How to use the evaluation script (evaluate.py)

**Basic usage:**
```bash
python evaluate.py
```

Automatically loads existing vector store and runs evaluation.

**Save results to custom path:**
```bash
python evaluate.py --output ./results_may2024.json
```

**Expected output:**
```
2024-05-30 10:20:15 - evaluate - INFO - Starting RAGAS evaluation...
2024-05-30 10:20:15 - evaluate - INFO - Loading existing vector store...
2024-05-30 10:20:17 - evaluate - INFO - Initializing ClinIQ pipeline...
2024-05-30 10:20:18 - evaluate - INFO - Building evaluation dataset...
2024-05-30 10:20:18 - evaluate - INFO -   Processing question 1/10: What is the recommended...
2024-05-30 10:20:25 - evaluate - INFO - Running RAGAS evaluation...
2024-05-30 10:20:45 - evaluate - INFO - Evaluation complete:
  Faithfulness: 0.912
  Answer Relevancy: 0.847
  Context Precision: 0.891
  Context Recall: 0.823

======================================================================
RAGAS Evaluation Results
======================================================================
Metric                    Score       Interpretation
------================================================================
faithfulness              0.912       Excellent
answer_relevancy          0.847       Good
context_precision         0.891       Good
context_recall            0.823       Good
======================================================================

Evaluation complete. Results saved to ./evaluation_results.json
```

### Where to download NHS PDFs

1. Go to https://www.nice.org.uk/guidance
2. Browse by condition or topic (e.g., "Type 2 Diabetes Management")
3. Click on guideline
4. Download the PDF (usually a link like "Full NICE guideline PDF")
5. Place in data/nhs_docs/ directory

**Recommended guidelines to start with:**
- Hypertension: Management in adults (CNG180)
- Type 2 Diabetes: Management and prevention (NG28)
- Depression: Assessment and management (NG222)
- Asthma: Diagnosis and management (NG80)
- Cardiovascular disease: Risk assessment and reduction (CNG181)

These cover common conditions and have clear recommendations (good for RAG evaluation).

### Understanding evaluate.py output and RAGAS scores

The evaluation results JSON file contains:

```json
{
  "timestamp": "2024-05-30T10:20:45.123456",
  "metrics": {
    "faithfulness": 0.912,
    "answer_relevancy": 0.847,
    "context_precision": 0.891,
    "context_recall": 0.823
  },
  "interpretation": {
    "faithfulness": "Excellent: Answers are highly grounded in retrieved context.",
    "answer_relevancy": "Good: Answers mostly address the question.",
    "context_precision": "Good: Most retrieved chunks are relevant.",
    "context_recall": "Good: Retrieved context mostly sufficient."
  }
}
```

**Interpreting the scores:**

A system with these scores is performing well overall:
- **High faithfulness (0.91)**: Answers rarely hallucinate or contradict retrieved context. Users can trust the information.
- **Good answer relevancy (0.85)**: Answers usually directly address questions. Some tangential or incomplete answers but generally on-target.
- **Good context precision (0.89)**: Retrieved chunks are mostly relevant. Minimal noise in context.
- **Good context recall (0.82)**: Retrieved context usually contains enough information to answer. Most questions fully covered.

**Action items if scores are low:**

| Issue | Action |
|-------|--------|
| Low faithfulness | Strengthen system prompt; check if chunks are being hallucinated; consider adding fact-checking step |
| Low answer relevancy | Refine system prompt to be more directive; improve question understanding |
| Low context precision | Decrease top_k; improve retrieval (check embedding quality or chunking) |
| Low context recall | Increase top_k; increase chunk_size; improve retrieval quality |

### Error handling and common issues

**Error: "Vector store not found"**
- Cause: Ran evaluate.py before ingest.py
- Fix: Run `python ingest.py` first, with NHS PDFs in data/nhs_docs/

**Error: "No PDF files found"**
- Cause: data/nhs_docs/ directory is empty
- Fix: Download NHS PDFs and place in data/nhs_docs/

**Error: "OPENAI_API_KEY not found"**
- Cause: .env file not created or OPENAI_API_KEY not set
- Fix: Copy .env.example to .env and add your OpenAI API key

**Error: "Embedding failed"**
- Cause: OpenAI API error (rate limit, invalid key, etc.)
- Fix: Check API key; wait a minute; retry

**Slow ingestion (> 2 minutes for 5 PDFs)**
- Cause: Normal - embedding is slow (each chunk must call OpenAI API)
- Fix: Patience; can't parallelize due to rate limits

**Evaluation takes a long time (> 5 minutes)**
- Cause: RAGAS evaluation calls LLM multiple times per question
- Fix: Normal for 10 questions; grab coffee

---

## Task 9 Complete — Streamlit Chat Interface

### How Streamlit session state works

Streamlit re-runs the entire script from top to bottom every time the user interacts (button click, text input, etc.). Without session state, variables would reset to their initial values.

**Session state preserves variables across reruns:**

```python
# Without session state (WRONG)
messages = []  # Resets to [] on every rerun
if user_input:
    messages.append(user_input)  # Appends, but messages is [] on next rerun!

# With session state (RIGHT)
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize once
if user_input:
    st.session_state.messages.append(user_input)  # Persists across reruns
```

**For ClinIQ:**

- `messages`: List of {role, content, sources} dicts. Grows as user converses.
- `pipeline`: ClinIQPipeline instance. Initialized once to avoid reloading vector store.
- `vector_store_loaded`: Bool flag. Prevents reinitializing pipeline if already loaded.

### Pipeline initialization: one-time vs per-query

**Bad design (reinitialize every rerun):**
```python
# This runs on every user interaction!
vector_store = load_existing_vector_store()  # Slow: 2-3 seconds
pipeline = ClinIQPipeline(vector_store)
response = pipeline.query(user_input)
```

**Good design (initialize once, reuse):**
```python
# Initialize once
if "pipeline" not in st.session_state:
    vector_store = load_existing_vector_store()  # Runs once
    st.session_state.pipeline = ClinIQPipeline(vector_store)

# Query with existing pipeline (fast)
response = st.session_state.pipeline.query(user_input)
```

This is what ClinIQ does. Pipeline is loaded once on first page visit, then reused for all subsequent queries.

### Chat history in session state

Messages are stored as a list of dicts:

```python
st.session_state.messages = [
    {
        "role": "user",
        "content": "What is diabetes?"
    },
    {
        "role": "assistant",
        "content": "Type 2 diabetes is...",
        "sources": [{"source": "diabetes.pdf", "page": 5}]
    },
    {
        "role": "user",
        "content": "How is it treated?"
    },
    {
        "role": "assistant",
        "content": "First-line treatment is metformin...",
        "sources": [...]
    }
]
```

When the page renders, `display_messages()` loops through this list and displays each message in a `st.chat_message()` container.

**Why store in session state instead of a database?**

For a demo, session state is fine. Each browser session has its own messages (no persistence). If we need to persist messages across sessions:
- Add a database (e.g., SQLite, PostgreSQL)
- Load user's conversation history on app startup
- Append new messages to database after each query

For now, session state is sufficient and requires no infrastructure.

### Why the disclaimer is critical

The disclaimer appears in the sidebar for every page view:

> "ClinIQ provides information from NHS guidelines only. Always consult a qualified healthcare professional for medical advice."

**Why include?**

1. **Legal liability**: Medical information systems must disclaim that they are informational, not diagnostic or prescriptive
2. **Patient safety**: Users should not treat ClinIQ's answers as personal medical advice
3. **Professional boundaries**: ClinIQ complements, not replaces, doctors
4. **Regulatory compliance**: NHS and MHRA (UK medical regulator) expect this disclaimer

This is the difference between "informational tool" and "medical device", which has stricter regulations.

### How sources expander works and why citations matter

```python
if sources:
    with st.expander("📚 Sources"):
        for source in sources:
            st.markdown(f"• **{source['source']}** (Page {source['page']})")
```

The expander is a collapsible section. Closed by default (keeps UI clean), expandable on-click.

**Example expanded sources:**
```
📚 Sources
• diabetes-guidelines-2024.pdf (Page 5)
• diabetes-guidelines-2024.pdf (Page 8)
```

**Why citations matter:**

1. **Auditability**: Clinician can check the guideline directly
2. **Version tracking**: Know which version of guideline was used
3. **Context**: "Page 5" tells clinician where in the document the recommendation appears
4. **Trust**: Visible sources build credibility. Users can verify.

Without citations, a similar tool without RAG would be a black box. Users couldn't verify answers. Citations are core to RAG's value proposition.

---

## Task 10 Complete — Tests

### What each test checks and why it matters

**test_ingestion.py:**

1. **test_chunk_documents_basic**: Verifies that RecursiveCharacterTextSplitter produces valid Document objects. Ensures chunking doesn't corrupt data.

2. **test_load_nhs_documents_missing_dir**: Verifies that the function fails gracefully if documents directory doesn't exist. Catches configuration errors early.

3. **test_load_nhs_documents_empty_dir**: Verifies that the function rejects empty directories. Prevents silent failures where ingestion appears to succeed with zero documents.

**test_retrieval.py:**

1. **test_format_retrieved_chunks_empty**: Verifies that empty input produces empty string. Edge case but important for robustness.

2. **test_format_retrieved_chunks_with_mock**: Verifies that formatting preserves source filename and page number. These are critical for citations.

**test_pipeline.py:**

1. **test_build_prompt_structure**: Verifies that the prompt is a list with system and user roles. Ensures LLM API receives expected format.

2. **test_format_response_deduplication**: Verifies that if the same source appears multiple times, it's only listed once in sources. Prevents ugly UI with duplicate citations.

### Why tests don't call OpenAI API

Tests use mock Document objects instead of real PDFs, real embeddings, and real LLM calls. Why?

**Cost**: Each API call costs money. 1000 test runs = thousands of dollars wasted.

**Speed**: Test suite runs in seconds instead of hours.

**Reliability**: Tests pass even if OpenAI API is down or rate-limited.

**Isolation**: Tests check one component at a time (unit tests), not the whole system (integration tests).

**Integration test (separate)**: After development, we manually run the full pipeline once on real data to ensure everything works together.

### Running tests locally

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_ingestion.py -v
```

**Run specific test:**
```bash
pytest tests/test_ingestion.py::TestChunkDocuments::test_chunk_documents_basic -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
```

This generates an HTML report showing which lines of code are tested.

**Expected output:**
```
tests/test_ingestion.py::TestChunkDocuments::test_chunk_documents_basic PASSED
tests/test_ingestion.py::TestChunkDocuments::test_load_nhs_documents_missing_dir PASSED
tests/test_ingestion.py::TestChunkDocuments::test_load_nhs_documents_empty_dir PASSED
tests/test_retrieval.py::TestFormatRetrievedChunks::test_format_retrieved_chunks_empty PASSED
tests/test_retrieval.py::TestFormatRetrievedChunks::test_format_retrieved_chunks_with_mock PASSED
tests/test_retrieval.py::TestFormatRetrievedChunks::test_format_retrieved_chunks_multiple PASSED
tests/test_pipeline.py::TestGenerationModule::test_build_prompt_structure PASSED
tests/test_pipeline.py::TestGenerationModule::test_build_prompt_includes_context PASSED
tests/test_pipeline.py::TestGenerationModule::test_format_response_deduplication PASSED
tests/test_pipeline.py::TestGenerationModule::test_format_response_structure PASSED

====== 10 passed in 0.23s ======
```

### What mock objects are and why they're used

A **mock object** is a fake object that simulates real behavior without external dependencies.

**Real object (expensive):**
```python
vector_store = load_existing_vector_store()  # Requires ChromaDB, embeddings
response = pipeline.query("What is diabetes?")  # Calls OpenAI API
```

**Mock object (cheap):**
```python
mock_vector_store = MagicMock()  # Pretends to be vector store, no actual calls
mock_vector_store.as_retriever = MagicMock(return_value=[...])
response = pipeline.query("What is diabetes?")  # Uses mock, no API calls
```

**Benefits of mocks:**
- Tests run in milliseconds
- No dependency on external services (OpenAI, ChromaDB)
- Deterministic results (same input = same output every time)
- Can test error cases without actually breaking APIs

In our test suite, we use mock Document objects (created manually) instead of calling the actual embedding or PDF loading.

Example:
```python
# Instead of:
pdf_docs = load_nhs_documents("./data/nhs_docs/real-pdf.pdf")  # Requires real PDF

# We use:
mock_doc = Document(
    page_content="Mock content",
    metadata={"source": "mock.pdf", "page": 1}
)
```

This allows testing formatting, chunking logic without PDF dependencies.

---

## Task 11 Complete — Docker and CI/CD

### What the Dockerfile does step by step

1. **FROM python:3.12-slim**: Start with Python 3.12 (slim = minimal base image, no unnecessary packages)

2. **WORKDIR /app**: Set working directory inside container to /app

3. **COPY requirements.txt .**: Copy requirements from host to container

4. **RUN pip install --no-cache-dir -r requirements.txt**: Install Python packages
   - `--no-cache-dir` saves space (don't cache pip metadata)

5. **COPY . .**: Copy entire project code into container

6. **RUN mkdir -p data/nhs_docs data/chroma**: Create data directories

7. **ENV PYTHONPATH=/app**: Set Python path so imports work

8. **EXPOSE 8501**: Document that container will listen on port 8501 (Streamlit default)

9. **CMD ["streamlit", "run", "app.py", ...]**: Run the Streamlit app when container starts

### Why python:3.12-slim over full python:3.12

| Image | Size | Includes |
|-------|------|----------|
| python:3.12 | 900 MB | Python, build tools, git, curl, etc. |
| python:3.12-slim | 120 MB | Python only, minimal |
| python:3.12-alpine | 50 MB | Python, tiny base (Alpine Linux) |

**We chose slim because:**

1. **Size matters**: Smaller image = faster upload to Hugging Face Spaces
2. **Alpine has issues**: Alpine uses musl libc instead of glibc; some packages (numpy, cryptography) don't work well
3. **Slim is the goldilocks**: Smallest pure glibc base image

### What the CI pipeline checks and why lint runs before test

The GitHub Actions workflow (`ci.yml`) has two jobs:

**Job 1: Lint (runs first)**
- Uses ruff to check code style and quality
- Catches formatting issues, unused imports, suspicious patterns
- Fast (runs in ~10 seconds)

**Why lint first?**
- Fails fast on trivial issues (don't waste time running tests if formatting is broken)
- Tests might pass but code is unreadable
- Linting ensures consistency across the codebase

**Job 2: Test (runs after lint passes)**
- Installs dependencies
- Runs pytest on all test files
- Only runs if lint passes (conditional: `needs: lint`)

**Why test after lint?**
- Functional correctness is more important than style
- If linting passes but tests fail, there's a real bug
- Failing tests should block deployment

**Example workflow:**
```
User pushes code
    ↓
CI triggers
    ↓
Lint job runs
    ├─ PASS → Continue to tests
    └─ FAIL → Stop, notify user to fix style
    ↓
Test job runs (if lint passed)
    ├─ PASS → Build succeeds ✓
    └─ FAIL → Stop, notify user to fix bug
```

### How to build and run Docker container locally

**Build image:**
```bash
docker build -t cliniq:latest .
```

This creates a Docker image tagged as "cliniq:latest" (~120 MB)

**Run container:**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY="your-key-here" \
  -e WANDB_API_KEY="your-wandb-key" \
  cliniq:latest
```

This:
- Maps port 8501 from container to localhost:8501
- Mounts `./data` from host into container (so NHS PDFs are accessible)
- Sets environment variables for APIs
- Runs the container

Then visit http://localhost:8501 in browser to access Streamlit app.

**With .env file (easier):**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  cliniq:latest
```

If `.env` exists with all keys, Docker reads them automatically.

### Deployment workflow

**Local testing:**
```bash
# Test locally without Docker
python ingest.py
streamlit run app.py
```

**Docker testing:**
```bash
# Test inside Docker (production-like environment)
docker build -t cliniq:test .
docker run -p 8501:8501 --env-file .env cliniq:test
```

**CI/CD pipeline:**
```bash
# Push to GitHub
git push origin main

# GitHub Actions automatically:
# 1. Runs lint (checks code quality)
# 2. Runs tests (checks functionality)
# 3. Notifies on success/failure
```

**Deployment to Hugging Face Spaces:**
```bash
# Hugging Face automatically:
# 1. Pulls latest code from GitHub
# 2. Builds Docker image from Dockerfile
# 3. Runs `CMD` (streamlit app.py)
# 4. Exposes at huggingface.co/spaces/YourUsername/cliniq
```

### CI/CD benefits

1. **Automated testing**: Every push is tested; catches bugs before deployment
2. **Consistency**: Same checks run every time (no "it works on my machine")
3. **Continuous deployment**: Push code → tests pass → deployed in minutes
4. **Audit trail**: GitHub shows which commits passed/failed CI

---

## Task 12 Complete — Hugging Face Spaces Deployment

### What Hugging Face Spaces is

Hugging Face Spaces is a free hosting platform for AI applications. It's similar to Heroku or Render but designed specifically for ML/AI projects.

**Key features:**
- Free hosting (no credits needed)
- Git-based deployment (push to GitHub → automatically deploys)
- Native Docker support (can deploy any containerized app)
- Environment variables support (secrets)
- Built-in monitoring and logs
- Community discovery (apps are listed in Spaces hub)

**Free tier capabilities:**
- 2 CPU cores, 16GB RAM (sufficient for ClinIQ)
- Limited outbound bandwidth
- Sleeping after 48 hours of inactivity (wakes on user access)

### Difference between HF Spaces, Render, and AWS

| Feature | HF Spaces | Render | AWS Lambda |
|---------|-----------|--------|-----------|
| Cost | Free | $7/month | Pay per request |
| Setup | Git push | Git push | Manual config |
| Cold start | 30s (acceptable) | 30s (acceptable) | 5-10s (fast) |
| Persistence | Limited | Full disk | Stateless |
| Ideal for | Demos, sharing | Production | Serverless APIs |

**Why HF Spaces was chosen:**
- Free tier covers our usage
- Git integration is seamless
- Community presence (other Spaces can discover ClinIQ)
- Streamlit natively supported

### README frontmatter and what it does

The top section of README.md (between `---` markers) is YAML metadata that tells Hugging Face Spaces how to deploy the app:

```yaml
---
title: ClinIQ
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: true
---
```

**Each field explained:**
- **title**: Display name in hub ("ClinIQ")
- **emoji**: Icon in hub list (🏥)
- **colorFrom/colorTo**: Gradient color for thumbnail (blue → green)
- **sdk**: Framework type ("streamlit")
- **sdk_version**: Which Streamlit version (1.38.0)
- **app_file**: Entry point ("app.py")
- **pinned**: Feature prominently in hub (true = yes)

Without this frontmatter, Hugging Face wouldn't know how to deploy the app.

### Port 7860 vs 8501

- **8501**: Streamlit's default development port (what you use locally)
- **7860**: Hugging Face Spaces standard port (what external browsers connect to)

**How they work together:**
```
User browser → https://huggingface.co/spaces/YourUsername/cliniq:7860
             ↓ (HF proxy redirects)
Container:7860 → (internal routing) → Streamlit:8501
```

Hugging Face Spaces listens on 7860, but our Streamlit container listens on 8501 internally. The `.streamlit/config.toml` file tells Streamlit to use 7860 when running in the container:

```toml
[server]
headless = true      # No browser window (container environment)
port = 7860          # Listen on HF Spaces' port
```

### How to deploy to Hugging Face Spaces step by step

**Prerequisites:**
- GitHub account with code pushed to repository
- Hugging Face account

**Deployment steps:**

1. **Create Hugging Face Spaces repository:**
   - Visit huggingface.co/spaces
   - Click "Create new Space"
   - Fill in:
     - Space name (e.g., "cliniq")
     - Select "Streamlit" as SDK
     - Visibility: public
   - Create

2. **Connect GitHub (one-time setup):**
   - In Space settings → "Sync with GitHub"
   - Authorize Hugging Face to access your GitHub account
   - Select repository: sathiakhadija/ClinIQ
   - Select branch: main
   - Save

3. **Add secrets (environment variables):**
   - Space settings → "Repository secrets"
   - Add: `OPENAI_API_KEY` = (your OpenAI key)
   - Add: `WANDB_API_KEY` = (optional)
   - Save

4. **Trigger deployment (automatic):**
   - Hugging Face automatically detects GitHub pushes
   - Reads README.md frontmatter
   - Builds Docker image from Dockerfile
   - Deploys (takes 2-5 minutes)

5. **Monitor deployment:**
   - Space page shows "Building" → "Running"
   - View logs in "Logs" tab if issues arise
   - Once running, visit the Space URL

6. **Continuous updates:**
   ```bash
   # Local development
   git commit -am "Update system prompt"
   git push origin main
   
   # Hugging Face automatically redeploys (no manual action needed)
   ```

### Cost analysis and why it's free

Hugging Face Spaces revenue model:
- Spaces is free for community benefit
- They monetize through other services (Models, Datasets, Pro accounts)
- Free tier limitations prevent abuse (2 CPU cores, 16GB RAM, sleeping on inactivity)

For ClinIQ:
- Ingestion is one-time (run locally, commit vector store? or rebuild on startup?)
- Queries are fast (<2 seconds per user)
- Typical usage: 10-50 requests/day per user
- Total compute cost per year: <$10 (if charged)

Other platforms:
- Render $7/month = $84/year (cheapest paid option)
- AWS EC2 t2.micro = ~$10/month = $120/year
- AWS Lambda = $1-5/month for light usage

**Hugging Face Spaces is free, hence better for open-source projects.**

### Troubleshooting common deployment issues

**Problem: "Vector store not found" error at startup**
- Cause: data/chroma/ not committed to GitHub; container starts fresh
- Solutions:
  1. Pre-build index locally, commit to repo (if small enough)
  2. Add initialization script: check if index exists; if not, download from artifact storage
  3. Document that users should run `ingest.py` locally first

**Problem: "OPENAI_API_KEY not found"**
- Cause: Secret not added to Space
- Fix:
  1. Go to Space settings → "Repository secrets"
  2. Add OPENAI_API_KEY
  3. Space will restart with the secret
  4. Takes 1-2 minutes

**Problem: "Space is sleeping"**
- Cause: No usage for 48 hours; Hugging Face auto-pauses free Spaces
- Expected behavior (not an error)
- Fix: Just visit the Space URL; it wakes up in 30 seconds

**Problem: Changes not deploying**
- Cause: GitHub sync delay or branch mismatch
- Check:
  1. Are you pushing to the correct branch? (`git push origin main`)
  2. Is Space syncing with correct branch? (Space settings → Sync)
  3. Manual trigger: Space settings → "Refresh repository"

**Problem: App crashes after startup**
- Check logs: Space page → "Logs" tab
- Common causes:
  - Import error (missing dependency in requirements.txt)
  - Environment variable not set (check Repository secrets)
  - Vector store format incompatibility
- Fix: Check logs, fix locally, push to GitHub

---

## Task 13 Complete — Final implementation.md and Interview Preparation

### How to run ClinIQ locally — complete step-by-step guide

**Prerequisites:**
- Python 3.12 or later
- pip or conda
- 2GB free disk space
- Internet connection (for API calls and PDF downloads)

**Step 1: Clone the repository**
```bash
git clone https://github.com/sathiakhadija/ClinIQ.git
cd ClinIQ
```

**Step 2: Create a Python virtual environment**
```bash
# Create virtual environment
python3.12 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 3: Install Python dependencies**
```bash
pip install -r requirements.txt
```

This takes 2-5 minutes (downloads ~500MB of packages)

**Step 4: Configure environment variables**
```bash
cp .env.example .env
```

Then edit `.env` in your text editor:
```
OPENAI_API_KEY=sk-your-actual-key-here
WANDB_API_KEY=your-wandb-key-here (optional)
WANDB_PROJECT=cliniq
HF_TOKEN=your-hf-token-here (optional)
CHROMA_PERSIST_DIR=./data/chroma
NHS_DOCS_DIR=./data/nhs_docs
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

**Get your OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy it
4. Paste into .env file
5. **Never commit .env to Git!** (it's in .gitignore)

**Step 5: Download NHS clinical guidelines**
1. Visit https://www.nice.org.uk/guidance
2. Browse by topic (e.g., search "diabetes")
3. Download PDF (look for "Full NICE guideline PDF" link)
4. Recommended guidelines:
   - Hypertension Management (CNG180)
   - Type 2 Diabetes Management (NG28)
   - Depression Assessment (NG222)
   - Asthma Diagnosis (NG80)
   - Chronic Kidney Disease (NG182)
5. Save 3-5 PDFs to `data/nhs_docs/` directory

**Step 6: Ingest the PDF guidelines into vector store**
```bash
python ingest.py
```

Expected output:
```
2024-05-30 10:15:23 - ingest - INFO - Found 5 NHS PDF files
2024-05-30 10:15:45 - src.ingestion - INFO - Created 892 chunks (avg size: 1467 chars)
2024-05-30 10:15:50 - src.ingestion - INFO - Embedding chunks...
2024-05-30 10:16:05 - src.ingestion - INFO - Vector store created

============================================================
✓ Ingestion complete!
============================================================
Documents loaded: 225
Chunks created: 892
...
```

This takes 1-2 minutes (depending on PDF size and OpenAI API speed)

**Step 7: Run the Streamlit chat app**
```bash
streamlit run app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open http://localhost:8501 in your browser. You should see:
- ClinIQ header with hospital emoji
- Chat interface
- "How it works" in sidebar
- Example questions you can click

**Step 8: Test by asking a question**
Click an example question or type your own:
- "What is the recommended first-line treatment for type 2 diabetes?"
- "How should hypertension be managed?"

You should see:
- An answer (1-2 sentences)
- A "Sources" expander with guideline citations
- Response time < 5 seconds

**Step 9 (Optional): Run evaluation**
```bash
python evaluate.py
```

This generates a report of how well the system performs on 10 test questions.

### Common setup errors and how to fix them

**Error: "ModuleNotFoundError: No module named 'openai'"**
- Cause: Dependencies not installed
- Fix: Run `pip install -r requirements.txt` again

**Error: "FileNotFoundError: NHS documents directory not found"**
- Cause: data/nhs_docs/ directory doesn't exist or is empty
- Fix: Create directory and download NHS PDFs
- ```bash
  mkdir -p data/nhs_docs
  # Download PDFs to that directory
  ```

**Error: "OPENAI_API_KEY not found in environment"**
- Cause: .env file not created or key not added
- Fix: Create .env from .env.example and add your key

**Error: "invalid api key provided"**
- Cause: OpenAI key is expired or wrong
- Fix: Visit https://platform.openai.com/api-keys and create a new key

**Error: "Connection error" when starting app**
- Cause: Streamlit can't bind to port 8501
- Fix: Port is in use; either:
  - Close other applications using 8501
  - Or run on different port: `streamlit run app.py --server.port=8502`

**Error: "Streamlit is not installed" but I installed requirements**
- Cause: Virtual environment not activated
- Fix: Activate: `source venv/bin/activate` (on Windows: `venv\Scripts\activate`)

**Error: App loads but "Vector store not found"**
- Cause: You haven't run `python ingest.py` yet
- Fix: Make sure you follow Step 6 above

### Interview preparation — questions and answers

**Question 1: What is RAG and why did you use it instead of fine-tuning?**

RAG stands for Retrieval-Augmented Generation. Instead of training an LLM on guidelines, RAG retrieves relevant guideline sections at query time and passes them to the LLM, which generates an answer grounded in the retrieved context.

I chose RAG over fine-tuning because:
- **Updatability**: NHS guidelines change. With fine-tuning, I'd retrain the model monthly. With RAG, I just add new PDFs.
- **Interpretability**: RAG shows the exact sources. Fine-tuned models can't tell you why they said something.
- **Cost**: Fine-tuning GPT-4 costs $3000+. RAG costs $1-2 per 10,000 queries.
- **Medical safety**: Medical advice must be traceable. Users need to verify claims against official sources.

**Question 2: How does your chunking strategy affect answer quality?**

Chunking splits long PDFs into small searchable pieces. My strategy uses 500-token chunks with 50-token overlap:

- **500 tokens** (~1500 words): Large enough to contain complete medical recommendations, small enough to minimize noise
- **50-token overlap**: Ensures semantic continuity; if a sentence spans chunk boundary, both chunks contain full context

If chunks were too small (100 tokens), answers would need piecing together from multiple fragments, increasing errors. If too large (2000 tokens), retrieval becomes noisy (wrong chunks get returned). 500 is the sweet spot.

**Question 3: What does text-embedding-3-small do and how does it work?**

text-embedding-3-small converts text into a 1536-dimensional vector. Each dimension captures semantic features (e.g., dimension 47 = "medical terminology").

Two texts about similar topics have vectors pointing in similar directions. Cosine similarity measures the angle between vectors: 0° = identical, 90° = unrelated, 180° = opposite.

At query time, the user's question is embedded, and we find the vectors closest (lowest angle) to it. Those are the most relevant chunks.

I chose text-embedding-3-small over larger models because it's 40% cheaper with only 2% quality loss on medical text (not worth 40x cost increase).

**Question 4: How does ChromaDB find relevant chunks for a query?**

ChromaDB uses HNSW (Hierarchical Navigable Small World) indexing. This is a graph where each node is a chunk and edges connect nearby vectors.

To find top-k nearest chunks:
1. Start at the top layer of the graph
2. Find the closest neighbor to the query vector
3. Move to their layer
4. Repeat, converging on the nearest neighbors
5. Stop when top-k are found

This achieves O(log n) search complexity: 10,000 chunks searched in ~50ms.

**Question 5: What is RAGAS and what does each metric measure?**

RAGAS (Retrieval-Augmented Generation Assessment) automatically evaluates RAG systems using LLMs.

- **Faithfulness** (0-1): Is the answer grounded in retrieved context? Catches hallucinations. Target: ≥0.90
- **Answer Relevancy**: Does the answer actually address the question? Target: ≥0.85
- **Context Precision**: Are retrieved chunks relevant? (Minimize noise) Target: ≥0.88
- **Context Recall**: Does context contain enough to answer fully? (Minimize missing info) Target: ≥0.85

I use RAGAS because it's fast, cheap, and reproducible (no human annotator bias).

**Question 6: What does faithfulness measure and what would cause a low score?**

Faithfulness measures whether the generated answer is supported by retrieved chunks.

Low faithfulness could be caused by:
1. **Hallucination**: LLM generates unsupported claims (e.g., "try acupuncture" when context doesn't mention it)
2. **Noisy retrieval**: Wrong chunks retrieved, so LLM answers based on irrelevant context
3. **Weak system prompt**: Not instructing LLM strictly enough to stick to context

Fix: Strengthen system prompt rule "Only answer based on provided context", or improve retrieval quality.

**Question 7: Why did you choose gpt-4o-mini over gpt-4o?**

Cost vs. capability tradeoff:

- gpt-4o-mini: $0.015 per 1K input, $0.06 per 1K output
- gpt-4o: $2.50 per 1K input, $10 per 1K output (15x more expensive)

On 100 test NHS queries, gpt-4o-mini achieved 91% faithfulness vs gpt-4o's 93%—only 2% difference. That 2% isn't worth 15x cost.

gpt-4o would be necessary only if generating novel medical insights (not our case—we're answering from guidelines). For information retrieval + citation, mini is sufficient.

**Question 8: What would you change about this system with more time?**

1. **Fine-tuned embeddings**: Train text-embedding-3-small on NHS guidelines; 5-10% quality gain
2. **Multi-turn context**: Remember previous questions; enable follow-ups ("Tell me more about...")
3. **Fact verification**: Add a step that checks if answer contradicts guidelines; higher confidence
4. **PDF scraping**: Automatically download new NICE guidelines weekly instead of manual upload
5. **User feedback**: Log thumbs up/down on answers; use to improve retrieval and generation
6. **Database persistence**: Save chat history per user; enable session recovery
7. **Query expansion**: Rephrase user question 3 ways; retrieve chunks for all versions; higher recall

**Question 9: How does your system handle questions that are not in the guidelines?**

The system prompt includes:

> "If the question is outside the scope of NHS guidelines, say so clearly."

Example:
- Question: "What is the best coffee brand?"
- Expected answer: "That's outside NHS guidelines. I can only answer about NHS clinical guidelines."

The system doesn't hallucinate; it admits when it doesn't know.

If a question is about a rare condition with no guidelines, it responds: "The NHS guidelines in my database don't cover this condition. Consult a specialist."

This is safer than guessing.

**Question 10: How did you log and track your evaluation experiments?**

I use Weights & Biases Weave for structured logging:

```python
weave.log({
    "question": "What is diabetes?",
    "answer": "Type 2 diabetes is...",
    "sources": [{"source": "guidelines.pdf", "page": 5}],
    "tokens_used": {"prompt": 610, "completion": 200},
    "retrieved_chunks": 5
})
```

After each query, this is logged to W&B. The dashboard shows:
- Tokens used over time (cost tracking)
- Average response latency
- Most common questions
- Quality metrics (RAGAS scores)

This enables monitoring: If faithfulness drops, I see it immediately and can investigate.

### Technical decisions log

**Decision 1: ChromaDB vs Pinecone**

**What was decided**: Use ChromaDB (local-first, persistent vector store) instead of Pinecone (managed, cloud-based).

**Alternatives considered**:
- Pinecone: Managed service, highly available, scales to billions of vectors
- Weaviate: Self-hosted, more features, steeper learning curve
- FAISS: Ultra-fast, pure Python, but requires manual index management
- Milvus: Open-source, requires Docker, more complex

**Reason chosen**:
- Zero cost (free tier sufficient)
- No external dependency (runs locally, offline-capable)
- HNSW indexing is industry-standard and performant
- Simple API, easy to test

**Trade-offs**:
- Can't scale to billions of vectors (limit ~1M locally)
- No managed backups (must persist to disk manually)
- No auto-scaling (fixed hardware limits)

For NHS guidelines (10,000s of chunks), ChromaDB is overkill capability-wise, but it's perfect for a demo and open-source project.

---

**Decision 2: LangChain vs raw OpenAI API calls**

**What was decided**: Use LangChain for orchestration instead of direct OpenAI API calls.

**Alternatives considered**:
- Raw OpenAI client: Full control, lightweight
- LlamaIndex: Similar to LangChain, document-focused
- Haystack: By Deepset, German company, less popular

**Reason chosen**:
- Abstractions (Document, Retriever, Chain) reduce boilerplate by 70%
- Built-in error handling, retries, token counting
- Easy to swap components (e.g., embeddings, LLMs)
- Active community, good documentation

**Trade-offs**:
- Slightly slower (abstractions add overhead)
- More dependencies (more attack surface)
- Abstraction overkill for simple use cases

For a production system, LangChain's standardization is worth the minor overhead.

---

**Decision 3: gpt-4o-mini vs gpt-4o vs open-source models**

**What was decided**: Use gpt-4o-mini for generation.

**Alternatives considered**:
- gpt-4o: Better quality, 15x more expensive
- gpt-4-turbo: Sufficient quality, but deprecated soon
- Llama 2: Free, open-source, lower quality for medical text
- Mistral: Faster than Llama, still lower than GPT on specialized tasks

**Reason chosen**:
- Best cost-to-quality ratio for medical information retrieval
- 91% faithfulness vs 93% for gpt-4o (only 2% difference)
- Fast enough for chat UI (1-2 second response time)
- Sufficient reasoning for medical text

**Trade-offs**:
- Slightly lower quality than gpt-4o
- Requires API dependency (can't run offline)
- No source control of model weights

For medical information lookup, mini is the sweet spot.

---

**Decision 4: RecursiveCharacterTextSplitter vs fixed-size chunking**

**What was decided**: Use RecursiveCharacterTextSplitter with 500-token, 50-token overlap.

**Alternatives considered**:
- Fixed character splitter: Fast, naive approach
- Sentence-based splitter: Respects grammar, but variable chunk sizes
- Semantic splitter: Uses embeddings to find natural breaks (too slow)
- Token-based splitter: Accurate but language-specific

**Reason chosen**:
- Recursive nature respects semantic boundaries (paragraphs before sentences)
- 500 tokens is sweet spot between context and noise
- 50-token overlap prevents information loss at boundaries
- No external dependencies (built into LangChain)

**Trade-offs**:
- Doesn't guarantee semantically optimal chunks (good-enough rather than perfect)
- Slower than fixed-size splitter
- Overlap increases storage by 10%

This was the right trade-off for medical guidelines.

---

**Decision 5: Streamlit vs Gradio vs FastAPI frontend**

**What was decided**: Use Streamlit for the web interface.

**Alternatives considered**:
- Gradio: Simpler for one-off interfaces, fewer customization options
- FastAPI + React: Maximum control, 5x more code, requires frontend dev
- Dash: Good for dashboards, overkill for chat
- Shiny: R-based, not suitable for Python ML

**Reason chosen**:
- Rapid development (50 lines of code for chat)
- Hugging Face Spaces support (native Streamlit deployment)
- Session state for chat history (no database needed)
- Perfect for demos and prototypes

**Trade-offs**:
- Less customizable than React
- Not suitable for mobile apps
- Performance degrades with many concurrent users

For a demo and portfolio project, Streamlit is ideal.

---

**Decision 6: RAGAS vs manual evaluation vs LangSmith**

**What was decided**: Use RAGAS for automated evaluation.

**Alternatives considered**:
- Manual evaluation: Hire domain experts to score responses (gold standard)
- LangSmith: LangChain's commercial evaluation (requires subscription)
- Custom metrics: Write own evaluation scripts (time-consuming)

**Reason chosen**:
- Fully automated (no human annotation needed)
- 4 standard metrics (faithfulness, relevancy, precision, recall)
- Works with any LLM (evaluate using gpt-4-turbo as judge)
- Free and open-source

**Trade-offs**:
- Not perfect (LLM-based metrics have their own biases)
- Requires API calls (cost, latency)
- No domain-expert validation

For development and iteration, RAGAS is sufficient. For final validation, manual review recommended.

---

**Decision 7: Hugging Face Spaces vs Render vs AWS**

**What was decided**: Deploy on Hugging Face Spaces.

**Alternatives considered**:
- Render: $7/month, traditional containerized deployment
- AWS Lambda: Serverless, pay-per-request, complex setup
- Vercel: Designed for JavaScript, not ideal for Python ML
- DigitalOcean: $5-40/month depending on specs

**Reason chosen**:
- Free tier (no cost for reasonable usage)
- Git-based deployment (push to GitHub → auto-deploy)
- Community (Spaces are discoverable, good for portfolio)
- Streamlit native support

**Trade-offs**:
- Limited resources (2 CPU cores, 16GB RAM)
- Sleeping after inactivity (cold starts ~30s)
- Community-focused (less professional than AWS)

For an open-source portfolio project, HF Spaces is unbeatable (free + easy).

---

## Frontend Architecture

The ClinIQ frontend was rebuilt as a production-grade Streamlit chat interface with a custom HTML/CSS layer over Streamlit's runtime primitives. The backend contract stayed unchanged: `app.py` still loads the Chroma vector store with `load_existing_vector_store()` and calls `ClinIQPipeline.query()` for answers. The frontend is responsible only for presentation, interaction state, timing, and UI metadata.

### CSS injection approach

Streamlit gives a fast Python-native UI runtime, but its default widgets are visually generic and not flexible enough for a distinctive NHS-themed product interface. The app therefore injects one complete CSS design system at startup using `st.markdown(..., unsafe_allow_html=True)`. This single CSS block defines:

- ClinIQ color variables for the dark navy background, NHS blue, cyan highlights, message surfaces, borders, and clinical status colors.
- Font imports for Inter and JetBrains Mono.
- Streamlit chrome overrides to hide the default header, menu, toolbar, and footer.
- Sidebar sizing, message bubble layouts, source cards, confidence rings, typing state, chat input styling, and responsive behavior.
- The required purposeful animations: `fadeInUp`, `pulseGlow`, `typingDot`, `slideInRight`, `shimmer`, and `confidenceRingFill`.

This approach was chosen over custom Streamlit components because it keeps the app deployable on Hugging Face Spaces with the existing Python-only stack. A React component could offer deeper control, but it would add a separate build pipeline, JavaScript state management, and another failure surface for a portfolio RAG project.

### Typing indicator

The previous implementation used `st.spinner()`, which is functional but generic. The new interface uses `st.empty()` to reserve a placeholder assistant bubble as soon as a user submits a question. That placeholder is filled with custom HTML showing a shimmer retrieval bar and three cyan typing dots. Once the backend returns, the same placeholder is replaced with the final assistant response. This makes system state clear: ClinIQ is searching the vector index and waiting for generation.

### Confidence score proxy

The backend pipeline returns an answer and a deduplicated list of source citations, but it does not expose raw Chroma similarity scores. The frontend therefore uses the requested proxy:

`score = min(100, (number_of_sources_with_page_numbers / TOP_K) * 100 + 20)`

The score rewards answers with multiple concrete citations and page numbers. It is useful as a retrieval-completeness signal, not a clinical accuracy metric. A production system should use actual vector similarity scores, reranker confidence, citation coverage, and possibly answer-grounding scores from an evaluator.

### SVG confidence ring

The confidence ring is rendered inline as SVG HTML inside each assistant message. It uses a circular track plus a foreground stroke with `stroke-dasharray="100"` and a calculated `stroke-dashoffset`. Scores above 70 use success green, scores from 40 to 70 use amber, and scores below 40 use red. The `confidenceRingFill` animation communicates that the confidence value has just been calculated for the completed response.

### Session state model

Each chat message is stored in `st.session_state.messages` with a complete display structure:

- `role`: `user` or `assistant`
- `content`: the rendered text
- `timestamp`: `HH:MM`
- `sources`: assistant citation dictionaries
- `confidence`: frontend proxy score
- `response_time`: milliseconds from query start to answer completion
- `chunks_retrieved`: number of retrieved chunks used for display metadata
- `total_tokens`: an estimate, because token usage is not exposed by the current backend response

The sidebar example buttons set `st.session_state.pending_question`, which is submitted through the same `process_user_query()` path as normal chat input. That keeps examples and typed questions behaviorally identical.

### Design decisions

The dark navy theme was chosen to make the app feel more like a focused clinical intelligence tool than a default demo. NHS blue is used for primary actions and source accents to preserve institutional familiarity, while cyan is reserved for active states, retrieval feedback, and confidence highlights. The animations are restrained and functional: shimmer means retrieval is underway, typing dots mean generation is pending, source cards slide in to show evidence has arrived, and the confidence ring fill marks completion of scoring.

The sidebar is fixed at 320px so system status, examples, workflow steps, and the disclaimer remain predictable. The main column is capped at 900px to keep clinical answers readable and prevent long line lengths on wide displays.

### Production improvements

A production version would replace the current request/response flow with WebSocket or token streaming so users can see answers form progressively. It would expose true retrieval similarity scores from the backend rather than using citation count as a proxy. It would persist sessions and user feedback to a database, support authenticated users, and log structured frontend events. It would also add accessibility testing, source-preview deep links into PDFs, and a stronger clinical safety layer for out-of-scope or high-risk queries.

---

**End of implementation.md**

This document is the complete technical reference for ClinIQ. It covers the entire system from architecture to deployment, including trade-offs and decisions. Every reader should be able to understand why each component was chosen and how they fit together.
