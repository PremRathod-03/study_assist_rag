Project Status — Subject-Wise Agentic RAG Study Assistant
Last updated: end of Session 2

This file exists so any session (with Claude, or just you re-reading it in six months) can pick up instantly without re-explaining the whole project. Update it at the end of every working session, right before you commit.


1. What This Project Is
A resume project: a RAG-based study assistant with an independent knowledge base per subject, built from your own course notes (PDF/PPTX/DOCX). It should be able to:

Answer doubt-style questions, grounded in your actual notes
Generate practice questions at a given mark-weightage, mapped to Course Outcomes (COs)
Generate full sample question papers (internals vs end-sem, matching real VTU-style structure), using PYQs + notes

The full target architecture is agentic RAG: multi-format ingestion → Contextual Retrieval → hybrid search (vector + BM25) → cross-encoder reranking → a LangGraph agent that can decompose compound questions and self-correct if retrieval comes back insufficient.

We are building this incrementally — a simple version working end-to-end first, then layering the advanced pieces on top. Nothing advanced is scaffolded until the simple version under it is tested and working.


2. Key Decisions Made (don't re-litigate these without a reason)
Package manager: uv, not pip/venv. Dependencies live in pyproject.toml; requirements.txt is kept in sync manually as a courtesy/reference (uv export --format requirements-txt --output-file requirements.txt).
LLM API: Switched from Anthropic (no free tier) to Google Gemini 2.5 Flash via google-genai, for free-tier usage. Not yet wired in — .env + client setup is a near-term next step.
Embeddings: Local, free model via sentence-transformers — BAAI/bge-small-en-v1.5 (upgraded from the initially-tried all-MiniLM-L6-v2 for better retrieval quality). No API cost, no rate limits.
Vector store: Chroma, PersistentClient, one collection per subject, saved to vectorstore/.
Subjects are dynamic, not hardcoded (decision made, not yet built): uploading a file should trigger an LLM guess at the subject, show it to the user, and only create/add to a knowledge base after the user confirms or corrects it. config.py's SUBJECTS list is a placeholder for now, will eventually be read live from existing Chroma collections instead.
Frontend vision (not yet built): a chat-style interface with drag-and-drop upload (like Claude.ai's own UI), not a rigid form. Plus a separate analytics page tracking token usage per document and per query.
Environment gotcha to remember: the machine has a pre-existing conda base environment that auto-activates in new terminals. It doesn't break anything (uv manages its own .venv regardless), but don't run conda install/conda activate inside this project — uv only.
Terminal workflow: multi-line inline python -c "..." commands proved unreliable in this environment (silent freezes / lost input). Standard practice now: write a small script file (test_*.py) using cat > filename.py << 'EOF' ... EOF, then run it with uv run python script.py. One command per terminal action, not chained/pasted in bulk.


3. Current Status — What's Built and Tested
Piece
File
Status
Config (subject list, exam blueprints)
app/config.py
✅ Built, tested
Multi-format file loading (PDF/PPTX/DOCX → text)
app/loaders.py
✅ Built, tested on a real PDF (90 pages, clean extraction incl. formulas)
Chunking
app/chunking.py
✅ Built, tested (105 chunks from 90 pages)
Ingestion (chunk → embed → store in Chroma)
app/ingest.py
✅ Built, tested — reinforcement_learning subject fully ingested
Retrieval (query → relevant chunks)
tested inline, not yet its own file
✅ Verified working — a real query returned genuinely relevant chunks, correctly ranked
Contextual chunk enrichment
app/contextual_chunks.py
❌ Not started — next up
Hybrid search (BM25 + vector fusion)
app/retrieval/hybrid.py
❌ Not started
Reranking
app/retrieval/rerank.py
❌ Not started
LangGraph agent (decompose/retrieve/grade/answer)
app/agent/graph.py
❌ Not started
Question generation
app/question_gen.py
❌ Not started
Paper generation
app/paper_gen.py
❌ Not started
FastAPI backend
app/main.py
❌ Not started (only uv's default main.py stub exists)
Streamlit / chat-style frontend
frontend/
❌ Not started
Dynamic subject detection + confirm flow
—
❌ Not started
Analytics page (token usage tracking)
—
❌ Not started
.env + Gemini API wiring
—
❌ Not started
Git / GitHub
—
✅ Repo live at PremRathod-03/study_assist_rag, .gitignore set up, first commit pushed


One real file currently ingested: unit1_rl.pdf (Reinforcement Learning, Unit 1) — everything so far has been tested against this single file. Other subjects/files haven't been added yet.


4. Immediate Next Step
Contextual chunk enrichment (app/contextual_chunks.py) — for each chunk, call the LLM with the full source document + that chunk, get back a 1–2 sentence blurb situating it, and prepend that to the chunk before embedding. This requires:

Setting up .env with a Gemini API key (get one free at Google AI Studio)
Writing the google-genai client call
Deciding whether to re-ingest unit1_rl.pdf with enrichment, or test enrichment standalone first before touching the existing ingested data


5. Longer-Term Roadmap (in rough order)
Contextual chunk enrichment (next)
Hybrid search (BM25 + vector, reciprocal rank fusion)
Cross-encoder reranking
LangGraph agentic layer (query decomposition, retrieval grading/retry loop)
Question generation + paper generation (reusing the retrieval stack above)
FastAPI backend exposing it all
Dynamic subject detection with confirm-before-create
Chat-style frontend with upload
Analytics page (token usage per document/query)
Evaluation (plain retrieval vs. full pipeline, logged pass/fail on a test question set)
README + architecture diagram + polish for resume presentation


6. How to Resume a Session
At the start of a new session, just say something like: "Continuing the RAG study assistant project — here's where we left off" and paste this file, or say "pick up where we left off on the RAG project." This file plus the two companion docs below should be enough context to continue without re-explaining anything:

RAG_Project_Build_Plan_v2.docx — the full phased build plan with code
File_By_File_Reference_Guide.docx — what every file/folder is for, inputs/outputs, who calls what

Keep this file in docs/PROJECT_STATUS.md in the repo, and update it (don't just append — rewrite the relevant sections) at the end of every session before your git commit.

