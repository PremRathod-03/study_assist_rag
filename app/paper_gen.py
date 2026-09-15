import os
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from app.config import EXAM_BLUEPRINTS
from app.retrieval.hybrid import hybrid_search
from app.retrieval.rerank import rerank

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
chroma_client = chromadb.PersistentClient(path="vectorstore")


def generate_paper(subject: str, exam_type: str) -> str:
    """
    Assembles a full sample question paper for a subject, matching the
    real exam structure defined in EXAM_BLUEPRINTS (internals vs end_sem).

    Uses PYQs (past-year questions) as a style reference if that subject
    has a separate PYQ collection ingested - if not, it falls back to
    generating purely from notes, without a style example to match.
    """
    blueprint = EXAM_BLUEPRINTS[exam_type]
    notes_collection = chroma_client.get_collection(
        name=subject, embedding_function=embedding_function
    )

    # Pull a broad sample of notes content to base questions on
    query = f"{subject} key concepts and topics"
    candidate_ids = hybrid_search(subject, query, notes_collection, top_k=15)
    notes_sample = notes_collection.get(ids=candidate_ids)["documents"]
    notes_context = "\n\n".join(notes_sample)

    # Try to also pull PYQs for this subject, if that collection exists
    pyq_context = ""
    try:
        pyq_collection = chroma_client.get_collection(
            name=f"{subject}_pyqs", embedding_function=embedding_function
        )
        pyq_sample = pyq_collection.get(limit=10)["documents"]
        pyq_context = "\n\n".join(pyq_sample)
    except Exception:
        # No PYQs ingested yet for this subject - that's fine, we just
        # won't have a real style example to match against.
        pass

    style_instruction = (
        f"Match the structure and phrasing style of these real past papers:\n{pyq_context}"
        if pyq_context
        else "No past papers are available yet - use standard university exam phrasing and structure."
    )

    prompt = f"""Generate a {exam_type} question paper for {subject}.
Total marks: {blueprint['total_marks']}. Cover {blueprint['num_units_covered']} units/topics.

{style_instruction}

Base the question content on this syllabus material:
{notes_context}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()
