import os
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from app.retrieval.hybrid import hybrid_search
from app.retrieval.rerank import rerank

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
chroma_client = chromadb.PersistentClient(path="vectorstore")


def generate_questions(subject: str, topic: str, marks: int = 10, count: int = 5) -> str:
    """
    Generates exam-style practice questions on a given topic, at a
    specified mark-weightage, grounded in the subject's actual notes.
    Reuses the same hybrid search + rerank retrieval used for answering
    questions - just retrieves content about the TOPIC instead of
    retrieving content that answers a specific QUESTION.
    """
    collection = chroma_client.get_collection(
        name=subject,
        embedding_function=embedding_function,
    )

    candidate_ids = hybrid_search(subject, topic, collection, top_k=10)
    candidates = collection.get(ids=candidate_ids)["documents"]
    top_chunks = rerank(topic, candidates, top_k=6)
    context = "\n\n".join(top_chunks)

    prompt = f"""Using ONLY the context below, write {count} exam-style
questions worth {marks} marks each, on the topic "{topic}".
Match the depth and phrasing style typical of university exam papers.
Number each question.

Context:
{context}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
    )
    return response.choices[0].message.content.strip()
