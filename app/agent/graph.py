import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Remember: gpt-oss models spend tokens on internal reasoning before
# writing the visible answer, so max_tokens needs to stay generous
# or the response comes back empty (we hit this exact bug earlier).
MODEL = "openai/gpt-oss-20b"


class AgentState(TypedDict):
    """
    The 'clipboard' that travels through every step of the agent.
    Each node below reads from this and writes its result back onto it
    before handing off to the next node.
    """
    subject: str
    original_question: str
    sub_queries: list[str]
    retrieved_chunks: list[str]
    attempts: int
    final_answer: str


def analyze_and_decompose(state: AgentState) -> AgentState:
    """
    Looks at the question and decides if it's really asking more than
    one thing at once. If so, splits it into separate sub-questions,
    each of which will be retrieved independently.
    """
    prompt = f"""Does this question need to be split into multiple
independent sub-questions to be answered well?
Question: {state['original_question']}

If yes, list each sub-question on its own line, nothing else.
If no, just repeat the original question unchanged, as a single line.
Do not add numbering, explanation, or commentary."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()
    state["sub_queries"] = [q.strip() for q in raw.split("\n") if q.strip()]
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """
    For every sub-query, runs hybrid search then reranking, and collects
    all the resulting chunks together.
    """
    from app.retrieval.hybrid import hybrid_search
    from app.retrieval.rerank import rerank
    import chromadb
    from chromadb.utils import embedding_functions

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )
    chroma_client = chromadb.PersistentClient(path="vectorstore")
    collection = chroma_client.get_collection(
        name=state["subject"],
        embedding_function=embedding_function,
    )

    all_chunks = []
    for sub_q in state["sub_queries"]:
        candidate_ids = hybrid_search(state["subject"], sub_q, collection, top_k=10)
        candidates = collection.get(ids=candidate_ids)["documents"]
        top_chunks = rerank(sub_q, candidates, top_k=4)
        all_chunks.extend(top_chunks)

    state["retrieved_chunks"] = all_chunks
    return state


def grade_retrieval(state: AgentState) -> str:
    """
    Asks the LLM to judge whether the retrieved chunks actually contain
    enough to answer the original question. If not (and we haven't
    already retried twice), loop back to re-analyze the question.
    Otherwise, move on to writing the final answer.
    """
    context = "\n\n".join(state["retrieved_chunks"])
    prompt = f"""Context:
{context}

Question: {state['original_question']}

Does the context above contain enough information to answer this
question well? Answer with exactly one word: YES or NO."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    answer = response.choices[0].message.content.strip().upper()
    sufficient = "YES" in answer

    if sufficient or state["attempts"] >= 2:
        return "synthesize"

    state["attempts"] += 1
    return "retry"


def synthesize_answer(state: AgentState) -> AgentState:
    """
    Writes the actual final answer for the student, using only the
    retrieved context - explicitly told not to guess beyond it.
    """
    context = "\n\n".join(state["retrieved_chunks"])
    prompt = f"""Answer the student's question using ONLY the context
below. If the context doesn't fully cover it, say so honestly instead
of guessing.

Context:
{context}

Question: {state['original_question']}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    from app.usage_tracker import log_usage
    log_usage(state["subject"], "ask", response.usage.prompt_tokens, response.usage.completion_tokens)
    state["final_answer"] = response.choices[0].message.content.strip()
    return state


# --- Wire the graph together ---
graph = StateGraph(AgentState)
graph.add_node("analyze", analyze_and_decompose)
graph.add_node("retrieve", retrieve_node)
graph.add_node("synthesize", synthesize_answer)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "retrieve")
graph.add_conditional_edges(
    "retrieve", grade_retrieval, {"synthesize": "synthesize", "retry": "analyze"}
)
graph.add_edge("synthesize", END)

agentic_rag = graph.compile()


def ask_agent(subject: str, question: str) -> str:
    """
    The one function the rest of the project should call. Everything
    above this is internal - main.py will only ever call this function.
    """
    result = agentic_rag.invoke({
        "subject": subject,
        "original_question": question,
        "sub_queries": [],
        "retrieved_chunks": [],
        "attempts": 0,
        "final_answer": "",
    })
    return result["final_answer"]
