import json
import sys
sys.path.insert(0, ".")

import chromadb
from chromadb.utils import embedding_functions
from app.agent.graph import ask_agent
from app.groq_pool import call_with_pool, estimate_tokens


embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
chroma_client = chromadb.PersistentClient(path="vectorstore")


def baseline_answer(subject: str, question: str) -> str:
    """
    Deliberately simple baseline: plain vector search only, top-3
    chunks straight to the LLM - no hybrid keyword search, no
    reranking, no query decomposition, no retry-on-insufficient-
    retrieval. This is what most basic RAG tutorials do.
    """
    collection = chroma_client.get_collection(subject, embedding_function=embedding_function)
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""Answer using ONLY this context. If it's not covered, say so.

Context:
{context}

Question: {question}"""

    response = call_with_pool(
        lambda client: client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        ),
        estimate_tokens(prompt) + 500,
    )
    return response.choices[0].message.content.strip()


def score_answer(answer: str, expected_keywords: list[str]) -> tuple[int, int]:
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits, len(expected_keywords)


def run_eval(limit: int = 15, subject_filter: str = None):
    with open("tests/eval_questions.json") as f:
        all_questions = json.load(f)

    questions = all_questions
    if subject_filter:
        questions = [q for q in questions if q["subject"] == subject_filter]
    questions = questions[:limit]

    print(f"Running eval on {len(questions)} of {len(all_questions)} total questions.\n")

    baseline_hits, baseline_possible = 0, 0
    agent_hits, agent_possible = 0, 0
    results = []

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['subject']}: {q['question']}")

        b_answer = baseline_answer(q["subject"], q["question"])
        b_h, b_t = score_answer(b_answer, q["expected_keywords"])
        baseline_hits += b_h
        baseline_possible += b_t

        a_answer = ask_agent(q["subject"], q["question"])
        a_h, a_t = score_answer(a_answer, q["expected_keywords"])
        agent_hits += a_h
        agent_possible += a_t

        print(f"  Baseline: {b_h}/{b_t}  |  Agent: {a_h}/{a_t}")

        results.append({
            "question": q["question"], "subject": q["subject"],
            "baseline_score": f"{b_h}/{b_t}", "agent_score": f"{a_h}/{a_t}",
            "baseline_answer": b_answer, "agent_answer": a_answer,
        })

    print("\n" + "="*60)
    print("EVAL SUMMARY")
    print("="*60)
    if baseline_possible:
        print(f"Baseline (plain vector search):   {baseline_hits}/{baseline_possible} ({100*baseline_hits/baseline_possible:.0f}%)")
    if agent_possible:
        print(f"Full agent (hybrid+rerank+agent): {agent_hits}/{agent_possible} ({100*agent_hits/agent_possible:.0f}%)")

    with open("tests/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to tests/eval_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--subject", type=str, default=None)
    args = parser.parse_args()
    run_eval(limit=args.limit, subject_filter=args.subject)