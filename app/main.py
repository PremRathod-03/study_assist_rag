from fastapi import FastAPI

from app.agent.graph import ask_agent
from app.question_gen import generate_questions
from app.paper_gen import generate_paper

app = FastAPI(title="RAG Study Assistant")


@app.get("/ask")
def ask(subject: str, question: str):
    """
    Answers a student's question, grounded in that subject's notes.
    Example: /ask?subject=reinforcement_learning&question=What is a value function?
    """
    answer = ask_agent(subject, question)
    return {"answer": answer}


@app.get("/generate-questions")
def questions(subject: str, topic: str, marks: int = 10, count: int = 5):
    """
    Generates exam-style practice questions on a topic.
    Example: /generate-questions?subject=reinforcement_learning&topic=bandits&marks=5&count=3
    """
    result = generate_questions(subject, topic, marks, count)
    return {"questions": result}


@app.get("/generate-paper")
def paper(subject: str, exam_type: str):
    """
    Generates a full sample question paper.
    exam_type must be "internals" or "end_sem".
    Example: /generate-paper?subject=reinforcement_learning&exam_type=internals
    """
    result = generate_paper(subject, exam_type)
    return {"paper": result}
