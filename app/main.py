import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import json

from app.agent.graph import ask_agent
from app.question_gen import generate_questions
from app.paper_gen import generate_paper
from app.subject_detect import detect_subject
from app.ingest import ingest_single_file_streaming, chroma_client
from app.loaders import load_file

app = FastAPI(title="RAG Study Assistant")


@app.get("/ask")
def ask(subject: str, question: str):
    answer = ask_agent(subject, question)
    return {"answer": answer}


@app.get("/generate-questions")
def questions(subject: str, topic: str, marks: int = 10, count: int = 5):
    result = generate_questions(subject, topic, marks, count)
    return {"questions": result}


@app.get("/generate-paper")
def paper(subject: str, exam_type: str):
    result = generate_paper(subject, exam_type)
    return {"paper": result}


@app.get("/subjects")
def list_subjects():
    collections = chroma_client.list_collections()
    names = [c.name for c in collections if not c.name.endswith("_pyqs")]
    return {"subjects": names}


@app.post("/detect-subject")
def detect(file: UploadFile = File(...)):
    os.makedirs("data/uploads_tmp", exist_ok=True)
    temp_path = f"data/uploads_tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = load_file(temp_path)
    snippet = "\n".join(p["text"] for p in pages[:3] if p.get("text"))

    existing = [c.name for c in chroma_client.list_collections() if not c.name.endswith("_pyqs")]
    guess = detect_subject(snippet, existing)

    return {"temp_path": temp_path, "guessed_subject": guess, "existing_subjects": existing}


@app.post("/confirm-ingest")
def confirm_ingest(temp_path: str = Form(...), subject: str = Form(...)):
    """
    Streams progress updates as newline-delimited JSON, one line per
    chunk enriched, so the frontend can show a live progress bar instead
    of waiting silently for minutes.
    """
    def event_stream():
        for update in ingest_single_file_streaming(subject, temp_path):
            yield json.dumps(update) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
