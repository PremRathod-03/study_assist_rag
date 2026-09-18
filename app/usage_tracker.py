import json
import os
from datetime import datetime, timezone

LOG_PATH = "data/usage_log.jsonl"


def log_usage(subject: str, operation: str, prompt_tokens: int, completion_tokens: int):
    """
    Appends one usage record to a JSON-lines log file - one line per
    LLM call made anywhere in the project. 'operation' identifies what
    kind of call it was (e.g. "ask", "enrich_chunk", "generate_questions",
    "generate_paper", "detect_subject") so the analytics page can break
    down usage by both subject and operation type.
    """
    os.makedirs("data", exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "operation": operation,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
