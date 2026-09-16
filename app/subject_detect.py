import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"


def detect_subject(file_text: str, existing_subjects: list[str]) -> str:
    """
    Looks at a snippet of an uploaded file's text and guesses which
    subject it belongs to. Prefers matching an existing subject over
    inventing a new one, unless it genuinely doesn't fit any of them.
    Returns a subject name as a lowercase_underscore string.
    """
    existing_list = ", ".join(existing_subjects) if existing_subjects else "(none yet)"

    prompt = f"""Here is a snippet from a course notes file:
<snippet>
{file_text[:2000]}
</snippet>

Existing subjects already in the system: {existing_list}

Which subject does this file belong to? If it clearly matches one of
the existing subjects, respond with that EXACT existing name. If it
doesn't match any, suggest a new subject name in lowercase_underscore
format (e.g. "big_data_analytics").

Respond with ONLY the subject name, nothing else."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    guess = response.choices[0].message.content.strip().lower()
    guess = guess.replace(" ", "_").strip('"').strip("'")
    return guess
