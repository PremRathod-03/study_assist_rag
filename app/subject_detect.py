import difflib
from app.groq_pool import call_with_pool, estimate_tokens

MODEL = "openai/gpt-oss-20b"


def detect_subject(file_text: str, existing_subjects: list[str]) -> str:
    """
    Looks at a snippet of an uploaded file's text and guesses which
    subject it belongs to. Prefers matching an existing subject over
    inventing a new one.

    Uses fuzzy string matching as a safety net: the LLM's raw guess
    might not exactly match an existing subject's string, so we snap
    close guesses to the real existing name rather than treating
    near-matches as brand new subjects.
    """
    existing_list = ", ".join(existing_subjects) if existing_subjects else "(none yet)"

    prompt = f"""Here is a snippet from a course notes file:
<snippet>
{file_text[:2000]}
</snippet>

Existing subjects already in the system: {existing_list}

Which subject does this file belong to? If it clearly matches one of
the existing subjects, respond with that EXACT existing name, character
for character. If it doesn't match any, suggest a new subject name in
lowercase_underscore format (e.g. "big_data_analytics").

Respond with ONLY the subject name, nothing else."""

    response = call_with_pool(
        lambda c: c.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        ),
        estimate_tokens(prompt) + 500,
    )
    raw_guess = response.choices[0].message.content.strip().lower()
    raw_guess = raw_guess.replace(" ", "_").strip('"').strip("'")

    if existing_subjects:
        close_matches = difflib.get_close_matches(raw_guess, existing_subjects, n=1, cutoff=0.6)
        if close_matches:
            return close_matches[0]

    return raw_guess