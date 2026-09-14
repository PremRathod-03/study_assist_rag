import os
from groq import Groq
from dotenv import load_dotenv

# Loads GROQ_API_KEY from your .env file into the environment
# so we don't hardcode the key anywhere in this file.
load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def add_context(full_document_text: str, chunk_text: str) -> str:
    """
    Takes a chunk of text plus the full document it came from, and asks
    the LLM to write a short 1-2 sentence blurb situating that chunk
    within the document - so the chunk makes sense retrieved on its own,
    without needing the surrounding text around it.

    Returns the chunk with that blurb stuck in front of it.
    """
    prompt = f"""Here is a document:
<document>
{full_document_text[:6000]}
</document>

Here is one specific chunk from that document:
<chunk>
{chunk_text}
</chunk>

Write a 1-2 sentence context blurb that situates this chunk within the
overall document, to help it be understood correctly when retrieved on
its own later. Answer with ONLY the blurb, nothing else."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )

    context_blurb = response.choices[0].message.content.strip()
    return f"{context_blurb}\n\n{chunk_text}"
