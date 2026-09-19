from app.groq_pool import get_client_with_capacity, record_usage, estimate_tokens
from app.usage_tracker import log_usage


def add_context(full_document_text: str, chunk_text: str) -> str:
    """
    Takes a chunk of text plus the full document it came from, and asks
    the LLM to write a short 1-2 sentence blurb situating that chunk
    within the document - so the chunk makes sense retrieved on its own,
    without needing the surrounding text around it.

    Uses a pool of Groq API keys (see app/groq_pool.py) so multiple
    accounts' rate limits combine into one larger effective budget.

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

    estimated = estimate_tokens(prompt) + 400
    client, key_index = get_client_with_capacity(estimated)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )

    record_usage(key_index, response.usage.total_tokens)
    log_usage("_enrichment", "enrich_chunk", response.usage.prompt_tokens, response.usage.completion_tokens)

    context_blurb = response.choices[0].message.content.strip()
    return f"{context_blurb}\n\n{chunk_text}"