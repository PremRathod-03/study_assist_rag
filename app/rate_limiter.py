import time
from collections import deque

# Groq free tier limit for our model - tokens per minute.
# Kept slightly under the real 8000 limit as a safety margin.
TPM_LIMIT = 7500

# Tracks (timestamp, tokens_used) for calls made in the last 60 seconds
_usage_window = deque()


def _purge_old_entries():
    """Removes usage records older than 60 seconds from the tracking window."""
    cutoff = time.time() - 60
    while _usage_window and _usage_window[0][0] < cutoff:
        _usage_window.popleft()


def wait_for_capacity(estimated_tokens: int):
    """
    Blocks (sleeps) if making a call right now, with the given estimated
    token cost, would push us over the per-minute limit. Called BEFORE
    every Groq request - this is a queue in spirit: calls wait their
    turn instead of firing and hitting a 429.
    """
    while True:
        _purge_old_entries()
        current_usage = sum(tokens for _, tokens in _usage_window)

        if current_usage + estimated_tokens <= TPM_LIMIT:
            return

        # Over capacity - wait until the oldest entry ages out of the window
        oldest_timestamp = _usage_window[0][0]
        wait_time = max(1, 60 - (time.time() - oldest_timestamp))
        print(f"  Approaching rate limit ({current_usage}/{TPM_LIMIT} tokens used) - pausing {wait_time:.0f}s...")
        time.sleep(wait_time)


def record_usage(actual_tokens: int):
    """Called AFTER a call completes, with the real token count from the response."""
    _usage_window.append((time.time(), actual_tokens))


def estimate_tokens(text: str) -> int:
    """
    Rough estimate: ~4 characters per token on average. Used only to
    decide whether to wait before a call - the real count (from the
    API response) is what actually gets recorded afterward.
    """
    return len(text) // 4
