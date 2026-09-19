import os
import time
from collections import deque
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TPM_LIMIT = 7500  # per-key budget, kept under Groq's real 8000 limit

# Load however many GROQ_API_KEY(_N) variables exist in .env - lets you
# add more accounts later just by adding more keys, no code changes.
_keys = []
i = 1
while True:
    key_name = "GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"
    key = os.environ.get(key_name)
    if not key:
        break
    _keys.append(key)
    i += 1

if not _keys:
    raise RuntimeError("No GROQ_API_KEY found in .env")

_clients = [Groq(api_key=k) for k in _keys]
_usage_windows = [deque() for _ in _clients]  # one usage-tracking window per key


def _purge_old(window):
    cutoff = time.time() - 60
    while window and window[0][0] < cutoff:
        window.popleft()


def _capacity_used(index):
    window = _usage_windows[index]
    _purge_old(window)
    return sum(tokens for _, tokens in window)


def get_client_with_capacity(estimated_tokens: int):
    """
    Returns (client, index) for whichever key currently has room for
    this call, checking all keys before waiting. If every key is full,
    sleeps briefly and rechecks - effectively pooling capacity across
    all your accounts instead of just waiting on one.
    """
    while True:
        for index in range(len(_clients)):
            if _capacity_used(index) + estimated_tokens <= TPM_LIMIT:
                return _clients[index], index

        # All keys are currently at capacity - wait for the least-busy
        # one's oldest entry to age out, then recheck
        soonest_free = min(
            (_usage_windows[i][0][0] for i in range(len(_clients)) if _usage_windows[i]),
            default=time.time(),
        )
        wait_time = max(1, 60 - (time.time() - soonest_free))
        print(f"  All {len(_clients)} API key(s) at capacity - pausing {wait_time:.0f}s...")
        time.sleep(wait_time)


def record_usage(index: int, actual_tokens: int):
    _usage_windows[index].append((time.time(), actual_tokens))


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def pool_size() -> int:
    return len(_clients)
