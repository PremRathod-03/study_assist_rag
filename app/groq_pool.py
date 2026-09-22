import os
import re
import time
from collections import deque
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

TPM_LIMIT = 7500

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
_usage_windows = [deque() for _ in _clients]
_daily_exhausted_until = [None] * len(_clients)


def _purge_old(window):
    cutoff = time.time() - 60
    while window and window[0][0] < cutoff:
        window.popleft()


def _capacity_used(index):
    window = _usage_windows[index]
    _purge_old(window)
    return sum(tokens for _, tokens in window)


def _is_daily_exhausted(index):
    exhausted_until = _daily_exhausted_until[index]
    if exhausted_until is None:
        return False
    if time.time() >= exhausted_until:
        _daily_exhausted_until[index] = None
        return False
    return True


def _mark_daily_exhausted(index, error_message: str):
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", error_message)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        wait_seconds = minutes * 60 + seconds
    else:
        wait_seconds = 3600

    _daily_exhausted_until[index] = time.time() + wait_seconds
    print(f"  Key {index + 1} hit its DAILY limit - marking unusable for ~{wait_seconds:.0f}s")


def get_client_with_capacity(estimated_tokens: int):
    while True:
        for index in range(len(_clients)):
            if _is_daily_exhausted(index):
                continue
            if _capacity_used(index) + estimated_tokens <= TPM_LIMIT:
                return _clients[index], index

        active_indices = [i for i in range(len(_clients)) if not _is_daily_exhausted(i)]
        if not active_indices:
            soonest_recovery = min(t for t in _daily_exhausted_until if t is not None)
            wait_time = max(1, soonest_recovery - time.time())
            print(f"  ALL {len(_clients)} keys are daily-exhausted - waiting {wait_time:.0f}s...")
        else:
            soonest_free = min(
                (_usage_windows[i][0][0] for i in active_indices if _usage_windows[i]),
                default=time.time(),
            )
            wait_time = max(1, 60 - (time.time() - soonest_free))
            print(f"  All active keys at per-minute capacity - pausing {wait_time:.0f}s...")

        time.sleep(wait_time)


def call_with_pool(make_request_fn, estimated_tokens: int):
    while True:
        client, index = get_client_with_capacity(estimated_tokens)
        try:
            response = make_request_fn(client)
            record_usage(index, response.usage.total_tokens)
            return response
        except RateLimitError as e:
            error_message = str(e)
            if "tokens per day" in error_message.lower() or "TPD" in error_message:
                _mark_daily_exhausted(index, error_message)
                continue
            raise


def record_usage(index: int, actual_tokens: int):
    _usage_windows[index].append((time.time(), actual_tokens))


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def pool_size() -> int:
    return len(_clients)