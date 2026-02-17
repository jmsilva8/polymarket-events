"""Thread-safe token-bucket rate limiter for API calls."""

import time
import threading
from typing import Optional


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Args:
        rate:  Maximum sustained requests per second.
        burst: Maximum burst size (defaults to int(rate) or 1).
    """

    def __init__(self, rate: float, burst: Optional[int] = None):
        self.rate = rate
        self.burst = burst if burst is not None else max(1, int(rate))
        self.tokens = float(self.burst)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_refill = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return

            time.sleep(0.05)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        pass
