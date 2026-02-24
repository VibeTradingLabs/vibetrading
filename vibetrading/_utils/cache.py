"""
Cached API call wrapper with rate-limit detection, caching, and retry logic.
"""

import time
import logging

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when API rate limit is hit."""
    pass


class CachedAPICall:
    """
    Robust API call wrapper with rate-limit detection, caching, and retry logic.

    Features:
    - Automatic rate-limit detection (429 errors)
    - Fallback to cached values on rate limits
    - Exponential backoff retry for transient errors
    - Configurable cache TTL
    """

    def __init__(self, cache_ttl: float = 1.0, max_retries: int = 2, initial_backoff: float = 0.5):
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.cache_value = None
        self.cache_time = 0.0
        self.cache_valid = False

    def is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error (429)."""
        error_str = str(error)
        return '429' in error_str or 'Too Many Requests' in error_str or 'rate limit' in error_str.lower()

    def is_transient_error(self, error: Exception) -> bool:
        """Check if error is transient (network issues, timeouts)."""
        error_str = str(error).lower()
        transient_indicators = ['timeout', 'connection', 'network', '502', '503', '504']
        return any(indicator in error_str for indicator in transient_indicators)

    def __call__(self, api_func, *args, use_cache_on_error: bool = True, cache_key: str = None, **kwargs):
        """
        Execute API call with robust error handling.

        Args:
            api_func: The API function to call
            *args: Positional arguments for api_func
            use_cache_on_error: Whether to return cached value on errors
            cache_key: Optional cache key for debugging
            **kwargs: Keyword arguments for api_func

        Returns:
            API response or cached value on rate limit

        Raises:
            Exception: If all retries fail and no cache available
        """
        current_time = time.time()
        cache_age = current_time - self.cache_time

        if self.cache_valid and cache_age < self.cache_ttl:
            return self.cache_value

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = api_func(*args, **kwargs)
                self.cache_value = result
                self.cache_time = current_time
                self.cache_valid = True
                return result

            except Exception as e:
                last_error = e

                if self.is_rate_limit_error(e):
                    logger.warning(f"Rate limit hit for {cache_key}: {e}")
                    if use_cache_on_error and self.cache_valid:
                        return self.cache_value
                    else:
                        raise RateLimitError(f"Rate limit hit and no cache available for {cache_key}") from e

                if self.is_transient_error(e) and attempt < self.max_retries:
                    backoff_time = self.initial_backoff * (2 ** attempt)
                    logger.warning(f"Transient error for {cache_key}, retry {attempt + 1}/{self.max_retries}: {e}")
                    time.sleep(backoff_time)
                    continue

                if use_cache_on_error and self.cache_valid:
                    logger.warning(f"Error for {cache_key}, using cached value: {e}")
                    return self.cache_value
                else:
                    raise

        if use_cache_on_error and self.cache_valid:
            return self.cache_value
        else:
            raise last_error
