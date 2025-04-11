from fastapi import Request, HTTPException
from cachetools import TTLCache
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # window in seconds

# Initialize rate limit cache
rate_limit_cache = TTLCache(maxsize=10000, ttl=RATE_WINDOW)

class RateLimiter:
    def __init__(self, rate_limit: int = RATE_LIMIT, window: int = RATE_WINDOW):
        self.rate_limit = rate_limit
        self.window = window
        self.cache = TTLCache(maxsize=10000, ttl=window)

    async def check_rate_limit(self, request: Request):
        """Check if request should be rate limited."""
        client_ip = request.client.host
        current_time = time.time()
        
        # Get current request count for this IP
        request_count = self.cache.get(client_ip, 0)
        
        if request_count >= self.rate_limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
        
        # Increment request count
        self.cache[client_ip] = request_count + 1
        logger.debug(f"Request count for IP {client_ip}: {request_count + 1}")

# Create rate limiter instance
rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    """Middleware to apply rate limiting."""
    await rate_limiter.check_rate_limit(request)
    response = await call_next(request)
    return response 