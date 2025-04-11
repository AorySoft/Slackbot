from cachetools import TTLCache, LRUCache
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configurations
LLM_CACHE_SIZE = 1000
LLM_CACHE_TTL = 3600  # 1 hour
EMBEDDING_CACHE_SIZE = 10000
PROCESSED_MESSAGES_CACHE_SIZE = 10000
PROCESSED_MESSAGES_TTL = 86400  # 24 hours

# Initialize caches
llm_cache = TTLCache(maxsize=LLM_CACHE_SIZE, ttl=LLM_CACHE_TTL)
embedding_cache = LRUCache(maxsize=EMBEDDING_CACHE_SIZE)
processed_messages = TTLCache(maxsize=PROCESSED_MESSAGES_CACHE_SIZE, ttl=PROCESSED_MESSAGES_TTL)

def get_cached_llm_response(text: str) -> str:
    """Get cached LLM response if available."""
    return llm_cache.get(text)

def set_cached_llm_response(text: str, response: str):
    """Cache LLM response."""
    llm_cache[text] = response

def get_cached_embedding(text: str):
    """Get cached embedding if available."""
    return embedding_cache.get(text)

def set_cached_embedding(text: str, embedding):
    """Cache embedding."""
    embedding_cache[text] = embedding

def is_message_processed(message_id: str) -> bool:
    """Check if message has been processed."""
    return message_id in processed_messages

def mark_message_processed(message_id: str):
    """Mark message as processed."""
    processed_messages[message_id] = True

@lru_cache(maxsize=1000)
def get_cached_similar_questions(question: str, k: int = 5):
    """Cache similar questions lookup results."""
    # This is a placeholder - implement actual similar questions lookup
    pass

def clear_caches():
    """Clear all caches."""
    llm_cache.clear()
    embedding_cache.clear()
    processed_messages.clear()
    logger.info("All caches cleared") 