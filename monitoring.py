from prometheus_client import Counter, Histogram, Gauge
import psutil
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
LLM_REQUESTS = Counter('llm_requests_total', 'Total LLM requests')
EMBEDDING_REQUESTS = Counter('embedding_requests_total', 'Total embedding requests')
DB_CONNECTIONS = Gauge('db_connections_current', 'Current number of database connections')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])
ERROR_COUNT = Counter('errors_total', 'Total errors', ['type'])

class MetricsCollector:
    @staticmethod
    def record_request(method: str, endpoint: str, duration: float):
        """Record HTTP request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    @staticmethod
    def record_llm_request():
        """Record LLM request."""
        LLM_REQUESTS.inc()

    @staticmethod
    def record_embedding_request():
        """Record embedding request."""
        EMBEDDING_REQUESTS.inc()

    @staticmethod
    def record_db_connections(count: int):
        """Record current database connections."""
        DB_CONNECTIONS.set(count)

    @staticmethod
    def record_cache_hit(cache_type: str):
        """Record cache hit."""
        CACHE_HITS.labels(cache_type=cache_type).inc()

    @staticmethod
    def record_cache_miss(cache_type: str):
        """Record cache miss."""
        CACHE_MISSES.labels(cache_type=cache_type).inc()

    @staticmethod
    def record_error(error_type: str):
        """Record error."""
        ERROR_COUNT.labels(type=error_type).inc()

    @staticmethod
    def get_system_metrics():
        """Get current system metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }

# Create metrics collector instance
metrics = MetricsCollector()

async def metrics_middleware(request, call_next):
    """Middleware to collect request metrics."""
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            duration=duration
        )
        return response
    except Exception as e:
        metrics.record_error(type(e).__name__)
        raise 