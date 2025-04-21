import asyncio
from typing import Optional
from datetime import datetime, timedelta

# Cache configuration
CACHE_TTL = timedelta(hours=24)  # 24-hour cache lifetime
MAX_CACHE_SIZE = 10_000  # Maximum cached entries

class AnalysisCache:
    """Production-grade cache with TTL and size limits"""
    def __init__(self):
        self.store = {}
        self.timestamps = {}
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[dict]:
        async with self.lock:
            if key in self.store:
                if datetime.now() - self.timestamps[key] < CACHE_TTL:
                    return self.store[key]
                del self.store[key]
                del self.timestamps[key]
            return None

    async def set(self, key: str, value: dict):
        async with self.lock:
            if len(self.store) >= MAX_CACHE_SIZE:
                oldest = min(self.timestamps, key=self.timestamps.get)
                del self.store[oldest]
                del self.timestamps[oldest]
            self.store[key] = value
            self.timestamps[key] = datetime.now()

analysis_cache = AnalysisCache()