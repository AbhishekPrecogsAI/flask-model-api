# complex_test.py

# This file contains various complex Python constructs to test the CodeChunker.extract_chunks function.

import os
import math
from functools import wraps
from dataclasses import dataclass

# Simple function
def foo(x, y=10):
    '''Adds two numbers with default second parameter.'''
    return x + y

# Function with nested function and lambda
def outer(a):
    def inner(b):
        return b * 2
    sq = lambda z: z ** 2
    return inner(a) + sq(a)

# Async function with await
async def fetch_data(url: str) -> dict:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

# Class with methods, static and class methods
decorators = []

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@dataclass
class Container:
    items: list

    @staticmethod
    def empty():
        return Container([])

    @classmethod
    def singleton(cls, item):
        return cls([item])

    @log_calls
    def add(self, item):
        self.items.append(item)
        return self.items

# Generator function with yield
def countdown(n: int):
    while n > 0:
        yield n
        n -= 1

# Function with complex signature and type hints
def process_data(data: list[dict[str, int]], *, flag: bool = False) -> tuple[int, int]:
    """Processes a list of dicts, returns min and max of 'value' keys."""
    values = [d['value'] for d in data if 'value' in d]
    if flag:
        values = [v for v in values if v > 0]
    return min(values), max(values)


for s, e in uncovered:
    gap = code_bytes[s:e].decode('utf8', errors='replace').strip()
    if gap:
        for chunk in self.fallback_semantic_chunks(gap, file_path, language_name):
            chunks.append(chunk)

return chunks

# Main guard
if __name__ == '__main__':
    print(foo(3))
    print(outer(5))
    cnt = Container.singleton(42)
    cnt.add(99)
    print(list(countdown(5)))
    print(process_data([{'value': -1}, {'value': 10}], flag=True))
