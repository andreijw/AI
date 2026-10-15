"""Benchmark challenges: the interface every challenge implements, and the registry that finds them."""

from .challenge import BenchmarkChallenge
from .registry import (
    CHALLENGES,
    discover_challenges,
    import_failures_text,
    register_challenge,
    unknown_challenge_message,
)

__all__ = [
    "BenchmarkChallenge",
    "CHALLENGES",
    "discover_challenges",
    "import_failures_text",
    "register_challenge",
    "unknown_challenge_message",
]
