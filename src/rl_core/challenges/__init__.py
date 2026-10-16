"""Benchmark challenges: the interface every challenge implements, and the registry that finds them."""

from .challenge import BenchmarkChallenge, ChallengeEnv
from .registry import (
    CHALLENGES,
    discover_challenges,
    import_failures_text,
    register_challenge,
    select_challenge,
    unknown_challenge_message,
)

__all__ = [
    "BenchmarkChallenge",
    "CHALLENGES",
    "ChallengeEnv",
    "discover_challenges",
    "import_failures_text",
    "register_challenge",
    "select_challenge",
    "unknown_challenge_message",
]
