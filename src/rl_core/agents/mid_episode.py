"""Updates from part of an episode, for agents that learn from a replay buffer.

The trainer hands an agent its transitions after each episode, or every ``update_every`` steps
inside it (``training.update_every``). An agent that turns ``n`` transitions into
``max(1, int(n * updates_per_step))`` gradient steps per call would, called once per step, take
one step per transition whatever ``updates_per_step`` says. :class:`MidEpisodeUpdates` counts the
transitions trained on instead, so the steps owed are the same however the episode is cut up.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

# Checkpoint key for the transitions counted, written only by an agent updating mid-episode, so
# every other checkpoint keeps its keys.
TRANSITIONS_KEY = "update_transitions"


def updates_due(counted: int, transitions: int, per_transition: float) -> int:
    """Gradient steps owed for ``transitions`` more after ``counted``, at ``per_transition`` each:
    ``floor((counted + transitions) * per_transition) - floor(counted * per_transition)``.

    The floors are taken in floating point, so at a rate such as 0.29 a step can come one transition
    late (100 * 0.29 = 28.999...); the total over all calls is still the floor of its own product
    and does not drift."""
    return int((counted + transitions) * per_transition) - int(counted * per_transition)


class MidEpisodeUpdates:
    """Mixin for an agent whose ``update`` may get part of an episode.

    Before :meth:`enable_mid_episode_updates` it rounds each call as it always has; after it,
    gradient steps follow the transitions counted since, carried across calls and checkpoints.
    """

    _mid_episode: bool = False
    _update_transitions: int = 0

    def enable_mid_episode_updates(self, update_every: int) -> None:
        """Count gradient steps over every transition trained on from now on (the trainer calls
        this when ``training.update_every`` is set; the count does not depend on its value)."""
        self._mid_episode = True

    def _updates_for(self, transitions: int, per_transition: float) -> int:
        """Gradient steps for a call that stored ``transitions`` and is past warm-up: at least one
        per call as before, or, updating mid-episode, those the running count has earned (maybe
        none)."""
        if not self._mid_episode:
            return max(1, int(transitions * per_transition))
        due = updates_due(self._update_transitions, transitions, per_transition)
        self._update_transitions += transitions
        return due

    def _mid_episode_checkpoint_arrays(self) -> dict[str, Any]:
        if not self._mid_episode:
            return {}
        return {TRANSITIONS_KEY: np.array(self._update_transitions)}

    def _load_mid_episode_count(self, data: Mapping[str, Any]) -> None:
        if TRANSITIONS_KEY in data:
            self._update_transitions = int(data[TRANSITIONS_KEY])
