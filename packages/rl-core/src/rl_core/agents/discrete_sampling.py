"""Acting from a discrete policy's action probabilities, for the agents that have one."""

import numpy as np


class DiscreteSampling:
    """Mixin for an agent whose policy gives a probability per discrete action (REINFORCE and the
    discrete Actor-Critics, ``ActorCriticBase``). Listed before ``BaseAgent``, which sets
    ``action_dim``; the agent creates ``_explore_rng`` (AGENT_EXPLORATION_STREAM)."""

    action_dim: int
    _explore_rng: np.random.Generator

    def _select_action_from_probs(self, probs: np.ndarray, training: bool) -> int:
        """
        Sample or greedily select an action from a probability distribution, sampling with the
        subclass's ``_explore_rng``.

        Args:
            probs:    Action probability distribution.
            training: If True, sample stochastically; otherwise act greedily.

        Returns:
            Selected action index.
        """
        if training:
            return int(self._explore_rng.choice(self.action_dim, p=probs))
        return int(np.argmax(probs))
