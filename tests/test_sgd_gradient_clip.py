"""``max_grad_norm`` with plain SGD: every agent that takes both optimizers clips the mean gradient.

Each case builds two agents with the same seed, one clipped and one not, and takes exactly one
optimizer step on the same batch. The clipped step must be the unclipped one scaled down to
``max_grad_norm``: the same direction, and the size Adam's clip would give the same gradient.
"""

from typing import Any

import numpy as np
import pytest

from rl_core.agents import ActorCriticAgent, DQNAgent, PPOAgent, ReinforceAgent
from tests.agent_internals import weights

OBS_DIM, ACTION_DIM, N = 4, 2, 6
LR, CLIP = 0.1, 1e-3

# Each agent with what makes one update exactly one optimizer step on the whole batch, and the
# unclipped twin's extra settings (DQN's element-wise clip set too high to bite).
CASES: list[tuple[type, dict[str, Any], dict[str, Any]]] = [
    (ReinforceAgent, {}, {}),
    (ActorCriticAgent, {}, {}),
    (PPOAgent, {"ppo_epochs": 1, "rollout_steps": N}, {}),
    (DQNAgent, {"batch_size": N, "min_buffer_size": N}, {"grad_clip_value": 1e12}),
]
IDS = [agent_cls.__name__ for agent_cls, _, _ in CASES]


def _batch() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {
        "observations": rng.standard_normal((N, OBS_DIM)),
        "actions": rng.integers(0, ACTION_DIM, size=N),
        "rewards": rng.uniform(-1.0, 2.0, size=N),
        "next_observations": rng.standard_normal((N, OBS_DIM)),
        "dones": np.array([False] * (N - 1) + [True]),
    }


def _agent(agent_cls: type, config: dict[str, Any], seed: int = 11) -> Any:
    base = {"seed": seed, "hidden_dim": 8, "learning_rate": LR, "optimizer": "sgd"}
    return agent_cls(OBS_DIM, ACTION_DIM, config={**base, **config})


def _step(agent: Any, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    before = weights(agent)
    agent.update(batch)
    after = weights(agent)
    return {name: after[name] - before[name] for name in before}


@pytest.mark.parametrize(("agent_cls", "config", "unclipped_extra"), CASES, ids=IDS)
def test_sgd_scales_the_mean_gradient_down_to_max_grad_norm(agent_cls, config, unclipped_extra):
    batch = _batch()
    unclipped = _step(_agent(agent_cls, {**config, **unclipped_extra}), batch)
    clipped = _step(_agent(agent_cls, {**config, "max_grad_norm": CLIP}), batch)

    # One plain SGD step moves the weights by -lr times the mean gradient.
    mean_grad_norm = np.sqrt(sum(np.sum(np.square(step)) for step in unclipped.values())) / LR
    assert mean_grad_norm > 10 * CLIP, (
        f"the fixture's gradient ({mean_grad_norm}) must exceed the clip"
    )
    scale = CLIP / (mean_grad_norm + 1e-6)  # Adam's clip coefficient for the same gradient
    for name, step in unclipped.items():
        np.testing.assert_allclose(
            clipped[name], scale * step, rtol=1e-7, atol=1e-14,
            err_msg=f"{name} should move by the unclipped step scaled to max_grad_norm",
        )  # fmt: skip


@pytest.mark.parametrize(("agent_cls", "config", "_"), CASES, ids=IDS)
def test_a_clipped_sgd_agent_resumes_from_its_checkpoint(agent_cls, config, _, tmp_path):
    """Clipped SGD keeps no state of its own: the loaded agent takes the same next step."""
    config = {**config, "max_grad_norm": CLIP}
    batch = _batch()
    saved = _agent(agent_cls, config)
    saved.update(batch)
    path = str(tmp_path / "clipped")
    saved.save(path)
    loaded = _agent(agent_cls, config, seed=99)
    loaded.update(batch)
    loaded.load(path)

    saved_step, loaded_step = _step(saved, batch), _step(loaded, batch)
    for name, step in saved_step.items():
        np.testing.assert_allclose(loaded_step[name], step, rtol=1e-12, atol=1e-15, err_msg=name)
