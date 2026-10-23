"""DQN's optional linear learning-rate decay (``anneal_steps``), as PPO's: the step size, not the
gradient, is scaled by ``max(0, 1 - steps_trained / anneal_steps)``, where ``steps_trained`` counts
the transitions given to ``update`` before the call (CHANGELOG.md, 2026-10-23)."""

import numpy as np
import pytest

from rl_core.agents import DQNAgent
from tests.agent_internals import assert_same_weights, counters, optimizer, weights

OBS_DIM, ACTION_DIM = 4, 2
ANNEAL = {"anneal_steps": 100, "min_buffer_size": 8, "batch_size": 8}


def _agent(**config) -> DQNAgent:
    return DQNAgent(OBS_DIM, ACTION_DIM, config={"seed": 5, "hidden_dim": 8, **config})


def _batch(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OBS_DIM)),
        "actions": rng.integers(0, ACTION_DIM, size=n),
        "rewards": rng.standard_normal(n),
        "next_observations": rng.standard_normal((n, OBS_DIM)),
        "dones": np.array([False] * (n - 1) + [True]),
        "terminated": np.array([False] * (n - 1) + [True]),
    }


def _step(agent: DQNAgent, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run one update and return how much each weight moved."""
    before = weights(agent)
    agent.update(batch)
    after = weights(agent)
    return {name: after[name] - before[name] for name in before}


def test_the_fraction_falls_linearly_with_the_transitions_trained_before_each_update():
    agent = _agent(**ANNEAL)
    fractions = [agent.update(_batch(40, seed=i))["anneal_fraction"] for i in range(4)]
    # Before each call 0, 40, 80 and 120 transitions were trained: 1 - 0/100, 1 - 40/100, 1 - 80/100,
    # and 1 - 120/100 clamped at zero.
    assert fractions == pytest.approx([1.0, 0.6, 0.2, 0.0]), "the rate decays to zero and stays"


def test_warm_up_transitions_count_towards_the_decay():
    agent = _agent(**ANNEAL)
    first = agent.update(_batch(5))  # 5 stored, below min_buffer_size 8: no gradient step
    second = agent.update(_batch(5, seed=1))
    assert "loss" not in first, "the first call is warm-up"
    assert first["anneal_fraction"] == pytest.approx(1.0), "warm-up reports the fraction too"
    assert second["anneal_fraction"] == pytest.approx(0.95), "the 5 warm-up transitions count"


@pytest.mark.parametrize("max_grad_norm", [None, 0.05])
def test_adam_takes_a_scaled_step_from_unscaled_moments(max_grad_norm):
    adam = {"optimizer": "adam", "learning_rate": 0.01, "max_grad_norm": max_grad_norm}
    plain, annealed = _agent(**adam, min_buffer_size=8, batch_size=8), _agent(**adam, **ANNEAL)
    for agent in (plain, annealed):
        agent.update(_batch(40))  # fraction 1.0: both take the same first step

    plain_moved, annealed_moved = (
        _step(plain, _batch(40, seed=1)),
        _step(annealed, _batch(40, seed=1)),
    )

    for name, moved in plain_moved.items():
        np.testing.assert_allclose(
            annealed_moved[name], 0.6 * moved, rtol=1e-12, atol=1e-15,
            err_msg=f"at fraction 0.6 the step on {name} should be 0.6 times the full one",
        )  # fmt: skip
    plain_adam, annealed_adam = optimizer(plain), optimizer(annealed)
    for name in plain_adam.m:
        np.testing.assert_array_equal(annealed_adam.m[name], plain_adam.m[name], err_msg=name)
        np.testing.assert_array_equal(annealed_adam.v[name], plain_adam.v[name], err_msg=name)
    # Adam's moments see the full gradient: only the step is scaled.


def test_sgd_scales_the_clipped_step():
    sgd = {"learning_rate": 0.1, "grad_clip_value": 1e-3}
    plain, annealed = _agent(**sgd, min_buffer_size=8, batch_size=8), _agent(**sgd, **ANNEAL)
    for agent in (plain, annealed):
        agent.update(_batch(40))

    plain_moved, annealed_moved = (
        _step(plain, _batch(40, seed=1)),
        _step(annealed, _batch(40, seed=1)),
    )

    largest = max(float(np.max(np.abs(moved))) for moved in plain_moved.values())
    assert largest == pytest.approx(0.1 * 1e-3), "the element clip should bind in this check"
    for name, moved in plain_moved.items():
        np.testing.assert_allclose(
            annealed_moved[name], 0.6 * moved, rtol=1e-12, atol=1e-15,
            err_msg=f"the clipped step on {name} should be scaled after the clip",
        )  # fmt: skip


def test_at_fraction_zero_the_weights_stop_moving():
    agent = _agent(optimizer="adam", **ANNEAL)
    for i in range(3):
        agent.update(_batch(40, seed=i))  # 120 transitions trained: the next call's fraction is 0
    moved = _step(agent, _batch(40, seed=9))
    assert all(not np.any(m) for m in moved.values()), "a decayed rate of zero takes no step"


def test_without_anneal_steps_nothing_changes():
    unset, null = _agent(optimizer="adam"), _agent(optimizer="adam", anneal_steps=None)
    for agent in (unset, null):
        for i in range(3):
            metrics = agent.update(_batch(20, seed=i))
        assert "anneal_fraction" not in metrics, "no decay, no extra metric in the log"
    assert_same_weights(null, unset, rtol=0.0)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_anneal_steps_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="anneal_steps"):
        _agent(anneal_steps=bad)


def test_a_checkpoint_resumes_the_decay(tmp_path):
    agent = _agent(**ANNEAL)
    for i in range(2):
        agent.update(_batch(40, seed=i))
    path = str(tmp_path / "dqn.npz")
    agent.save(path)

    loaded = _agent(**ANNEAL)
    loaded.load(path)

    assert counters(loaded)["steps_trained"] == 80, "the decay counter is checkpointed"
    assert loaded.update(_batch(40, seed=2))["anneal_fraction"] == pytest.approx(0.2)


def test_a_checkpoint_from_before_the_decay_loads_with_the_counter_at_zero(tmp_path):
    agent = _agent(**ANNEAL)
    agent.update(_batch(40))
    path = tmp_path / "dqn.npz"
    agent.save(str(path))
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key != "steps_trained"}
    old = tmp_path / "old.npz"
    np.savez(old, **arrays)

    loaded = _agent(**ANNEAL)
    loaded.load(str(old))

    assert counters(loaded)["steps_trained"] == 0, "an older checkpoint starts the decay afresh"


def test_a_resumed_decay_trains_exactly_as_an_uninterrupted_one(tmp_path):
    uninterrupted, stopped = _agent(optimizer="adam", **ANNEAL), _agent(optimizer="adam", **ANNEAL)
    for i in range(2):
        uninterrupted.update(_batch(40, seed=i))
        stopped.update(_batch(40, seed=i))
    path = str(tmp_path / "dqn.npz")
    stopped.save(path)
    resumed = _agent(optimizer="adam", **ANNEAL)
    resumed.load(path)

    uninterrupted.update(_batch(40, seed=2))
    resumed.update(_batch(40, seed=2))

    assert_same_weights(resumed, uninterrupted, rtol=0.0)


def test_train_py_logs_the_decay_fraction(tmp_path, monkeypatch):
    import json

    import yaml

    from tests.helpers import minimal_config, run_main

    path = minimal_config(tmp_path, training={"num_episodes": 10})
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["agent"] = {"type": "dqn", "config": {"seed": 1, "anneal_steps": 1000}}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    run_main(["--config", path], monkeypatch)

    (metrics,) = (tmp_path / "logs" / "cartpole" / "dqn").glob("metrics_*.jsonl")
    records = [json.loads(line) for line in metrics.read_text().splitlines()]
    updates = [r["update"] for r in records if "update" in r]
    assert updates and all("anneal_fraction" in u for u in updates), (
        "every logged update average should carry anneal_fraction when anneal_steps is set"
    )
    assert 0.0 < updates[-1]["anneal_fraction"] < 1.0, "the logged fraction should be decaying"
