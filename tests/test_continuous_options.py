"""Options of the continuous agents added for Pendulum: initial mean scale, reward scale, time-limit
bootstrapping, annealing, the clip key and the log-ratio guard (CHANGELOG.md, 2026-10-15)."""

import copy

import numpy as np
import pytest

from rl_core.agents.continuous_actor_critic import ContinuousActorCriticAgent
from rl_core.agents.continuous_ppo import ContinuousPPOAgent
from rl_core.training.trainer import Trainer
from tests.agent_internals import (
    assert_same_weights,
    gaussian_forward,
    log_prob,
    ppo_minibatch_loss_and_grads,
    spy_on_adam_step,
    spy_on_ppo_minibatches,
    weights,
)

AGENTS = [ContinuousActorCriticAgent, ContinuousPPOAgent]


def _agent(agent_cls, **config):
    settings = {"hidden_dim": 8, "seed": 5, "action_low": -2.0, "action_high": 2.0}
    return agent_cls(observation_dim=3, action_dim=1, config={**settings, **config})


def _batch(n: int = 12, seed: int = 0, last_done: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n, 3))
    return {
        "observations": states,
        "actions": rng.uniform(-2, 2, size=(n, 1)),
        "rewards": rng.uniform(-16, 0, size=n),
        "next_observations": np.roll(states, -1, axis=0),
        "dones": np.r_[np.zeros(n - 1, bool), last_done],
    }


# ---------------------------------------------------------------------------
# Initial scale of the mean head
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_cls", AGENTS)
def test_mean_head_starts_small_by_default_without_changing_other_weights(agent_cls):
    small, full = _agent(agent_cls), _agent(agent_cls, mean_init_scale=1.0)
    small_weights, full_weights = weights(small), weights(full)
    np.testing.assert_allclose(small_weights["W_mu"], 0.01 * full_weights["W_mu"], rtol=1e-12)
    for name in ("W1", "W_v", "b_mu", "log_std"):
        np.testing.assert_array_equal(small_weights[name], full_weights[name], err_msg=name)


def test_small_mean_head_keeps_initial_actions_inside_the_bounds_at_pendulum_scale():
    agent = _agent(ContinuousPPOAgent, hidden_dim=128)
    states = np.array(
        [[np.cos(a), np.sin(a), w] for a in np.linspace(-3, 3, 7) for w in (-8, 0, 8)]
    )
    mu, _, _, _ = gaussian_forward(agent, states)
    assert np.abs(mu).max() < 0.5


@pytest.mark.parametrize("agent_cls", AGENTS)
@pytest.mark.parametrize("key", ["mean_init_scale", "reward_scale"])
def test_scales_must_be_positive(agent_cls, key):
    with pytest.raises(ValueError, match=f"{key} must be positive"):
        _agent(agent_cls, **{key: 0.0})


# ---------------------------------------------------------------------------
# Reward scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_cls", AGENTS)
def test_reward_scale_trains_as_if_the_rewards_were_scaled(agent_cls):
    scaled, reference = _agent(agent_cls, reward_scale=0.1), _agent(agent_cls)
    batch = _batch()
    scaled.update(batch)
    reference.update({**batch, "rewards": batch["rewards"] * 0.1})
    assert_same_weights(scaled, reference, rtol=1e-10)


def test_reward_scale_leaves_the_batch_unchanged():
    batch = _batch()
    before = batch["rewards"].copy()
    _agent(ContinuousPPOAgent, reward_scale=0.1).update(batch)
    np.testing.assert_array_equal(batch["rewards"], before)


# ---------------------------------------------------------------------------
# Time-limit truncation bootstraps
# ---------------------------------------------------------------------------


def test_truncated_final_step_bootstraps_like_an_unfinished_episode():
    """terminated False on a done final step: bootstrap from V(next state), as with dones[-1] False."""
    truncated, unfinished = _agent(ContinuousPPOAgent), _agent(ContinuousPPOAgent)
    batch = _batch(last_done=True)
    terminated = np.zeros(len(batch["rewards"]), bool)
    truncated.update({**batch, "terminated": terminated})
    unfinished.update({**batch, "dones": np.r_[batch["dones"][:-1], False]})
    assert_same_weights(truncated, unfinished, rtol=0.0)


def test_terminated_final_step_does_not_bootstrap():
    with_key, without_key = _agent(ContinuousPPOAgent), _agent(ContinuousPPOAgent)
    batch = _batch(last_done=True)
    with_key.update({**batch, "terminated": batch["dones"].copy()})
    without_key.update(batch)
    assert_same_weights(with_key, without_key, rtol=0.0)


def test_trainer_batch_reports_terminated_apart_from_truncated():
    """A time-limit cut is done but not terminated in the batch the Trainer builds."""
    from rl_pendulum.environments.pendulum_env import PendulumEnv

    seen = []

    class Recorder(ContinuousPPOAgent):
        def update(self, trajectory):
            seen.append(trajectory)
            return {}

    env = PendulumEnv(seed=1, max_episode_steps=5)
    agent = Recorder(3, 1, {"hidden_dim": 8, "seed": 1, "action_low": -2.0, "action_high": 2.0})
    Trainer(agent=agent, env=env, config={"num_episodes": 1, "save_frequency": 99}).train()
    env.close()
    batch = seen[0]
    assert list(batch["dones"]) == [False] * 4 + [True]
    assert list(batch["terminated"]) == [False] * 5


# ---------------------------------------------------------------------------
# Annealing for continuous PPO
# ---------------------------------------------------------------------------


def test_annealing_passes_scaled_learning_rate_and_clip(monkeypatch):
    agent = _agent(
        ContinuousPPOAgent,
        anneal_steps=48,
        learning_rate=0.01,
        clip_epsilon=0.2,
        ppo_epochs=1,
        batch_size=64,
    )
    seen_lr, seen_clip = [], []
    spy_on_adam_step(
        monkeypatch,
        agent,
        lambda adam_step, grads, lr: (seen_lr.append(lr), adam_step(grads, lr))[1],
    )
    spy_on_ppo_minibatches(
        monkeypatch,
        agent,
        lambda loss, *a, **k: (seen_clip.append(k.get("clip_epsilon")), loss(*a, **k))[1],
    )
    first = agent.update(_batch(12))
    second = agent.update(_batch(12, seed=1))
    assert seen_lr == pytest.approx([0.01, 0.0075])
    assert seen_clip == pytest.approx([0.2, 0.15])
    assert (first["anneal_fraction"], second["anneal_fraction"]) == pytest.approx((1.0, 0.75))


def test_annealing_stops_learning_once_spent():
    agent = _agent(ContinuousPPOAgent, anneal_steps=10)
    agent.update(_batch(12))
    before = copy.deepcopy(weights(agent))
    agent.update(_batch(12, seed=1))
    for name, value in weights(agent).items():
        np.testing.assert_array_equal(value, before[name], err_msg=name)


def test_without_anneal_steps_the_fraction_is_one():
    assert _agent(ContinuousPPOAgent).update(_batch())["anneal_fraction"] == 1.0


def test_annealing_progress_survives_a_checkpoint(tmp_path):
    agent = _agent(ContinuousPPOAgent, anneal_steps=100)
    agent.update(_batch(12))
    path = str(tmp_path / "ppo.npz")
    agent.save(path)
    resumed = _agent(ContinuousPPOAgent, anneal_steps=100, seed=9)
    resumed.load(path)
    assert resumed.update(_batch(12, seed=1))["anneal_fraction"] == pytest.approx(0.88), (
        "The resumed agent must continue the schedule after the 12 transitions already trained."
    )


def test_anneal_steps_must_be_positive():
    with pytest.raises(ValueError, match="anneal_steps must be a positive integer"):
        _agent(ContinuousPPOAgent, anneal_steps=0)


# ---------------------------------------------------------------------------
# Clip key and log-ratio guard
# ---------------------------------------------------------------------------


def test_clip_epsilon_is_read_and_clip_ratio_still_works():
    assert _agent(ContinuousPPOAgent, clip_epsilon=0.3).clip_ratio == 0.3
    assert _agent(ContinuousPPOAgent, clip_ratio=0.25).clip_ratio == 0.25
    assert _agent(ContinuousPPOAgent).clip_ratio == 0.2


def test_conflicting_clip_keys_are_rejected():
    with pytest.raises(ValueError, match="clip_epsilon and clip_ratio"):
        _agent(ContinuousPPOAgent, clip_epsilon=0.3, clip_ratio=0.2)


def test_clip_must_be_positive():
    with pytest.raises(ValueError, match="clip_epsilon must be positive"):
        _agent(ContinuousPPOAgent, clip_epsilon=0.0)


def test_an_extreme_log_ratio_does_not_overflow():
    agent = _agent(ContinuousPPOAgent)
    b = _batch(8)
    with np.errstate(over="raise", invalid="raise"):
        components, grads = ppo_minibatch_loss_and_grads(
            agent, b["observations"], b["actions"], np.zeros(8), np.ones(8), np.full(8, -1e4)
        )
    assert np.isfinite(components["total"])
    assert all(np.all(np.isfinite(g)) for g in grads.values())


def test_the_loss_uses_the_clip_range_it_is_given():
    """Ratios of 1.1 with advantage 1: clip 0.2 leaves them (loss -1.1), clip 0.05 caps them (-1.05)."""
    agent = _agent(ContinuousPPOAgent)
    b = _batch(6)
    mu, log_std, _, _ = gaussian_forward(agent, b["observations"])
    old = log_prob(agent, b["actions"], mu, log_std) - np.log(1.1)
    args = (b["observations"], b["actions"], np.zeros(6), np.ones(6), old)
    wide = ppo_minibatch_loss_and_grads(agent, *args, clip_epsilon=0.2)[0]["policy"]
    narrow = ppo_minibatch_loss_and_grads(agent, *args, clip_epsilon=0.05)[0]["policy"]
    assert wide == pytest.approx(-1.1, rel=1e-9)
    assert narrow == pytest.approx(-1.05, rel=1e-9)


def test_a_clamped_log_ratio_gives_no_policy_gradient_even_with_negative_advantages():
    """A step whose log-ratio is outside ±20 is stale: it must not dominate the update."""
    agent = _agent(ContinuousPPOAgent, entropy_coef=0.0, value_coef=0.0)
    b = _batch(8)
    stale = np.full(8, -1e4)
    _, grads = ppo_minibatch_loss_and_grads(
        agent, b["observations"], b["actions"], np.zeros(8), -np.ones(8), stale
    )
    for name in ("W_mu", "b_mu", "log_std"):
        np.testing.assert_array_equal(grads[name], 0.0, err_msg=name)


def test_update_does_not_modify_the_callers_dones():
    batch = {**_batch(), "terminated": np.zeros(12, bool)}
    before = batch["dones"].copy()
    _agent(ContinuousPPOAgent).update(batch)
    np.testing.assert_array_equal(batch["dones"], before)


def test_terminated_is_read_from_a_list_of_transitions_too():
    as_dict, as_list = _agent(ContinuousPPOAgent), _agent(ContinuousPPOAgent)
    b = _batch()
    terminated = np.zeros(12, bool)
    as_dict.update({**b, "terminated": terminated})
    as_list.update(
        [
            {"state": s, "action": a, "reward": r, "done": d, "terminated": t, "next_state": n}
            for s, a, r, d, t, n in zip(
                b["observations"],
                b["actions"],
                b["rewards"],
                b["dones"],
                terminated,
                b["next_observations"],
                strict=True,
            )
        ]
    )
    assert_same_weights(as_dict, as_list, rtol=0.0)


def test_an_empty_terminated_array_is_ignored():
    with_empty, without = _agent(ContinuousPPOAgent), _agent(ContinuousPPOAgent)
    b = _batch()
    with_empty.update({**b, "terminated": np.array([], dtype=bool)})
    without.update(b)
    assert_same_weights(with_empty, without, rtol=0.0)
