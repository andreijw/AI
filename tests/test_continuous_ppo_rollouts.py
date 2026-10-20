"""Continuous PPO trained on multi-episode rollouts (``rollout_steps``), each time-limit cut bootstrapped.

With rollout_steps N > 1 the agent buffers whole episodes until it holds at least N transitions and then
trains once on all of them. A cut episode's last (scaled) reward gains gamma * V(next state), and its last
step ends the advantage estimate so that no episode borrows from the next (SB3's timeout handling).
"""

import numpy as np
import pytest

from rl_core.agents import ContinuousPPOAgent
from tests.agent_internals import (
    assert_weights_unchanged,
    gaussian_forward,
    ppo_rollout_episode,
    rollout_targets,
    set_constant_critic,
    weights,
)

GAMMA, LAMBDA, SCALE = 0.9, 0.95, 0.1


def _agent(**config) -> ContinuousPPOAgent:
    settings = {
        "seed": 3, "hidden_dim": 8, "batch_size": 4, "ppo_epochs": 2, "gamma": GAMMA,
        "gae_lambda": LAMBDA, "reward_scale": SCALE, "action_low": -2.0, "action_high": 2.0,
    }  # fmt: skip
    return ContinuousPPOAgent(3, 1, config={**settings, **config})


def _episode(seed: int, length: int = 7, terminated: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((length, 3)),
        "actions": rng.uniform(-2.0, 2.0, (length, 1)),
        "rewards": -rng.uniform(0.0, 16.0, length),
        "next_observations": rng.standard_normal((length, 3)),
        "dones": np.array([False] * (length - 1) + [True]),
        "terminated": np.array([False] * (length - 1) + [terminated]),
    }


def _as_list(episode: dict[str, np.ndarray]) -> list[dict[str, object]]:
    return [
        {
            "state": episode["observations"][i], "action": episode["actions"][i],
            "reward": episode["rewards"][i], "done": bool(episode["dones"][i]),
            "next_state": episode["next_observations"][i],
            "terminated": bool(episode["terminated"][i]),
        }
        for i in range(len(episode["rewards"]))
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# Buffering
# ---------------------------------------------------------------------------


def test_episodes_are_buffered_until_the_rollout_is_full():
    agent = _agent(rollout_steps=15, anneal_steps=42)
    before = weights(agent)
    assert agent.update(_episode(1)) == {}  # 7 transitions
    assert agent.update(_episode(2)) == {}  # 14
    assert_weights_unchanged(agent, before, "while buffering")
    metrics = agent.update(_episode(3))  # 21 >= 15: train on all three
    assert metrics["rollout_steps"] == 21
    assert {"policy_loss", "value_loss", "entropy", "anneal_fraction"} <= set(metrics)
    assert agent.update(_episode(4)) == {}, "the buffer starts again after training"
    assert agent.update(_episode(5)) == {}
    assert agent.update(_episode(6))["anneal_fraction"] == pytest.approx(0.5), (
        "All 21 transitions of the first rollout must count towards the schedule."
    )


def test_the_default_trains_on_every_episode():
    agent = _agent()
    assert agent.rollout_steps == 1
    assert agent.update(_episode(1))["rollout_steps"] == 7


def test_a_list_of_transitions_is_buffered_like_a_batch():
    as_dict, as_list = _agent(rollout_steps=10), _agent(rollout_steps=10)
    for seed in (1, 2):
        as_dict.update(_episode(seed))
        as_list.update(_as_list(_episode(seed)))
    for name, value in weights(as_dict).items():
        np.testing.assert_array_equal(weights(as_list)[name], value)


@pytest.mark.parametrize("bad", [0, -2, 2.5, "many"])
def test_rollout_steps_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="rollout_steps"):
        _agent(rollout_steps=bad)


@pytest.mark.parametrize(
    "key,value",
    [
        ("ppo_epochs", 2.5),  # was truncated to 2
        ("ppo_epochs", 0),  # was accepted, and then the agent never trained
        ("batch_size", 2.5),
        ("anneal_steps", 2.5),
    ],
)
def test_integer_settings_must_be_positive_integers(key, value):
    with pytest.raises(ValueError, match=f"{key} must be a positive integer"):
        _agent(**{key: value})


def test_loading_a_checkpoint_empties_a_partly_filled_rollout(tmp_path):
    agent = _agent(rollout_steps=15)
    agent.update(_episode(1))
    path = str(tmp_path / "ppo.pt")
    agent.save(path)
    agent.update(_episode(2))  # 14 buffered
    agent.load(path)
    assert agent.update(_episode(3)) == {}, "7 transitions after loading: still buffering"


# ---------------------------------------------------------------------------
# Bootstrapping and episode boundaries
# ---------------------------------------------------------------------------


def test_a_cut_episode_bootstraps_its_last_scaled_reward():
    agent = _agent()
    set_constant_critic(agent, 2.0)
    episode = _episode(1, length=2)
    episode["rewards"] = np.array([-1.0, -3.0])
    _, _, rewards, dones = ppo_rollout_episode(agent, episode)
    # scaled: [-0.1, -0.3]; the cut adds gamma * V(s') = 0.9 * 2.0 to the last one
    np.testing.assert_allclose(rewards, [-0.1, -0.3 + 0.9 * 2.0])
    np.testing.assert_array_equal(dones, [False, True])


def test_a_terminated_episode_does_not_bootstrap():
    agent = _agent()
    set_constant_critic(agent, 2.0)
    episode = _episode(1, length=2, terminated=True)
    episode["rewards"] = np.array([-1.0, -3.0])
    _, _, rewards, dones = ppo_rollout_episode(agent, episode)
    np.testing.assert_allclose(rewards, [-0.1, -0.3])
    np.testing.assert_array_equal(dones, [False, True])


def test_without_next_observations_the_last_step_is_terminal():
    agent = _agent()
    set_constant_critic(agent, 2.0)
    episode = _episode(1, length=2)
    episode["rewards"] = np.array([-1.0, -3.0])
    del episode["next_observations"]
    _, _, rewards, _ = ppo_rollout_episode(agent, episode)
    np.testing.assert_allclose(rewards, [-0.1, -0.3])


def test_advantages_match_a_hand_computation_across_two_episodes():
    agent = _agent()
    set_constant_critic(agent, 2.0)
    cut, ended = _episode(1, length=2), _episode(2, length=1, terminated=True)
    cut["rewards"], ended["rewards"] = np.array([-1.0, -3.0]), np.array([-5.0])
    parts = [ppo_rollout_episode(agent, episode) for episode in (cut, ended)]
    states, _, rewards, dones = (np.concatenate(column) for column in zip(*parts, strict=True))
    advantages, returns = rollout_targets(agent, states, rewards, dones)
    # V = 2 everywhere. Cut episode: d1 = (-0.3 + 1.8) - 2 = -0.5 (no bootstrap past its end);
    # d0 = -0.1 + 0.9 * 2 - 2 = -0.3, A0 = d0 + 0.9 * 0.95 * d1. Terminated episode: d = -0.5 - 2.
    a1 = -0.5
    a0 = -0.3 + GAMMA * LAMBDA * a1
    np.testing.assert_allclose(advantages, [a0, a1, -2.5])
    np.testing.assert_allclose(returns, advantages + 2.0)


def test_one_episodes_advantages_ignore_the_next_episode():
    agent = _agent()
    first = ppo_rollout_episode(agent, _episode(1))
    targets = []
    for second_seed in (2, 3):
        parts = [first, ppo_rollout_episode(agent, _episode(second_seed))]
        states, _, rewards, dones = (np.concatenate(c) for c in zip(*parts, strict=True))
        targets.append(rollout_targets(agent, states, rewards, dones)[0][:7])
    np.testing.assert_array_equal(targets[0], targets[1])


def test_rollout_episode_leaves_the_callers_batch_unchanged():
    agent = _agent()
    episode = _episode(1)
    copies = {key: value.copy() for key, value in episode.items()}
    ppo_rollout_episode(agent, episode)
    for key, value in copies.items():
        np.testing.assert_array_equal(episode[key], value)


@pytest.mark.parametrize("trajectory", [[], {"observations": []}, "not a batch"])
def test_an_empty_or_unrecognised_episode_buffers_nothing(trajectory):
    agent = _agent(rollout_steps=7)
    assert agent.update(trajectory) == {}
    assert agent.update(_episode(1))["rollout_steps"] == 7, "Nothing may have been buffered."


def test_the_default_matches_the_previous_bootstrap_bit_for_bit():
    """rollout_steps 1 folds gamma * V(s') into the last reward; the previous code passed it to GAE as
    last_value with the last step not done. Both must give the very same floats."""
    from rl_core.agents.returns import gae

    agent = _agent()
    episode = _episode(1)
    _, _, values, _ = gaussian_forward(agent, episode["observations"])
    _, _, next_value, _ = gaussian_forward(agent, episode["next_observations"][-1])
    previous = gae(
        episode["rewards"] * SCALE, np.asarray(values, dtype=np.float64),
        np.zeros(7, dtype=bool), GAMMA, LAMBDA, last_value=float(next_value),
    )  # fmt: skip
    states, _, rewards, dones = ppo_rollout_episode(agent, episode)
    advantages, _ = rollout_targets(agent, states, rewards, dones)
    np.testing.assert_array_equal(advantages, previous)
