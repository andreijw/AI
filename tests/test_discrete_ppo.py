"""Standard PPO for the discrete agent: rollouts across episodes, GAE and shuffled mini-batches.

The agent buffers whole episodes until it holds ``rollout_steps`` transitions, then computes GAE over the
rollout, records the old log-probabilities and values once, and takes ``ppo_epochs`` passes of shuffled
mini-batch steps on a mean-reduced loss. These tests check the pieces one at a time: the batched forward
pass, the mini-batch loss and its gradients, the buffering, and a full update replayed independently.
"""

import copy

import numpy as np
import pytest

from rl_core.agents import PPOAgent
from rl_core.agents.returns import gae
from tests.helpers import ReferenceAdam, assert_gradients_match

OBS_DIM, HIDDEN_DIM, ACTION_DIM = 4, 6, 3
EPS = 1e-5  # perturbation used by tests.helpers.numerical_grad


def _agent(**config) -> PPOAgent:
    settings = {"hidden_dim": HIDDEN_DIM, "seed": 11, "value_coef": 0.5, "entropy_coef": 0.3}
    return PPOAgent(OBS_DIM, ACTION_DIM, {**settings, **config})


def _episode(length: int, seed: int, done: bool | None = True) -> dict[str, np.ndarray]:
    """One episode as the Trainer builds it; ``done=None`` leaves out the dones key."""
    rng = np.random.default_rng(seed)
    batch = {
        "observations": rng.standard_normal((length, OBS_DIM)),
        "actions": rng.integers(0, ACTION_DIM, size=length),
        "rewards": rng.uniform(0.0, 2.0, size=length),
    }
    if done is not None:
        batch["dones"] = np.array([False] * (length - 1) + [done])
    return batch


def _cut_episode(
    rewards, terminated: bool = False, done: bool = True, next_observations: bool = True
) -> dict[str, np.ndarray]:
    """An episode shaped as the Trainer builds it, with chosen rewards and a chosen ending.

    ``done=True, terminated=False`` is a time-limit cut; ``done=False, terminated=False`` is a cut by
    ``max_steps_per_episode``; ``terminated=True`` is a true end.
    """
    length = len(rewards)
    rng = np.random.default_rng(length)
    batch = {
        "observations": rng.standard_normal((length, OBS_DIM)),
        "actions": rng.integers(0, ACTION_DIM, size=length),
        "rewards": np.array(rewards, dtype=np.float64),
        "dones": np.array([False] * (length - 1) + [done]),
        "terminated": np.array([False] * (length - 1) + [terminated]),
    }
    if next_observations:
        batch["next_observations"] = rng.standard_normal((length, OBS_DIM))
    return batch


def _constant_critic(agent: PPOAgent, value: float) -> None:
    """Make V(s) = value for every state."""
    agent._W_v[...] = 0.0
    agent._b_v = value


def _weights(agent) -> dict[str, np.ndarray]:
    return {
        name: np.array(value, dtype=np.float64, copy=True)
        for name, value in agent._parameters().items()
    }


def _assert_same_weights(actual, expected, rtol: float = 1e-12) -> None:
    a, e = _weights(actual), _weights(expected)
    for name in e:
        np.testing.assert_allclose(
            a[name], e[name], rtol=rtol, atol=1e-15, err_msg=f"{name} differs."
        )


# ---------------------------------------------------------------------------
# Batched forward pass
# ---------------------------------------------------------------------------


def test_batched_forward_matches_the_single_observation_forward_row_by_row():
    agent = _agent()
    observations = np.random.default_rng(1).standard_normal((5, OBS_DIM))
    # One row's logits thousands above the others': subtracting a single global max would underflow
    # every other row to 0/0, so the max must be taken per row.
    observations[0] *= 20000.0

    probs, values, hidden = agent._forward_batch(observations)

    assert (
        probs.shape == (5, ACTION_DIM) and values.shape == (5,) and hidden.shape == (5, HIDDEN_DIM)
    )
    # Batched and single matrix products may sum in a different order (BLAS differs by CPU), and
    # logits near 1000 turn last-bit differences into ~1e-10 relative differences after exp. A
    # global max gives 0/0 instead, which no tolerance accepts.
    for row, obs in enumerate(observations):
        single_probs, single_value, single_hidden = agent._forward(obs)
        assert np.all(np.isfinite(probs[row]))
        np.testing.assert_allclose(probs[row], single_probs, rtol=1e-9, atol=1e-300)
        assert values[row] == pytest.approx(single_value, rel=1e-10)
        np.testing.assert_allclose(hidden[row], single_hidden, rtol=1e-10)


# ---------------------------------------------------------------------------
# Mini-batch loss: gradients and a value worked out by hand
# ---------------------------------------------------------------------------


def _parameters(agent) -> tuple[dict[str, np.ndarray], np.ndarray]:
    boxed_value_bias = np.array([agent._b_v], dtype=np.float64)
    params = {
        "W1": agent._W1,
        "b1": agent._b1,
        "W_pi": agent._W_pi,
        "b_pi": agent._b_pi,
        "W_v": agent._W_v,
        "b_v": boxed_value_bias,
    }
    return params, boxed_value_bias


def test_minibatch_gradients_match_finite_differences():
    agent = _agent(seed=24)
    rng = np.random.default_rng(4)
    observations = rng.standard_normal((6, OBS_DIM))
    actions = np.array([0, 1, 2, 0, 1, 2])
    returns = rng.uniform(-1.0, 2.0, size=6)
    advantages = np.array([1.0, -0.8, 0.6, -1.2, 0.9, -0.4])
    probs, _, _ = agent._forward_batch(observations)
    current = np.log(probs[np.arange(6), actions])
    # Ratios of 1.5, 0.6, 1.0, 0.7, 1.4, 1.1: some steps clip, some do not, none sits on a clip edge.
    old_log_probs = current - np.log([1.5, 0.6, 1.0, 0.7, 1.4, 1.1])

    pre_activation = observations @ agent._W1 + agent._b1
    assert np.abs(pre_activation).min() > 100 * EPS * np.abs(observations).max()
    active = pre_activation > 0
    assert active.any() and not active.all()

    components, grads, clipped = agent._minibatch_loss_and_grads(
        observations, actions, returns, advantages, old_log_probs
    )
    assert 0 < clipped < 6, "Some steps must clip and some must not."

    params, boxed = _parameters(agent)

    def loss() -> float:
        agent._b_v = float(boxed[0])
        return agent._minibatch_loss_and_grads(
            observations, actions, returns, advantages, old_log_probs
        )[0]["total"]

    grads = {**grads, "b_v": np.atleast_1d(grads["b_v"])}
    assert_gradients_match(loss, params, grads)


def test_minibatch_loss_matches_hand_computation():
    """Trunk zeroed, so every step sees probabilities (0.8, 0.2) and value 0.5.

    Ratios are 1.5, 1.0 and 0.5 with advantages 1, -1 and 0.5:
      step 0: min(1.5*1, 1.2*1)      = 1.2   (clipped branch)
      step 1: min(1.0*-1, 1.0*-1)    = -1.0
      step 2: min(0.5*0.5, 0.8*0.5)  = 0.25  (unclipped branch is smaller)
      policy loss = -(1.2 - 1.0 + 0.25) / 3 = -0.15
      value loss  = 0.5 * mean((0.5 - [1.75, 1.5, 3.0])^2) = 0.5 * (1.5625 + 1 + 6.25) / 3 = 1.46875
    """
    agent = PPOAgent(2, 2, {"hidden_dim": 3, "seed": 0, "value_coef": 0.5, "entropy_coef": 0.1})
    agent._W1[:] = 0.0
    agent._b1[:] = 0.0
    agent._b_pi = np.log(np.array([0.8, 0.2]))
    agent._b_v = 0.5
    actions = np.array([0, 1, 1])
    log_probs = np.log(np.array([0.8, 0.2, 0.2]))
    old_log_probs = log_probs - np.log([1.5, 1.0, 0.5])
    entropy = -(0.8 * np.log(0.8) + 0.2 * np.log(0.2))

    components, _, clipped = agent._minibatch_loss_and_grads(
        np.ones((3, 2)),
        actions,
        np.array([1.75, 1.5, 3.0]),
        np.array([1.0, -1.0, 0.5]),
        old_log_probs,
    )

    assert components["policy"] == pytest.approx(-0.15, rel=1e-12)
    assert components["value"] == pytest.approx(1.46875, rel=1e-12)
    assert components["entropy"] == pytest.approx(entropy, rel=1e-12)
    assert components["total"] == pytest.approx(-0.15 + 0.5 * 1.46875 - 0.1 * entropy, rel=1e-12)
    assert clipped == 1


def test_minibatch_loss_and_grads_changes_nothing():
    agent = _agent()
    before = _weights(agent)
    episode = _episode(5, seed=2)
    agent._minibatch_loss_and_grads(
        episode["observations"], episode["actions"], np.ones(5), np.ones(5), np.zeros(5)
    )
    for name, value in _weights(agent).items():
        np.testing.assert_array_equal(value, before[name])


# ---------------------------------------------------------------------------
# Buffering whole episodes into a rollout
# ---------------------------------------------------------------------------


def test_episodes_are_buffered_until_the_rollout_is_full():
    agent = _agent(rollout_steps=10)
    before = _weights(agent)

    assert agent.update(_episode(6, seed=1)) == {}
    for name, value in _weights(agent).items():
        np.testing.assert_array_equal(value, before[name], err_msg=f"{name} moved while buffering.")

    metrics = agent.update(_episode(6, seed=2))
    assert metrics["rollout_steps"] == 12, "Trains on everything buffered, overshoot included."
    assert any(not np.array_equal(v, before[n]) for n, v in _weights(agent).items())

    assert agent.update(_episode(3, seed=3)) == {}, "The buffer starts again empty after training."


@pytest.mark.parametrize("last_done", [False, None])
def test_each_buffered_episode_ends_its_gae_trace(last_done):
    """An episode cut short (last done False) or sent without dones, and without a next state to
    bootstrap from, still ends where it ends.

    Otherwise GAE would bootstrap from the next episode's first state.
    """
    marked, unmarked = _agent(rollout_steps=9), _agent(rollout_steps=9)
    marked.update(_episode(4, seed=1, done=True))
    unmarked.update(_episode(4, seed=1, done=last_done))
    marked.update(_episode(5, seed=2, done=True))
    unmarked.update(_episode(5, seed=2, done=last_done))
    _assert_same_weights(unmarked, marked, rtol=0.0)


# ---------------------------------------------------------------------------
# Episode ends: a time-limit cut bootstraps, a true termination does not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ending",
    [
        {"done": True, "terminated": False},  # cut by the environment's time limit
        {"done": False, "terminated": False},  # cut by the Trainer's max_steps_per_episode
    ],
)
def test_a_cut_episode_bootstraps_its_last_reward(ending):
    agent = _agent(gamma=0.9)  # the seeded critic gives every state its own value
    episode = _cut_episode([1.0, 3.0], **ending)
    next_value = agent._forward(episode["next_observations"][-1])[1]
    last_value = agent._forward(episode["observations"][-1])[1]
    assert next_value != pytest.approx(last_value), "the test needs V(s') and V(s_T) to differ"

    _, _, rewards, dones = agent._rollout_episode(episode)

    # The cut adds gamma * V(next state), not the value of the last state, to the last reward.
    np.testing.assert_allclose(rewards, [1.0, 3.0 + 0.9 * next_value], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(dones, [False, True])


@pytest.mark.parametrize(
    "ending",
    [
        {"terminated": True},  # a true end
        {"next_observations": False},  # a cut whose next state is unknown
    ],
)
def test_an_episode_without_a_state_to_bootstrap_from_ends_as_terminal(ending):
    agent = _agent(gamma=0.9)
    _, _, rewards, dones = agent._rollout_episode(_cut_episode([1.0, 3.0], **ending))
    np.testing.assert_array_equal(rewards, [1.0, 3.0])
    np.testing.assert_array_equal(dones, [False, True])


def test_rollout_episode_leaves_the_callers_batch_unchanged():
    agent = _agent(gamma=0.9)
    episode = _cut_episode([1.0, 3.0, 2.0])  # float64 rewards, as the Trainer builds them
    copies = {key: value.copy() for key, value in episode.items()}
    agent._rollout_episode(episode)
    for key, value in copies.items():
        np.testing.assert_array_equal(episode[key], value, err_msg=f"{key} was modified")


def test_the_bootstrap_reaches_the_value_loss():
    """One update on one cut episode, with V(s) = 2 everywhere, reports the hand-computed loss.

    gamma 0.9, lambda 0.8, rewards [1, 3], cut, so the last reward becomes 3 + 0.9 * 2 = 4.8:
      A1 = 4.8 - 2 = 2.8                         (the last step ends the trace)
      A0 = (1 + 0.9 * 2 - 2) + 0.9 * 0.8 * A1 = 0.8 + 0.72 * 2.8 = 2.816
    The returns are A + V, so the value loss before the only step is 0.5 * mean(A^2).
    Without the bootstrap A1 would be 1 and A0 = 1.52.
    """
    agent = _agent(gamma=0.9, gae_lambda=0.8, ppo_epochs=1)
    _constant_critic(agent, 2.0)
    metrics = agent.update(_cut_episode([1.0, 3.0]))
    a1 = 2.8
    a0 = 0.8 + 0.9 * 0.8 * a1
    assert metrics["value_loss"] == pytest.approx(0.5 * (a0**2 + a1**2) / 2, rel=1e-12)


def test_empty_batch_changes_nothing_and_returns_no_metrics():
    agent = _agent(rollout_steps=1)
    before = _weights(agent)
    empty = {
        "observations": np.zeros((0, OBS_DIM)),
        "actions": np.zeros(0, dtype=int),
        "rewards": np.zeros(0),
    }
    assert agent.update(empty) == {}
    for name, value in _weights(agent).items():
        np.testing.assert_array_equal(value, before[name])


def test_loading_a_checkpoint_drops_the_partial_rollout(tmp_path):
    """Buffered episodes came from the old weights; after a load they are stale."""
    agent = _agent(rollout_steps=10)
    path = str(tmp_path / "ppo.npz")
    agent.save(path)
    agent.update(_episode(6, seed=1))
    agent.load(path)
    assert agent.update(_episode(6, seed=2)) == {}, "Only the 6 steps after the load are buffered."


# ---------------------------------------------------------------------------
# A full update, replayed independently
# ---------------------------------------------------------------------------


def _replay(
    reference: PPOAgent, episodes, rng, adam: ReferenceAdam | None, lr: float, mb: int, epochs: int
):
    """The update written out from the plan: GAE, normalise once, shuffled mini-batches."""
    obs = np.concatenate([e["observations"] for e in episodes])
    acts = np.concatenate([e["actions"] for e in episodes])
    rews = np.concatenate([e["rewards"] for e in episodes])
    dones = np.concatenate(
        [np.arange(len(e["rewards"])) == len(e["rewards"]) - 1 for e in episodes]
    )
    n = len(rews)
    probs, values, _ = reference._forward_batch(obs)
    old_log_probs = np.log(probs[np.arange(n), acts])
    raw = gae(rews, values, dones, reference.gamma, reference.gae_lambda)
    returns = raw + values
    advantages = (raw - raw.mean()) / (raw.std() + 1e-8)
    batch_sizes, clipped_total, parts = [], 0, []
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, mb):
            idx = order[start : start + mb]
            batch_sizes.append(len(idx))
            components, grads, clipped = reference._minibatch_loss_and_grads(
                obs[idx], acts[idx], returns[idx], advantages[idx], old_log_probs[idx]
            )
            clipped_total += clipped
            parts.append(components)
            steps = adam.step(grads) if adam is not None else {k: lr * g for k, g in grads.items()}
            for name, step in steps.items():
                attribute = f"_{name}"
                current = getattr(reference, attribute)
                setattr(
                    reference, attribute, float(current - step) if name == "b_v" else current - step
                )
    means = {key: float(np.mean([c[key] for c in parts])) for key in ("policy", "value", "entropy")}
    return batch_sizes, clipped_total / (n * epochs), means


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_update_replays_gae_and_shuffled_minibatches(optimizer):
    """Two episodes (9 steps) in mini-batches of 4: the last mini-batch has 1 step and its own mean."""
    config = {
        "rollout_steps": 9,
        "minibatch_size": 4,
        "ppo_epochs": 2,
        "gamma": 0.9,
        "gae_lambda": 0.8,
        "learning_rate": 0.05,
        "optimizer": optimizer,
    }
    if optimizer == "adam":
        config["max_grad_norm"] = 0.5
    agent, reference = _agent(**config), _agent(**config)
    rng = copy.deepcopy(agent._shuffle_rng)
    adam = ReferenceAdam(0.05, 0.5) if optimizer == "adam" else None
    episodes = [_episode(4, seed=1), _episode(5, seed=2)]

    agent.update(episodes[0])
    metrics = agent.update(episodes[1])
    sizes, clip_fraction, means = _replay(reference, episodes, rng, adam, lr=0.05, mb=4, epochs=2)

    assert sizes == [4, 4, 1, 4, 4, 1]
    _assert_same_weights(agent, reference)
    assert metrics["rollout_steps"] == 9
    assert metrics["clip_fraction"] == pytest.approx(clip_fraction, rel=1e-12)
    assert 0.0 < clip_fraction < 1.0, "The second epoch must clip some steps for this to be tested."
    assert metrics["policy_loss"] == pytest.approx(means["policy"], rel=1e-12)
    assert metrics["value_loss"] == pytest.approx(means["value"], rel=1e-12)
    assert metrics["entropy_bonus"] == pytest.approx(means["entropy"], rel=1e-12)


def test_minibatch_larger_than_the_rollout_takes_one_step_per_epoch():
    config = {"rollout_steps": 5, "minibatch_size": 64, "ppo_epochs": 3, "learning_rate": 0.05}
    agent, reference = _agent(**config), _agent(**config)
    rng = copy.deepcopy(agent._shuffle_rng)
    episode = _episode(5, seed=3)

    agent.update(episode)
    sizes, _, _ = _replay(reference, [episode], rng, None, lr=0.05, mb=64, epochs=3)

    assert sizes == [5, 5, 5]
    _assert_same_weights(agent, reference)


def test_same_seed_gives_the_same_weights():
    a, b = _agent(rollout_steps=8, minibatch_size=3), _agent(rollout_steps=8, minibatch_size=3)
    for agent in (a, b):
        agent.update(_episode(4, seed=1))
        agent.update(_episode(5, seed=2))
    _assert_same_weights(a, b, rtol=0.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_defaults_train_on_every_episode_with_monte_carlo_advantages_on_the_whole_episode():
    """Without the new keys, PPO keeps its earlier update: each episode alone, full batch, lambda 1.

    With lambda 1 the GAE of a finished episode is its return minus V(s), the earlier advantage.
    """
    agent = _agent(learning_rate=0.05, ppo_epochs=2)
    reference = _agent(learning_rate=0.05, ppo_epochs=2)
    rng = copy.deepcopy(agent._shuffle_rng)
    assert (agent.rollout_steps, agent.minibatch_size, agent.gae_lambda) == (1, None, 1.0)
    episode = _episode(70, seed=1)  # longer than any fixed mini-batch default would be

    agent.update(episode)
    sizes, _, _ = _replay(reference, [episode], rng, None, lr=0.05, mb=70, epochs=2)

    assert sizes == [70, 70]
    _assert_same_weights(agent, reference)


def test_a_one_step_rollout_has_zero_advantage_after_centring():
    """A single advantage cannot be scaled to unit spread; it is centred to zero instead."""
    agent = _agent(entropy_coef=0.0, value_coef=0.0, learning_rate=0.05)
    before = _weights(agent)
    agent.update(_episode(1, seed=3))
    for name, value in _weights(agent).items():
        np.testing.assert_array_equal(value, before[name], err_msg=f"{name} moved.")


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("rollout_steps", 0, "rollout_steps must be a positive integer"),
        ("rollout_steps", 2.5, "rollout_steps must be a positive integer"),  # was truncated to 2
        ("minibatch_size", 0, "minibatch_size must be a positive integer"),
        ("gae_lambda", -0.1, "gae_lambda must be in"),
        ("gae_lambda", 1.1, "gae_lambda must be in"),
    ],
)
def test_invalid_settings_are_rejected(key, value, message):
    with pytest.raises(ValueError, match=message):
        _agent(**{key: value})


# ---------------------------------------------------------------------------
# Linear annealing of the learning rate and the clip range
# ---------------------------------------------------------------------------


def test_annealing_scales_learning_rate_and_clip_by_the_remaining_fraction():
    """Rollout k trains at lr·f and clip·f with f = 1 − (transitions trained before it) / anneal_steps."""
    config = {
        "rollout_steps": 5,
        "ppo_epochs": 2,
        "learning_rate": 0.05,
        "anneal_steps": 20,
        "clip_epsilon": 0.2,
        "gae_lambda": 0.9,
    }
    agent, reference = _agent(**config), _agent(**config)
    rng = copy.deepcopy(agent._shuffle_rng)
    episodes = [_episode(5, seed=1), _episode(5, seed=2)]

    for episode in episodes:
        agent.update(episode)
    for fraction, episode in zip([1.0, 0.75], episodes, strict=True):
        reference.clip_epsilon = 0.2 * fraction
        _replay(reference, [episode], rng, None, lr=0.05 * fraction, mb=5, epochs=2)

    _assert_same_weights(agent, reference)
    assert agent._steps_trained == 10


def test_annealing_stops_learning_once_the_schedule_is_spent():
    """Past the schedule the fraction stays at zero; it must not turn negative and ascend."""
    agent = _agent(rollout_steps=5, anneal_steps=4, learning_rate=0.05)
    agent.update(_episode(5, seed=1))
    before = _weights(agent)
    agent.update(_episode(5, seed=2))
    for name, value in _weights(agent).items():
        np.testing.assert_array_equal(value, before[name], err_msg=f"{name} moved at fraction 0.")


def test_without_anneal_steps_the_learning_rate_stays_constant():
    config = {"rollout_steps": 5, "ppo_epochs": 1, "learning_rate": 0.05}
    agent, reference = _agent(**config), _agent(**config)
    rng = copy.deepcopy(agent._shuffle_rng)
    episodes = [_episode(5, seed=1), _episode(5, seed=2)]
    for episode in episodes:
        agent.update(episode)
        _replay(reference, [episode], rng, None, lr=0.05, mb=5, epochs=1)
    _assert_same_weights(agent, reference)


def test_checkpoint_keeps_the_annealing_progress(tmp_path):
    agent = _agent(rollout_steps=5, anneal_steps=20)
    agent.update(_episode(5, seed=1))
    path = str(tmp_path / "ppo.npz")
    agent.save(path)
    resumed = _agent(rollout_steps=5, anneal_steps=20, seed=99)
    resumed.load(path)
    assert resumed._steps_trained == 5


def test_anneal_steps_must_be_positive():
    with pytest.raises(ValueError, match="anneal_steps must be a positive integer"):
        _agent(anneal_steps=0)


def test_each_minibatch_gets_the_annealed_clip_range(monkeypatch):
    agent = _agent(
        rollout_steps=5, anneal_steps=20, clip_epsilon=0.2, minibatch_size=5, ppo_epochs=1
    )
    seen = []
    original = agent._minibatch_loss_and_grads

    def spy(*args, **kwargs):
        seen.append(kwargs.get("clip_epsilon"))
        return original(*args, **kwargs)

    monkeypatch.setattr(agent, "_minibatch_loss_and_grads", spy)
    agent.update(_episode(5, seed=1))
    agent.update(_episode(5, seed=2))
    assert seen == pytest.approx([0.2, 0.15])


def test_a_checkpoint_without_the_annealing_counter_restarts_the_schedule(tmp_path):
    """Checkpoints written before annealing existed load with the schedule at its start."""
    agent = _agent(rollout_steps=5, anneal_steps=20)
    agent.update(_episode(5, seed=1))
    path = tmp_path / "old.npz"
    agent.save(str(path))
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key != "ppo_steps_trained"}
    np.savez(path, **arrays)

    resumed = _agent(rollout_steps=5, anneal_steps=20)
    resumed._steps_trained = 7
    resumed.load(str(path))
    assert resumed._steps_trained == 0


def test_metrics_report_the_annealing_fraction():
    agent = _agent(rollout_steps=5, anneal_steps=20)
    assert agent.update(_episode(5, seed=1))["anneal_fraction"] == 1.0
    assert agent.update(_episode(5, seed=2))["anneal_fraction"] == pytest.approx(0.75)
    assert _agent(rollout_steps=5).update(_episode(5, seed=1))["anneal_fraction"] == 1.0
