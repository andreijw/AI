"""Running observation normalisation (``agent.config.normalize_observations``): the running
statistics, the agent that owns and checkpoints them, and the trainer that applies them once per
observation, updating in training and frozen in evaluation."""

import numpy as np
import pytest

from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import (
    ContinuousPPOAgent,
    DDPGAgent,
    DQNAgent,
    PPOAgent,
    ReinforceAgent,
    TD3Agent,
)
from rl_core.agents.base_agent import OBS_NORM_KEY_PREFIX
from rl_core.agents.observation_normalizer import RunningMeanStd
from rl_core.agents.replay_buffer import ReplayBuffer
from rl_core.training import Trainer
from rl_pendulum.environments.pendulum_env import PendulumEnv
from tests.agent_internals import (
    forward_batch,
    gaussian_forward,
    log_prob,
    spy_on_ppo_minibatches,
)
from tests.helpers import minimal_config, run_main
from tests.test_agent_harness import (
    ALL_AGENTS,
    ARCHITECTURE_KEYS,
    CONTINUOUS_AGENTS,
    agent_id,
)

ON = {"normalize_observations": True}

# Every agent (each may normalise), with a CartPole (discrete) or Pendulum (continuous) size.
NORMALISING_AGENTS = [pytest.param(c, cfg, 4, 2, id=agent_id(c, cfg)) for c, cfg in ALL_AGENTS] + [
    pytest.param(c, cfg, 3, 1, id=c.__name__) for c, cfg in CONTINUOUS_AGENTS
]


def _chan(mean, var, count, xs):
    """The reference: Chan et al.'s combination of a prior (mean, var, count) with a sample."""
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    delta = xs.mean(0) - mean
    total = count + n
    new_mean = mean + delta * n / total
    m2 = var * count + xs.var(0) * n + delta**2 * count * n / total
    return new_mean, m2 / total, total


def _entries(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def _assert_refused_load_changes_nothing(loader, path, match, tmp_path):
    loader.save(str(tmp_path / "before.npz"))
    with pytest.raises(ValueError, match=match):
        loader.load(path)
    loader.save(str(tmp_path / "after.npz"))
    before, after = _entries(tmp_path / "before.npz"), _entries(tmp_path / "after.npz")
    assert sorted(after) == sorted(before)
    for key, value in before.items():
        np.testing.assert_array_equal(after[key], value, err_msg=f"a refused load changed {key}")


def _rewrite(path, change):
    entries = _entries(path)
    change(entries)
    with open(path, "wb") as f:
        np.savez_compressed(f, **entries)


# --- RunningMeanStd ------------------------------------------------------------------------------


def test_running_statistics_start_as_sb3s_and_update_by_hand():
    stats = RunningMeanStd(1)
    np.testing.assert_array_equal(stats.mean, [0.0])
    np.testing.assert_array_equal(stats.var, [1.0])
    assert stats.count == 1e-4
    stats.update(np.array([2.0]))
    # count 1.0001; mean = 0 + 2 * 1 / 1.0001; M2 = 1 * 1e-4 + 0 + 4 * 1e-4 * 1 / 1.0001
    np.testing.assert_allclose(stats.mean, [2.0 / 1.0001], rtol=1e-12)
    np.testing.assert_allclose(stats.var, [(1e-4 + 4e-4 / 1.0001) / 1.0001], rtol=1e-12)
    assert stats.count == pytest.approx(1.0001, rel=1e-12)


def test_running_statistics_match_chans_combination_over_many_draws():
    rng = np.random.default_rng(7)
    xs = rng.normal([1.0, -3.0, 0.0], [2.0, 0.1, 5.0], size=(1000, 3))
    stats = RunningMeanStd(3)
    for x in xs:
        stats.update(x)
    mean, var, count = _chan(np.zeros(3), np.ones(3), 1e-4, xs)
    np.testing.assert_allclose(stats.mean, mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(stats.var, var, rtol=1e-10)
    assert stats.count == pytest.approx(count, rel=1e-12)


def test_normalising_subtracts_the_mean_and_divides_by_the_standard_deviation():
    stats = RunningMeanStd(2)
    for x in ([1.0, 10.0], [3.0, 30.0]):
        stats.update(np.array(x))
    expected = (np.array([2.0, 20.0]) - stats.mean) / np.sqrt(stats.var + 1e-8)
    np.testing.assert_allclose(stats.normalize(np.array([2.0, 20.0])), expected, rtol=1e-12)


def test_a_constant_entry_normalises_to_zero_and_a_near_constant_one_is_clipped():
    """A masked entry is always 0, so its mean stays exactly 0 and it normalises to exactly 0."""
    stats = RunningMeanStd(2)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        stats.update(np.array([0.0, 5.0 + 1e-6 * rng.standard_normal()]))
    out = stats.normalize(np.array([0.0, 5.0 + 1.0]))
    assert out[0] == 0.0, "a constant entry must stay exactly 0"
    assert out[1] == 10.0, "an entry far outside a tiny spread is clipped at +10, not exploded"
    assert stats.normalize(np.array([0.0, 4.0]))[1] == -10.0


def test_normalising_returns_a_float64_copy():
    stats = RunningMeanStd(2)
    x = np.array([1.0, 2.0], dtype=np.float32)
    out = stats.normalize(x)
    assert out.dtype == np.float64 and out is not x


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda s: s.update(mean=np.zeros(3)), "mean"),
        (lambda s: s.update(var=np.array([1.0, np.nan])), "var"),
        (lambda s: s.update(var=np.array([1.0, -0.5])), "var"),
        (lambda s: s.update(count=np.array(0.0)), "count"),
        (lambda s: s.update(count=np.array([1.0, 2.0])), "count"),
        (lambda s: s.update(mean=np.array([np.inf, 0.0])), "mean"),
    ],
    ids=["mean shape", "nan var", "negative var", "zero count", "count not scalar", "inf mean"],
)
def test_check_state_refuses_malformed_statistics(change, match):
    stats = RunningMeanStd(2)
    stats.update(np.array([1.0, 2.0]))
    state = stats.state()
    change(state)
    with pytest.raises(ValueError, match=match):
        RunningMeanStd.check_state(state, dim=2)


def test_the_state_round_trips():
    stats = RunningMeanStd(2)
    for x in ([1.0, 2.0], [3.0, -1.0]):
        stats.update(np.array(x))
    copy = RunningMeanStd(2)
    copy.load_state(stats.state())
    probe = np.array([0.5, 0.5])
    np.testing.assert_array_equal(copy.normalize(probe), stats.normalize(probe))
    assert copy.count == stats.count


# --- The agent -------------------------------------------------------------------------------------


def test_without_the_setting_an_observation_passes_through_unchanged():
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config={"seed": 1})
    obs = np.array([0.1, 0.2, 0.3, 0.4])
    assert agent.observation_normalizer is None
    assert agent.normalize_observation(obs, training=True) is obs, "off must be the identity"


def test_training_updates_the_statistics_and_evaluation_only_reads_them():
    agent = ReinforceAgent(observation_dim=2, action_dim=2, config={"seed": 1, **ON})
    stats = agent.observation_normalizer
    agent.normalize_observation(np.array([1.0, 3.0]), training=True)
    agent.normalize_observation(np.array([3.0, 5.0]), training=True)
    count, mean = stats.count, stats.mean.copy()

    out = agent.normalize_observation(np.array([2.0, 4.0]), training=False)
    assert stats.count == count, "evaluation must not update the statistics"
    np.testing.assert_array_equal(stats.mean, mean)
    np.testing.assert_allclose(out, stats.normalize(np.array([2.0, 4.0])), rtol=1e-12)


def test_training_updates_before_normalising():
    """As SB3's VecNormalize: the first observation is already part of the mean it is centred on."""
    agent = ReinforceAgent(observation_dim=1, action_dim=2, config={"seed": 1, **ON})
    out = agent.normalize_observation(np.array([2.0]), training=True)
    stats = agent.observation_normalizer
    np.testing.assert_allclose(out, (2.0 - stats.mean) / np.sqrt(stats.var + 1e-8), rtol=1e-12)
    assert abs(out[0]) < 0.02, "updated first, the first observation lands near 0"


# --- The replay-buffer agents: raw observations stored, each sampled batch normalised -----------

REPLAY_AGENTS = [(DQNAgent, (4, 2)), (DDPGAgent, (3, 1)), (TD3Agent, (3, 1))]


def _raw_batch(rng, n, obs_dim, action_dim, discrete):
    """A trainer batch from a normalising agent: raw observations beside (here dummy) inputs."""
    raw = rng.normal(5.0, 2.0, (n + 1, obs_dim))
    actions = rng.integers(0, 2, n) if discrete else rng.uniform(-1.0, 1.0, (n, action_dim))
    terminated = np.zeros(n, dtype=bool)
    return {
        "observations": np.zeros((n, obs_dim)),  # what the agent acted on; never stored
        "actions": actions,
        "rewards": rng.normal(size=n),
        "next_observations": np.zeros((n, obs_dim)),
        "dones": terminated.copy(),
        "terminated": terminated,
        "raw_observations": raw[:-1],
        "raw_next_observations": raw[1:],
    }


def _assert_same_rows(actual, expected, message):
    """The two arrays hold the same rows in some order (a uniform sample is a permutation)."""
    order = lambda a: a[np.lexsort(a.T[::-1])]  # noqa: E731
    np.testing.assert_array_equal(order(actual), order(expected), err_msg=message)


def test_a_normalising_buffer_samples_its_raw_rows_with_the_current_statistics():
    statistics = RunningMeanStd(3)
    buffer = ReplayBuffer(capacity=10, observation_dim=3, seed=4, normalizer=statistics)
    rng = np.random.default_rng(0)
    batch = _raw_batch(rng, 4, 3, 1, discrete=True)
    assert buffer.add_transitions(batch) == 4
    np.testing.assert_array_equal(buffer.observations[:4], batch["raw_observations"])
    np.testing.assert_array_equal(buffer.next_observations[:4], batch["raw_next_observations"])

    for x in rng.normal(-3.0, 0.5, (20, 3)):  # the statistics move after the rows were stored
        statistics.update(x)
    sample = buffer.sample(4)

    _assert_same_rows(sample["observations"], statistics.normalize(batch["raw_observations"]),
                      "a sample must be normalised by the statistics at sampling time")  # fmt: skip
    _assert_same_rows(sample["next_observations"],
                      statistics.normalize(batch["raw_next_observations"]),
                      "next observations must be normalised the same way")  # fmt: skip
    np.testing.assert_array_equal(buffer.observations[:4], batch["raw_observations"],
                                  err_msg="sampling must not change what is stored")  # fmt: skip


def test_a_normalising_buffer_refuses_a_batch_without_raw_observations():
    buffer = ReplayBuffer(capacity=10, observation_dim=3, seed=4, normalizer=RunningMeanStd(3))
    batch = _raw_batch(np.random.default_rng(0), 4, 3, 1, discrete=True)
    del batch["raw_next_observations"]
    with pytest.raises(ValueError, match="raw_next_observations"):
        buffer.add_transitions(batch)
    assert len(buffer) == 0


def test_a_plain_buffer_stores_the_observations_it_is_given():
    buffer = ReplayBuffer(capacity=10, observation_dim=3, seed=4)
    batch = _raw_batch(np.random.default_rng(0), 4, 3, 1, discrete=True)
    buffer.add_transitions(batch)
    np.testing.assert_array_equal(buffer.observations[:4], batch["observations"])


@pytest.mark.parametrize("agent_cls,dims", REPLAY_AGENTS)
def test_the_replay_buffer_agents_store_raw_and_learn_on_normalised_inputs(agent_cls, dims):
    obs_dim, action_dim = dims
    warm_up = {"batch_size": 4, "min_buffer_size": 4, "learning_starts": 0}  # DQN's and DDPG's keys
    agent = agent_cls(obs_dim, action_dim, {"seed": 1, **warm_up, **ON})
    rng = np.random.default_rng(2)
    for x in rng.normal(5.0, 2.0, (30, obs_dim)):
        agent.normalize_observation(x, training=True)
    batch = _raw_batch(rng, 6, obs_dim, action_dim, discrete=agent_cls is DQNAgent)
    metrics = agent.update(batch)

    assert metrics, "the agent must have taken a gradient step on the sampled batch"
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, float))
    np.testing.assert_array_equal(agent.replay_buffer.observations[:6], batch["raw_observations"])
    _assert_same_rows(agent.replay_buffer.sample(6)["observations"],
                      agent.observation_normalizer.normalize(batch["raw_observations"]),
                      "the agent's buffer must normalise with the agent's statistics")  # fmt: skip


@pytest.mark.parametrize("agent_cls,dims", REPLAY_AGENTS)
def test_a_loaded_replay_agent_samples_with_the_loaded_statistics(agent_cls, dims, tmp_path):
    obs_dim, action_dim = dims
    config = {"batch_size": 4, "min_buffer_size": 4, **ON}
    saved = agent_cls(obs_dim, action_dim, {"seed": 1, **config})
    rng = np.random.default_rng(3)
    for x in rng.normal(5.0, 2.0, (30, obs_dim)):
        saved.normalize_observation(x, training=True)
    saved.update(_raw_batch(rng, 6, obs_dim, action_dim, discrete=agent_cls is DQNAgent))
    path = str(tmp_path / "agent.npz")
    saved.save(path)
    loaded = agent_cls(obs_dim, action_dim, {"seed": 9, **config})
    loaded.normalize_observation(np.full(obs_dim, -50.0), training=True)  # statistics to replace
    loaded.load(path)

    raw = loaded.replay_buffer.observations[:6]
    _assert_same_rows(loaded.replay_buffer.sample(6)["observations"],
                      saved.observation_normalizer.normalize(raw),
                      "after a load the buffer must read the loaded statistics")  # fmt: skip


@pytest.mark.parametrize("value", ["true", 1, None])
def test_the_setting_must_be_a_boolean(value):
    with pytest.raises(ValueError, match="normalize_observations"):
        ReinforceAgent(observation_dim=4, action_dim=2, config={"normalize_observations": value})


@pytest.mark.parametrize("agent_cls,config,obs_dim,action_dim", NORMALISING_AGENTS)
def test_the_statistics_are_saved_and_restored(agent_cls, config, obs_dim, action_dim, tmp_path):
    saved = agent_cls(obs_dim, action_dim, {**config, **ON})
    rng = np.random.default_rng(3)
    for _ in range(50):
        saved.normalize_observation(rng.normal(2.0, 3.0, obs_dim), training=True)
    path = str(tmp_path / "agent.pt")
    saved.save(path)
    assert {f"{OBS_NORM_KEY_PREFIX}{k}" for k in ("mean", "var", "count")} <= set(_entries(path))

    architecture = {k: config[k] for k in ARCHITECTURE_KEYS if k in config}
    loaded = agent_cls(obs_dim, action_dim, {**architecture, "seed": 99, **ON})
    loaded.load(path)
    probe = rng.normal(2.0, 3.0, obs_dim)
    np.testing.assert_array_equal(
        loaded.normalize_observation(probe, training=False),
        saved.normalize_observation(probe, training=False),
    )
    assert loaded.observation_normalizer.count == saved.observation_normalizer.count
    obs = saved.normalize_observation(probe, training=False)
    assert np.all(
        np.asarray(loaded.select_action(obs, training=False))
        == np.asarray(saved.select_action(obs, training=False))
    ), "the loaded agent acts differently"


def test_an_agent_without_the_setting_saves_no_statistics(tmp_path):
    ReinforceAgent(observation_dim=4, action_dim=2, config={"seed": 1}).save(
        str(tmp_path / "a.npz")
    )
    assert not [k for k in _entries(tmp_path / "a.npz") if k.startswith(OBS_NORM_KEY_PREFIX)]


@pytest.mark.parametrize(
    "saved_on,loader_on,match",
    [
        (True, False, r"saved with agent.config.normalize_observations: true"),
        (False, True, r"saved with agent.config.normalize_observations: false"),
    ],
    ids=["normalised checkpoint, plain agent", "plain checkpoint, normalising agent"],
)
def test_a_checkpoint_loads_only_with_the_same_setting(saved_on, loader_on, match, tmp_path):
    path = str(tmp_path / "agent.npz")
    ReinforceAgent(4, 2, {"seed": 1, **(ON if saved_on else {})}).save(path)
    loader = ReinforceAgent(4, 2, {"seed": 2, **(ON if loader_on else {})})
    _assert_refused_load_changes_nothing(loader, path, match, tmp_path)


def test_malformed_saved_statistics_are_refused_before_any_weight_changes(tmp_path):
    path = str(tmp_path / "agent.npz")
    trained = ReinforceAgent(4, 2, {"seed": 1, **ON})
    trained.normalize_observation(np.ones(4), training=True)
    trained.save(path)
    _rewrite(path, lambda e: e.update({f"{OBS_NORM_KEY_PREFIX}var": np.array([1, np.nan, 1, 1])}))

    loader = ReinforceAgent(4, 2, {"seed": 2, **ON})
    _assert_refused_load_changes_nothing(loader, path, f"{OBS_NORM_KEY_PREFIX}var", tmp_path)


def test_a_checkpoint_missing_part_of_the_statistics_is_refused_naming_it(tmp_path):
    path = str(tmp_path / "agent.npz")
    ReinforceAgent(4, 2, {"seed": 1, **ON}).save(path)
    _rewrite(path, lambda e: e.pop(f"{OBS_NORM_KEY_PREFIX}count"))
    with pytest.raises(ValueError, match=f"lacks {OBS_NORM_KEY_PREFIX}count"):
        ReinforceAgent(4, 2, {"seed": 2, **ON}).load(path)


def test_save_refuses_another_entry_under_the_statistics_prefix(tmp_path):
    from tests.agent_internals import with_extra_checkpoint_entry

    key = f"{OBS_NORM_KEY_PREFIX}extra"
    agent = with_extra_checkpoint_entry(ReinforceAgent, key)(4, 2, {"seed": 1})
    with pytest.raises(ValueError, match=key):
        agent.save(str(tmp_path / "agent.npz"))


# --- The trainer --------------------------------------------------------------------------------


class _SpyReinforce(ReinforceAgent):
    """Records what the trainer hands it: the observations it acts on and the batches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acted_on, self.batches = [], []

    def select_action(self, observation, training=True):
        self.acted_on.append(np.array(observation, copy=True))
        return super().select_action(observation, training)

    def update(self, batch):
        self.batches.append({k: np.array(v, copy=True) for k, v in batch.items()})
        return super().update(batch)


def _trainer(agent, directory, episodes=3, steps=30):
    training = {"num_episodes": episodes, "max_steps_per_episode": steps, "eval_frequency": 1000,
                "save_frequency": 1000, "checkpoint_dir": str(directory)}  # fmt: skip
    return Trainer(env=CartPoleEnv(seed=5), agent=agent, config=training,
                   eval_env=CartPoleEnv(seed=5), eval_seed=500)  # fmt: skip


def test_the_trainer_normalises_each_observation_once_and_uses_it_everywhere(tmp_path):
    agent = _SpyReinforce(4, 2, {"seed": 1, **ON})
    trainer = _trainer(agent, tmp_path, episodes=1, steps=30)
    trainer.train()
    (batch,) = agent.batches
    n = len(batch["actions"])

    assert agent.observation_normalizer.count == pytest.approx(1e-4 + n + 1), (
        "every raw observation (the reset and each step's) must update the statistics once"
    )
    np.testing.assert_array_equal(batch["observations"], np.array(agent.acted_on))
    np.testing.assert_array_equal(batch["observations"][1:], batch["next_observations"][:-1])
    assert np.abs(batch["observations"]).max() <= 10.0
    # The same seeded environment's first observation, updated into fresh statistics and normalised.
    reference_env = CartPoleEnv(seed=5)
    first_raw, _ = reference_env.reset()
    reference_env.close()
    fresh = RunningMeanStd(4)
    fresh.update(first_raw)
    np.testing.assert_array_equal(batch["observations"][0], fresh.normalize(first_raw))


@pytest.mark.parametrize("on", [True, False], ids=["normalising", "plain"])
def test_the_trainer_hands_raw_observations_beside_the_inputs_only_when_normalising(on, tmp_path):
    agent = _SpyReinforce(4, 2, {"seed": 1, **(ON if on else {})})
    _trainer(agent, tmp_path, episodes=1, steps=30).train()
    (batch,) = agent.batches

    if not on:
        assert "raw_observations" not in batch and "raw_next_observations" not in batch, (
            "a run without normalisation must hand the agent the same batch as before"
        )
        return
    reference_env = CartPoleEnv(seed=5)
    first_raw, _ = reference_env.reset()
    reference_env.close()
    np.testing.assert_array_equal(batch["raw_observations"][0], first_raw)
    np.testing.assert_array_equal(
        batch["raw_observations"][1:], batch["raw_next_observations"][:-1]
    )
    assert np.abs(batch["raw_observations"]).max() > 0 and not np.array_equal(
        batch["raw_observations"], batch["observations"]
    ), "raw observations must be the environment's, not the normalised inputs"


def test_evaluation_leaves_the_statistics_unchanged(tmp_path):
    agent = ReinforceAgent(4, 2, {"seed": 1, **ON})
    trainer = _trainer(agent, tmp_path, episodes=2)
    trainer.train()
    state = agent.observation_normalizer.state()
    trainer.evaluate(3)
    for key, value in agent.observation_normalizer.state().items():
        np.testing.assert_array_equal(value, state[key], err_msg=f"evaluation changed {key}")


def _policy_log_probs(agent, observations, actions):
    """log pi(a|s) for a batch under the agent's current weights, as its PPO update computes them."""
    if isinstance(agent, ContinuousPPOAgent):
        mu, log_std, _, _ = gaussian_forward(agent, observations)
        return log_prob(agent, actions, mu, log_std)
    probs, _, _ = forward_batch(agent, observations)
    return np.log(np.clip(probs[np.arange(len(actions)), actions], 1e-8, 1.0))


# Each PPO with a rollout of several short episodes, so the statistics move between and during
# rollouts; a small network and few epochs keep it fast.
PPO_ROLLOUT_CASES = [
    pytest.param(
        ContinuousPPOAgent,
        {
            "hidden_dim": 8,
            "rollout_steps": 60,
            "batch_size": 16,
            "ppo_epochs": 2,
            "action_low": -2.0,
            "action_high": 2.0,
        },
        lambda: PendulumEnv(seed=5, max_episode_steps=20),
        3,
        1,
        id="ContinuousPPOAgent-Pendulum",
    ),
    pytest.param(
        PPOAgent,
        {"hidden_dim": 8, "rollout_steps": 60, "minibatch_size": 16, "ppo_epochs": 2},
        lambda: CartPoleEnv(seed=5),
        4,
        2,
        id="PPOAgent-CartPole",
    ),
]


@pytest.mark.parametrize("agent_cls,config,make_env,obs_dim,action_dim", PPO_ROLLOUT_CASES)
def test_ppo_old_log_probs_are_those_of_collection_under_moving_statistics(
    agent_cls, config, make_env, obs_dim, action_dim, tmp_path, monkeypatch
):
    """
    PPO recomputes its old log-probabilities at the start of an update, on the inputs it stored.
    With the statistics moving on every training observation, those must still be exactly the
    log-probabilities of the actions the policy took on the inputs it acted on, and the first
    mini-batch's ratio must be 1: nothing about the input may change between collection and
    update (diagnosis of the Pendulum PPO collapse with normalisation, 2026-11-09).
    """
    agent = agent_cls(obs_dim, action_dim, {"seed": 3, **config, **ON})
    collected = {}  # (input, action) -> log pi(a|s) when the action was drawn
    select_action = agent.select_action

    def acting(observation, training=True):
        action = select_action(observation, training=training)
        if training:
            row = np.asarray(observation, dtype=np.float64)[None]
            acts = np.asarray(action)[None]
            collected[row.tobytes() + acts.tobytes()] = float(
                _policy_log_probs(agent, row, acts)[0]
            )
        return action

    monkeypatch.setattr(agent, "select_action", acting)
    updates = {"first_minibatch_checked": 0}
    means_at_update = []  # the normaliser's mean when each update starts
    update = agent.update

    def updating(batch):
        normalizer = agent.observation_normalizer
        assert normalizer is not None
        means_at_update.append(normalizer.mean.copy())
        updates["pending"] = True
        return update(batch)

    monkeypatch.setattr(agent, "update", updating)

    def minibatch(original, observations, actions, returns, advantages, old_log_probs, *a, **kw):
        for row, act, old in zip(observations, actions, old_log_probs, strict=True):
            key = row[None].tobytes() + np.asarray(act)[None].tobytes()
            assert key in collected, (
                "PPO trained on an input the policy never acted on: the stored observation must "
                "be the one normalised at collection, under the statistics of that moment"
            )
            assert abs(old - collected[key]) <= 1e-12, (
                f"old log-probability {old!r} differs from the collection-time {collected[key]!r}"
            )
        if updates.pop("pending", False):
            ratio = np.exp(_policy_log_probs(agent, observations, actions) - old_log_probs)
            assert np.max(np.abs(ratio - 1.0)) <= 1e-12, (
                f"the first mini-batch's ratio must be 1 before any step, got {ratio!r}"
            )
            updates["first_minibatch_checked"] += 1
        return original(observations, actions, returns, advantages, old_log_probs, *a, **kw)

    spy_on_ppo_minibatches(monkeypatch, agent, minibatch)
    env, eval_env = make_env(), make_env()
    try:
        training = {
            "num_episodes": 9,
            "max_steps_per_episode": 20,
            "eval_frequency": 1000,
            "save_frequency": 1000,
            "checkpoint_dir": str(tmp_path),
        }
        Trainer(env=env, agent=agent, config=training, eval_env=eval_env, eval_seed=700).train()
    finally:
        env.close()
        eval_env.close()

    assert updates["first_minibatch_checked"] >= 2, (
        "at least two rollouts must be trained on, the second under statistics that have moved"
    )
    assert not np.allclose(means_at_update[0], means_at_update[-1]), (
        "the statistics must have moved between the first and last update, or the test pins nothing"
    )


def test_a_resumed_normalising_run_ends_exactly_like_one_that_never_stopped(tmp_path):
    def trainer(directory, episodes):
        agent = PPOAgent(4, 2, {"seed": 1, "rollout_steps": 64, **ON})
        training = {"num_episodes": episodes, "max_steps_per_episode": 40, "eval_frequency": 1000,
                    "save_frequency": 4, "checkpoint_dir": str(directory)}  # fmt: skip
        return Trainer(env=CartPoleEnv(seed=3), agent=agent, config=training,
                       eval_env=CartPoleEnv(seed=3), eval_seed=900)  # fmt: skip

    full = trainer(tmp_path / "full", 8)
    full.train()
    trainer(tmp_path / "split", 4).train()
    resumed = trainer(tmp_path / "split", 8)
    resumed.resume(str(tmp_path / "split" / "agent_episode_4.pt"))
    resumed.train()

    assert resumed.episode_rewards == full.episode_rewards
    want, got = (
        _entries(tmp_path / "full" / "agent_episode_8.pt"),
        _entries(tmp_path / "split" / "agent_episode_8.pt"),
    )
    assert sorted(got) == sorted(want)
    for key in want:
        np.testing.assert_array_equal(got[key], want[key], err_msg=f"{key} differs after resume")


@pytest.mark.usefixtures("clear_sdl")
def test_train_and_record_run_with_normalised_stacked_masked_observations(tmp_path, monkeypatch):
    pytest.importorskip("moviepy")
    config_path = minimal_config(
        tmp_path,
        environment={"frame_stack": 4, "mask_velocity": True},
        training={"save_frequency": 2},
        agent={"type": "reinforce", "config": {"seed": 1, **ON}},
    )
    run_main(["--config", config_path], monkeypatch)
    checkpoint = tmp_path / "checkpoints" / "cartpole" / "reinforce" / "agent_episode_4.pt"
    entries = _entries(checkpoint)
    assert entries[f"{OBS_NORM_KEY_PREFIX}mean"].shape == (16,)
    assert entries[f"{OBS_NORM_KEY_PREFIX}count"] > 1, "train.py fed the statistics nothing"

    video_dir = tmp_path / "videos"
    run_main(
        ["--config", config_path, "--checkpoint", str(checkpoint), "--episodes", "1"]
        + ["--video-dir", str(video_dir)],
        monkeypatch,
        module="record",
    )
    assert (video_dir / "cartpole-playback-full.mp4").exists(), "record.py wrote no video"
