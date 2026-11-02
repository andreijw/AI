"""Running observation normalisation (``agent.config.normalize_observations``): the running
statistics, the agent that owns and checkpoints them, and the trainer that applies them once per
observation, updating in training and frozen in evaluation."""

import numpy as np
import pytest

from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import DDPGAgent, DQNAgent, PPOAgent, ReinforceAgent, TD3Agent
from rl_core.agents.base_agent import OBS_NORM_KEY_PREFIX
from rl_core.agents.observation_normalizer import RunningMeanStd
from rl_core.training import Trainer
from tests.helpers import minimal_config, run_main
from tests.test_agent_harness import (
    ALL_AGENTS,
    ARCHITECTURE_KEYS,
    CONTINUOUS_AGENTS,
    agent_id,
)

ON = {"normalize_observations": True}

# Every agent that may normalise, with a CartPole (discrete) or Pendulum (continuous) size.
NORMALISING_AGENTS = [
    pytest.param(c, cfg, 4, 2, id=agent_id(c, cfg))
    for c, cfg in ALL_AGENTS
    if c.SUPPORTS_OBSERVATION_NORMALIZATION
] + [
    pytest.param(c, cfg, 3, 1, id=c.__name__)
    for c, cfg in CONTINUOUS_AGENTS
    if c.SUPPORTS_OBSERVATION_NORMALIZATION
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


@pytest.mark.parametrize(
    "agent_cls,dims", [(DQNAgent, (4, 2)), (DDPGAgent, (3, 1)), (TD3Agent, (3, 1))]
)
def test_the_replay_buffer_agents_refuse_the_setting(agent_cls, dims):
    with pytest.raises(ValueError, match="normalize_observations"):
        agent_cls(*dims, {"seed": 1, **ON})


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


def test_evaluation_leaves_the_statistics_unchanged(tmp_path):
    agent = ReinforceAgent(4, 2, {"seed": 1, **ON})
    trainer = _trainer(agent, tmp_path, episodes=2)
    trainer.train()
    state = agent.observation_normalizer.state()
    trainer.evaluate(3)
    for key, value in agent.observation_normalizer.state().items():
        np.testing.assert_array_equal(value, state[key], err_msg=f"evaluation changed {key}")


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
