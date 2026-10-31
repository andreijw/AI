"""A run stopped after a checkpoint and resumed from it ends exactly as one that never stopped
(``Trainer.resume``, ``train.py --resume``; CHANGELOG.md, 2026-10-21)."""

import os

import gymnasium as gym
import numpy as np
import pytest

from rl_acrobot.environments import AcrobotEnv
from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import ContinuousActorCriticAgent, RandomAgent
from rl_core.environments import ResumableEnv
from rl_core.training import Trainer
from rl_mountaincar.environments import MountainCarEnv
from rl_pendulum.environments import PendulumEnv
from tests.test_agent_harness import ALL_AGENTS, CONTINUOUS_AGENTS, agent_id

EPISODE_STEPS = 15


def _named(name):
    """The harness entry for one agent class, by name rather than list position."""
    return next((c, cfg) for c, cfg in ALL_AGENTS + CONTINUOUS_AGENTS if c.__name__ == name)


def _cartpole(seed=3):
    return CartPoleEnv(seed=seed, max_episode_steps=EPISODE_STEPS, obs_noise_std=0.01,
                       action_noise_prob=0.1, domain_randomization={"length": (0.4, 0.6)})  # fmt: skip


def _pendulum(seed=3):
    return PendulumEnv(seed=seed, max_episode_steps=EPISODE_STEPS, obs_noise_std=0.01,
                       action_noise_std=0.1)  # fmt: skip


AGENTS = [pytest.param(c, cfg, _cartpole, 4, 2, id=agent_id(c, cfg)) for c, cfg in ALL_AGENTS] + [
    pytest.param(c, cfg, _pendulum, 3, 1, id=c.__name__) for c, cfg in CONTINUOUS_AGENTS
]


class _RecordingLogger:
    def __init__(self):
        self.records, self.lines = [], []

    def log(self, metrics):
        self.records.append(metrics)

    def info(self, message):
        self.lines.append(message)

    def warning(self, message):
        self.lines.append(message)


def _trainer(agent_cls, config, make_env, dims, directory, logger=None, eval_seed=1000, **budget):
    training = {"max_steps_per_episode": EPISODE_STEPS, "eval_frequency": 5, "save_frequency": 5,
                "checkpoint_dir": str(directory), **budget}  # fmt: skip
    return Trainer(
        env=make_env(),
        agent=agent_cls(*dims, dict(config)),
        config=training,
        logger=_RecordingLogger() if logger is None else logger,
        eval_env=make_env(),
        eval_seed=eval_seed,
    )


def _assert_same_file(a, b):
    with np.load(a) as x, np.load(b) as y:
        assert sorted(x.files) == sorted(y.files), f"{a} and {b} hold different keys"
        for key in x.files:
            np.testing.assert_array_equal(y[key], x[key], err_msg=f"{os.path.basename(a)}: {key}")


# ---------------------------------------------------------------------------
# Stop after checkpoint 5, resume, and end where an uninterrupted run ends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("agent_cls", "config", "make_env", "obs_dim", "action_dim"), AGENTS)
def test_a_resumed_run_ends_exactly_like_one_that_never_stopped(
    agent_cls, config, make_env, obs_dim, action_dim, tmp_path
):
    """Checkpoint 5 falls inside the 10-episode log window, so the update metrics gathered so far
    must survive the stop too."""
    dims = (obs_dim, action_dim)
    full = _trainer(agent_cls, config, make_env, dims, tmp_path / "full", num_episodes=10)
    full_stats = full.train()

    first = _trainer(agent_cls, config, make_env, dims, tmp_path / "split", num_episodes=5)
    first.train()
    resumed = _trainer(agent_cls, config, make_env, dims, tmp_path / "split", num_episodes=10)
    resumed.resume(str(tmp_path / "split" / "agent_episode_5.pt"))
    resumed_stats = resumed.train()

    assert resumed.episode_rewards == full.episode_rewards
    assert resumed.episode_lengths == full.episode_lengths
    assert resumed_stats == full_stats
    assert first.logger.records + resumed.logger.records == full.logger.records
    for name in ("agent_episode_10.pt", "trainer_episode_10.npz"):
        _assert_same_file(tmp_path / "full" / name, tmp_path / "split" / name)


def test_a_resumed_a2c_run_updating_mid_episode_ends_like_one_that_never_stopped(tmp_path):
    """The continuous Actor-Critic with ``update_every`` equal to ``n_steps`` (true A2C): 15-step
    episodes in chunks of 4, 4, 4, 3, stopped at checkpoint 5."""
    config = {"seed": 2, "hidden_dim": 8, "n_steps": 4, "action_low": -2.0, "action_high": 2.0}
    args = (ContinuousActorCriticAgent, config, _pendulum, (3, 1))
    full = _trainer(*args, tmp_path / "full", num_episodes=10, update_every=4)
    full_stats = full.train()
    _trainer(*args, tmp_path / "split", num_episodes=5, update_every=4).train()
    resumed = _trainer(*args, tmp_path / "split", num_episodes=10, update_every=4)
    resumed.resume(str(tmp_path / "split" / "agent_episode_5.pt"))
    assert resumed.train() == full_stats
    assert resumed.episode_rewards == full.episode_rewards
    _assert_same_file(tmp_path / "full" / "agent_episode_10.pt",
                      tmp_path / "split" / "agent_episode_10.pt")  # fmt: skip


def test_a_step_budget_counts_the_steps_trained_before_the_stop(tmp_path):
    cls, config = _named("PPOAgent")
    full = _trainer(cls, config, _cartpole, (4, 2), tmp_path / "full", total_steps=120)
    full.train()
    first = _trainer(cls, config, _cartpole, (4, 2), tmp_path / "split", num_episodes=5)
    first.train()
    resumed = _trainer(cls, config, _cartpole, (4, 2), tmp_path / "split", total_steps=120)
    resumed.resume(str(tmp_path / "split" / "agent_episode_5.pt"))
    resumed.train()
    assert resumed.episode_lengths == full.episode_lengths
    assert resumed.total_steps_trained == full.total_steps_trained >= 120


def test_checkpoint_numbering_continues_after_a_resume(tmp_path):
    cls, config = _named("RandomAgent")
    _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=5).train()
    before = os.path.getmtime(tmp_path / "agent_episode_5.pt")
    resumed = _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=10)
    resumed.resume(str(tmp_path / "agent_episode_5.pt"))
    assert resumed.current_episode == 5, "video clips are named from it, so they continue too"
    resumed.train()
    assert resumed.current_episode == 10
    assert os.path.getmtime(tmp_path / "agent_episode_5.pt") == before, "checkpoint 5 is kept"
    assert sorted(os.listdir(tmp_path)) == [
        "agent_episode_10.pt",
        "agent_episode_5.pt",
        "trainer_episode_10.npz",
        "trainer_episode_5.npz",
    ], "both files per checkpoint and no temporary file left behind"


def test_resuming_a_run_that_met_its_budget_trains_nothing(tmp_path):
    cls, config = _named("RandomAgent")
    _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=5).train()
    resumed = _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=5)
    resumed.resume(str(tmp_path / "agent_episode_5.pt"))
    stats = resumed.train()
    assert stats["total_episodes"] == 5 and resumed.logger.records == []
    assert any("already" in line for line in resumed.logger.lines), "says why nothing trained"


def test_the_saved_evaluation_seed_replaces_the_new_runs(tmp_path):
    """An unseeded run draws its evaluation seed from entropy; a resume must not draw another."""
    cls, config = _named("RandomAgent")
    _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=5, eval_seed=111).train()
    resumed = _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=10, eval_seed=222)
    resumed.resume(str(tmp_path / "agent_episode_5.pt"))
    assert resumed.eval_seed == 111


def _recorded_run(agent_cls, config, directory, video_dir, num_episodes, resume_from=None):
    """Train with every 2nd training episode recorded to ``video_dir``, as train.py attaches it."""
    from rl_core.utils.video import TrainingVideoRecorder

    def make_env():
        return CartPoleEnv(render_mode="rgb_array", seed=3, max_episode_steps=EPISODE_STEPS)

    trainer = _trainer(agent_cls, config, make_env, (4, 2), directory, num_episodes=num_episodes)
    if resume_from is not None:
        trainer.resume(str(resume_from))
    trainer.env.wrap_env(
        TrainingVideoRecorder(
            trainer.env.env, video_folder=str(video_dir), progress=trainer, every=2
        )
    )
    try:
        trainer.train()
    finally:
        trainer.env.close()
        trainer.eval_env.close()


def test_a_resumed_run_records_the_same_clips_under_the_same_numbers(tmp_path):
    pytest.importorskip("moviepy")
    cls, config = _named("PPOAgent")
    _recorded_run(cls, config, tmp_path / "full", tmp_path / "full_videos", 10)
    # A second folder for the resumed half: RecordVideo warns about any existing folder.
    _recorded_run(cls, config, tmp_path / "split", tmp_path / "videos_1", 5)
    _recorded_run(cls, config, tmp_path / "split", tmp_path / "videos_2", 10,
                  resume_from=tmp_path / "split" / "agent_episode_5.pt")  # fmt: skip

    def clips(*folders):
        return {
            name: (folder / name).read_bytes() for folder in folders for name in os.listdir(folder)
        }

    full, split = (
        clips(tmp_path / "full_videos"),
        clips(tmp_path / "videos_1", tmp_path / "videos_2"),
    )
    assert (
        sorted(split)
        == sorted(full)
        == sorted(f"cartpole-training-episode-{n}.mp4" for n in (2, 4, 6, 8, 10))
    ), "the resumed run records episodes 6 to 10, numbered as in the uninterrupted run"
    assert split == full, "and the clips show the same episodes, byte for byte"


# ---------------------------------------------------------------------------
# Checkpoints that cannot be resumed
# ---------------------------------------------------------------------------


def test_a_checkpoint_without_trainer_state_cannot_be_resumed(tmp_path):
    """Checkpoints from before 2026-10-21 hold only the agent."""
    cls, config = _named("RandomAgent")
    cls(4, 2, dict(config)).save(str(tmp_path / "agent_episode_5.pt"))
    trainer = _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=10)
    with pytest.raises(FileNotFoundError, match="trainer_episode_5.npz"):
        trainer.resume(str(tmp_path / "agent_episode_5.pt"))


def test_a_checkpoint_not_named_by_the_trainer_cannot_be_resumed(tmp_path):
    cls, config = _named("RandomAgent")
    trainer = _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=10)
    with pytest.raises(ValueError, match="agent_episode_"):
        trainer.resume(str(tmp_path / "model.pt"))


def test_a_checkpoint_of_another_agent_is_rejected_before_anything_loads(tmp_path):
    random_cls, random_config = _named("RandomAgent")
    _trainer(random_cls, random_config, _cartpole, (4, 2), tmp_path, num_episodes=5).train()
    ppo_cls, ppo_config = _named("PPOAgent")
    trainer = _trainer(ppo_cls, ppo_config, _cartpole, (4, 2), tmp_path, num_episodes=10)
    with pytest.raises(ValueError, match="RandomAgent"):
        trainer.resume(str(tmp_path / "agent_episode_5.pt"))
    assert trainer.episode_rewards == [], "nothing restored"


def test_a_checkpoint_of_another_environment_is_rejected(tmp_path):
    cls, config = _named("RandomAgent")
    _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=5).train()
    trainer = _trainer(cls, config, _pendulum, (4, 2), tmp_path, num_episodes=10)
    with pytest.raises(ValueError, match="CartPole-v1"):
        trainer.resume(str(tmp_path / "agent_episode_5.pt"))


def test_a_plain_gymnasium_environment_trains_and_checkpoints_without_its_generators(tmp_path):
    """Only an environment that can save its generators has them saved; a resume of such a run
    into it then fails clearly, as its start states would differ."""
    cls, config = _named("RandomAgent")
    training = {"num_episodes": 5, "max_steps_per_episode": EPISODE_STEPS, "save_frequency": 5,
                "eval_frequency": 100, "checkpoint_dir": str(tmp_path / "plain")}  # fmt: skip
    plain = gym.make("CartPole-v1")
    try:
        Trainer(env=plain, agent=cls(4, 2, dict(config)), config=training).train()
    finally:
        plain.close()
    assert os.path.exists(tmp_path / "plain" / "trainer_episode_5.npz")

    _trainer(cls, config, _cartpole, (4, 2), tmp_path / "wrapped", num_episodes=5).train()
    plain = gym.make("CartPole-v1")
    try:
        trainer = Trainer(env=plain, agent=cls(4, 2, dict(config)), config=training)
        with pytest.raises(ValueError, match="generator"):
            trainer.resume(str(tmp_path / "wrapped" / "agent_episode_5.pt"))
    finally:
        plain.close()

    plain = gym.make("CartPole-v1")
    try:
        trainer = Trainer(
            env=plain, agent=cls(4, 2, dict(config)), config={**training, "num_episodes": 10}
        )
        trainer.resume(str(tmp_path / "plain" / "agent_episode_5.pt"))
        assert trainer.train()["total_episodes"] == 10, (
            "a plain run resumes into a plain environment"
        )
    finally:
        plain.close()


class _DivergingAgent(RandomAgent):
    """Its update reports a non-finite loss from episode 3 on."""

    def update(self, batch):
        self.updates = getattr(self, "updates", 0) + 1
        return {"loss": float("nan") if self.updates >= 3 else 1.0}


def test_the_non_finite_bookkeeping_survives_a_resume(tmp_path):
    """A diverging run warns once per metric, also across a stop, and keeps the episode it began."""

    def run(n):
        trainer = _trainer(
            _DivergingAgent, {"seed": 1}, _cartpole, (4, 2), tmp_path, num_episodes=n
        )
        return trainer

    first = run(5)
    first.train()
    assert first.first_non_finite_episode == 3
    resumed = run(10)
    resumed.resume(str(tmp_path / "agent_episode_5.pt"))
    resumed.train()  # its update counter restarts, so episodes 8 to 10 report NaN again
    assert resumed.first_non_finite_episode == 3
    assert not any("first appeared" in line for line in resumed.logger.lines), "warned twice"


def _corrupt(tmp_path, change):
    """A run checkpointed at episode 5 whose trainer file ``change`` alters; the resuming trainer."""
    cls, config = _named("PPOAgent")
    _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=5).train()
    path = tmp_path / "trainer_episode_5.npz"
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    change(arrays)
    np.savez(path, **arrays)
    return _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=10)


def _set(key, value):
    return lambda a: a.__setitem__(key, np.asarray(value))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        pytest.param(lambda a: a.pop("episode_lengths"), "episode_lengths", id="missing-key"),
        pytest.param(
            lambda a: a.__setitem__("episode_lengths", a["episode_lengths"][:3]),
            "episode_lengths",
            id="lengths-short",
        ),  # fmt: skip
        pytest.param(
            lambda a: a.__setitem__("episode_rewards", a["episode_rewards"][:, None]),
            "episode_rewards",
            id="rewards-2d",
        ),  # fmt: skip
        pytest.param(
            lambda a: a.__setitem__("total_steps_trained", a["total_steps_trained"] + 1),
            "total_steps_trained",
            id="steps-disagree",
        ),  # fmt: skip
        pytest.param(_set("update_metric_window", "[1, 2]"), "update_metric_window", id="window"),
        pytest.param(_set("non_finite_metrics_seen", "[3]"), "non_finite_metrics_seen", id="seen"),
        pytest.param(
            _set("agent_checkpoint", "agent_episode_6.pt"),
            "agent_checkpoint",
            id="other-checkpoint",
        ),  # fmt: skip
        pytest.param(_set("agent_sha256", "0" * 64), "agent_sha256", id="agent-file-replaced"),
        pytest.param(
            _set("env_start_rng", '{"bit_generator": "MT19937", "state": {}}'),
            "env_start_rng",
            id="env-generator",
        ),  # fmt: skip
        pytest.param(_set("agent_class", 3), "agent_class", id="class-not-text"),
        pytest.param(_set("eval_seed", True), "eval_seed", id="seed-bool"),
        pytest.param(_set("eval_seed", -1), "eval_seed", id="seed-negative"),
        pytest.param(_set("total_steps_trained", 75.0), "total_steps_trained", id="steps-float"),
        pytest.param(
            lambda a: a.__setitem__("total_steps_trained", a["total_steps_trained"].reshape(1)),
            "single value",
            id="steps-array",
        ),  # fmt: skip
        pytest.param(
            lambda a: a.__setitem__("episode_lengths", a["episode_lengths"] + 0.4),
            "episode_lengths",
            id="lengths-float",
        ),  # fmt: skip
        pytest.param(
            lambda a: a.__setitem__("episode_rewards", a["episode_rewards"].astype(str)),
            "episode_rewards",
            id="rewards-text",
        ),  # fmt: skip
        pytest.param(
            _set("first_non_finite_episode", 999),
            "first_non_finite_episode",
            id="first-non-finite-out-of-range",
        ),  # fmt: skip
        pytest.param(_set("env_has_reset", [True, False]), "env_has_reset", id="has-reset-array"),
        pytest.param(_set("env_has_reset", "False"), "env_has_reset", id="has-reset-text"),
        pytest.param(_set("env_typo", 1), "env_typo", id="unexpected-key"),
    ],
)
def test_a_malformed_trainer_file_is_rejected_before_anything_changes(change, match, tmp_path):
    trainer = _corrupt(tmp_path, change)
    before = trainer.agent.select_action(np.zeros(4), training=False), trainer.env.rng_state()
    with pytest.raises(ValueError, match=match):
        trainer.resume(str(tmp_path / "agent_episode_5.pt"))
    assert trainer.episode_rewards == [] and trainer.total_steps_trained == 0
    for key, value in before[1].items():
        np.testing.assert_array_equal(trainer.env.rng_state()[key], value, err_msg=key)
    fresh = _named("PPOAgent")[0](4, 2, dict(_named("PPOAgent")[1]))
    trainer.agent.save(str(tmp_path / "after.npz"))
    fresh.save(str(tmp_path / "fresh.npz"))
    _assert_same_file(tmp_path / "fresh.npz", tmp_path / "after.npz")


def test_a_run_evaluated_on_its_own_environment_needs_one_to_resume(tmp_path):
    cls, config = _named("RandomAgent")
    _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=5).train()
    training = {"num_episodes": 10, "max_steps_per_episode": EPISODE_STEPS, "eval_frequency": 5,
                "save_frequency": 5, "checkpoint_dir": str(tmp_path)}  # fmt: skip
    trainer = Trainer(env=_cartpole(), agent=cls(4, 2, dict(config)), config=training)
    with pytest.raises(ValueError, match="evaluation"):
        trainer.resume(str(tmp_path / "agent_episode_5.pt"))


def test_a_run_evaluated_on_its_training_environment_resumes_without_another(tmp_path):
    cls, config = _named("RandomAgent")
    training = {"num_episodes": 5, "max_steps_per_episode": EPISODE_STEPS, "eval_frequency": 5,
                "save_frequency": 5, "checkpoint_dir": str(tmp_path)}  # fmt: skip
    Trainer(env=_cartpole(), agent=cls(4, 2, dict(config)), config=training).train()
    trainer = _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=10)
    with pytest.raises(ValueError, match="evaluation"):
        trainer.resume(str(tmp_path / "agent_episode_5.pt"))


def test_a_trainer_state_whose_environment_keys_clash_is_not_written():
    from rl_core.training.checkpoints import TrainerState

    state = TrainerState(
        agent_checkpoint="agent_episode_0.pt", agent_sha256="", agent_class="A", env_id="E",
        episode_rewards=[], episode_lengths=[], total_steps_trained=0, update_metric_window={},
        non_finite_metrics_seen=[], env_state={"env_id": np.array("other")},
    )  # fmt: skip
    with pytest.raises(ValueError, match="env_id"):
        state.to_arrays()


def test_an_interrupted_agent_save_leaves_no_checkpoint(tmp_path, monkeypatch):
    cls, config = _named("RandomAgent")
    trainer = _trainer(cls, config, _cartpole, (4, 2), tmp_path, num_episodes=5)
    real_save = trainer.agent.save

    def crash(path):
        real_save(path)
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer.agent, "save", crash)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    assert os.listdir(tmp_path) == [], "no agent file, and no trainer file beside a missing one"


# ---------------------------------------------------------------------------
# Environment generators
# ---------------------------------------------------------------------------


def _episodes(env, n, action):
    out = []
    for _ in range(n):
        obs, _ = env.reset()
        out.append(obs)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(action)
            out.append(obs)
            done = terminated or truncated
    return np.array(out)


@pytest.mark.parametrize(
    ("make_env", "action"),
    [
        (_cartpole, 1),
        (_pendulum, np.array([0.5])),
        (lambda: AcrobotEnv(seed=3, max_episode_steps=EPISODE_STEPS, obs_noise_std=0.01), 2),
        (lambda: MountainCarEnv(seed=3, max_episode_steps=EPISODE_STEPS, obs_noise_std=0.01), 2),
        (lambda: _cartpole(seed=None), 1),
    ],
    ids=["cartpole", "pendulum", "acrobot", "mountaincar", "unseeded"],
)
def test_an_environment_continues_from_its_saved_generators(make_env, action):
    """Start states, observation and action noise and domain randomization all continue."""
    env, fresh = make_env(), make_env()
    try:
        assert isinstance(env, ResumableEnv)
        _episodes(env, 3, action)
        fresh.set_rng_state(env.rng_state())
        np.testing.assert_array_equal(_episodes(fresh, 3, action), _episodes(env, 3, action))
    finally:
        env.close()
        fresh.close()


@pytest.mark.parametrize("broken", ["missing", "other-kind"])
def test_an_unusable_environment_state_is_rejected_and_changes_nothing(broken):
    from rl_core.seeding import generator_state

    env, reference = _cartpole(), _cartpole()
    try:
        state = env.rng_state()
        if broken == "missing":
            del state["env_start_rng"]
        else:
            state["env_start_rng"] = generator_state(np.random.Generator(np.random.PCG64DXSM(1)))
        _episodes(env, 1, 1)
        _episodes(reference, 1, 1)
        with pytest.raises(ValueError, match="env_start_rng"):
            env.set_rng_state(state)
        np.testing.assert_array_equal(_episodes(env, 2, 1), _episodes(reference, 2, 1))
    finally:
        env.close()
        reference.close()


# ---------------------------------------------------------------------------
# train.py --resume
# ---------------------------------------------------------------------------


def _config(directory, num_episodes):
    import yaml

    from tests.helpers import minimal_config

    directory.mkdir()
    path = minimal_config(directory)
    with open(path) as f:
        config = yaml.safe_load(f)
    config["agent"]["type"] = "ppo"
    config["agent"]["config"]["seed"] = 42  # as committed configs do; unseeded runs never match
    config["training"].update(num_episodes=num_episodes, save_frequency=5, eval_frequency=5)
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def test_train_py_resume_ends_like_an_uninterrupted_run(tmp_path, monkeypatch):
    from tests.helpers import run_main

    run_main(["--config", _config(tmp_path / "full", 10)], monkeypatch)
    split = _config(tmp_path / "split", 5)
    run_main(["--config", split], monkeypatch)
    checkpoints = tmp_path / "split" / "checkpoints" / "cartpole" / "ppo"
    written = os.stat(checkpoints / "agent_episode_5.pt").st_mtime_ns
    run_main(
        [
            "--config",
            split,
            "--num-episodes",
            "10",
            "--resume",
            str(checkpoints / "agent_episode_5.pt"),
        ],
        monkeypatch,
    )
    for name in ("agent_episode_10.pt", "trainer_episode_10.npz"):
        _assert_same_file(tmp_path / "full" / "checkpoints" / "cartpole" / "ppo" / name,
                          checkpoints / name)  # fmt: skip
    # A fresh seeded run would reach the same checkpoint 10, but would rewrite checkpoint 5.
    assert os.stat(checkpoints / "agent_episode_5.pt").st_mtime_ns == written, "episodes 1-5 rerun"


def test_train_py_resume_from_a_missing_checkpoint_is_a_usage_error(tmp_path, monkeypatch, capsys):
    from tests.helpers import run_main

    with pytest.raises(SystemExit):
        run_main(["--config", _config(tmp_path / "a", 5), "--resume", str(tmp_path / "nope.pt")],
                 monkeypatch)  # fmt: skip
    assert "--resume: no checkpoint at" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# rl_core.training.checkpoints
# ---------------------------------------------------------------------------


def test_an_interrupted_write_leaves_no_file_under_the_real_name(tmp_path):
    from rl_core.training.checkpoints import write_atomically

    def crash(path):
        with open(path, "wb") as f:
            f.write(b"half")
        raise KeyboardInterrupt

    target = tmp_path / "agent_episode_5.pt"
    with pytest.raises(KeyboardInterrupt):
        write_atomically(str(target), crash)
    assert os.listdir(tmp_path) == [], "neither a truncated checkpoint nor the temporary file"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("agent_episode_40.pt", "trainer_episode_40.npz"),
        ("agent_episode_7.npz", "trainer_episode_7.npz"),
    ],
)
def test_the_trainer_file_is_named_after_the_agent_checkpoint(name, expected, tmp_path):
    from rl_core.training.checkpoints import trainer_state_path

    assert trainer_state_path(str(tmp_path / name)) == str(tmp_path / expected)
