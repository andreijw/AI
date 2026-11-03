"""Curriculum scheduling of the wrappers' noise and randomization between training episodes
(``rl_core.training.curriculum``; the trainer's ``curriculum`` argument; ``environment.curriculum``
in a config). The environment section is the target, which evaluation always plays."""

import contextlib
import json
import os

import gymnasium as gym
import numpy as np
import pytest

from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import RandomAgent
from rl_core.training import Trainer
from rl_core.training.curriculum import CurriculumScheduler, curriculum_from_config, interpolate
from rl_pendulum.environments import PendulumEnv

EPISODE_STEPS = 15
TARGET = {"obs_noise_std": 0.05, "domain_randomization": {"length": (0.25, 1.0)}}


def _difficulties(scheduler, rewards):
    """The difficulty before each episode, recording ``rewards`` one by one, and after the last."""
    seen = [scheduler.difficulty]
    for reward in rewards:
        scheduler.record(reward)
        seen.append(scheduler.difficulty)
    return seen


# ---------------------------------------------------------------------------
# Interpolation: the one rule, exact at both ends
# ---------------------------------------------------------------------------


def test_interpolate_a_scalar_by_hand():
    # (1 - 0.25) * 0.0 + 0.25 * 0.05 = 0.0125
    np.testing.assert_allclose(interpolate(0.0, 0.05, 0.25), 0.0125, rtol=1e-15)


def test_interpolate_a_range_moves_each_end_separately():
    # low: 0.5 * 0.5 + 0.5 * 0.25 = 0.375; high: 0.5 * 0.5 + 0.5 * 1.0 = 0.75
    got = interpolate({"length": (0.5, 0.5)}, {"length": (0.25, 1.0)}, 0.5)
    assert got == {"length": (0.375, 0.75)}, "each end of the range is interpolated on its own"


@pytest.mark.parametrize(("difficulty", "expected"), [(0.0, "start"), (1.0, "target")])
def test_interpolate_is_exact_at_both_ends(difficulty, expected):
    start, target = {"m": (0.1, 0.3), "l": (0.7, 0.7)}, {"m": (0.05, 0.2), "l": (0.1, 1.3)}
    got = interpolate(start, target, difficulty)
    assert got == (start if expected == "start" else target), f"d={difficulty} gives the {expected}"
    assert interpolate(0.1, 0.7, difficulty) == (0.1 if expected == "start" else 0.7)


# ---------------------------------------------------------------------------
# Pacing: each mode's difficulty sequence by hand
# ---------------------------------------------------------------------------


def _scheduler(**section):
    return CurriculumScheduler({"start": {"obs_noise_std": 0.0}, **section}, TARGET)


def test_linear_reaches_the_target_after_its_episodes_and_stays():
    scheduler = _scheduler(mode="linear", episodes=4)
    assert _difficulties(scheduler, [0.0] * 5) == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    assert scheduler.stage is None, "linear has no stages"


def test_step_climbs_in_equal_stages():
    # stage = floor(2 * episodes_done / 4): 0, 0, 1, 1, 2, then held at 2
    scheduler = _scheduler(mode="step", episodes=4, stages=2)
    assert _difficulties(scheduler, [0.0] * 5) == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
    assert scheduler.stage == 2


def test_reward_threshold_advances_on_the_stage_average_never_falls_and_waits_its_dwell():
    scheduler = _scheduler(
        mode="reward_threshold", stages=2, threshold=10.0, window=2, min_episodes_per_stage=3
    )
    rewards = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0]
    # Episodes 1-2 average 10 but have not dwelt 3 episodes; episode 3 advances. Episodes 4-7
    # average 0 at stage 1 and the difficulty holds; episode 8 makes the stage's last two (0, 20)
    # average 10 and advances to the target, where it stays.
    assert _difficulties(scheduler, rewards) == [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]


def test_reward_threshold_averages_only_the_current_stages_episodes():
    """The episodes before an advance count towards neither the next stage's average nor its
    dwell, so every stage plays at least min_episodes_per_stage episodes of its own."""
    scheduler = _scheduler(
        mode="reward_threshold", stages=3, threshold=10.0, window=2, min_episodes_per_stage=2
    )
    # Episode 2 advances. Episode 3 is stage 1's first: counting stage 0's episodes, its window
    # (10, 10) and dwell of 3 would advance again at once; on its own it waits for episode 4.
    assert _difficulties(scheduler, [10.0] * 4) == [0, 0, 1 / 3, 1 / 3, 2 / 3]


def test_record_reports_each_advance():
    scheduler = _scheduler(mode="step", episodes=4, stages=2)
    assert [scheduler.record(0.0) for _ in range(5)] == [False, True, False, True, False]


def test_settings_interpolate_every_scheduled_setting_and_keep_the_rest_at_the_target():
    target = {**TARGET, "action_noise_prob": 0.2}
    scheduler = CurriculumScheduler(
        {
            "mode": "linear",
            "episodes": 2,
            "start": {"obs_noise_std": 0.0, "domain_randomization": {"length": (0.5, 0.5)}},
        },
        target,
    )
    scheduler.record(0.0)
    assert scheduler.settings() == {
        "obs_noise_std": 0.025,
        "domain_randomization": {"length": (0.375, 0.75)},
        "action_noise_prob": 0.2,
    }


def test_a_start_randomization_may_name_some_of_the_targets_parameters():
    target = {"domain_randomization": {"length": (0.25, 1.0), "masspole": (0.05, 0.2)}}
    scheduler = CurriculumScheduler(
        {
            "mode": "linear",
            "episodes": 1,
            "start": {"domain_randomization": {"masspole": (0.1, 0.1)}},
        },
        target,
    )
    assert scheduler.settings() == {
        "domain_randomization": {"length": (0.25, 1.0), "masspole": (0.1, 0.1)}
    }, "an unnamed parameter stays at its target range, in the target's order"
    assert list(scheduler.settings()["domain_randomization"]) == ["length", "masspole"]


def test_replay_reaches_the_state_recording_did():
    rewards = [12.0, 3.0, 14.0, 15.0, 2.0, 16.0, 16.0]
    section = {"mode": "reward_threshold", "stages": 4, "threshold": 12.0, "window": 2,
               "min_episodes_per_stage": 2}  # fmt: skip
    recorded, replayed = _scheduler(**section), _scheduler(**section)
    for reward in rewards:
        recorded.record(reward)
    replayed.record(99.0)  # replay starts again from the start
    replayed.replay(rewards)
    assert (replayed.difficulty, replayed.stage) == (recorded.difficulty, recorded.stage)
    assert replayed.metrics() == recorded.metrics()
    for reward in (16.0, 16.0):
        assert replayed.record(reward) == recorded.record(reward)


def test_metrics_name_the_difficulty_and_the_stage():
    assert _scheduler(mode="linear", episodes=4).metrics() == {"curriculum_difficulty": 0.0}
    assert _scheduler(mode="step", episodes=4, stages=2).metrics() == {
        "curriculum_difficulty": 0.0,
        "curriculum_stage": 0,
    }


def test_the_spec_is_json_and_differs_for_a_different_curriculum():
    a = _scheduler(mode="step", episodes=4, stages=2)
    assert json.loads(a.spec())["mode"] == "step"
    assert a.spec() == _scheduler(mode="step", episodes=4, stages=2).spec()
    assert a.spec() != _scheduler(mode="step", episodes=4, stages=4).spec()
    other_target = CurriculumScheduler(
        {"mode": "step", "episodes": 4, "stages": 2, "start": {"obs_noise_std": 0.0}},
        {**TARGET, "obs_noise_std": 0.1},
    )
    assert a.spec() != other_target.spec(), "the target is part of the curriculum"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

LINEAR = {"mode": "linear", "episodes": 4, "start": {"obs_noise_std": 0.0}}
THRESHOLD = {"mode": "reward_threshold", "stages": 2, "threshold": 10.0, "window": 2,
             "min_episodes_per_stage": 2, "start": {"obs_noise_std": 0.0}}  # fmt: skip


@pytest.mark.parametrize(
    ("section", "match"),
    [
        ("linear", "environment.curriculum must be a mapping"),
        ({**LINEAR, "mode": "cosine"}, "curriculum.mode must be one of linear, step, reward_thr"),
        ({k: v for k, v in LINEAR.items() if k != "mode"}, "curriculum.mode must be one of"),
        ({**LINEAR, "warmup": 3}, "curriculum has unknown key"),
        (
            {**LINEAR, "start": {"domain_randomization": 0.5}},
            "start.domain_randomization must be a",
        ),
        ({**LINEAR, "start": {}}, "curriculum.start must be a mapping of at least one setting"),
        ({**LINEAR, "start": None}, "curriculum.start must be a mapping of at least one setting"),
        ({k: v for k, v in LINEAR.items() if k != "episodes"}, "curriculum.episodes is required"),
        ({**LINEAR, "stages": 2}, "curriculum.stages is not used by mode linear"),
        ({**LINEAR, "episodes": 0}, "curriculum.episodes must be a positive integer"),
        ({**LINEAR, "episodes": 2.5}, "curriculum.episodes must be a positive integer"),
        ({**LINEAR, "episodes": True}, "curriculum.episodes must be a positive integer"),
        ({**THRESHOLD, "threshold": float("nan")}, "curriculum.threshold must be a finite number"),
        ({**THRESHOLD, "threshold": "400"}, "curriculum.threshold must be a finite number"),
        ({**THRESHOLD, "window": 3}, r"min_episodes_per_stage \(2\) must be at least .*window \(3"),
        ({**THRESHOLD, "episodes": 4}, "curriculum.episodes is not used by mode reward_threshold"),
        ({**LINEAR, "start": {"action_noise_prob": 0.0}}, "action_noise_prob, which the environ"),
        (
            {**LINEAR, "start": {"domain_randomization": {"masscart": (1.0, 1.0)}}},
            "curriculum.start randomizes masscart, which the environment section does not",
        ),
        (
            {**LINEAR, "start": {"domain_randomization": {"length": 0.5}}},
            "range for 'length' must be",
        ),
    ],
)
def test_a_malformed_curriculum_is_refused(section, match):
    with pytest.raises(ValueError, match=match):
        CurriculumScheduler(section, TARGET)


# ---------------------------------------------------------------------------
# curriculum_from_config: the config section, checked against the environment
# ---------------------------------------------------------------------------


def _cartpole(**settings):
    defaults = {"seed": 3, "max_episode_steps": EPISODE_STEPS, "obs_noise_std": 0.05,
                "domain_randomization": {"length": (0.25, 1.0)}}  # fmt: skip
    return CartPoleEnv(**{**defaults, **settings})


@pytest.mark.parametrize("section", [None, {}])
def test_no_curriculum_section_gives_no_scheduler(section):
    with contextlib.closing(_cartpole()) as env:
        assert curriculum_from_config({"curriculum": section}, env) is None
        assert curriculum_from_config({}, env) is None


def test_the_target_is_the_environment_as_built():
    with contextlib.closing(_cartpole()) as env:
        scheduler = curriculum_from_config({"curriculum": dict(LINEAR, episodes=1)}, env)
        assert scheduler is not None
        scheduler.record(0.0)
        assert scheduler.settings() == env.scheduled_settings()


def test_a_start_setting_the_environment_cannot_schedule_is_refused():
    with contextlib.closing(_cartpole()) as env, pytest.raises(ValueError, match="frame_stack"):
        curriculum_from_config({"curriculum": {**LINEAR, "start": {"frame_stack": 1}}}, env)


def test_a_start_value_is_checked_by_the_wrappers_own_rule():
    start = {"domain_randomization": {"length": (0.9, 0.5)}}
    with contextlib.closing(_cartpole()) as env, pytest.raises(ValueError, match="min > max"):
        curriculum_from_config({"curriculum": {**LINEAR, "start": start}}, env)


def test_a_start_randomization_is_canonicalised_through_the_wrapper():
    """Pendulum accepts 'length' for its 'l': the start is matched to the target under 'l'."""
    with contextlib.closing(
        PendulumEnv(seed=3, domain_randomization={"l": (0.5, 1.5), "m": (0.5, 1.5)})
    ) as env:
        start = {"domain_randomization": {"Length": (1.0, 1.0)}}
        scheduler = curriculum_from_config({"curriculum": {**LINEAR, "start": start}}, env)
        assert scheduler is not None
        assert scheduler.settings() == {"domain_randomization": {"l": (1.0, 1.0), "m": (0.5, 1.5)}}


def test_an_environment_that_cannot_change_its_settings_is_refused():
    env = gym.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="cannot change its settings between episodes"):
            curriculum_from_config({"curriculum": LINEAR}, env)
    finally:
        env.close()


def test_the_config_reader_passes_the_curriculum_through():
    """Every make_env caller (record.py, export.py, the benchmark) builds the target and ignores
    the curriculum; only the trainer's scripts read it."""
    from rl_cartpole.environments.make_env import make_env_from_config

    env = make_env_from_config({"name": "CartPole-v1", "obs_noise_std": 0.05,
                                "curriculum": LINEAR})  # fmt: skip
    with contextlib.closing(env):
        assert env.scheduled_settings() == {"obs_noise_std": 0.05}


# ---------------------------------------------------------------------------
# The trainer: settings before each training episode, evaluation at the target
# ---------------------------------------------------------------------------


class _Logger:
    def __init__(self):
        self.records, self.lines, self.warnings = [], [], []

    def log(self, metrics):
        self.records.append(metrics)

    def info(self, message):
        self.lines.append(message)

    def warning(self, message):
        self.warnings.append(message)


class _ResetsRecorded(CartPoleEnv):
    """A CartPole wrapper that keeps its scheduled settings at every reset, and every
    apply_settings call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen, self.applied = [], 0

    def reset(self, **kwargs):
        self.seen.append(self.scheduled_settings())
        return super().reset(**kwargs)

    def apply_settings(self, settings):
        self.applied += 1
        super().apply_settings(settings)


def _recorded(**settings):
    defaults = {"seed": 3, "max_episode_steps": EPISODE_STEPS, "obs_noise_std": 0.05,
                "domain_randomization": {"length": (0.25, 1.0)}}  # fmt: skip
    return _ResetsRecorded(**{**defaults, **settings})


def _trainer(env, eval_env, section, directory, num_episodes=5, **training):
    config = {"num_episodes": num_episodes, "max_steps_per_episode": EPISODE_STEPS,
              "eval_frequency": 5, "save_frequency": 5, "checkpoint_dir": str(directory),
              **training}  # fmt: skip
    curriculum = None if section is None else curriculum_from_config({"curriculum": section}, env)
    return Trainer(
        env=env,
        agent=RandomAgent(4, 2, {"seed": 7}),
        config=config,
        logger=_Logger(),
        eval_env=eval_env,
        eval_seed=1000,
        curriculum=curriculum,
    )


def test_each_training_episode_starts_at_the_scheduled_difficulty(tmp_path):
    env, eval_env = _recorded(), _recorded()
    try:
        _trainer(env, eval_env, LINEAR, tmp_path).train()
        noise = [seen["obs_noise_std"] for seen in env.seen]
        np.testing.assert_allclose(noise, [0.0, 0.0125, 0.025, 0.0375, 0.05], rtol=1e-12)
        assert all(seen == eval_env.seen[0] for seen in eval_env.seen)
        assert eval_env.seen[0] == TARGET, "evaluation always plays the target"
    finally:
        env.close()
        eval_env.close()


def test_without_a_curriculum_the_settings_are_never_touched(tmp_path):
    env, eval_env = _recorded(), _recorded()
    try:
        _trainer(env, eval_env, None, tmp_path).train()
        assert env.applied == 0 and eval_env.applied == 0
    finally:
        env.close()
        eval_env.close()


def test_a_curriculum_whose_start_is_its_target_trains_the_run_with_none(tmp_path):
    runs = []
    for section in (None, {**LINEAR, "start": TARGET}):
        env, eval_env = _cartpole(), _cartpole()
        try:
            trainer = _trainer(env, eval_env, section, tmp_path / str(len(runs)), num_episodes=10)
            stats = trainer.train()
            stats.pop("curriculum_difficulty", None)
            records = [
                {k: v for k, v in r.items() if not k.startswith("curriculum")}
                for r in trainer.logger.records
            ]
            runs.append((trainer.episode_rewards, trainer.episode_lengths, stats, records))
        finally:
            env.close()
            eval_env.close()
    assert runs[0][:3] == runs[1][:3]
    assert runs[0][3] == runs[1][3]
    with (
        np.load(tmp_path / "0" / "agent_episode_10.pt") as a,
        np.load(tmp_path / "1" / "agent_episode_10.pt") as b,
    ):
        for key in a.files:
            np.testing.assert_array_equal(b[key], a[key], err_msg=key)


def test_the_periodic_record_carries_the_difficulty_and_an_advance_is_logged(tmp_path):
    env, eval_env = _cartpole(), _cartpole()
    try:
        trainer = _trainer(env, eval_env, {**LINEAR, "mode": "step", "stages": 2, "episodes": 10},
                           tmp_path, num_episodes=20)  # fmt: skip
        trainer.train()
    finally:
        env.close()
        eval_env.close()
    periodic = [r for r in trainer.logger.records if "episode" in r]
    assert [(r["curriculum_difficulty"], r["curriculum_stage"]) for r in periodic] == [
        (0.5, 1),
        (1.0, 2),
    ], "each 10-episode record has the difficulty its last episode trained at"
    advances = [line for line in trainer.logger.lines if line.startswith("Curriculum")]
    assert advances == [
        "Curriculum stage 1/2 (difficulty 0.50) from episode 6",
        "Curriculum stage 2/2 (difficulty 1.00) from episode 11",
    ]
    assert not trainer.logger.warnings


def test_a_run_that_ends_below_the_target_warns(tmp_path):
    env, eval_env = _cartpole(), _cartpole()
    try:
        trainer = _trainer(env, eval_env, {**LINEAR, "episodes": 10}, tmp_path, num_episodes=5)
        stats = trainer.train()
    finally:
        env.close()
        eval_env.close()
    assert stats["curriculum_difficulty"] == 0.4, "the last episode trained at 4/10"
    assert trainer.logger.warnings == [
        "Training ended at curriculum difficulty 0.40, below the target its evaluation played"
    ]


def test_a_curriculum_needs_its_own_evaluation_environment():
    env = _cartpole()
    try:
        with pytest.raises(ValueError, match="needs an evaluation environment"):
            Trainer(
                env=env,
                agent=RandomAgent(4, 2, {"seed": 7}),
                config={"num_episodes": 2},
                curriculum=curriculum_from_config({"curriculum": LINEAR}, env),
            )
    finally:
        env.close()


def test_a_curriculum_needs_an_environment_that_can_change_its_settings():
    env, eval_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="cannot change its settings between episodes"):
            Trainer(
                env=env,
                agent=RandomAgent(4, 2, {"seed": 7}),
                config={"num_episodes": 2},
                eval_env=eval_env,
                eval_seed=1000,
                curriculum=CurriculumScheduler(LINEAR, TARGET),
            )
    finally:
        env.close()
        eval_env.close()


# ---------------------------------------------------------------------------
# Resume: the spec is saved, the difficulty replayed from the rewards
# ---------------------------------------------------------------------------

THRESHOLD_RUN = {"mode": "reward_threshold", "stages": 3, "threshold": 14.0, "window": 2,
                 "min_episodes_per_stage": 2, "start": {"obs_noise_std": 0.0}}  # fmt: skip


def _run(directory, section, num_episodes, resume_from=None):
    env, eval_env = _cartpole(), _cartpole()
    try:
        trainer = _trainer(env, eval_env, section, directory, num_episodes=num_episodes)
        if resume_from is not None:
            trainer.resume(str(resume_from))
        trainer.train()
        return trainer
    finally:
        env.close()
        eval_env.close()


def test_a_resumed_curriculum_continues_the_difficulty_an_unbroken_run_has(tmp_path):
    full = _run(tmp_path / "full", THRESHOLD_RUN, 10)
    first = _run(tmp_path / "split", THRESHOLD_RUN, 5)
    resumed = _run(tmp_path / "split", THRESHOLD_RUN, 10, tmp_path / "split" / "agent_episode_5.pt")

    stages = [r["curriculum_stage"] for r in full.logger.records if "episode" in r]
    assert stages and stages[-1] > 0, "the fixture must advance at least once to test anything"
    assert resumed.episode_rewards == full.episode_rewards, "the resumed run trains differently"
    assert first.logger.records + resumed.logger.records == full.logger.records, (
        "the resumed run's records (difficulty and stage included) differ from the unbroken run's"
    )
    with (
        np.load(tmp_path / "full" / "trainer_episode_10.npz") as a,
        np.load(tmp_path / "split" / "trainer_episode_10.npz") as b,
    ):
        assert sorted(a.files) == sorted(b.files), "the two trainer files hold different keys"
        for key in a.files:
            np.testing.assert_array_equal(b[key], a[key], err_msg=f"trainer file: {key}")


def test_a_resume_with_nothing_left_to_train_reports_the_difficulty_last_played(tmp_path):
    """The run ended at episode 5 having played 4/10, and the scheduler then stood at 5/10: the
    resumed trainer reports and warns about what was played, as the stopped one did."""
    section = {**LINEAR, "episodes": 10}
    stopped = _run(tmp_path, section, 5)
    env, eval_env = _cartpole(), _cartpole()
    try:
        resumed = _trainer(env, eval_env, section, tmp_path, num_episodes=5)
        resumed.resume(str(tmp_path / "agent_episode_5.pt"))
        stats = resumed.train()
    finally:
        env.close()
        eval_env.close()
    assert stats["curriculum_difficulty"] == 0.4, "the last episode played 4/10, not the next 5/10"
    assert resumed.logger.warnings == stopped.logger.warnings


@pytest.mark.parametrize(
    ("saved", "resumed", "match"),
    [
        (THRESHOLD_RUN, {**THRESHOLD_RUN, "threshold": 10.0}, "a different curriculum"),
        (THRESHOLD_RUN, None, "trained with a curriculum; give this trainer the same one"),
        (None, THRESHOLD_RUN, "trained without a curriculum"),
    ],
)
def test_a_resume_with_a_changed_curriculum_is_refused(saved, resumed, match, tmp_path):
    _run(tmp_path, saved, 5)
    env, eval_env = _cartpole(), _cartpole()
    try:
        trainer = _trainer(env, eval_env, resumed, tmp_path, num_episodes=10)
        with pytest.raises(ValueError, match=match):
            trainer.resume(str(tmp_path / "agent_episode_5.pt"))
        assert trainer.episode_rewards == [], "nothing is restored from a refused checkpoint"
    finally:
        env.close()
        eval_env.close()


# ---------------------------------------------------------------------------
# train.py and the benchmark build the curriculum from the config
# ---------------------------------------------------------------------------


def test_train_py_trains_through_the_configs_curriculum(tmp_path, monkeypatch):
    from tests.helpers import minimal_config, run_main

    curriculum = {"mode": "step", "episodes": 10, "stages": 2, "start": {"obs_noise_std": 0.0}}
    path = minimal_config(
        tmp_path,
        environment={"obs_noise_std": 0.05, "curriculum": curriculum},
        training={"num_episodes": 20, "eval_frequency": 10},
    )
    run_main(["--config", path], monkeypatch)
    log_dir = tmp_path / "logs" / "cartpole" / "random"
    (metrics_file,) = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    with open(log_dir / metrics_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    difficulties = [r["curriculum_difficulty"] for r in records if "curriculum_difficulty" in r]
    assert difficulties == [0.5, 1.0], "the difficulty rises to the target over the run"


def test_the_benchmark_trains_each_run_through_its_configs_curriculum(tmp_path, monkeypatch):
    import benchmark
    from benchmark import BenchmarkRunner, BenchmarkSettings
    from rl_cartpole.benchmark import CartPoleChallenge

    (tmp_path / CartPoleChallenge.CONFIG_MAP["random"]).write_text(
        "environment:\n  obs_noise_std: 0.05\n  curriculum:\n    mode: linear\n    episodes: 4\n"
        "    start: {obs_noise_std: 0.0}\nagent:\n  config: {}\n",
        encoding="utf-8",
    )
    built = []

    class _Seen(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self.curriculum)

    monkeypatch.setattr(benchmark, "Trainer", _Seen)
    BenchmarkRunner(CartPoleChallenge(config_dir=str(tmp_path))).run(
        settings=BenchmarkSettings(
            agents=("random",),
            seeds=(1,),
            episodes=4,
            eval_episodes=1,
            output_dir=str(tmp_path / "out"),
        ),
        report_path=str(tmp_path / "report.md"),
        save_plot=False,
    )
    (curriculum,) = built
    assert curriculum is not None and curriculum.difficulty == 1.0
