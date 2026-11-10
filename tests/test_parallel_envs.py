"""Training from several environments in lockstep (``training.num_envs``; parallel environments, PR 1).

Scripted environments make each one's episode lengths, ends and observations known, so a test can
say which steps every update must hold, in which order the episodes are recorded, and how many steps
each environment is asked for. Real CartPole runs cover determinism, the normaliser, checkpoints,
resume and the curriculum. Training with one environment is pinned by tests/test_trainer_golden.py.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_core.agents import (
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DDPGAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
    SACAgent,
    TD3Agent,
)
from rl_core.training import Trainer
from rl_core.training.curriculum import curriculum_from_config
from tests.trainer_internals import force_lockstep

EVAL_SEED = 900


class _LenEnv:
    """Episodes of the given lengths in turn. The observation is ``[env_id, episode, step]``; the
    last step terminates (or, with ``truncate``, is cut by the time limit) and reports success when
    the episode number is even. Counts every ``step`` and ``reset`` it is asked for."""

    def __init__(self, env_id: int, lengths: list[int], truncate: bool = False, render_mode=None):
        self.env_id, self.lengths, self.truncate = env_id, lengths, truncate
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float64)
        self.action_space = gym.spaces.Discrete(2)
        self.episode, self.t = -1, 0
        self.steps = self.resets = 0

    def _obs(self) -> np.ndarray:
        return np.array([self.env_id, self.episode, self.t], dtype=np.float64)

    def reset(self, seed=None, options=None):
        self.resets += 1
        self.episode += 1
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        self.t += 1
        end = self.t >= self.lengths[self.episode % len(self.lengths)]
        info = {"is_success": self.episode % 2 == 0} if end else {}
        return self._obs(), 1.0, end and not self.truncate, end and self.truncate, info

    def close(self):
        return None


class _Recorder(RandomAgent):
    """A random agent that acts 0 and keeps every observation it acted on and every batch."""

    def __init__(self) -> None:
        super().__init__(3, 2, {"seed": 0})
        self.acted: list[np.ndarray] = []
        self.batches: list[dict[str, Any]] = []

    def select_action(self, observation, training=True):
        if training:
            self.acted.append(np.asarray(observation).copy())
        return 0

    def update(self, batch):
        self.batches.append(batch)
        return {}


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, metrics):
        return None

    def info(self, message):
        self.messages.append(message)

    def warning(self, message):
        self.messages.append(message)


def _scripted(lengths_per_env, agent=None, truncate=False, **training):
    envs = [_LenEnv(i, lengths, truncate) for i, lengths in enumerate(lengths_per_env)]
    config = {"eval_frequency": 1000, "save_frequency": 1000, "num_envs": len(envs), **training}
    trainer = Trainer(
        env=envs[0],
        agent=agent or _Recorder(),
        config=config,
        logger=_Logger(),
        eval_env=_LenEnv(99, [2]),
        eval_seed=EVAL_SEED,
        extra_envs=envs[1:],
    )
    return trainer, envs


# ----------------------------------------------------------------------------------------------
# Configuration and refusals
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, 0, -1, "2", 2.5])
def test_num_envs_is_read_strictly(value):
    with pytest.raises(ValueError, match="num_envs"):
        Trainer(env=_LenEnv(0, [3]), agent=_Recorder(), config={"num_envs": value})


def test_the_number_of_environments_must_match_num_envs():
    with pytest.raises(ValueError, match="num_envs"):
        Trainer(env=_LenEnv(0, [3]), agent=_Recorder(), config={"num_envs": 2})
    with pytest.raises(ValueError, match="num_envs"):
        Trainer(env=_LenEnv(0, [3]), agent=_Recorder(), config={}, extra_envs=[_LenEnv(1, [3])])


def test_several_environments_need_an_evaluation_environment():
    with pytest.raises(ValueError, match="eval_env"):
        Trainer(
            env=_LenEnv(0, [3]),
            agent=_Recorder(),
            config={"num_envs": 2},
            extra_envs=[_LenEnv(1, [3])],
        )


def test_update_every_is_refused_with_several_environments():
    with pytest.raises(ValueError, match="update_every"):
        _scripted([[3], [3]], update_every=2)


def test_a_rendering_environment_is_refused_with_several_environments():
    envs = [_LenEnv(0, [3], render_mode="rgb_array"), _LenEnv(1, [3])]
    with pytest.raises(ValueError, match="render"):
        Trainer(
            env=envs[0],
            agent=_Recorder(),
            config={"num_envs": 2},
            eval_env=_LenEnv(9, [2]),
            eval_seed=EVAL_SEED,
            extra_envs=envs[1:],
        )


ACCEPTING = [
    (ReinforceAgent, 4, 2, {}),
    (ActorCriticAgent, 4, 2, {}),
    (ContinuousActorCriticAgent, 3, 1, {}),
    (RandomAgent, 4, 2, {}),
    (ContinuousRandomAgent, 3, 1, {}),
]
REFUSING = [
    (PPOAgent, 4, 2, {}),
    (ContinuousPPOAgent, 3, 1, {}),
    (DQNAgent, 4, 2, {}),
    (DDPGAgent, 3, 1, {}),
    (TD3Agent, 3, 1, {}),
    (SACAgent, 3, 1, {}),
]


@pytest.mark.parametrize(
    ("cls", "obs", "act", "config"), ACCEPTING, ids=lambda v: getattr(v, "__name__", "")
)
def test_the_episode_agents_accept_several_environments(cls, obs, act, config):
    cls(obs, act, {"seed": 1, **config}).enable_parallel_envs(4)


@pytest.mark.parametrize(
    ("cls", "obs", "act", "config"), REFUSING, ids=lambda v: getattr(v, "__name__", "")
)
def test_ppo_and_the_replay_buffer_agents_refuse_several_environments(cls, obs, act, config):
    with pytest.raises(ValueError, match="num_envs"):
        cls(obs, act, {"seed": 1, **config}).enable_parallel_envs(2)


def test_the_trainer_asks_the_agent_before_training():
    with pytest.raises(ValueError, match="num_envs"):
        envs = [CartPoleEnv(seed=1), CartPoleEnv(seed=2)]
        Trainer(
            env=envs[0],
            agent=PPOAgent(4, 2, {"seed": 1}),
            config={"num_envs": 2},
            eval_env=CartPoleEnv(seed=3),
            eval_seed=EVAL_SEED,
            extra_envs=envs[1:],
        )


# ----------------------------------------------------------------------------------------------
# What each update holds, and when
# ----------------------------------------------------------------------------------------------


def _finish_order(lengths_per_env, num_episodes):
    """(env, episode, length) of each episode in the order lockstep collection finishes them."""
    clocks = [0] * len(lengths_per_env)
    episodes = []
    for env_id, lengths in enumerate(lengths_per_env):
        for k in range(num_episodes):
            clocks[env_id] += lengths[k % len(lengths)]
            episodes.append((clocks[env_id], env_id, k, lengths[k % len(lengths)]))
    episodes.sort()  # by the vector step it ends at, then by env index
    return [(env, k, n) for _, env, k, n in episodes]


def test_each_update_is_one_environments_whole_episode_in_finish_order():
    lengths = [[3, 5], [2], [4, 1]]
    trainer, _ = _scripted(lengths, num_episodes=7)
    trainer.train()
    agent = trainer.agent
    expected = _finish_order(lengths, 7)[:7]
    handed = [
        (int(b["observations"][0, 0]), int(b["observations"][0, 1]), len(b["rewards"]))
        for b in agent.batches
    ]
    assert handed == expected, "updates should come one finished episode at a time, in finish order"
    for batch in agent.batches:
        obs, nxt = batch["observations"], batch["next_observations"]
        assert (obs[:, 0] == obs[0, 0]).all() and (obs[:, 1] == obs[0, 1]).all(), (
            "a batch should hold one environment's one episode"
        )
        np.testing.assert_array_equal(obs[:, 2], np.arange(len(obs)), "steps in order, from 0")
        assert nxt[-1, 2] == len(obs) and nxt[-1, 1] == obs[0, 1], (
            "the last next observation should be the episode's final one, not the next reset"
        )
        assert batch["terminated"][-1] and batch["dones"][-1]
    assert trainer.episode_lengths == [n for _, _, n in expected]


def test_a_time_limit_and_a_trainer_cut_are_not_terminations():
    trainer, _ = _scripted([[3], [9]], truncate=True, num_episodes=4, max_steps_per_episode=5)
    trainer.train()
    for batch in trainer.agent.batches:
        assert not batch["terminated"][-1], "a time limit or a trainer cut should bootstrap"
    cut = [b for b in trainer.agent.batches if len(b["rewards"]) == 5]
    assert cut and not cut[0]["dones"][-1], "the trainer's cut leaves done False, as with one env"


def test_actions_are_chosen_in_environment_order_every_vector_step():
    trainer, _ = _scripted([[4], [4], [4]], num_episodes=3)
    trainer.train()
    env_ids = [int(o[0]) for o in trainer.agent.acted]
    assert env_ids == [0, 1, 2] * 4


def test_each_episode_records_its_own_success():
    trainer, _ = _scripted([[2], [3]], num_episodes=4)
    trainer.train()
    # env 0 ends episodes 0, 1 at steps 2, 4; env 1 ends episode 0 at step 3: order e0k0, e1k0, e0k1
    assert trainer.episode_successes == [True, True, False, False]


# ----------------------------------------------------------------------------------------------
# Budgets
# ----------------------------------------------------------------------------------------------


def test_num_episodes_is_exact_and_idle_environments_are_not_stepped():
    trainer, envs = _scripted([[2], [7], [3]], num_episodes=5)
    trainer.train()
    assert len(trainer.episode_rewards) == 5
    assert sum(env.steps for env in envs) == sum(trainer.episode_lengths), (
        "every step asked of an environment should belong to a recorded episode"
    )
    assert trainer.total_steps_trained == sum(trainer.episode_lengths)


def test_a_step_budget_starts_no_episode_once_met_and_finishes_those_in_flight():
    trainer, envs = _scripted([[2], [9]], total_steps=8)
    trainer.train()
    # Env 0 ends at vector steps 2 and 4. At step 3 it restarts (2 done + 2 in flight < 8); at
    # step 5 it does not (4 done + 4 in flight of env 1's episode = 8). Env 1 plays its 9 steps.
    assert trainer.episode_lengths == [2, 2, 9]
    assert [env.steps for env in envs] == [4, 9]
    assert trainer.total_steps_trained == 13


@pytest.mark.parametrize("budget", [{"total_steps": 11}, {"num_episodes": 4}])
def test_lockstep_at_one_environment_keeps_the_budget_rule(budget):
    runs = []
    for lockstep in (False, True):
        trainer, envs = _scripted([[3, 5]], **budget)
        if lockstep:
            force_lockstep(trainer)
        trainer.train()
        runs.append((trainer.episode_lengths, envs[0].steps, envs[0].resets))
    assert runs[0] == runs[1], "collected plus in-flight steps should reduce to today's rule"


# ----------------------------------------------------------------------------------------------
# Evaluation, checkpoints and resume
# ----------------------------------------------------------------------------------------------


def test_evaluation_and_saving_run_once_at_the_end_of_the_crossing_vector_step(tmp_path):
    envs = [_LenEnv(i, [3]) for i in range(2)]
    logger = _Logger()
    trainer = Trainer(
        env=envs[0],
        agent=_Recorder(),
        config={
            "num_envs": 2,
            "num_episodes": 6,
            "eval_frequency": 3,
            "save_frequency": 3,
            "checkpoint_dir": str(tmp_path),
        },
        logger=logger,
        eval_env=_LenEnv(9, [2]),
        eval_seed=EVAL_SEED,
        extra_envs=envs[1:],
        save_final=True,
    )
    trainer.train()
    evaluations = [m for m in logger.messages if m.startswith("Evaluation at episode")]
    # Pairs finish together: 2, 4, 6. Multiples of 3 are crossed at 4 and reached at 6.
    assert [m.split(":")[0] for m in evaluations] == [
        "Evaluation at episode 4",
        "Evaluation at episode 6",
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "agent_episode_4.pt",
        "agent_episode_6.pt",
        "trainer_episode_4.npz",
        "trainer_episode_6.npz",
    ], "checkpoints should be named by the episodes recorded when they were saved"


def _cartpole_run(tmp_path, num_envs, seed=1, resume=None, **training):
    envs = [CartPoleEnv(seed=seed if i == 0 else seed * 100 + i) for i in range(num_envs)]
    eval_env = CartPoleEnv(seed=77)
    agent = ActorCriticAgent(
        4, 2, {"seed": 5, "hidden_dim": 8, "learning_rate": 0.02, "normalize_observations": True}
    )
    config = {
        "num_envs": num_envs,
        "num_episodes": 8,
        "eval_frequency": 4,
        "save_frequency": 4,
        "checkpoint_dir": str(tmp_path),
        **training,
    }
    trainer = Trainer(
        env=envs[0],
        agent=agent,
        config=config,
        logger=_Logger(),
        eval_env=eval_env,
        eval_seed=EVAL_SEED,
        extra_envs=envs[1:],
    )
    try:
        if resume is not None:
            trainer.resume(str(resume))
        trainer.train()
    finally:
        for env in (*envs, eval_env):
            env.close()
    return trainer


def _checkpoint_arrays(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def test_two_runs_with_one_seed_are_identical(tmp_path):
    first = _cartpole_run(tmp_path / "a", 3)
    second = _cartpole_run(tmp_path / "b", 3)
    assert first.episode_rewards == second.episode_rewards
    a = _checkpoint_arrays(next((tmp_path / "a").glob("agent_episode_8*")))
    b = _checkpoint_arrays(next((tmp_path / "b").glob("agent_episode_8*")))
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_the_normaliser_sees_every_training_observation_from_every_environment(tmp_path):
    trainer = _cartpole_run(tmp_path, 3)
    normalizer = trainer.agent.observation_normalizer
    assert normalizer is not None
    episodes = len(trainer.episode_lengths)
    # Every step's next observation, and each start state (resets of envs that finished and were
    # not restarted are not made). Evaluation reads the statistics without adding to them.
    resets = episodes  # every recorded episode started with one reset; none is left in flight
    # The statistics start from a count of 1e-4 (RunningMeanStd's prior).
    assert normalizer.count == pytest.approx(sum(trainer.episode_lengths) + resets, abs=1e-3)


def test_resume_restores_every_environment_and_two_resumes_agree(tmp_path):
    _cartpole_run(tmp_path / "run", 2)
    checkpoint = next((tmp_path / "run").glob("agent_episode_4*"))
    trainer_file = tmp_path / "run" / "trainer_episode_4.npz"
    keys = set(np.load(trainer_file).files)
    assert "num_envs" in keys and {"env1_env_start_rng", "env1_env_wrapper_rng"} <= keys
    first = _cartpole_run(tmp_path / "r1", 2, resume=checkpoint)
    second = _cartpole_run(tmp_path / "r2", 2, resume=checkpoint)
    assert first.episode_rewards[:4] == _cartpole_run(tmp_path / "again", 2).episode_rewards[:4]
    assert len(first.episode_rewards) == 8
    assert first.episode_rewards == second.episode_rewards, "a resume should be reproducible"


def test_resume_refuses_another_number_of_environments(tmp_path):
    _cartpole_run(tmp_path / "run", 2)
    checkpoint = next((tmp_path / "run").glob("agent_episode_4*"))
    with pytest.raises(ValueError, match="num_envs"):
        _cartpole_run(tmp_path / "three", 3, resume=checkpoint)


def test_a_one_environment_trainer_file_gains_no_key(tmp_path):
    _cartpole_run(tmp_path, 1)
    keys = set(np.load(tmp_path / "trainer_episode_4.npz").files)
    assert "num_envs" not in keys and not any(k.startswith("env1_") for k in keys)


# ----------------------------------------------------------------------------------------------
# Curriculum
# ----------------------------------------------------------------------------------------------


class _Scheduled(CartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.applied: list[float] = []

    def apply_settings(self, settings):
        self.applied.append(float(settings["obs_noise_std"]))
        super().apply_settings(settings)

    def reset(self, seed=None, options=None):
        self.reset_count = getattr(self, "reset_count", 0) + 1
        return super().reset(seed=seed, options=options)


def test_each_environment_takes_the_curriculum_settings_before_each_of_its_resets(tmp_path):
    envs = [_Scheduled(seed=1 + i, obs_noise_std=0.05) for i in range(2)]
    section = {"mode": "linear", "start": {"obs_noise_std": 0.0}, "episodes": 4}
    trainer = Trainer(
        env=envs[0],
        agent=ReinforceAgent(4, 2, {"seed": 2}),
        config={"num_envs": 2, "num_episodes": 6, "eval_frequency": 100, "save_frequency": 100},
        logger=_Logger(),
        eval_env=CartPoleEnv(seed=9, obs_noise_std=0.05),
        eval_seed=EVAL_SEED,
        curriculum=curriculum_from_config({"curriculum": section}, envs[0]),
        extra_envs=envs[1:],
    )
    trainer.train()
    for env in envs:
        assert len(env.applied) == env.reset_count, "settings before every reset of every env"
    assert envs[0].applied[0] == 0.0 and max(envs[0].applied + envs[1].applied) > 0.0
    assert trainer.curriculum is not None
    assert trainer.curriculum.difficulty == 1.0, "every finished episode is recorded once"


# ----------------------------------------------------------------------------------------------
# Seeds, and building the environments
# ----------------------------------------------------------------------------------------------


def test_env_seeds_keep_the_run_seed_first_and_are_distinct_and_small():
    from rl_core.seeding import env_seed

    assert env_seed(7, 0) == 7
    assert env_seed(None, 3) is None
    seeds = [env_seed(7, i) for i in range(1, 9)]
    assert len(set(seeds)) == 8 and all(0 <= s < 2**31 for s in seeds)
    assert env_seed(7, 1) != env_seed(8, 0), "run 7's second env must not replay run 8's first"
    assert env_seed(7, 2) == env_seed(7, 2)


def test_the_challenge_builds_the_extra_environments_without_changing_the_section():
    from rl_cartpole.benchmark import CartPoleChallenge
    from rl_core.seeding import env_seed

    section = {"name": "CartPole-v1", "seed": 3, "max_episode_steps": 50}
    envs = CartPoleChallenge().make_extra_envs(section, 3)
    try:
        assert [env.seed for env in envs] == [env_seed(3, 1), env_seed(3, 2)]
        assert all(env.render_mode is None for env in envs)
        assert section == {"name": "CartPole-v1", "seed": 3, "max_episode_steps": 50}
    finally:
        for env in envs:
            env.close()


def test_single_env_training_drops_num_envs():
    from rl_core.training.parallel import single_env_training

    training = {"num_envs": 8, "num_episodes": 3}
    assert single_env_training(training) == {"num_episodes": 3}
    assert training == {"num_envs": 8, "num_episodes": 3}


# ----------------------------------------------------------------------------------------------
# train.py
# ----------------------------------------------------------------------------------------------


def test_train_py_trains_from_several_environments(tmp_path, monkeypatch):
    from tests.helpers import minimal_config, run_main

    config = minimal_config(
        tmp_path,
        agent={"type": "reinforce", "config": {"seed": 3, "hidden_dim": 8}},
        training={"num_envs": 3, "num_episodes": 6, "save_frequency": 6},
    )
    run_main(["--config", config], monkeypatch)
    trainer_files = sorted((tmp_path / "checkpoints").rglob("trainer_episode_*.npz"))
    assert trainer_files, "train.py should save a checkpoint"
    with np.load(trainer_files[-1]) as data:
        assert int(data["num_envs"]) == 3 and "env2_env_start_rng" in data.files


def test_train_py_refuses_to_render_several_environments(tmp_path, monkeypatch):
    from tests.helpers import minimal_config, run_main

    config = minimal_config(
        tmp_path, environment={"render_mode": "rgb_array"}, training={"num_envs": 2}
    )
    with pytest.raises(ValueError, match="render"):
        run_main(["--config", config], monkeypatch)


# ----------------------------------------------------------------------------------------------
# The benchmark
# ----------------------------------------------------------------------------------------------


def _benchmark_report(tmp_path, num_envs):
    import shutil
    from pathlib import Path

    import yaml

    from benchmark import BenchmarkRunner, BenchmarkSettings
    from rl_cartpole.benchmark import CartPoleChallenge

    configs = tmp_path / f"configs-{num_envs}"
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", configs)
    path = configs / "cartpole_reinforce.yaml"
    config = yaml.safe_load(path.read_text())
    config["training"]["num_envs"] = num_envs
    config["agent"]["config"]["hidden_dim"] = 8
    path.write_text(yaml.safe_dump(config))
    report = tmp_path / f"report-{num_envs}.md"
    BenchmarkRunner(CartPoleChallenge(config_dir=str(configs))).run(
        settings=BenchmarkSettings(
            agents=("reinforce",),
            seeds=(1,),
            steps=200,
            eval_episodes=1,
            output_dir=str(tmp_path / f"out-{num_envs}"),
        ),
        report_path=str(report),
        save_plot=False,
    )
    return report.read_text(encoding="utf-8")


def test_the_benchmark_report_states_the_overshoot_of_several_environments(tmp_path):
    one = _benchmark_report(tmp_path, 1)
    three = _benchmark_report(tmp_path, 3)
    single = "overshoot its budget by up to one episode."
    assert single in one and "N - 1 more episodes" not in one, "one environment: as before"
    assert single in three
    assert "(`training.num_envs`, reinforce: 3)" in three and "up to N - 1 more episodes" in three


def test_setup_extra_envs_builds_a_benchmark_runs_other_environments(tmp_path):
    from rl_cartpole.benchmark import CartPoleChallenge
    from rl_core.seeding import env_seed

    challenge = CartPoleChallenge()
    assert challenge.setup_extra_envs("reinforce", 4, 1) == []
    envs = challenge.setup_extra_envs("reinforce", 4, 3)
    try:
        assert [env.seed for env in envs] == [env_seed(4, 1), env_seed(4, 2)]
        first, _, _ = challenge.setup_run("reinforce", 4, None)
        try:
            assert first.seed == 4, "the run's first environment keeps the run seed"
        finally:
            first.close()
    finally:
        for env in envs:
            env.close()


def test_the_robustness_rows_evaluate_a_config_that_trains_several_environments():
    from rl_cartpole.benchmark import CartPoleChallenge
    from rl_core.training.robustness import robustness_rows

    challenge = CartPoleChallenge()
    config = {
        "environment": {"name": "CartPole-v1", "seed": 3, "max_episode_steps": 30},
        "training": {"num_envs": 4, "num_episodes": 10},
    }
    env = challenge.make_env(config["environment"], render_mode=None)
    try:
        agent = challenge.make_agent("reinforce", {"seed": 1}, env)
    finally:
        env.close()
    rows = robustness_rows(challenge, config, agent, [], episodes=2, eval_seed=11)
    assert len(rows) == 1, "one environment evaluates the nominal row; num_envs is not asked for"


class _NaNOnEnv1(_Recorder):
    """Returns a non-finite metric for every episode of environment 1."""

    def update(self, batch):
        super().update(batch)
        return {"loss": float("nan") if batch["observations"][0, 0] == 1 else 0.0}


def test_a_non_finite_metric_names_the_episode_that_produced_it():
    trainer, _ = _scripted([[2], [2]], agent=_NaNOnEnv1(), num_episodes=2)
    trainer.train()
    # Both end at vector step 2: env 0's episode is recorded as 1, env 1's as 2.
    assert trainer.first_non_finite_episode == 2


def test_episodes_in_flight_at_a_checkpoint_are_dropped_on_resume(tmp_path):
    def build(lengths):
        envs = [_LenEnv(i, n) for i, n in enumerate(lengths)]
        trainer = Trainer(
            env=envs[0],
            agent=_Recorder(),
            config={
                "num_envs": 2,
                "num_episodes": 3,
                "eval_frequency": 100,
                "save_frequency": 1,
                "checkpoint_dir": str(tmp_path),
            },
            logger=_Logger(),
            eval_env=_LenEnv(9, [2]),
            eval_seed=EVAL_SEED,
            extra_envs=envs[1:],
        )
        return trainer, envs

    first, _ = build([[2], [5]])
    first.train()  # saves after episode 1 (vector step 2), with env 1 two steps into its episode
    resumed, envs = build([[2], [5]])
    resumed.resume(str(tmp_path / "agent_episode_1.pt"))
    resumed.train()
    assert envs[1].resets == 1 and envs[1].steps == 5, (
        "env 1's episode in flight at the checkpoint is not continued: it starts afresh"
    )
    env1 = [b for b in resumed.agent.batches if b["observations"][0, 0] == 1]
    assert len(env1[0]["rewards"]) == 5 and env1[0]["observations"][0, 2] == 0
    assert resumed.total_steps_trained == sum(resumed.episode_lengths)
