"""Tests for the robustness benchmark: a checkpoint played greedily over perturbed environments
(``rl_core.training.robustness``, ``Trainer.evaluate_episodes`` and robustness.py)."""

import hashlib
import importlib
import os
import re

import numpy as np
import pytest
import yaml

from rl_cartpole.benchmark import CartPoleChallenge
from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import DDPGAgent, ReinforceAgent
from rl_core.training import Trainer
from rl_core.training.curriculum import curriculum_from_config
from rl_core.training.robustness import (
    Perturbation,
    RobustnessRow,
    difficulty_reached,
    parse_sweep,
    perturbed_section,
    physics_defaults,
    robustness_rows,
    training_range,
)
from rl_pendulum.benchmark import PendulumChallenge
from rl_pendulum.environments import PendulumEnv
from tests.helpers import minimal_config, run_main

CARTPOLE_DEFAULTS = {"gravity": 9.8, "masscart": 1.0, "masspole": 0.1, "length": 0.5}


@pytest.fixture
def cartpole():
    """A CartPole wrapper with every schedulable setting on, for the setting rules."""
    env = CartPoleEnv(
        obs_noise_std=0.05, action_noise_prob=0.1, domain_randomization={"length": (0.25, 1.0)}
    )
    yield env
    env.close()


@pytest.fixture
def pendulum():
    env = PendulumEnv(domain_randomization={"gravity": (8.0, 12.0)})
    yield env
    env.close()


# ---------------------------------------------------------------------------
# --sweep values
# ---------------------------------------------------------------------------


def test_parse_sweep_gives_one_perturbation_per_value_in_order():
    assert parse_sweep("length=0.25,0.5, 2") == [
        Perturbation("length", 0.25),
        Perturbation("length", 0.5),
        Perturbation("length", 2.0),
    ]


@pytest.mark.parametrize("text", ["length", "length=", "=1.0", "length=a", "length=1,,2", "x=inf"])
def test_parse_sweep_refuses_a_malformed_sweep(text):
    with pytest.raises(ValueError, match="SETTING=V1,V2"):
        parse_sweep(text)


# ---------------------------------------------------------------------------
# One environment section per perturbation
# ---------------------------------------------------------------------------


def test_a_noise_setting_is_set_and_the_curriculum_dropped_without_changing_the_input(cartpole):
    section = {"obs_noise_std": 0.05, "curriculum": {"mode": "linear"}, "seed": 3}
    before = yaml.safe_dump(section)

    setting, perturbed = perturbed_section(section, cartpole, Perturbation("obs_noise_std", 0.2))

    assert setting == "obs_noise_std"
    assert perturbed == {"obs_noise_std": 0.2, "seed": 3}, "evaluation plays one fixed setting"
    assert yaml.safe_dump(section) == before, "the config's section must not be changed"


def test_a_physics_parameter_is_pinned_and_the_other_ranges_kept(cartpole):
    section = {"domain_randomization": {"length": [0.25, 1.0], "masscart": [0.5, 1.5]}}

    setting, perturbed = perturbed_section(section, cartpole, Perturbation("length", 2.0))

    assert setting == "length"
    assert perturbed["domain_randomization"] == {"length": (2.0, 2.0), "masscart": (0.5, 1.5)}


def test_a_physics_parameter_the_target_does_not_randomize_is_added(cartpole):
    _, perturbed = perturbed_section({}, cartpole, Perturbation("masspole", 0.3))
    assert perturbed["domain_randomization"] == {"masspole": (0.3, 0.3)}


@pytest.mark.parametrize(
    ("target", "sweep", "expected"),
    [
        ({"gravity": [8.0, 12.0]}, "g", {"g": (10.0, 10.0)}),
        ({"g": [8.0, 12.0]}, "gravity", {"g": (10.0, 10.0)}),
        ({"gravity": [8.0, 12.0], "length": [0.8, 1.2]}, "g", {"g": (10.0, 10.0), "l": (0.8, 1.2)}),
        # Any letter case, as the wrapper reads a config's names.
        ({"g": [8.0, 12.0]}, "Gravity", {"g": (10.0, 10.0)}),
    ],
)
def test_pendulum_aliases_pin_the_canonical_parameter(pendulum, target, sweep, expected):
    """The target is read under the wrapper's own names first, so an alias is not given twice."""
    setting, perturbed = perturbed_section(
        {"domain_randomization": target}, pendulum, Perturbation(sweep, 10.0)
    )
    assert setting == "g"
    assert perturbed["domain_randomization"] == expected


def test_an_unknown_setting_is_refused_naming_what_can_be_swept(cartpole):
    with pytest.raises(ValueError, match=r"friction.*obs_noise_std.*length"):
        perturbed_section({}, cartpole, Perturbation("friction", 1.0))


def test_domain_randomization_itself_is_not_a_sweepable_setting(cartpole):
    with pytest.raises(ValueError, match="domain_randomization"):
        perturbed_section({}, cartpole, Perturbation("domain_randomization", 1.0))


@pytest.mark.parametrize(
    ("setting", "value"), [("length", 0.0), ("action_noise_prob", 1.5), ("obs_noise_std", -0.1)]
)
def test_a_value_the_wrapper_refuses_is_refused(cartpole, setting, value):
    with pytest.raises(ValueError, match=setting):
        perturbed_section({}, cartpole, Perturbation(setting, value))


def test_a_pinned_parameter_holds_on_every_reset(cartpole):
    _, perturbed = perturbed_section(
        {"domain_randomization": {"length": [0.25, 1.0]}},
        cartpole,
        Perturbation("length", 1.5),
    )
    env = CartPoleChallenge().make_env(perturbed, render_mode=None)
    try:
        for seed in (1, 2, 3):
            env.reset(seed=seed)
            assert env.env.unwrapped.length == pytest.approx(1.5), "every episode plays 1.5"
    finally:
        env.close()


@pytest.mark.parametrize(
    ("wrapper", "challenge"),
    [(CartPoleEnv, CartPoleChallenge), (PendulumEnv, PendulumChallenge)],
    ids=["cartpole", "pendulum"],
)
def test_every_randomizable_parameter_is_accepted_and_is_a_physics_attribute(wrapper, challenge):
    env = wrapper()
    try:
        assert wrapper.RANDOMIZABLE_PARAMETERS, "the wrapper lists what it can randomize"
        for name, canonical in wrapper.RANDOMIZABLE_PARAMETERS.items():
            checked = env.check_setting("domain_randomization", {name: (1.0, 1.0)})
            assert checked == {canonical: (1.0, 1.0)}, f"{name} is read as {canonical}"
    finally:
        env.close()
    # physics_defaults reads each canonical name off the unwrapped environment.
    assert set(physics_defaults(challenge(), {})) == set(wrapper.RANDOMIZABLE_PARAMETERS.values())


# ---------------------------------------------------------------------------
# The range a checkpoint trained on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("challenge", "expected"),
    [(CartPoleChallenge, CARTPOLE_DEFAULTS), (PendulumChallenge, {"g": 10.0, "m": 1.0, "l": 1.0})],
    ids=["cartpole", "pendulum"],
)
def test_physics_defaults_ignore_the_configs_randomization(challenge, expected):
    """A randomization that excludes the default must not leak into it (the wrappers set a fresh
    draw on each reset and never put the default back)."""
    key = "length" if challenge is CartPoleChallenge else "l"
    section = {"domain_randomization": {key: [2.0, 3.0]}, "seed": 1}
    assert physics_defaults(challenge(), section) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("setting", "target", "start", "difficulty", "expected"),
    [
        # Without a curriculum a noise level trained only at its target.
        ("obs_noise_std", {"obs_noise_std": 0.05}, None, 1.0, (0.05, 0.05)),
        # Off at construction: trained at 0.
        ("obs_noise_std", {}, None, 1.0, (0.0, 0.0)),
        # From start 0 to the target.
        ("obs_noise_std", {"obs_noise_std": 0.05}, {"obs_noise_std": 0.0}, 1.0, (0.0, 0.05)),
        # Stopped half way: 0.5 * 0 + 0.5 * 0.05.
        ("obs_noise_std", {"obs_noise_std": 0.05}, {"obs_noise_std": 0.0}, 0.5, (0.0, 0.025)),
        # A randomized parameter covers both its start and target ranges.
        (
            "length",
            {"domain_randomization": {"length": (0.25, 1.0)}},
            {"domain_randomization": {"length": (0.5, 0.5)}},
            1.0,
            (0.25, 1.0),
        ),
        # Half way: low 0.5 * 0.5 + 0.5 * 0.25 = 0.375, high 0.5 * 0.5 + 0.5 * 1.0 = 0.75.
        (
            "length",
            {"domain_randomization": {"length": (0.25, 1.0)}},
            {"domain_randomization": {"length": (0.5, 0.5)}},
            0.5,
            (0.375, 0.75),
        ),
        # A parameter left fixed trained only at its default.
        ("masscart", {"domain_randomization": {"length": (0.25, 1.0)}}, None, 1.0, (1.0, 1.0)),
    ],
)
def test_training_range(setting, target, start, difficulty, expected):
    got = training_range(setting, target, start, difficulty, CARTPOLE_DEFAULTS)
    assert got == pytest.approx(expected)


def _train_with_curriculum(tmp_path, num_episodes, save_frequency, curriculum=True):
    """Train a REINFORCE agent on CartPole, through a linear curriculum over 8 episodes unless
    ``curriculum`` is False, saving every ``save_frequency``; return the environment section and
    the checkpoint directory."""
    section = {
        "obs_noise_std": 0.05,
        "max_episode_steps": 20,
        "seed": 4,
        "curriculum": {"mode": "linear", "episodes": 8, "start": {"obs_noise_std": 0.0}},
    }
    if not curriculum:
        del section["curriculum"]
    challenge = CartPoleChallenge()
    env = challenge.make_env(section, render_mode=None)
    eval_env = challenge.make_env(section, render_mode=None)
    agent = challenge.make_agent("reinforce", {"seed": 4}, env)
    directory = tmp_path / "checkpoints"
    trainer = Trainer(
        agent=agent,
        env=env,
        eval_env=eval_env,
        eval_seed=7,
        config={
            "num_episodes": num_episodes,
            "max_steps_per_episode": 20,
            "eval_frequency": 100,
            "save_frequency": save_frequency,
            "checkpoint_dir": str(directory),
        },
        curriculum=curriculum_from_config(section, env),
    )
    try:
        trainer.train()
    finally:
        env.close()
        eval_env.close()
    return section, directory


def test_the_difficulty_reached_is_replayed_from_the_trainer_file(tmp_path):
    section, directory = _train_with_curriculum(tmp_path, num_episodes=4, save_frequency=4)
    env = CartPoleChallenge().make_env(section, render_mode=None)
    try:
        curriculum = curriculum_from_config(section, env)
        # Linear over 8 episodes: the 4th trained at 3 / 8.
        reached = difficulty_reached(str(directory / "agent_episode_4.pt"), curriculum)
    finally:
        env.close()
    assert reached == pytest.approx(3 / 8)


def test_no_trainer_file_gives_no_difficulty(tmp_path):
    section, directory = _train_with_curriculum(tmp_path, num_episodes=4, save_frequency=4)
    renamed = tmp_path / "policy.pt"
    os.replace(directory / "agent_episode_4.pt", renamed)
    env = CartPoleChallenge().make_env(section, render_mode=None)
    try:
        curriculum = curriculum_from_config(section, env)
        assert difficulty_reached(str(renamed), curriculum) is None
    finally:
        env.close()


def test_a_trainer_named_checkpoint_without_its_trainer_file_gives_no_difficulty(tmp_path):
    section, directory = _train_with_curriculum(tmp_path, num_episodes=4, save_frequency=4)
    os.remove(directory / "trainer_episode_4.npz")
    env = CartPoleChallenge().make_env(section, render_mode=None)
    try:
        curriculum = curriculum_from_config(section, env)
        assert difficulty_reached(str(directory / "agent_episode_4.pt"), curriculum) is None
    finally:
        env.close()


@pytest.mark.parametrize(
    "curriculum",
    [None, {"mode": "linear", "episodes": 9, "start": {"obs_noise_std": 0.0}}],
    ids=["config-without-one", "another-curriculum"],
)
def test_a_config_whose_curriculum_differs_from_the_checkpoints_is_refused(tmp_path, curriculum):
    """As a resume refuses it: replaying the rewards through another curriculum gives a wrong
    range, and a config without one would call trained values "beyond"."""
    section, directory = _train_with_curriculum(tmp_path, num_episodes=4, save_frequency=4)
    other = {k: v for k, v in section.items() if k != "curriculum"}
    if curriculum is not None:
        other["curriculum"] = curriculum
    env = CartPoleChallenge().make_env(other, render_mode=None)
    try:
        scheduler = curriculum_from_config(other, env)
        with pytest.raises(ValueError, match="trained with a different curriculum"):
            difficulty_reached(str(directory / "agent_episode_4.pt"), scheduler)
    finally:
        env.close()


def test_a_malformed_trainer_file_is_refused_not_ignored(tmp_path):
    section, directory = _train_with_curriculum(tmp_path, num_episodes=4, save_frequency=4)
    np.savez(directory / "trainer_episode_4.npz", unrelated=np.array(1))
    env = CartPoleChallenge().make_env(section, render_mode=None)
    try:
        with pytest.raises(ValueError, match="missing"):
            difficulty_reached(
                str(directory / "agent_episode_4.pt"), curriculum_from_config(section, env)
            )
    finally:
        env.close()


def test_a_run_without_a_curriculum_has_no_difficulty(tmp_path):
    section, directory = _train_with_curriculum(
        tmp_path, num_episodes=4, save_frequency=4, curriculum=False
    )
    assert difficulty_reached(str(directory / "agent_episode_4.pt"), None) is None


# ---------------------------------------------------------------------------
# Rows: greedy evaluation per perturbation
# ---------------------------------------------------------------------------


def test_a_row_counts_the_episodes_at_or_above_the_threshold():
    row = RobustnessRow("length", 1.0, (0.5, 0.5), (10.0, 500.0, 475.0, 474.0), (10, 500, 475, 474))
    assert row.solved_fraction(475.0) == 0.5, "475 and 500 of four episodes reach 475"
    assert row.solved_fraction(None) is None, "no threshold, no fraction"
    assert row.mean_reward == pytest.approx(364.75)
    assert row.mean_length == pytest.approx(364.75)
    assert row.in_range is False, "1.0 is beyond a range of 0.5 to 0.5"


@pytest.mark.parametrize(("value", "expected"), [(0.25, True), (0.5, True), (0.2, False)])
def test_in_range_includes_both_ends(value, expected):
    assert RobustnessRow("length", value, (0.25, 0.5), (1.0,), (1,)).in_range is expected


def test_the_nominal_row_has_no_setting_and_no_range():
    row = RobustnessRow(None, None, None, (1.0,), (1,))
    assert row.in_range is None


@pytest.mark.parametrize(
    ("successes", "expected"),
    [((True, False, True, True), 0.75), (None, None)],
    ids=["reported", "not-reported"],
)
def test_a_row_gives_the_share_of_episodes_that_succeeded(successes, expected):
    row = RobustnessRow("length", 1.0, (0.5, 0.5), (1.0,) * 4, (1,) * 4, successes)
    assert row.success_rate == expected


def _report(successes):
    rows = [RobustnessRow(None, None, None, (9.0, -1.0), (50, 20), successes)]
    module = importlib.import_module("robustness")
    return module.render_report(rows, title="t", header=[], threshold=7.0)


def test_the_report_has_a_success_column_only_when_episodes_report_success():
    with_success, without = _report((True, False)), _report(None)
    assert "| Success |" in with_success and "| 50% |" in with_success
    assert "Success" not in without, "an environment without is_success gets no column"
    assert "Reward ≥ 7" in without, "the threshold column counts rewards, not successes"


def _cartpole_config(**environment):
    return {
        "environment": {"name": "CartPole-v1", "max_episode_steps": 50, "seed": 5, **environment},
        "training": {"num_episodes": 1, "max_steps_per_episode": 50},
    }


def _agent(challenge, config, agent_type, seed):
    """An untrained agent of ``agent_type`` sized for the config's environment."""
    env = challenge.make_env(config["environment"], render_mode=None)
    try:
        return challenge.make_agent(agent_type, {"seed": seed}, env)
    finally:
        env.close()


def test_rows_start_with_the_nominal_target_and_follow_the_sweeps_in_order():
    config = _cartpole_config(domain_randomization={"length": [0.25, 1.0]})
    challenge = CartPoleChallenge()
    agent = _agent(challenge, config, "reinforce", seed=1)
    rows = robustness_rows(
        challenge,
        config,
        agent,
        [Perturbation("length", 0.5), Perturbation("length", 2.0), Perturbation("masscart", 1.0)],
        episodes=3,
        eval_seed=11,
    )
    assert [(r.setting, r.value) for r in rows] == [
        (None, None),
        ("length", 0.5),
        ("length", 2.0),
        ("masscart", 1.0),
    ], "the nominal row first, then each perturbation in the order given"
    assert [r.in_range for r in rows] == [None, True, False, True], (
        "length trained on 0.25 to 1.0, masscart fixed at its default 1.0"
    )
    assert all(len(r.rewards) == len(r.lengths) == 3 for r in rows), "3 episodes per row"


def test_rows_refuse_zero_episodes():
    config = _cartpole_config()
    challenge = CartPoleChallenge()
    agent = _agent(challenge, config, "reinforce", seed=1)
    with pytest.raises(ValueError, match="episodes must be a positive integer"):
        robustness_rows(challenge, config, agent, [], episodes=0, eval_seed=1)


def test_every_row_plays_the_same_start_states():
    """Without noise, pinning a parameter at its default plays the nominal episodes exactly: the
    same reset seeds, the same greedy agent, the same physics."""
    config = _cartpole_config()
    challenge = CartPoleChallenge()
    agent = _agent(challenge, config, "dqn", seed=2)
    nominal, at_default = robustness_rows(
        challenge, config, agent, [Perturbation("length", 0.5)], episodes=4, eval_seed=21
    )
    assert nominal.lengths == at_default.lengths, "episode i of every row starts alike"
    assert len(set(nominal.lengths)) > 1, "the four episodes start from different states"


def test_evaluate_episodes_returns_each_episode_that_evaluate_averages():
    config = _cartpole_config()
    challenge = CartPoleChallenge()
    env = challenge.make_env(config["environment"], render_mode=None)
    try:
        agent = challenge.make_agent("dqn", {"seed": 3}, env)
        trainer = Trainer(
            agent=agent, env=env, config=config["training"], eval_env=env, eval_seed=30
        )
        episodes = trainer.evaluate_episodes(4)
        stats = trainer.evaluate(num_episodes=4)
    finally:
        env.close()
    assert len(episodes) == 4
    rewards = [episode.reward for episode in episodes]
    assert stats["mean_reward"] == pytest.approx(np.mean(rewards))
    assert stats["mean_length"] == pytest.approx(np.mean([e.length for e in episodes]))
    assert {e.success for e in episodes} == {None}, "CartPole reports no success"


# ---------------------------------------------------------------------------
# robustness.py
# ---------------------------------------------------------------------------


def _checkpoint(tmp_path):
    path = str(tmp_path / "agent.pt")
    ReinforceAgent(observation_dim=4, action_dim=2, config={}).save(path)
    return path


def _run(tmp_path, monkeypatch, *args, environment=None, agent="reinforce"):
    config = minimal_config(tmp_path, environment=environment, agent={"type": agent})
    run_main(["--config", config, *args], monkeypatch, module="robustness")


def _files(root):
    return sorted(
        os.path.relpath(os.path.join(d, f), root) for d, _, files in os.walk(root) for f in files
    )


def test_cli_writes_a_report_with_a_row_per_perturbation(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    report = tmp_path / "out" / "robustness.md"
    _run(
        tmp_path,
        monkeypatch,
        "--checkpoint", checkpoint,
        "--sweep", "length=0.5,1.5",
        "--sweep", "obs_noise_std=0.1",
        "--episodes", "2",
        "--report", str(report),
    )  # fmt: skip
    md = report.read_text(encoding="utf-8")
    rows = [line for line in md.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 1 + 4, "a header row, then nominal and three perturbations"
    assert "nominal" in rows[1]
    assert "| length | 0.5 | in range |" in md
    assert "| length | 1.5 | beyond |" in md
    assert "| obs_noise_std | 0.1 | beyond |" in md
    with open(checkpoint, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() in md, "the report names the checkpoint"


def test_cli_without_report_prints_and_writes_nothing(tmp_path, monkeypatch, capsys):
    checkpoint = _checkpoint(tmp_path)
    config = minimal_config(tmp_path, agent={"type": "reinforce"})
    before = _files(tmp_path)
    run_main(
        [
            "--config",
            config,
            "--checkpoint",
            checkpoint,
            "--sweep",
            "length=1.0",
            "--episodes",
            "1",
        ],
        monkeypatch,
        module="robustness",
    )
    assert "| length | 1 | beyond |" in capsys.readouterr().out
    assert _files(tmp_path) == before, "without --report nothing is written"


def test_cli_names_the_eval_seed_of_a_config_without_a_seed(tmp_path, monkeypatch, capsys):
    _run(
        tmp_path,
        monkeypatch,
        "--checkpoint", _checkpoint(tmp_path),
        "--sweep", "length=1.0",
        "--episodes", "1",
        environment={"seed": None},
    )  # fmt: skip
    match = re.search(r"\*\*Eval seed\*\*: (\d+) ", capsys.readouterr().out)
    assert match, "the header names the eval seed"
    # Drawn from entropy for a run without a seed (rl_core.seeding.eval_seed_for), in this range.
    assert 2**31 <= int(match.group(1)) < 2**32


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--sweep", "length=1.0", "--episodes", "0"], "--episodes"),
        (["--sweep", "length"], "SETTING=V1,V2"),
        (["--sweep", "length=1.0", "--sweep", "length=2.0"], "one --sweep"),
        (["--sweep", "friction=1.0"], "friction"),
        (["--sweep", "length=0"], "length"),
        ([], "--sweep"),
    ],
)
def test_cli_refuses_bad_arguments_before_evaluating(tmp_path, monkeypatch, capsys, args, message):
    evaluated = []
    monkeypatch.setattr(Trainer, "evaluate_episodes", lambda self, n: evaluated.append(n))
    with pytest.raises(SystemExit):
        _run(tmp_path, monkeypatch, "--checkpoint", _checkpoint(tmp_path), *args)
    assert message in capsys.readouterr().err
    assert not evaluated, "nothing is evaluated after a bad argument"


def test_cli_loads_a_torch_backend_checkpoint_into_the_numpy_agent(tmp_path, monkeypatch, capsys):
    """A config with ``backend: torch`` still runs where PyTorch is not installed."""
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    agent = DDPGAgent(
        observation_dim=3,
        action_dim=1,
        config={"action_low": env.action_space.low, "action_high": env.action_space.high},
    )
    env.close()
    checkpoint = str(tmp_path / "ddpg.pt")
    agent.save(checkpoint)
    config = minimal_config(
        tmp_path,
        environment={"name": "Pendulum-v1", "max_episode_steps": 10},
        training={"max_steps_per_episode": 10},
        agent={"type": "ddpg", "config": {"backend": "torch"}},
    )
    run_main(
        ["--config", config, "--checkpoint", checkpoint, "--sweep", "g=12", "--episodes", "1"],
        monkeypatch,
        module="robustness",
    )
    assert "| g | 12 | beyond |" in capsys.readouterr().out


def test_cli_reports_a_curriculum_runs_range_as_trained(tmp_path, monkeypatch, capsys):
    section, directory = _train_with_curriculum(tmp_path, num_episodes=4, save_frequency=4)
    config = minimal_config(
        tmp_path,
        environment={k: v for k, v in section.items() if k != "max_episode_steps"},
        agent={"type": "reinforce", "config": {"seed": 4}},
    )
    run_main(
        [
            "--config", config,
            "--checkpoint", str(directory / "agent_episode_4.pt"),
            "--sweep", "obs_noise_std=0.01,0.04",
            "--episodes", "1",
        ],
        monkeypatch,
        module="robustness",
    )  # fmt: skip
    out = capsys.readouterr().out
    # Trained up to 3 / 8 of the way to 0.05: 0 to 0.01875.
    assert "| obs_noise_std | 0.01 | in range |" in out
    assert "| obs_noise_std | 0.04 | beyond |" in out
    assert "difficulty 0.38" in out, "the header says how far the curriculum got"


def test_cli_refuses_a_config_without_the_checkpoints_curriculum(tmp_path, monkeypatch, capsys):
    section, directory = _train_with_curriculum(tmp_path, num_episodes=4, save_frequency=4)
    evaluated = []
    monkeypatch.setattr(Trainer, "evaluate_episodes", lambda self, n: evaluated.append(n))
    with pytest.raises(SystemExit):
        _run(
            tmp_path,
            monkeypatch,
            "--checkpoint", str(directory / "agent_episode_4.pt"),
            "--sweep", "obs_noise_std=0.01",
            environment={"obs_noise_std": 0.05, "seed": 4},
        )  # fmt: skip
    assert "trained with a different curriculum" in capsys.readouterr().err
    assert not evaluated, "nothing is evaluated for the wrong config"


def test_rows_on_an_environment_reporting_success_carry_each_episodes_outcome():
    from rl_navigation.benchmark import NavigationChallenge

    challenge = NavigationChallenge()
    config = {
        "environment": {"name": "Navigation-v0", "seed": 3, "max_episode_steps": 30},
        "training": {"max_steps_per_episode": 30},
    }
    env = challenge.make_env(config["environment"], render_mode=None)
    try:
        agent = challenge.make_agent("random", {"seed": 3}, env)
    finally:
        env.close()
    rows = robustness_rows(
        challenge, config, agent, [Perturbation("sensor_noise_std", 0.1)], episodes=3, eval_seed=5
    )
    for row in rows:
        assert row.successes is not None and len(row.successes) == 3, "one outcome per episode"
        assert all(isinstance(s, bool) for s in row.successes)
