"""Tests for exporting an agent's greedy policy: ``InferencePolicy``, the C header and export.py."""

import dataclasses
import hashlib
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_core.agents import (  # noqa: E402
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DDPGAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
    TD3Agent,
)
from rl_core.agents.inference import DenseLayer, InferencePolicy  # noqa: E402
from rl_core.export import c_header, rollout, verify  # noqa: E402
from tests.agent_internals import (  # noqa: E402
    actor_forward,
    ddpg_networks,
    gaussian_forward,
    probabilities,
    q_values,
)
from tests.helpers import minimal_config, needs_cc, needs_torch, run_main  # noqa: E402

OBS_DIM = 5
BOUNDS = {"action_low": [-2.0, -0.5], "action_high": [2.0, 1.5]}
DISCRETE = [
    (ReinforceAgent, {"seed": 3, "hidden_dim": 6}),
    (ActorCriticAgent, {"seed": 3, "hidden_dim": 6}),
    (PPOAgent, {"seed": 3, "hidden_dim": 6}),
    (PPOAgent, {"seed": 3, "hidden_dim": 6, "normalize_observations": True}),
    (DQNAgent, {"seed": 3, "hidden_dim": 6}),
    (DQNAgent, {"seed": 3, "hidden_dim": 6, "dueling": True}),
]
CONTINUOUS = [
    (ContinuousActorCriticAgent, {"seed": 3, "hidden_dim": 6}),
    (ContinuousActorCriticAgent, {"seed": 3, "hidden_dim": 6, "bounded_mean": True, **BOUNDS}),
    (ContinuousPPOAgent, {"seed": 3, "hidden_dim": 6, **BOUNDS}),
    (ContinuousPPOAgent, {"seed": 3, "hidden_dim": 6, "normalize_observations": True, **BOUNDS}),
    (DDPGAgent, {"seed": 3, "hidden_sizes": [7, 5], **BOUNDS}),
    (TD3Agent, {"seed": 3, "hidden_sizes": [6], **BOUNDS}),
]
EXPORTABLE = DISCRETE + CONTINUOUS


def _id(case):
    cls, config = case
    options = [k for k in ("dueling", "bounded_mean", "normalize_observations") if config.get(k)]
    return "-".join([cls.__name__, *options])


def _action_dim(cls):
    return 3 if any(cls is c for c, _ in DISCRETE) else 2


def _observations(n=200, seed=0, scale=1.5):
    return np.random.default_rng(seed).normal(0.0, scale, size=(n, OBS_DIM)).astype(np.float32)


def _agent(cls, config):
    """An agent with its initial weights; a normalising one has seen 300 shifted observations, so
    its statistics are far from the identity."""
    agent = cls(observation_dim=OBS_DIM, action_dim=_action_dim(cls), config=dict(config))
    if config.get("normalize_observations"):
        for obs in _observations(300, seed=9, scale=3.0) + 2.0:
            agent.normalize_observation(obs, training=True)
    return agent


def _greedy(agent, raw_obs):
    return agent.select_action(agent.normalize_observation(raw_obs, training=False), training=False)


# ---------------------------------------------------------------------------------------------
# The NumPy reference: InferencePolicy reproduces every agent's greedy policy
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", EXPORTABLE, ids=[_id(c) for c in EXPORTABLE])
def test_policy_acts_as_the_agent_acts_in_evaluation(case):
    agent = _agent(*case)
    policy = agent.inference_policy()
    for obs in _observations():
        expected = _greedy(agent, obs)
        actual = policy.act(obs)
        if np.ndim(expected) == 0:
            assert actual == expected, f"greedy action differs on {obs}"
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "case", [c for c in DISCRETE if c[0] is not DQNAgent], ids=lambda c: _id(c)
)
def test_discrete_policy_outputs_are_the_agents_logits(case):
    """Softmax of the exported outputs is the agent's action distribution: the outputs are its
    logits, not some other monotone transform that happens to share the argmax."""
    agent = _agent(*case)
    policy = agent.inference_policy()
    for obs in _observations(20):
        logits = policy.outputs(obs)
        exp = np.exp(logits - logits.max())
        np.testing.assert_allclose(
            exp / exp.sum(),
            probabilities(agent, agent.normalize_observation(obs, training=False)),
            rtol=1e-10,
            err_msg="the exported outputs should be the policy head's logits",
        )


@pytest.mark.parametrize("dueling", [False, True])
def test_dqn_policy_outputs_are_the_q_values(dueling):
    """Dueling DQN is folded into one network whose outputs are Q, not the advantage stream."""
    agent = _agent(DQNAgent, {"seed": 3, "hidden_dim": 6, "dueling": dueling})
    observations = _observations(20).astype(np.float64)
    policy = agent.inference_policy()
    np.testing.assert_allclose(
        policy.outputs(observations), q_values(agent, observations), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize(
    "case",
    [c for c in CONTINUOUS if c[0] in (ContinuousActorCriticAgent, ContinuousPPOAgent)],
    ids=lambda c: _id(c),
)
def test_continuous_policy_action_is_the_clipped_mean(case):
    agent = _agent(*case)
    policy = agent.inference_policy()
    for obs in _observations(20):
        mean = gaussian_forward(agent, agent.normalize_observation(obs, training=False))[0]
        np.testing.assert_allclose(
            policy.act(obs), np.clip(mean, agent.action_low, agent.action_high), rtol=1e-10
        )


@pytest.mark.parametrize("cls", [DDPGAgent, TD3Agent])
def test_ddpg_policy_action_is_the_actor_output(cls):
    agent = _agent(cls, {"seed": 3, "hidden_sizes": [7, 5], **BOUNDS})
    observations = _observations(20).astype(np.float64)
    policy = agent.inference_policy()
    np.testing.assert_allclose(
        policy.act(observations), actor_forward(agent, observations)[0], rtol=1e-10
    )


def test_policy_outputs_for_a_batch_are_its_outputs_for_each_observation():
    policy = _agent(DDPGAgent, {"seed": 3, "hidden_sizes": [4], **BOUNDS}).inference_policy()
    observations = _observations(7)
    np.testing.assert_allclose(
        policy.outputs(observations), np.stack([policy.outputs(o) for o in observations])
    )


def test_discrete_policy_acts_on_a_batch_as_on_each_observation():
    policy = _agent(PPOAgent, {"seed": 3, "hidden_dim": 4}).inference_policy()
    observations = _observations(7)
    np.testing.assert_array_equal(policy.act(observations), [policy.act(o) for o in observations])


def test_policy_records_the_agents_observation_settings_and_name():
    agent = _agent(PPOAgent, {"seed": 3, "hidden_dim": 4})
    agent.record_observation_settings(
        {"frame_stack": 1, "mask_velocity": True}, {"frame_stack": 1, "mask_velocity": False}
    )
    policy = agent.inference_policy()
    assert policy.observation_settings == {"frame_stack": 1, "mask_velocity": True}
    assert policy.agent_name == "PPOAgent"


@pytest.mark.parametrize(
    "agent",
    [RandomAgent(OBS_DIM, 3, {}), ContinuousRandomAgent(OBS_DIM, 2, {})],
    ids=["RandomAgent", "ContinuousRandomAgent"],
)
def test_an_agent_without_a_network_is_refused(agent):
    with pytest.raises(ValueError, match=f"{type(agent).__name__} has no network to export"):
        agent.inference_policy()


def test_policy_arrays_are_read_only_copies():
    agent = _agent(DDPGAgent, {"seed": 3, "hidden_sizes": [4], **BOUNDS})
    policy = agent.inference_policy()
    obs = _observations(1)[0]
    before = policy.outputs(obs)
    with pytest.raises(ValueError, match="read-only"):
        policy.layers[0].weight[0, 0] = 1.0
    ddpg_networks(agent)["actor"]["W0"] += 1.0  # the agent trains on: the policy keeps its copy
    np.testing.assert_array_equal(policy.outputs(obs), before)


def _layer(n_in, n_out, relu=True, value=0.1):
    return DenseLayer(weight=np.full((n_in, n_out), value), bias=np.zeros(n_out), relu=relu)


def test_policy_refuses_layers_that_do_not_chain():
    with pytest.raises(ValueError, match="layer 1 takes 4 inputs.*layer 0 gives 3"):
        InferencePolicy(layers=(_layer(2, 3), _layer(4, 2, relu=False)), head="argmax")


def test_policy_refuses_a_hidden_activation_on_the_last_layer():
    with pytest.raises(ValueError, match="last layer"):
        InferencePolicy(layers=(_layer(2, 3), _layer(3, 2, relu=True)), head="argmax")


def test_a_layer_refuses_non_finite_weights():
    """A NaN would reach the header as a literal that is not valid C."""
    with pytest.raises(ValueError, match="finite"):
        DenseLayer(weight=np.array([[np.nan, 1.0]]), bias=np.zeros(2), relu=False)


@pytest.mark.parametrize(
    "head,low,high,match",
    [
        ("tanh_scale", None, None, "tanh_scale needs action bounds"),
        ("tanh_scale", [-np.inf, -1.0], [1.0, 1.0], "tanh_scale needs finite"),
        ("clip", [-1.0], [1.0], "bounds.*2 actions"),
        ("argmax", [-1.0, -1.0], [1.0, 1.0], "argmax takes no action bounds"),
        ("softmax", None, None, "head must be one of"),
    ],
)
def test_policy_refuses_a_head_that_does_not_fit(head, low, high, match):
    with pytest.raises(ValueError, match=match):
        InferencePolicy(
            layers=(_layer(2, 2, relu=False),),
            head=head,
            action_low=None if low is None else np.array(low),
            action_high=None if high is None else np.array(high),
        )


def test_normalisation_is_clipped_to_ten_as_the_agent_clips_it():
    """An observation 1000 standard deviations out reaches the network as 10, not 1000."""
    identity = DenseLayer(weight=np.eye(2), bias=np.zeros(2), relu=False)
    policy = InferencePolicy(
        layers=(identity,),
        head="clip",
        action_low=np.full(2, -np.inf),
        action_high=np.full(2, np.inf),
        obs_mean=np.zeros(2),
        obs_scale=np.ones(2),
    )
    np.testing.assert_array_equal(policy.outputs(np.array([1000.0, -3.0])), [10.0, -3.0])


def test_argmax_takes_the_first_of_tied_outputs():
    policy = InferencePolicy(layers=(_layer(2, 3, relu=False, value=0.0),), head="argmax")
    assert policy.act(np.array([1.0, 2.0])) == 0


# ---------------------------------------------------------------------------------------------
# The header text
# ---------------------------------------------------------------------------------------------

HEX_FLOAT = re.compile(r"(-?0x[0-9a-f]+(?:\.[0-9a-f]*)?p[+-]\d+)f")


def test_header_holds_every_weight_exactly_as_float32():
    policy = _agent(DDPGAgent, {"seed": 3, "hidden_sizes": [7, 5], **BOUNDS}).inference_policy()
    text = c_header.render_c_header(policy)
    literals = [float.fromhex(m) for m in HEX_FLOAT.findall(text)]
    expected = np.concatenate(  # stored [outputs][inputs] per layer, then the bias
        [np.concatenate([layer.weight.T.ravel(), layer.bias]) for layer in policy.layers]
    ).astype(np.float32)
    weights = np.array(literals[: expected.size], dtype=np.float64)
    np.testing.assert_array_equal(
        weights, expected.astype(np.float64), err_msg="every weight should round-trip float32"
    )


def test_header_declares_its_dimensions_and_guard():
    policy = _agent(PPOAgent, {"seed": 3, "hidden_dim": 6}).inference_policy()
    text = c_header.render_c_header(policy)
    assert "#ifndef POLICY_WEIGHTS_H" in text and "#define POLICY_WEIGHTS_H" in text
    assert f"#define POLICY_INPUT_DIM {OBS_DIM}" in text
    assert "#define POLICY_OUTPUT_DIM 3" in text
    assert "int policy_act(const float *obs)" in text


def test_header_clips_only_to_finite_bounds():
    """An infinite bound has no C literal: the header must not clip on that side at all."""
    layer = DenseLayer(weight=np.ones((2, 2)), bias=np.zeros(2), relu=False)
    policy = InferencePolicy(
        layers=(layer,),
        head="clip",
        action_low=np.array([-np.inf, -1.0]),
        action_high=np.array([np.inf, 1.0]),
    )
    text = c_header.render_c_header(policy)
    assert "action[0] <" not in text and "action[0] >" not in text, "action 0 is unbounded"
    assert "action[1] <" in text and "action[1] >" in text, "action 1 is clipped to ±1"
    assert not re.search(r"\binf", text, flags=re.IGNORECASE), "no inf literal"


def test_header_says_it_cannot_be_linked_twice():
    text = c_header.render_c_header(
        _agent(PPOAgent, {"seed": 3, "hidden_dim": 4}).inference_policy()
    )
    assert "one policy per program" in text


def test_continuous_header_acts_into_an_array():
    policy = _agent(DDPGAgent, {"seed": 3, "hidden_sizes": [4], **BOUNDS}).inference_policy()
    text = c_header.render_c_header(policy)
    assert "#define POLICY_ACTION_DIM 2" in text
    assert "void policy_act(const float *obs, float *action)" in text


def test_header_names_the_agent_settings_and_source():
    agent = _agent(PPOAgent, {"seed": 3, "hidden_dim": 4})
    agent.record_observation_settings({"frame_stack": 4}, {"frame_stack": 1})
    text = c_header.render_c_header(agent.inference_policy(), source="agent.pt sha256 abc123")
    assert "PPOAgent" in text
    assert "frame_stack: 4" in text
    assert "agent.pt sha256 abc123" in text


def test_header_is_reproducible():
    policy = _agent(DQNAgent, {"seed": 3, "hidden_dim": 4, "dueling": True}).inference_policy()
    assert c_header.render_c_header(policy) == c_header.render_c_header(policy)
    assert not re.search(r"20\d\d-\d\d-\d\d", c_header.render_c_header(policy)), "no date"


# ---------------------------------------------------------------------------------------------
# Compiled: the C forward pass agrees with NumPy
# ---------------------------------------------------------------------------------------------


@needs_cc
@pytest.mark.parametrize("case", EXPORTABLE, ids=[_id(c) for c in EXPORTABLE])
def test_compiled_header_agrees_with_numpy(case, tmp_path):
    policy = _agent(*case).inference_policy()
    header = c_header.render_c_header(policy)
    report = verify.verify_c_header(header, policy, _observations(), workdir=tmp_path, werror=True)
    assert report.passed, report.summary()
    assert report.action_mismatches == 0
    assert report.max_abs_error < 1e-5, report.summary()


SQUASHING = [
    c for c in CONTINUOUS if c[1].get("bounded_mean") or c[1].get("normalize_observations")
]
SQUASHING += [c for c in CONTINUOUS if c[0] is DDPGAgent]


@needs_cc
@pytest.mark.parametrize("case", SQUASHING, ids=[_id(c) for c in SQUASHING])
def test_compiled_header_stays_finite_and_clipped_on_huge_observations(case, tmp_path):
    """tanh of a huge input saturates rather than overflowing; normalisation clips to ±10."""
    policy = _agent(*case).inference_policy()
    observations = np.array([[1e6, -1e6, 1e6, -1e6, 1e6], [-1e6] * OBS_DIM], dtype=np.float32)
    report = verify.verify_c_header(
        c_header.render_c_header(policy), policy, observations, workdir=tmp_path
    )
    assert np.isfinite(report.c_outputs).all(), "non-finite C outputs"
    assert report.passed, report.summary()


@needs_cc
def test_compiled_header_takes_the_first_of_tied_actions(tmp_path):
    policy = InferencePolicy(layers=(_layer(OBS_DIM, 4, relu=False, value=0.0),), head="argmax")
    report = verify.verify_c_header(
        c_header.render_c_header(policy), policy, _observations(5), workdir=tmp_path
    )
    np.testing.assert_array_equal(report.c_actions, np.zeros(5))


@needs_cc
def test_compiled_header_clips_to_finite_bounds_only(tmp_path):
    """Infinite bounds compile (no inf literal) and leave the mean unclipped on that side."""
    layer = DenseLayer(weight=np.full((OBS_DIM, 2), 10.0), bias=np.zeros(2), relu=False)
    policy = InferencePolicy(
        layers=(layer,),
        head="clip",
        action_low=np.array([-np.inf, -1.0]),
        action_high=np.array([np.inf, 1.0]),
    )
    report = verify.verify_c_header(
        c_header.render_c_header(policy),
        policy,
        np.ones((1, OBS_DIM)),
        workdir=tmp_path,
        werror=True,
    )
    np.testing.assert_allclose(report.c_actions[0], [50.0, 1.0], rtol=1e-6)


def test_a_discrete_near_tie_is_counted_not_failed():
    """C (float32) may pick the other of two outputs within the tolerance; that is reported as a
    near-tie, while a disagreement on a clear gap is a mismatch. Two near-ties and one clear
    mismatch, so the counts cannot be swapped unnoticed."""
    reference = np.array([[1.0, 1.0 + 1e-7, 0.0], [0.0, 3.0, 3.0 - 1e-7], [2.0, 0.0, 0.0]])
    report = verify.compare(
        reference_outputs=reference,
        reference_actions=np.array([1, 1, 0]),
        c_outputs=reference.astype(np.float32).astype(np.float64),
        c_actions=np.array([0, 2, 1]),
        discrete=True,
    )
    assert report.near_ties == 2, "the two near-tied rows should count as near-ties"
    assert report.action_mismatches == 1, "the clear-gap row should count as a mismatch"
    assert not report.passed


def test_outputs_beyond_the_tolerance_fail():
    reference = np.array([[1.0, 2.0]])
    report = verify.compare(
        reference_outputs=reference,
        reference_actions=reference,
        c_outputs=reference + np.array([[0.0, 1e-3]]),
        c_actions=reference,
        discrete=False,
    )
    assert not report.passed
    assert report.max_abs_error == pytest.approx(1e-3)


def test_report_gives_the_worst_error_as_a_share_of_its_tolerance():
    """A large absolute error on a large output can pass; the share of the tolerance says so."""
    reference = np.array([[100.0, 0.0]])
    # Tolerances: 1e-5 + 1e-5 * 100 = 1.01e-3, and 1e-5. Errors: 5.05e-4 (half) and 2.5e-6 (a quarter).
    report = verify.compare(
        reference_outputs=reference,
        reference_actions=np.array([0]),
        c_outputs=reference + np.array([[5.05e-4, 2.5e-6]]),
        c_actions=np.array([0]),
        discrete=True,
    )
    assert report.worst_tolerance_share == pytest.approx(0.5)
    assert report.passed
    assert "0.5 of the tolerance" in report.summary()


def test_compiler_comes_from_cc_split_as_a_shell_would(monkeypatch):
    monkeypatch.setenv("CC", "ccache gcc")
    assert verify.find_c_compiler() == ["ccache", "gcc"]


def test_ci_has_a_c_compiler():
    """CI's Linux runner has gcc: the compiled tests must not all skip there unnoticed."""
    if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux":
        pytest.skip("only asserted on CI's Linux runner")
    assert verify.find_c_compiler() is not None


# ---------------------------------------------------------------------------------------------
# The greedy rollout and the NumPy cross-check
# ---------------------------------------------------------------------------------------------


class _ScriptedEnv:
    """Returns the given observations in turn, ending the episode after the last."""

    def __init__(self, observations):
        self.observations = observations
        self.step_count = 0
        self.actions = []

    def reset(self, seed=None):
        self.step_count = 0
        return self.observations[0].copy(), {}

    def step(self, action):
        self.actions.append(action)
        self.step_count += 1
        done = self.step_count == len(self.observations)
        return (
            self.observations[min(self.step_count, len(self.observations) - 1)].copy(),
            0.0,
            done,
            False,
            {},
        )


def _rollout(case, steps=12):
    agent = _agent(*case)
    observations = _observations(steps, seed=4, scale=3.0)
    env = _ScriptedEnv(observations)
    return agent, observations, rollout.greedy_rollout(env, agent, episodes=1, seed=0)


def test_greedy_rollout_records_raw_observations_and_the_agents_actions():
    """The C check feeds the rollout's observations to code that normalises them itself: they must
    be the raw ones, or a normalising agent's export is checked against inputs normalised twice."""
    case = (PPOAgent, {"seed": 3, "hidden_dim": 6, "normalize_observations": True})
    agent, observations, result = _rollout(case)
    np.testing.assert_array_equal(result.observations, observations)
    assert result.actions == [_greedy(agent, obs) for obs in observations]


NORMALISING = [c for c in EXPORTABLE if c[1].get("normalize_observations")]


@pytest.mark.parametrize("case", NORMALISING, ids=[_id(c) for c in NORMALISING])
def test_policy_disagreements_count_a_policy_that_acts_unlike_the_agent(case):
    agent, _, result = _rollout(case)
    policy = agent.inference_policy()
    assert rollout.policy_disagreements(policy, result).mismatches == 0
    last = policy.layers[-1]
    shifted = DenseLayer(last.weight, last.bias + np.linspace(-5.0, 5.0, last.outputs), False)
    wrong = dataclasses.replace(policy, layers=(*policy.layers[:-1], shifted))
    assert rollout.policy_disagreements(wrong, result).mismatches > 0, (
        "a policy with shifted outputs should act unlike the agent on some step"
    )


def test_policy_disagreements_do_not_fail_on_a_near_tie():
    """Tied outputs: the agent's other choice is a near-tie, not a disagreement."""
    tied = InferencePolicy(layers=(_layer(OBS_DIM, 3, relu=False, value=0.0),), head="argmax")
    result = rollout.GreedyRollout(_observations(2), [1, 0])
    agreement = rollout.policy_disagreements(tied, result)
    assert (agreement.mismatches, agreement.near_ties) == (0, 1)


# ---------------------------------------------------------------------------------------------
# export.py
# ---------------------------------------------------------------------------------------------


def _checkpoint(tmp_path, cls=PPOAgent, config=None):
    path = str(tmp_path / "agent.pt")
    cls(observation_dim=4, action_dim=2, config=config or {}).save(path)
    return path


def _export(tmp_path, monkeypatch, agent_type="ppo", extra=(), checkpoint=None):
    output = tmp_path / "out" / "policy_weights.h"
    run_main(
        ["--config", minimal_config(tmp_path), "--agent-type", agent_type]
        + ["--checkpoint", checkpoint or _checkpoint(tmp_path), "--output", str(output)]
        + list(extra),
        monkeypatch,
        module="export",
    )
    return output


def test_export_writes_the_header_without_verification(tmp_path, monkeypatch):
    output = _export(tmp_path, monkeypatch, extra=["--verify-episodes", "0"])
    text = output.read_text(encoding="utf-8")
    assert "#define POLICY_INPUT_DIM 4" in text
    digest = hashlib.sha256((tmp_path / "agent.pt").read_bytes()).hexdigest()
    assert digest in text, "the header should name the checkpoint it came from"
    assert sorted(os.listdir(tmp_path)) == ["agent.pt", "config.yaml", "out"], (
        "export.py should write only --output (no logs or checkpoints directories)"
    )


@needs_cc
def test_export_verifies_on_environment_observations(tmp_path, monkeypatch, capsys):
    output = _export(tmp_path, monkeypatch, extra=["--verify-episodes", "1"])
    assert output.exists()
    out = capsys.readouterr().out
    assert "passed" in out and "max abs error" in out


def test_export_without_a_compiler_writes_the_header_and_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(verify, "find_c_compiler", lambda: None)
    with pytest.raises(SystemExit) as stop:
        _export(tmp_path, monkeypatch, extra=["--verify-episodes", "1"])
    assert stop.value.code == 1
    assert (tmp_path / "out" / "policy_weights.h").exists()
    assert "--verify-episodes 0" in capsys.readouterr().err


@needs_cc
def test_export_verifies_a_normalising_agent(tmp_path, monkeypatch, capsys):
    """Statistics far from the identity, so normalising twice (or not at all) would show."""
    config = {"normalize_observations": True}
    agent = PPOAgent(observation_dim=4, action_dim=2, config=dict(config))
    for obs in _observations(300, seed=9, scale=0.5)[:, :4] + 0.3:
        agent.normalize_observation(obs, training=True)
    path = str(tmp_path / "agent.pt")
    agent.save(path)
    output = tmp_path / "policy.h"
    run_main(
        ["--config", minimal_config(tmp_path, agent={"type": "ppo", "config": config})]
        + ["--checkpoint", path, "--output", str(output), "--verify-episodes", "1"],
        monkeypatch,
        module="export",
    )
    assert "policy_obs_mean" in output.read_text(encoding="utf-8")
    assert "passed" in capsys.readouterr().out


def test_export_keeps_no_header_that_failed_verification(tmp_path, monkeypatch):
    def failing(header, policy, observations, **_):
        n = len(observations)
        return verify.compare(np.zeros((n, 2)), np.zeros(n), np.ones((n, 2)), np.zeros(n), True)

    monkeypatch.setattr(verify, "find_c_compiler", lambda: ["cc"])
    monkeypatch.setattr(verify, "verify_c_header", failing)
    with pytest.raises(SystemExit) as stop:
        _export(tmp_path, monkeypatch, extra=["--verify-episodes", "1"])
    assert stop.value.code == 1
    assert not (tmp_path / "out" / "policy_weights.h").exists(), (
        "a header that failed verification should not be left at --output"
    )


def test_export_refuses_a_negative_episode_count(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        _export(tmp_path, monkeypatch, extra=["--verify-episodes", "-1"])
    assert stop.value.code == 2


@pytest.mark.parametrize("saved_by", [RandomAgent, PPOAgent], ids=["random", "ppo"])
def test_export_refuses_the_random_agent(tmp_path, monkeypatch, saved_by):
    # Refused before the checkpoint is read, so another agent's checkpoint gives the same message.
    checkpoint = _checkpoint(tmp_path, cls=saved_by)
    with pytest.raises(ValueError, match="RandomAgent has no network to export"):
        _export(tmp_path, monkeypatch, agent_type="random", checkpoint=checkpoint)
    assert not (tmp_path / "out").exists()


@needs_torch
def test_export_reads_a_torch_checkpoint_through_the_numpy_agent(tmp_path, monkeypatch):
    from rl_core.agents.torch_ddpg_agent import TorchDDPGAgent

    config = {"hidden_sizes": [8], "backend": "torch", "device": "cpu"}
    path = str(tmp_path / "agent.pt")
    TorchDDPGAgent(observation_dim=3, action_dim=1, config=config).save(path)
    output = tmp_path / "policy.h"
    run_main(
        [
            "--config",
            minimal_config(
                tmp_path,
                environment={"name": "Pendulum-v1"},
                agent={"type": "ddpg", "config": config},
            ),
        ]
        + ["--checkpoint", path, "--output", str(output), "--verify-episodes", "0"],
        monkeypatch,
        module="export",
    )
    assert "#define POLICY_ACTION_DIM 1" in output.read_text(encoding="utf-8")
