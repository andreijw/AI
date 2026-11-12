"""SAC: a squashed Gaussian actor, twin critics and a learned temperature (SB3's SAC)."""

import copy
import math

import numpy as np
import pytest

from rl_core.agents import DDPGAgent, SACAgent, TD3Agent
from rl_core.agents.continuous_actor_critic import LOG_STD_MAX, LOG_STD_MIN
from rl_core.agents.sac_agent import log_one_minus_tanh_squared
from rl_core.challenges import discover_challenges
from tests.agent_internals import (
    assert_same_weights,
    counters,
    critic_forward,
    critic_loss_and_grads,
    critic_step,
    ent_coef_loss_and_grads,
    exploration_generator,
    make_constant,
    off_policy_networks,
    policy_noise_generator,
    sac_actor_loss_and_grads,
    sac_policy,
    soft_targets,
    weights,
)
from tests.helpers import (
    ReferenceAdam,
    assert_gradients_match,
    assert_same_checkpoint,
)

OBS_DIM, ACT_DIM = 3, 2
LOW, HIGH = np.array([-2.0, -1.0]), np.array([1.0, 3.0])  # scale [1.5, 2], bias [-0.5, 1]
SCALE, BIAS = (HIGH - LOW) / 2.0, (HIGH + LOW) / 2.0
SMALL = {
    "seed": 4,  # every hidden layer of the actor and both critics has live and dead units
    "hidden_sizes": [4, 3],
    "batch_size": 4,
    "learning_starts": 0,
    "action_low": LOW,
    "action_high": HIGH,
}
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _agent(**config) -> SACAgent:
    return SACAgent(OBS_DIM, ACT_DIM, config={**SMALL, **config})


def _batch(n: int = 6, seed: int = 2) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OBS_DIM)),
        "actions": rng.uniform(LOW, HIGH, size=(n, ACT_DIM)).astype(np.float32),
        "rewards": -rng.uniform(0.0, 5.0, size=n),
        "next_observations": rng.standard_normal((n, OBS_DIM)),
        "dones": np.array([False] * (n - 1) + [True]),
        "terminated": np.array([False] * (n - 2) + [True, False]),  # one true end mid-batch
    }


def _naive_log_prob(eps, log_std, pre_tanh) -> np.ndarray:
    """log pi of the squashed sample, written from the textbook form: the Gaussian density of u
    minus log(1 - tanh(u)²) per dimension, summed over the action. Fine at moderate u only."""
    gaussian = -0.5 * eps**2 - log_std - HALF_LOG_2PI
    return np.sum(gaussian - np.log(1.0 - np.tanh(pre_tanh) ** 2), axis=1)


# ---------------------------------------------------------------------------
# Defaults, validation, construction
# ---------------------------------------------------------------------------


def test_the_defaults_are_sb3s_sac_settings():
    agent = SACAgent(OBS_DIM, ACT_DIM, {"action_low": LOW, "action_high": HIGH})
    assert agent.hidden_sizes == [256, 256]
    assert (agent.learning_rate, agent.gamma, agent.tau) == pytest.approx((3e-4, 0.99, 0.005))
    assert (agent.batch_size, agent.buffer_capacity, agent.learning_starts) == (
        256,
        1_000_000,
        100,
    )
    assert agent.updates_per_step == 1.0
    assert agent.learns_ent_coef and agent.ent_coef == pytest.approx(1.0)
    assert agent.target_entropy == -2.0, "target entropy 'auto' is minus the action size"


@pytest.mark.parametrize(
    "setting, learned, alpha",
    [("auto", True, 1.0), ("auto_0.1", True, 0.1), (0.2, False, 0.2), ("0.05", False, 0.05)],
)
def test_ent_coef_is_learned_from_an_initial_value_or_fixed(setting, learned, alpha):
    agent = _agent(ent_coef=setting)
    assert agent.learns_ent_coef is learned
    assert agent.ent_coef == pytest.approx(alpha, rel=1e-12)


@pytest.mark.parametrize(
    "bad",
    ["autox", "auto_", "auto_-1", "auto_0", "auto_nan", 0, -0.1, float("nan"), float("inf"), True],
)
def test_an_invalid_ent_coef_is_rejected_by_name(bad):
    with pytest.raises(ValueError, match="ent_coef"):
        _agent(ent_coef=bad)


def test_target_entropy_is_minus_the_action_size_or_the_given_number():
    assert _agent(target_entropy="auto").target_entropy == -2.0
    assert _agent(target_entropy=-0.5).target_entropy == -0.5


@pytest.mark.parametrize("bad", ["minus_two", float("nan"), float("-inf"), True])
def test_an_invalid_target_entropy_is_rejected_by_name(bad):
    with pytest.raises(ValueError, match="target_entropy"):
        _agent(target_entropy=bad)


def test_the_actor_outputs_mean_and_log_std_and_the_critics_follow_it_on_the_stream():
    """Actor (O -> H -> 2A), critic 1, then critic 2, all from the seed's weights stream."""
    from rl_core.agents.mlp import init_mlp
    from rl_core.seeding import AGENT_WEIGHTS_STREAM, derive_rng

    agent = _agent()
    rng = derive_rng(SMALL["seed"], AGENT_WEIGHTS_STREAM)
    expected = {
        "actor": init_mlp([OBS_DIM, 4, 3, 2 * ACT_DIM], rng),
        "critic": init_mlp([OBS_DIM + ACT_DIM, 4, 3, 1], rng),
        "critic_2": init_mlp([OBS_DIM + ACT_DIM, 4, 3, 1], rng),
    }
    nets = off_policy_networks(agent)
    for net, params in expected.items():
        for name, value in params.items():
            np.testing.assert_array_equal(nets[net][name], value, err_msg=f"{net}_{name}")
            np.testing.assert_array_equal(nets.get(f"target_{net}", nets[net])[name], value)
    assert "target_actor" not in nets, "SAC's actor has no target network"


def test_pendulum_builds_sac():
    challenge = discover_challenges()["pendulum"]()
    env = challenge.make_env({"seed": 1}, render_mode=None)
    try:
        agent = challenge.make_agent("sac", {"seed": 1, "hidden_sizes": [4]}, env)
    finally:
        env.close()
    assert type(agent) is SACAgent
    np.testing.assert_array_equal(agent.scale, [2.0])
    assert agent.target_entropy == -1.0


# ---------------------------------------------------------------------------
# The squashed Gaussian, worked by hand
# ---------------------------------------------------------------------------


def test_the_stable_tanh_correction_equals_the_naive_form_and_stays_finite():
    u = np.linspace(-3.0, 3.0, 61)
    np.testing.assert_allclose(log_one_minus_tanh_squared(u), np.log(1.0 - np.tanh(u) ** 2))
    with np.errstate(divide="ignore"):
        assert np.isneginf(np.log(1.0 - np.tanh(30.0) ** 2)), "the naive form underflows here"
    far = log_one_minus_tanh_squared(np.array([-30.0, 30.0]))
    # 1 - tanh² u = 4 / (e^u + e^-u)², so at |u| = 30 it is log 4 - 60 to within e^-60.
    np.testing.assert_allclose(far, [math.log(4.0) - 60.0] * 2, rtol=1e-14)


def test_the_policy_sample_and_its_log_prob_match_the_textbook_form():
    agent = _agent()
    make_constant(off_policy_networks(agent)["actor"], [0.4, -0.3, math.log(0.5), 0.2])
    eps = np.array([[0.5, -1.0], [-2.0, 0.3]])
    sample = sac_policy(agent, np.zeros((2, OBS_DIM)), eps)
    log_std = np.array([math.log(0.5), 0.2])
    u = np.array([0.4, -0.3]) + np.exp(log_std) * eps
    np.testing.assert_allclose(sample.pre_tanh, u)
    np.testing.assert_allclose(sample.action, BIAS + SCALE * np.tanh(u))
    # The density is of the squashed value in [-1, 1], not of the scaled action: no log(scale).
    np.testing.assert_allclose(sample.log_prob, _naive_log_prob(eps, log_std, u), rtol=1e-12)


def test_the_squashed_density_integrates_to_one():
    """In one dimension, exp(log pi) over t = tanh(u) in (-1, 1) integrates to 1: the density is of
    the squashed value, with its Jacobian."""
    agent = SACAgent(OBS_DIM, 1, {**SMALL, "action_low": [-2.0], "action_high": [2.0]})
    make_constant(off_policy_networks(agent)["actor"], [0.3, math.log(0.8)])
    eps = np.linspace(-9.0, 9.0, 40001)[:, None]
    sample = sac_policy(agent, np.zeros((len(eps), OBS_DIM)), eps)
    t = sample.squashed[:, 0]
    assert np.all(np.diff(t) >= 0.0)
    total = np.trapezoid(np.exp(sample.log_prob), t)
    assert total == pytest.approx(1.0, abs=1e-4), f"the density integrates to {total}"


def test_the_log_std_is_clipped_to_the_shared_bounds():
    agent = _agent()
    make_constant(off_policy_networks(agent)["actor"], [0.0, 0.0, 50.0, -50.0])
    sample = sac_policy(agent, np.zeros((1, OBS_DIM)), np.zeros((1, ACT_DIM)))
    np.testing.assert_array_equal(sample.log_std, [[LOG_STD_MAX, LOG_STD_MIN]])
    assert (LOG_STD_MIN, LOG_STD_MAX) == (-20.0, 2.0)


# ---------------------------------------------------------------------------
# Targets and losses, worked by hand
# ---------------------------------------------------------------------------


def test_the_soft_target_is_the_smaller_target_critic_minus_alpha_log_pi():
    agent = _agent(gamma=0.9)
    nets = off_policy_networks(agent)
    make_constant(nets["target_critic"], 5.0)
    make_constant(nets["target_critic_2"], 3.0)
    make_constant(nets["actor"], [0.2, -0.1, math.log(0.6), 0.1])
    rewards, terminated = np.array([1.0, 2.0, 3.0]), np.array([False, True, False])
    eps_next = np.array([[0.3, -0.7], [1.1, 0.2], [-0.4, 0.9]])
    alpha = 0.4
    log_std = np.array([math.log(0.6), 0.1])
    u = np.array([0.2, -0.1]) + np.exp(log_std) * eps_next
    log_pi = _naive_log_prob(eps_next, log_std, u)
    expected = rewards + 0.9 * np.array([1.0, 0.0, 1.0]) * (3.0 - alpha * log_pi)  # min(5, 3)
    y = soft_targets(agent, rewards, np.zeros((3, OBS_DIM)), terminated, eps_next, alpha)
    np.testing.assert_allclose(y, expected, rtol=1e-12)
    assert y[1] == 2.0, "a terminal transition does not bootstrap"

    make_constant(nets["target_critic"], -4.0)  # now the first is the smaller
    y = soft_targets(agent, rewards, np.zeros((3, OBS_DIM)), terminated, eps_next, alpha)
    np.testing.assert_allclose(y[0], 1.0 + 0.9 * (-4.0 - alpha * log_pi[0]), rtol=1e-12)


def test_the_temperature_loss_and_its_gradient_by_hand():
    agent = _agent()
    off_policy_networks(agent)["log_ent_coef"]["value"][...] = math.log(0.5)
    log_prob = np.array([-1.0, 0.5, -3.5])  # mean -4/3
    components, grads = ent_coef_loss_and_grads(agent, log_prob)
    # L = -log(alpha) · mean(log pi + target), target = -2: -log 0.5 · (-4/3 - 2).
    assert components["ent_coef_loss"] == pytest.approx(-math.log(0.5) * (-4.0 / 3.0 - 2.0))
    # dL/dlog(alpha) = -mean(log pi) - target = A - mean(log pi) = 2 + 4/3.
    np.testing.assert_allclose(grads["value"], [2.0 + 4.0 / 3.0])
    assert grads["value"].shape == (1,), "log alpha is a one-parameter group of shape (1,)"


@pytest.mark.parametrize("target, falls", [("auto", True), (5.0, False)])
def test_alpha_falls_when_the_policy_beats_its_target_entropy_and_rises_when_it_falls_short(
    target, falls
):
    """A fresh actor's entropy (about 1 here) is above -2 and below an unreachable 5."""
    agent = _agent(target_entropy=target, max_updates_per_call=1)
    first = agent.update(_batch())
    second = agent.update(_batch(seed=3))
    assert -2.0 < first["entropy"] < 5.0, "the premise: between the two targets"
    assert (second["ent_coef"] < first["ent_coef"]) is falls, (
        f"alpha should {'fall' if falls else 'rise'}: {first['ent_coef']} -> {second['ent_coef']}"
    )
    assert first["ent_coef"] == pytest.approx(1.0), "the first step uses alpha from before it"


def test_the_temperature_step_is_never_clipped():
    """max_grad_norm clips the actor and critics only: log alpha takes plain Adam steps on
    dL/dlog(alpha) = -mean(log pi) - target_entropy = entropy + 2 (A = 2, target 'auto')."""
    agent = _agent(max_grad_norm=1e-9, max_updates_per_call=1)
    reference, log_alpha = ReferenceAdam(agent.learning_rate), 0.0
    for seed in range(3):
        entropy = agent.update(_batch(seed=seed))["entropy"]  # one step: its own -mean log pi
        log_alpha -= float(reference.step({"value": entropy + 2.0})["value"])
    np.testing.assert_allclose(weights(agent)["log_ent_coef_value"], [log_alpha], rtol=1e-10)


def test_a_fixed_ent_coef_is_never_trained_and_reports_no_temperature_loss():
    agent = _agent(ent_coef=0.2)
    before = weights(agent)["log_ent_coef_value"]
    metrics = agent.update(_batch())
    np.testing.assert_array_equal(weights(agent)["log_ent_coef_value"], before)
    assert "ent_coef_loss" not in metrics
    assert metrics["ent_coef"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Gradient checks
# ---------------------------------------------------------------------------


def _assert_relus_mixed_and_off_the_kink(*caches) -> None:
    """Every hidden layer has active and inactive units on the batch (or a missing ReLU mask would
    pass), and no pre-activation sits on the kink."""
    for cache in caches:
        for layer, pre in enumerate(cache["pre"][:-1]):
            assert 0.0 < np.mean(pre > 0.0) < 1.0, f"hidden layer {layer} should be mixed"
            assert np.min(np.abs(pre)) > 1e-4, f"a unit of hidden layer {layer} is on the kink"


def _balance_twin_critics(agent: SACAgent, obs: np.ndarray, actions: np.ndarray) -> None:
    """Shift critic 2's output so it is the smaller on about half the batch: both branches of the
    minimum then carry gradient."""
    nets = off_policy_networks(agent)
    q1, _ = critic_forward(agent, obs, actions, nets["critic"])
    q2, _ = critic_forward(agent, obs, actions, nets["critic_2"])
    last = max(int(name[1:]) for name in nets["critic_2"])
    nets["critic_2"][f"b{last}"] += float(np.median(q1 - q2)) + 1e-3


@pytest.mark.parametrize(
    "clip_edge", [None, LOG_STD_MAX, LOG_STD_MIN], ids=["interior", "near-max", "near-min"]
)
def test_the_actor_gradients_match_finite_differences(clip_edge):
    agent = _agent(hidden_sizes=[4, 3])
    rng = np.random.default_rng(5)
    obs = rng.standard_normal((7, OBS_DIM))
    eps = rng.standard_normal((7, ACT_DIM))
    alpha = 0.3
    actor = off_policy_networks(agent)["actor"]
    actor["b2"] += np.array([0.0, 0.0, -0.7, 0.4])  # spread the log stds apart
    if clip_edge is not None:
        # The first log std straddles the edge: some samples clipped (no gradient), some not.
        raw = sac_policy(agent, obs, eps).log_std_raw[:, 0]
        actor["b2"][ACT_DIM] += clip_edge - float(np.median(raw)) + 1e-3
    sample = sac_policy(agent, obs, eps)
    raw = sample.log_std_raw
    for edge in (LOG_STD_MIN, LOG_STD_MAX):
        assert np.min(np.abs(raw - edge)) > 1e-4, "no log std may sit on a clip edge"
    if clip_edge is not None:
        assert np.any(raw[:, 0] > clip_edge) and np.any(raw[:, 0] < clip_edge)
    _balance_twin_critics(agent, obs, sample.action)
    nets = off_policy_networks(agent)
    q1, c1 = critic_forward(agent, obs, sample.action, nets["critic"])
    q2, c2 = critic_forward(agent, obs, sample.action, nets["critic_2"])
    assert np.any(q1 < q2) and np.any(q2 < q1), "both critics should be the minimum somewhere"
    assert np.min(np.abs(q1 - q2)) > 1e-4, "no sample may sit on the minimum's kink"
    _assert_relus_mixed_and_off_the_kink(sample.cache, c1, c2)

    _, grads = sac_actor_loss_and_grads(agent, obs, eps, alpha)
    assert_gradients_match(
        lambda: sac_actor_loss_and_grads(agent, obs, eps, alpha)[0]["actor_loss"], actor, grads
    )


@pytest.mark.parametrize("critic", ["critic", "critic_2"])
def test_each_critics_gradients_match_finite_differences_on_soft_targets(critic):
    agent = _agent()
    s = _batch(n=8, seed=4)
    eps_next = np.random.default_rng(6).standard_normal((8, ACT_DIM))
    targets = soft_targets(
        agent, s["rewards"], s["next_observations"], s["terminated"], eps_next, alpha=0.37
    )
    params = off_policy_networks(agent)[critic]
    obs, actions = s["observations"], s["actions"].astype(np.float64)
    _assert_relus_mixed_and_off_the_kink(critic_forward(agent, obs, actions, params)[1])
    _, grads = critic_loss_and_grads(agent, obs, actions, targets, params)
    assert_gradients_match(
        lambda: critic_loss_and_grads(agent, obs, actions, targets, params)[0]["critic_loss"],
        params,
        grads,
    )


def test_the_temperature_gradient_matches_finite_differences():
    agent = _agent()
    log_ent_coef = off_policy_networks(agent)["log_ent_coef"]
    log_ent_coef["value"][...] = -0.8
    log_prob = np.array([-2.5, -0.4, 1.2, -3.0])
    _, grads = ent_coef_loss_and_grads(agent, log_prob)
    assert_gradients_match(
        lambda: ent_coef_loss_and_grads(agent, log_prob)[0]["ent_coef_loss"],
        log_ent_coef,
        grads,
    )


# ---------------------------------------------------------------------------
# The update: order, noise, metrics
# ---------------------------------------------------------------------------


def _replay(reference: SACAgent, batch, updates: int) -> None:
    """update() written out in SB3's order. Per step: sample; draw the actor's noise, then the
    next state's; log pi at the current actor; with alpha from before it, the temperature's Adam
    step; the soft targets; both critics' Adam steps; the actor's step against the updated
    critics; Polyak on the target critics only.

    It checks the order, the draws and what moves; the losses it borrows from the agent are
    checked by the hand-worked and gradient tests above."""
    nets = off_policy_networks(reference)
    reference.replay_buffer.add_batch(
        batch["observations"], batch["actions"], batch["rewards"],
        batch["next_observations"], batch["terminated"],
    )  # fmt: skip
    trained = ("actor", "critic", "critic_2", "log_ent_coef")
    adams = {net: ReferenceAdam(reference.learning_rate) for net in trained}
    noise_rng = policy_noise_generator(reference)
    shape = (reference.batch_size, ACT_DIM)
    for _ in range(updates):
        s = reference.replay_buffer.sample(reference.batch_size)
        obs = s["observations"]
        eps, eps_next = noise_rng.standard_normal(shape), noise_rng.standard_normal(shape)
        log_pi = sac_policy(reference, obs, eps).log_prob
        alpha = math.exp(float(nets["log_ent_coef"]["value"][0]))
        grad = -np.mean(log_pi + reference.target_entropy)
        nets["log_ent_coef"]["value"] = (
            nets["log_ent_coef"]["value"] - adams["log_ent_coef"].step({"value": grad})["value"]
        )
        y = soft_targets(
            reference, s["rewards"], s["next_observations"], s["dones"], eps_next, alpha
        )
        for net in ("critic", "critic_2"):
            _, grads = critic_loss_and_grads(reference, obs, s["actions"], y, nets[net])
            for name, delta in adams[net].step(grads).items():
                nets[net][name] = nets[net][name] - delta
        _, grads = sac_actor_loss_and_grads(reference, obs, eps, alpha)
        for name, delta in adams["actor"].step(grads).items():
            nets["actor"][name] = nets["actor"][name] - delta
        for net in ("critic", "critic_2"):
            for name in nets[net]:
                nets[f"target_{net}"][name] = (
                    reference.tau * nets[net][name]
                    + (1.0 - reference.tau) * nets[f"target_{net}"][name]
                )


def test_update_replays_sb3s_step_order():
    agent = _agent(tau=0.2, learning_rate=0.01)
    reference = copy.deepcopy(agent)
    batch = _batch()

    metrics = agent.update(batch)

    _replay(reference, batch, updates=6)
    assert_same_weights(agent, reference, rtol=1e-10)
    assert counters(agent)["gradient_steps"] == 6
    assert set(metrics) == {
        "critic_loss", "mean_q", "critic_grad_norm", "actor_loss", "actor_grad_norm",
        "saturation", "entropy", "ent_coef", "ent_coef_loss",
    }  # fmt: skip
    assert all(np.isfinite(value) for value in metrics.values())


def test_the_q_loss_falls_on_a_fixed_batch():
    agent = _agent(learning_rate=0.01)
    s = _batch(n=8, seed=9)
    eps_next = np.random.default_rng(1).standard_normal((8, ACT_DIM))
    targets = soft_targets(
        agent, s["rewards"], s["next_observations"], s["terminated"], eps_next, alpha=0.5
    )
    losses = [
        critic_step(agent, s["observations"], s["actions"], targets)["critic_loss"]
        for _ in range(60)
    ]
    assert losses[-1] < 0.5 * losses[0], f"the critic loss should fall: {losses[0]} -> {losses[-1]}"


def test_the_metrics_report_alpha_entropy_and_saturation():
    agent = _agent(max_updates_per_call=1)
    make_constant(off_policy_networks(agent)["actor"], [40.0, 0.0, -20.0, -20.0])
    metrics = agent.update(_batch())
    assert metrics["saturation"] == 0.5, "the first action saturates, the second does not"
    assert metrics["ent_coef"] == pytest.approx(1.0)
    # sigma = e^-20 and a saturated tanh: log pi is huge, so the entropy estimate -log pi is
    # far below zero.
    assert metrics["entropy"] < -30.0, "a near-deterministic policy has a very negative entropy"


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------


def test_warm_up_actions_are_uniform_in_the_bounds():
    agent = _agent(learning_starts=3)
    reference = exploration_generator(agent)
    actions = [agent.select_action(np.zeros(OBS_DIM), training=True) for _ in range(3)]
    expected = [reference.uniform(LOW, HIGH) for _ in range(3)]
    np.testing.assert_allclose(actions, np.array(expected, dtype=np.float32))


def test_a_training_action_is_a_squashed_sample_on_the_exploration_stream():
    agent = _agent()
    make_constant(off_policy_networks(agent)["actor"], [0.2, -0.4, math.log(0.5), 0.3])
    reference = exploration_generator(agent)
    updates_noise = copy.deepcopy(policy_noise_generator(agent))
    action = agent.select_action(np.zeros(OBS_DIM), training=True)
    eps = reference.standard_normal((1, ACT_DIM))[0]
    u = np.array([0.2, -0.4]) + np.exp([math.log(0.5), 0.3]) * eps
    np.testing.assert_allclose(action, (BIAS + SCALE * np.tanh(u)).astype(np.float32))
    assert policy_noise_generator(agent).bit_generator.state == updates_noise.bit_generator.state, (
        "acting must not draw from the update's policy-noise stream"
    )


def test_the_greedy_action_is_the_squashed_mean_and_the_inference_policys_action():
    agent = _agent()
    actor = off_policy_networks(agent)["actor"]
    actor["b2"] += np.array([0.3, -0.6, 1.5, -2.0])  # log stds must not leak into the action
    obs = np.random.default_rng(8).standard_normal((5, OBS_DIM))
    greedy = np.array([agent.select_action(o, training=False) for o in obs])
    mean = sac_policy(agent, obs, np.zeros((5, ACT_DIM))).mean
    np.testing.assert_allclose(greedy, BIAS + SCALE * np.tanh(mean), rtol=1e-6)
    policy = agent.inference_policy()
    np.testing.assert_allclose(policy.act(obs), greedy, rtol=1e-6)
    assert policy.output_dim == ACT_DIM and policy.head == "tanh_scale"


def test_evaluation_changes_nothing():
    agent = _agent()
    agent.update(_batch())
    before = weights(agent)
    states = (exploration_generator(agent).bit_generator.state, counters(agent))
    agent.select_action(np.zeros(OBS_DIM), training=False)
    for name, value in before.items():
        np.testing.assert_array_equal(weights(agent)[name], value)
    assert (exploration_generator(agent).bit_generator.state, counters(agent)) == states


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ent_coef", ["auto", 0.1])
def test_a_checkpoint_resumes_training_exactly(tmp_path, ent_coef):
    agent = _agent(ent_coef=ent_coef, max_updates_per_call=2)
    agent.update(_batch(seed=3))
    agent.save(str(tmp_path / "sac.pt"))
    resumed = _agent(seed=99, ent_coef=ent_coef, max_updates_per_call=2)
    resumed.update(_batch(seed=5))  # trained once, so a load that skips anything shows
    resumed.load(str(tmp_path / "sac.pt"))
    assert_same_checkpoint(agent, resumed, tmp_path)

    for run in (agent, resumed):
        run.update(_batch(seed=4))
    assert_same_weights(resumed, agent, rtol=0.0)
    with np.load(tmp_path / "sac.pt") as data:
        assert {
            "log_ent_coef_value",
            "log_ent_coef_adam_t",
            "critic_2_W0",
            "target_critic_2_W0",
            "rng_policy_noise_rng",
        } <= set(data.files), "log alpha, its Adam and the policy-noise stream are saved"
        assert data["log_ent_coef_value"].shape == (1,)


def test_a_td3_checkpoint_is_refused_by_sac(tmp_path):
    TD3Agent(OBS_DIM, ACT_DIM, SMALL).save(str(tmp_path / "td3.pt"))
    with pytest.raises(ValueError, match="not a SAC checkpoint.*log_ent_coef_value"):
        _agent().load(str(tmp_path / "td3.pt"))


@pytest.mark.parametrize("cls, name", [(TD3Agent, "TD3"), (DDPGAgent, "DDPG")])
def test_a_sac_checkpoint_is_refused_by_td3_and_ddpg(tmp_path, cls, name):
    _agent().save(str(tmp_path / "sac.pt"))
    with pytest.raises(ValueError, match=f"not a {name} checkpoint.*target_actor"):
        cls(OBS_DIM, ACT_DIM, SMALL).load(str(tmp_path / "sac.pt"))
