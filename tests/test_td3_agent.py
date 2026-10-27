"""TD3: twin critics, target-policy smoothing and delayed actor and target updates (SB3's TD3)."""

import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_core.agents import DDPGAgent, TD3Agent  # noqa: E402
from rl_core.challenges import discover_challenges  # noqa: E402
from tests.agent_internals import (  # noqa: E402
    actor_forward,
    actor_loss_and_grads,
    assert_same_weights,
    counters,
    critic_forward,
    critic_loss_and_grads,
    ddpg_networks,
    draw_target_noise,
    exploration_generator,
    replay_generator,
    target_actions,
    target_noise_generator,
    td3_targets,
    weights,
)
from tests.helpers import ReferenceAdam, assert_gradients_match  # noqa: E402

OBS_DIM, ACT_DIM = 3, 2
LOW, HIGH = np.array([-2.0, -1.0]), np.array([1.0, 3.0])  # scale [1.5, 2], bias [-0.5, 1]
SMALL = {
    "seed": 7,
    "hidden_sizes": [6, 5],
    "batch_size": 4,
    "learning_starts": 0,
    "action_low": LOW,
    "action_high": HIGH,
}


def _agent(**config) -> TD3Agent:
    return TD3Agent(OBS_DIM, ACT_DIM, config={**SMALL, **config})


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


def _constant(params: dict[str, np.ndarray], value: float) -> None:
    """Make a network output ``value`` for every input: zero weights, last bias = value."""
    for name in params:
        params[name][...] = 0.0
    last = max(int(name[1:]) for name in params)
    params[f"b{last}"][...] = value


# ---------------------------------------------------------------------------
# Defaults, validation, construction
# ---------------------------------------------------------------------------


def test_the_defaults_are_sb3s_td3_settings():
    agent = TD3Agent(OBS_DIM, ACT_DIM, {"action_low": LOW, "action_high": HIGH})
    assert (agent.policy_delay, agent.target_policy_noise, agent.target_noise_clip) == (2, 0.2, 0.5)
    assert (agent.learning_rate, agent.tau, agent.batch_size, agent.gamma) == (
        0.001,
        0.005,
        256,
        0.98,
    )


@pytest.mark.parametrize(
    "bad, key",
    [
        ({"policy_delay": 0}, "policy_delay"),
        ({"policy_delay": 1.5}, "policy_delay"),
        ({"target_policy_noise": -0.1}, "target_policy_noise"),
        ({"target_noise_clip": -0.5}, "target_noise_clip"),
        ({"target_policy_noise": float("nan")}, "target_policy_noise"),
        ({"target_noise_clip": float("inf")}, "target_noise_clip"),
    ],
)
def test_invalid_td3_settings_are_rejected_by_name(bad, key):
    with pytest.raises(ValueError, match=key):
        _agent(**bad)


def test_the_actor_and_first_critic_start_as_ddpgs_and_the_second_critic_differs():
    """Critic 2 is drawn after them from the same weights stream: a seed pairs TD3 with DDPG."""
    td3, ddpg = _agent(), DDPGAgent(OBS_DIM, ACT_DIM, SMALL)
    td3_nets, ddpg_nets = ddpg_networks(td3), ddpg_networks(ddpg)
    for net in ("actor", "critic", "target_actor", "target_critic"):
        for name, value in ddpg_nets[net].items():
            np.testing.assert_array_equal(td3_nets[net][name], value, err_msg=f"{net}_{name}")
    assert not np.allclose(td3_nets["critic_2"]["W0"], td3_nets["critic"]["W0"])
    for name, value in td3_nets["critic_2"].items():
        np.testing.assert_array_equal(td3_nets["target_critic_2"][name], value)


def test_pendulum_builds_td3():
    challenge = discover_challenges()["pendulum"]()
    env = challenge.make_env({"seed": 1}, render_mode=None)
    try:
        agent = challenge.make_agent("td3", {"seed": 1, "hidden_sizes": [4]}, env)
    finally:
        env.close()
    assert type(agent) is TD3Agent
    assert -2.0 <= agent.select_action(np.zeros(3), training=False)[0] <= 2.0


# ---------------------------------------------------------------------------
# Targets, worked by hand
# ---------------------------------------------------------------------------


def test_target_noise_is_clipped_then_the_action_is_clipped_to_the_bounds():
    agent = _agent()
    nets = ddpg_networks(agent)
    _constant(nets["target_actor"], 0.0)  # mu' = bias = [-0.5, 1]
    noise = np.array([[0.3, -0.25], [0.5, 0.5], [-0.5, -0.5]])  # already clipped to ±0.5
    expected = np.array(
        [
            [-0.5 + 1.5 * 0.3, 1.0 - 2.0 * 0.25],  # [-0.05, 0.5]
            [-0.5 + 1.5 * 0.5, 1.0 + 2.0 * 0.5],  # [0.25, 2.0]
            [-0.5 - 1.5 * 0.5, 1.0 - 2.0 * 0.5],  # [-1.25, 0.0]
        ]
    )
    np.testing.assert_allclose(target_actions(agent, np.zeros((3, OBS_DIM)), noise), expected)

    _constant(nets["target_actor"], 50.0)  # tanh saturates: mu' = high = [1, 3]
    saturated = target_actions(agent, np.zeros((3, OBS_DIM)), noise)
    # Noise pushing past the upper bound is cut there; noise pulling down is kept.
    np.testing.assert_allclose(saturated, [[1.0, 2.5], [1.0, 3.0], [0.25, 2.0]])


def test_the_drawn_noise_is_scaled_and_clipped_as_sb3_draws_it():
    agent = _agent(target_policy_noise=0.2, target_noise_clip=0.5)
    reference = copy.deepcopy(target_noise_generator(agent))
    noise = draw_target_noise(agent, 1000)
    expected = np.clip(reference.normal(0.0, 0.2, (1000, ACT_DIM)), -0.5, 0.5)
    np.testing.assert_array_equal(noise, expected)


def test_the_target_is_the_smaller_target_critic_with_a_terminal_mask():
    agent = _agent(gamma=0.9)
    nets = ddpg_networks(agent)
    _constant(nets["target_critic"], 5.0)
    _constant(nets["target_critic_2"], 3.0)
    rewards, terminated = np.array([1.0, 2.0, 3.0]), np.array([False, True, False])
    y = td3_targets(agent, rewards, np.zeros((3, OBS_DIM)), terminated, np.zeros((3, ACT_DIM)))
    np.testing.assert_allclose(y, [1.0 + 0.9 * 3.0, 2.0, 3.0 + 0.9 * 3.0])  # min(5, 3) = 3

    _constant(nets["target_critic"], -4.0)  # now the first is the smaller
    y = td3_targets(agent, rewards, np.zeros((3, OBS_DIM)), terminated, np.zeros((3, ACT_DIM)))
    np.testing.assert_allclose(y, [1.0 - 3.6, 2.0, 3.0 - 3.6])


def test_the_second_critics_gradients_match_finite_differences():
    agent = _agent()
    rng = np.random.default_rng(3)
    obs, actions, targets = (
        rng.standard_normal((8, OBS_DIM)),
        rng.uniform(LOW, HIGH, (8, 2)),
        rng.standard_normal(8),
    )
    critic_2 = ddpg_networks(agent)["critic_2"]
    _, grads = critic_loss_and_grads(agent, obs, actions, targets, critic_2)
    assert_gradients_match(
        lambda: critic_loss_and_grads(agent, obs, actions, targets, critic_2)[0]["critic_loss"],
        critic_2,
        grads,
    )


# ---------------------------------------------------------------------------
# The update: delay, order, metrics
# ---------------------------------------------------------------------------


def _replay(reference: TD3Agent, batch, updates: int) -> None:
    """update() written out: per step sample, draw target noise, both critics' Adam steps, and on
    every policy_delay-th step the actor's step against critic 1, then Polyak on all three targets.

    It checks the order, the delay, the noise draw and which targets move; the target and loss
    maths it borrows from the agent are checked by the hand-worked tests above."""
    nets = ddpg_networks(reference)
    reference.replay_buffer.add_batch(
        batch["observations"], batch["actions"], batch["rewards"],
        batch["next_observations"], batch["terminated"],
    )  # fmt: skip
    adams = {net: ReferenceAdam(reference.learning_rate) for net in ("actor", "critic", "critic_2")}
    noise_rng = target_noise_generator(reference)
    for step in range(1, updates + 1):
        s = reference.replay_buffer.sample(reference.batch_size)
        noise = np.clip(
            noise_rng.normal(0.0, reference.target_policy_noise, (reference.batch_size, ACT_DIM)),
            -reference.target_noise_clip,
            reference.target_noise_clip,
        )  # written out, not draw_target_noise: the replay checks the draw too
        y = td3_targets(reference, s["rewards"], s["next_observations"], s["dones"], noise)
        for net in ("critic", "critic_2"):
            _, grads = critic_loss_and_grads(
                reference, s["observations"], s["actions"], y, nets[net]
            )
            for name, delta in adams[net].step(grads).items():
                nets[net][name] = nets[net][name] - delta
        if step % reference.policy_delay == 0:
            _, grads = actor_loss_and_grads(reference, s["observations"])
            for name, delta in adams["actor"].step(grads).items():
                nets["actor"][name] = nets["actor"][name] - delta
            for net in ("actor", "critic", "critic_2"):
                for name in nets[net]:
                    nets[f"target_{net}"][name] = (
                        reference.tau * nets[net][name]
                        + (1.0 - reference.tau) * nets[f"target_{net}"][name]
                    )


@pytest.mark.parametrize("policy_delay", [1, 2, 3])
def test_update_replays_as_twin_critics_then_the_delayed_actor_and_targets(policy_delay):
    agent = _agent(tau=0.2, policy_delay=policy_delay)
    reference = copy.deepcopy(agent)
    batch = _batch()

    metrics = agent.update(batch)

    _replay(reference, batch, updates=6)
    assert_same_weights(agent, reference, rtol=1e-10)
    assert counters(agent)["gradient_steps"] == 6
    assert set(metrics) == {
        "critic_loss", "mean_q", "critic_grad_norm",
        "actor_loss", "saturation", "actor_grad_norm", "actor_updates",
    }  # fmt: skip
    assert metrics["actor_updates"] == 6 // policy_delay
    assert all(np.isfinite(value) for value in metrics.values())


def test_the_actor_and_targets_wait_for_every_second_step_across_updates():
    """SB3 counts the step before checking it: the first step trains the critics only."""
    agent = _agent(max_updates_per_call=1)
    before = weights(agent)
    first = agent.update(_batch(seed=3))
    after_first = weights(agent)
    for net in ("actor", "target_actor", "target_critic", "target_critic_2"):
        for name in ddpg_networks(agent)[net]:
            key = f"{net}_{name}"
            np.testing.assert_array_equal(after_first[key], before[key], err_msg=f"{key} moved")
    assert not np.allclose(after_first["critic_2_W0"], before["critic_2_W0"]), "critics train"
    assert "actor_loss" not in first, "no actor step, so no actor metrics in this call"

    second = agent.update(_batch(seed=4))
    after_second = weights(agent)
    for key in ("actor_W0", "target_actor_W0", "target_critic_W0", "target_critic_2_W0"):
        assert not np.allclose(after_second[key], after_first[key]), f"{key} should move on step 2"
    assert second["actor_updates"] == 1


def test_the_critic_loss_is_the_mean_of_the_two_critics():
    agent = _agent(max_updates_per_call=1)
    nets = ddpg_networks(agent)
    _constant(nets["critic"], 1.0)
    _constant(nets["critic_2"], 3.0)
    _constant(nets["target_critic"], 0.0)
    _constant(nets["target_critic_2"], 0.0)
    batch = _batch(n=4)
    batch["rewards"] = np.zeros(4)
    metrics = agent.update(batch)  # y = 0: losses ½·1 and ½·9, so their mean is 2.5
    assert metrics["critic_loss"] == pytest.approx(2.5)
    assert metrics["mean_q"] == pytest.approx(1.0), "mean_q is the first critic's"


# ---------------------------------------------------------------------------
# Randomness and checkpoints
# ---------------------------------------------------------------------------


def test_target_noise_has_its_own_stream():
    """Exploration and replay draw exactly as DDPG's with the same seed: target noise touches neither."""
    td3, ddpg = (
        _agent(max_updates_per_call=2),
        DDPGAgent(OBS_DIM, ACT_DIM, {**SMALL, "max_updates_per_call": 2}),
    )
    for agent in (td3, ddpg):
        for _ in range(3):
            agent.select_action(np.zeros(OBS_DIM), training=True)
        agent.update(_batch())
    for generator in (replay_generator, exploration_generator):
        assert generator(td3).bit_generator.state == generator(ddpg).bit_generator.state, (
            f"{generator.__name__} should have drawn exactly as DDPG's"
        )


def test_a_checkpoint_resumes_mid_delay(tmp_path):
    """Saved after an odd step, the next step is the delayed one: the actor and targets move."""
    agent = _agent(max_updates_per_call=1)
    agent.update(_batch(seed=3))
    agent.end_episode()
    agent.save(str(tmp_path / "td3.pt"))
    resumed = _agent(seed=99, max_updates_per_call=1)
    resumed.load(str(tmp_path / "td3.pt"))
    assert counters(resumed) == counters(agent)

    for run in (agent, resumed):
        run.update(_batch(seed=4))
    assert_same_weights(resumed, agent, rtol=0.0)
    with np.load(tmp_path / "td3.pt") as data:
        assert {
            "critic_2_W0",
            "target_critic_2_W0",
            "critic_2_adam_t",
            "rng_target_noise_rng",
        } <= set(data.files)


def test_a_ddpg_checkpoint_is_refused_by_td3(tmp_path):
    DDPGAgent(OBS_DIM, ACT_DIM, SMALL).save(str(tmp_path / "ddpg.pt"))
    with pytest.raises(ValueError, match="not a TD3 checkpoint"):
        _agent().load(str(tmp_path / "ddpg.pt"))


def test_q_of_the_second_critic_is_computed_from_its_own_weights():
    agent = _agent()
    nets = ddpg_networks(agent)
    _constant(nets["critic_2"], 7.0)
    q, _ = critic_forward(agent, np.zeros((2, OBS_DIM)), np.zeros((2, ACT_DIM)), nets["critic_2"])
    np.testing.assert_allclose(q, [7.0, 7.0])


def test_the_second_critic_continues_the_seeds_weights_stream():
    """Actor, critic 1, then critic 2, all from the seed's weights stream in that order."""
    from rl_core.agents.mlp import init_mlp
    from rl_core.seeding import AGENT_WEIGHTS_STREAM, derive_rng

    rng = derive_rng(SMALL["seed"], AGENT_WEIGHTS_STREAM)
    init_mlp([OBS_DIM, 6, 5, ACT_DIM], rng)  # the actor
    init_mlp([OBS_DIM + ACT_DIM, 6, 5, 1], rng)  # the first critic
    expected = init_mlp([OBS_DIM + ACT_DIM, 6, 5, 1], rng)
    critic_2 = ddpg_networks(_agent())["critic_2"]
    for name, value in expected.items():
        np.testing.assert_array_equal(critic_2[name], value, err_msg=f"critic_2_{name}")


def test_without_target_noise_the_target_action_is_the_target_actors():
    agent = _agent(target_policy_noise=0.0)
    noise = draw_target_noise(agent, 5)
    np.testing.assert_array_equal(noise, np.zeros((5, ACT_DIM)))
    next_obs = np.random.default_rng(1).standard_normal((5, OBS_DIM))
    mu = target_actions(agent, next_obs, noise)
    expected = np.clip(
        actor_forward(agent, next_obs, ddpg_networks(agent)["target_actor"])[0], LOW, HIGH
    )
    np.testing.assert_allclose(mu, expected)
