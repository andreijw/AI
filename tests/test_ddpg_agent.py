"""DDPG: deterministic actor, Q critic on [s, a], Polyak targets (SB3 DDPG as rl-zoo configures it).

Notation: actor u = tanh(mlp(s)), mu(s) = bias + scale * u with scale = (high - low) / 2 and
bias = (high + low) / 2; critic Q(s, a) = mlp([s, a]); target y = r + gamma (1 - terminated) Q'(s', mu'(s')).
Critic loss mean ½ (Q - y)²; actor loss -mean Q(s, mu(s)) with the critic held constant.
"""

import copy

import numpy as np
import pytest

from rl_core.agents import DDPGAgent
from tests.agent_internals import (
    actor_forward,
    actor_loss_and_grads,
    assert_same_weights,
    assert_weights_unchanged,
    bellman_targets,
    counters,
    critic_forward,
    critic_loss_and_grads,
    ddpg_networks,
    exploration_generator,
    polyak_update,
    replay_generator,
    weights,
)
from tests.helpers import ReferenceAdam, assert_gradients_match

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


def _agent(**config) -> DDPGAgent:
    return DDPGAgent(OBS_DIM, ACT_DIM, config={**SMALL, **config})


def _data(n: int = 8, seed: int = 1):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, OBS_DIM))
    actions = rng.uniform(LOW, HIGH, size=(n, ACT_DIM))
    targets = rng.standard_normal(n) * 2.0
    return obs, actions, targets


def _batch(n: int = 6, seed: int = 2, **extra) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = {
        "observations": rng.standard_normal((n, OBS_DIM)),
        "actions": rng.uniform(LOW, HIGH, size=(n, ACT_DIM)).astype(np.float32),
        "rewards": -rng.uniform(0.0, 5.0, size=n),
        "next_observations": rng.standard_normal((n, OBS_DIM)),
        "dones": np.array([False] * (n - 1) + [True]),  # Pendulum: the episode is cut ...
        "terminated": np.zeros(n, dtype=bool),  # ... never terminated
    }
    batch.update(extra)
    return batch


def _constant(params: dict[str, np.ndarray], value: float) -> None:
    """Make a network output ``value`` for every input: zero weights, last bias = value."""
    for name in params:
        params[name][...] = 0.0
    last = max(int(name[1:]) for name in params)
    params[f"b{last}"][...] = value


# ---------------------------------------------------------------------------
# Gradient checks and hand-computed values
# ---------------------------------------------------------------------------


def _assert_off_kinks(agent: DDPGAgent, obs, actions) -> None:
    _, critic_cache = critic_forward(agent, obs, actions)
    mu, u, actor_cache = actor_forward(agent, obs)
    _, mu_cache = critic_forward(agent, obs, mu)
    for cache in (critic_cache, actor_cache, mu_cache):
        for pre in cache["pre"][:-1]:
            assert np.abs(pre).min() > 1e-4, (
                "fixture sits on a ReLU kink (finite-difference step 1e-5)"
            )
    assert np.abs(u).max() < 0.95, "fixture saturates tanh"


def test_critic_gradients_match_finite_differences():
    agent = _agent()
    obs, actions, targets = _data()
    _assert_off_kinks(agent, obs, actions)
    _, grads = critic_loss_and_grads(agent, obs, actions, targets)
    assert_gradients_match(
        lambda: critic_loss_and_grads(agent, obs, actions, targets)[0]["critic_loss"],
        ddpg_networks(agent)["critic"],
        grads,
    )


def test_actor_gradients_flow_through_the_critic_and_match_finite_differences():
    agent = _agent()
    obs, actions, _ = _data()
    _assert_off_kinks(agent, obs, actions)
    _, grads = actor_loss_and_grads(agent, obs)
    assert_gradients_match(
        lambda: actor_loss_and_grads(agent, obs)[0]["actor_loss"],
        ddpg_networks(agent)["actor"],
        grads,
    )


def test_computing_losses_changes_no_weights():
    agent = _agent()
    nets = ddpg_networks(agent)
    obs, actions, targets = _data()
    before = copy.deepcopy(
        [
            nets["actor"],
            nets["critic"],
            nets["target_actor"],
            nets["target_critic"],
        ]
    )
    critic_loss_and_grads(agent, obs, actions, targets)
    actor_loss_and_grads(agent, obs)
    after = [
        nets["actor"],
        nets["critic"],
        nets["target_actor"],
        nets["target_critic"],
    ]
    for old, new in zip(before, after, strict=True):
        for name in old:
            np.testing.assert_array_equal(new[name], old[name])


def test_losses_match_a_hand_computation_with_a_constant_critic():
    agent = _agent()
    _constant(ddpg_networks(agent)["critic"], 0.5)
    obs, actions, _ = _data(n=4)
    components, _ = critic_loss_and_grads(agent, obs, actions, np.array([1.0, 2.0, 3.0, 4.0]))
    # ½ mean((0.5 - y)²) = ½ (0.25 + 2.25 + 6.25 + 12.25) / 4 = 2.625
    assert components["critic_loss"] == pytest.approx(2.625)
    assert components["mean_q"] == pytest.approx(0.5)
    assert actor_loss_and_grads(agent, obs)[0]["actor_loss"] == pytest.approx(-0.5)


def test_actor_output_is_tanh_scaled_to_the_bounds():
    agent = _agent()
    nets = ddpg_networks(agent)
    _constant(nets["actor"], 0.0)
    nets["actor"][f"b{len(agent.hidden_sizes)}"][...] = [
        np.arctanh(0.5),
        np.arctanh(-0.5),
    ]
    mu, u, _ = actor_forward(agent, np.zeros((1, OBS_DIM)))
    np.testing.assert_allclose(u, [[0.5, -0.5]])
    np.testing.assert_allclose(
        mu, [[-0.5 + 1.5 * 0.5, 1.0 - 2.0 * 0.5]], atol=1e-12
    )  # bias + scale·u


def test_bellman_targets_use_the_target_networks_and_bootstrap_unless_terminated():
    agent = _agent(gamma=0.9)
    nets = ddpg_networks(agent)
    _constant(nets["critic"], 5.0)  # the online critic must not be used
    _constant(nets["target_critic"], 2.0)
    targets = bellman_targets(
        agent, np.array([1.0, -1.0]), np.zeros((2, OBS_DIM)), np.array([False, True])
    )
    np.testing.assert_allclose(targets, [1.0 + 0.9 * 2.0, -1.0])


def test_polyak_moves_each_target_a_fraction_tau_towards_the_online_network():
    agent = _agent(tau=0.3)
    nets = ddpg_networks(agent)
    for online, target in (
        (nets["actor"], nets["target_actor"]),
        (nets["critic"], nets["target_critic"]),
    ):
        for name in online:
            online[name][...] = 1.0
            target[name][...] = -1.0
    polyak_update(agent)
    for target in (nets["target_actor"], nets["target_critic"]):
        for name in target:
            np.testing.assert_allclose(target[name], 0.3 * 1.0 + 0.7 * -1.0)
    assert not np.shares_memory(nets["target_actor"]["W0"], nets["actor"]["W0"])


def test_one_actor_step_raises_the_critic_value_of_the_policy():
    agent = _agent()
    nets = ddpg_networks(agent)
    obs, _, _ = _data()
    before, grads = actor_loss_and_grads(agent, obs)
    for name in nets["actor"]:
        nets["actor"][name] -= 1e-3 * grads[name]
    after, _ = actor_loss_and_grads(agent, obs)
    assert after["actor_loss"] < before["actor_loss"], "Q(s, mu(s)) should rise"


def test_the_critic_can_fit_a_fixed_batch():
    agent = _agent()
    obs, actions, targets = _data()
    adam = ReferenceAdam(0.01)
    first = critic_loss_and_grads(agent, obs, actions, targets)[0]["critic_loss"]
    for _ in range(500):
        _, grads = critic_loss_and_grads(agent, obs, actions, targets)
        for name, step in adam.step(grads).items():
            ddpg_networks(agent)["critic"][name] -= step
    last = critic_loss_and_grads(agent, obs, actions, targets)[0]["critic_loss"]
    assert last < 0.05 * first


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------

OBS = np.array([0.2, -0.4, 0.1])


def _mu(agent: DDPGAgent) -> np.ndarray:
    return actor_forward(agent, OBS[None, :])[0][0]


def test_evaluation_acts_with_mu_and_uses_up_no_exploration():
    agent = _agent(learning_starts=5)
    action = agent.select_action(OBS, training=False)
    assert action.dtype == np.float32 and action.shape == (ACT_DIM,)
    np.testing.assert_allclose(action, _mu(agent), rtol=1e-6)
    assert counters(agent)["training_actions"] == 0


def test_warm_up_actions_are_uniform_in_the_bounds_and_then_mu_plus_noise():
    agent = _agent(learning_starts=3, noise_std=0.0)
    warm = [agent.select_action(OBS, training=True) for _ in range(3)]
    for action in warm:
        assert np.all(action >= LOW) and np.all(action <= HIGH)
        assert not np.allclose(action, _mu(agent)), "warm-up must not act from the policy"
    np.testing.assert_allclose(agent.select_action(OBS, training=True), _mu(agent), rtol=1e-6)
    assert counters(agent)["training_actions"] == 4


def test_warm_up_covers_the_whole_action_range():
    agent = _agent(learning_starts=2000)
    actions = np.array([agent.select_action(OBS, training=True) for _ in range(2000)])
    np.testing.assert_allclose(actions.min(axis=0), LOW, atol=0.02)
    np.testing.assert_allclose(actions.max(axis=0), HIGH, atol=0.02)


def test_noise_is_gaussian_with_noise_std_in_the_scaled_action_space():
    agent = _agent(noise_std=0.1)
    _constant(ddpg_networks(agent)["actor"], 0.0)  # mu = bias, far from both bounds
    actions = np.array([agent.select_action(OBS, training=True) for _ in range(4000)])
    np.testing.assert_allclose(actions.mean(axis=0), [-0.5, 1.0], atol=0.02)
    np.testing.assert_allclose(actions.std(axis=0), 0.1 * np.array([1.5, 2.0]), rtol=0.08)


def test_gaussian_noise_is_drawn_exactly_as_before_the_noise_option():
    """The default path draws normal(0, noise_std, A) times scale once per action, so Pendulum DDPG
    runs and checkpoints repeat bit for bit."""
    agent = _agent(noise_std=0.1)
    _constant(ddpg_networks(agent)["actor"], 0.0)  # mu = bias
    rng = exploration_generator(agent)
    for _ in range(3):
        expected = np.clip(_mu(agent) + rng.normal(0.0, 0.1, ACT_DIM) * agent.scale, LOW, HIGH)
        np.testing.assert_array_equal(
            agent.select_action(OBS, training=True), expected.astype(np.float32)
        )


OU = {"noise_type": "ornstein_uhlenbeck", "noise_std": 0.2, "ou_theta": 0.5, "ou_dt": 0.1}


def _ou_increments(agent: DDPGAgent, steps: int) -> list[np.ndarray]:
    """The OU states the agent's next ``steps`` training actions should carry, from a zero start."""
    rng, x, states = exploration_generator(agent), np.zeros(ACT_DIM), []
    for _ in range(steps):
        x = x - 0.5 * x * 0.1 + 0.2 * np.sqrt(0.1) * rng.standard_normal(ACT_DIM)
        states.append(x.copy())
    return states


def _noise_of(agent: DDPGAgent) -> np.ndarray:
    """A training action minus the greedy one, in the scaled [-1, 1] space."""
    action = agent.select_action(OBS, training=True)
    return (action - agent.select_action(OBS, training=False)) / agent.scale


def test_ou_noise_is_added_to_mu_in_the_scaled_action_space():
    agent = _agent(**OU)
    _constant(ddpg_networks(agent)["actor"], 0.0)  # mu = bias, so no action is clipped
    expected = _ou_increments(agent, 5)
    for x in expected:
        np.testing.assert_allclose(_noise_of(agent), x, atol=1e-6, err_msg="mu + scale * x_t")


def test_ou_training_actions_are_correlated_from_step_to_step():
    agent = _agent(**{**OU, "ou_theta": 0.15, "ou_dt": 0.01, "noise_std": 0.5})
    _constant(ddpg_networks(agent)["actor"], 0.0)  # mu = bias, far from the bounds
    noise = np.array([_noise_of(agent)[0] for _ in range(400)])
    assert np.corrcoef(noise[:-1], noise[1:])[0, 1] > 0.9, (
        "OU noise should hold its direction across steps; white noise would not"
    )


@pytest.mark.parametrize("batch_change", ["empty", "too_few_to_sample", "trains"])
def test_end_episode_resets_the_ou_noise_for_the_next_episode(batch_change):
    """The Trainer calls end_episode after every training episode's update, so the next action
    starts the OU process from zero again, whether or not the update trained."""
    agent = _agent(**OU, batch_size=64 if batch_change == "too_few_to_sample" else 4)
    _constant(ddpg_networks(agent)["actor"], 0.0)  # mu = bias, so no action is clipped
    for _ in range(25):
        agent.select_action(OBS, training=True)
    batch = _batch()
    if batch_change == "empty":
        batch = {key: value[:0] for key, value in batch.items()}
    metrics = agent.update(batch)
    assert bool(metrics) == (batch_change == "trains"), "only the full batch should train"
    agent.end_episode()
    expected = _ou_increments(agent, 1)[0]  # before acting: it copies the generator's state
    np.testing.assert_allclose(_noise_of(agent), expected, atol=1e-6, err_msg="one step from zero")


def test_update_alone_does_not_reset_the_ou_noise():
    """Resetting is the episode's end, not the update's: an agent updating mid-episode keeps its
    noise process running."""
    agent = _agent(**OU, batch_size=4)
    _constant(ddpg_networks(agent)["actor"], 0.0)
    for _ in range(25):
        agent.select_action(OBS, training=True)
    agent.update(_batch())
    from_zero = _ou_increments(agent, 1)[0]
    assert not np.allclose(_noise_of(agent), from_zero, atol=1e-6), (
        "Without end_episode the OU process should continue from its state, not restart at zero."
    )


def test_evaluation_does_not_advance_the_ou_noise():
    agent, reference = _agent(**OU), _agent(**OU)
    # The twin never evaluates: if evaluation drew noise or stepped the OU state, they would part.
    for _ in range(4):
        agent.select_action(OBS, training=False)
        np.testing.assert_array_equal(
            agent.select_action(OBS, training=True), reference.select_action(OBS, training=True)
        )


def test_the_noise_option_adds_nothing_to_the_checkpoint(tmp_path):
    """The OU state is zero at every save (the Trainer saves after end_episode), so the format is
    unchanged."""
    paths = {}
    for name, extra in (("gaussian", {}), ("ou", OU)):
        agent = _agent(**extra)
        agent.select_action(OBS, training=True)
        agent.update(_batch())
        agent.end_episode()
        paths[name] = tmp_path / f"{name}.npz"
        agent.save(str(paths[name]))
    with np.load(paths["gaussian"]) as gaussian, np.load(paths["ou"]) as ou:
        assert sorted(ou.files) == sorted(gaussian.files)


def test_training_actions_stay_in_the_bounds_when_the_actor_is_saturated():
    agent = _agent(noise_std=0.5)
    nets = ddpg_networks(agent)
    _constant(nets["actor"], 0.0)
    nets["actor"][f"b{len(agent.hidden_sizes)}"][...] = [
        50.0,
        -50.0,
    ]  # mu at high[0] and low[1]
    actions = np.array([agent.select_action(OBS, training=True) for _ in range(200)])
    assert np.all(actions >= LOW.astype(np.float32)) and np.all(actions <= HIGH.astype(np.float32))
    assert (actions[:, 0] == np.float32(HIGH[0])).any(), "noise past the bound should be clipped"


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


def test_during_warm_up_update_stores_transitions_and_trains_nothing():
    agent = _agent(learning_starts=10)
    before = weights(agent)
    assert agent.update(_batch()) == {}
    assert len(agent.replay_buffer) == 6
    assert_weights_unchanged(agent, before, "without training")


@pytest.mark.parametrize("batch_change", ["empty", "no_next_observations"])
def test_a_batch_that_stores_nothing_trains_nothing(batch_change):
    agent = _agent()
    agent.update(_batch())  # warm buffer, past learning_starts
    before = weights(agent)
    batch = _batch()
    if batch_change == "empty":
        batch = {key: value[:0] for key, value in batch.items()}
    else:
        del batch["next_observations"]
    assert agent.update(batch) == {}
    assert_weights_unchanged(agent, before, "without training")


def test_training_starts_with_the_update_after_learning_starts_actions():
    agent = _agent(learning_starts=4)
    for _ in range(3):
        agent.select_action(OBS, training=True)
    assert agent.update(_batch()) == {}, "3 of 4 warm-up actions: still warming up"
    agent.select_action(OBS, training=True)
    assert agent.update(_batch(seed=3)) != {}


def test_a_time_limit_cut_is_stored_as_not_terminal():
    agent = _agent(learning_starts=100)
    agent.update(_batch())
    assert not agent.replay_buffer.dones[:6].any()


def _replay(reference: DDPGAgent, batch, updates: int) -> None:
    """update() written out: store; per step sample, targets, critic Adam step, then the actor
    step against the updated critic, then Polyak on both target networks."""
    reference_nets = ddpg_networks(reference)
    reference.replay_buffer.add_batch(
        batch["observations"], batch["actions"], batch["rewards"],
        batch["next_observations"], batch["terminated"],
    )  # fmt: skip
    critic_adam = ReferenceAdam(reference.learning_rate)
    actor_adam = ReferenceAdam(reference.learning_rate)
    for _ in range(updates):
        s = reference.replay_buffer.sample(reference.batch_size)
        y = bellman_targets(reference, s["rewards"], s["next_observations"], s["dones"])
        _, critic_grads = critic_loss_and_grads(reference, s["observations"], s["actions"], y)
        for name, step in critic_adam.step(critic_grads).items():
            reference_nets["critic"][name] = reference_nets["critic"][name] - step
        _, actor_grads = actor_loss_and_grads(reference, s["observations"])
        for name, step in actor_adam.step(actor_grads).items():
            reference_nets["actor"][name] = reference_nets["actor"][name] - step
        for online, target in (
            (reference_nets["actor"], reference_nets["target_actor"]),
            (reference_nets["critic"], reference_nets["target_critic"]),
        ):
            for name in online:
                target[name] = reference.tau * online[name] + (1 - reference.tau) * target[name]


@pytest.mark.parametrize(
    ("config", "expected_updates"),
    [
        ({"updates_per_step": 0.5}, 3),  # 6 transitions * 0.5
        ({"updates_per_step": 1.0, "max_updates_per_call": 2}, 2),  # capped
        ({}, 6),  # default: one gradient step per transition
    ],
)
def test_update_replays_as_critic_then_actor_then_polyak(config, expected_updates):
    agent = _agent(tau=0.2, **config)
    reference = copy.deepcopy(agent)
    batch = _batch()

    metrics = agent.update(batch)

    _replay(reference, batch, expected_updates)
    assert_same_weights(agent, reference, rtol=1e-10)
    assert counters(agent)["gradient_steps"] == expected_updates
    assert set(metrics) == {
        "critic_loss", "actor_loss", "mean_q", "saturation", "critic_grad_norm", "actor_grad_norm"
    }  # fmt: skip
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["saturation"] <= 1.0


# ---------------------------------------------------------------------------
# Configuration, randomness and checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"optimizer": "sgd"},
        {"tau": 0.0},
        {"tau": 1.5},
        {"gamma": 0.0},
        {"learning_rate": 0.0},
        {"learning_starts": -1},
        {"noise_std": -0.1},
        {"noise_type": "pink"},
        {"noise_type": "ornstein_uhlenbeck", "ou_theta": 20.0, "ou_dt": 0.1},
        {"ou_dt": 0.01},
        {"hidden_sizes": []},
        {"hidden_sizes": [4, 0]},
        {"hidden_sizes": 64},
        {"batch_size": 0},
        {"updates_per_step": 0.0},
        {"max_updates_per_call": 0},
        {"action_high": np.inf},
        {"action_low": 3.0},
    ],
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        _agent(**bad)


def test_the_defaults_are_rl_zoos_pendulum_settings():
    agent = DDPGAgent(OBS_DIM, ACT_DIM, config={})
    assert agent.hidden_sizes == [400, 300]
    assert (agent.learning_rate, agent.tau, agent.batch_size) == (1e-3, 0.005, 256)
    assert agent.noise_std == 0.1


def test_weights_exploration_and_replay_draw_from_separate_streams():
    """One seed must not make the noise repeat the numbers that initialised the weights."""
    agent = _agent(seed=11)
    plain = np.random.default_rng(11).standard_normal(4)
    explore = exploration_generator(agent).standard_normal(4)
    replay = replay_generator(agent).standard_normal(4)
    assert not np.allclose(explore, plain)
    assert not np.allclose(replay, plain)
    assert not np.allclose(replay, explore)


def test_two_agents_with_one_seed_act_and_learn_identically():
    first, second = _agent(seed=3), _agent(seed=3)
    for agent in (first, second):
        for _ in range(4):
            agent.select_action(OBS, training=True)
        agent.update(_batch())
    for name in ddpg_networks(first)["actor"]:
        np.testing.assert_array_equal(
            ddpg_networks(first)["actor"][name], ddpg_networks(second)["actor"][name]
        )


@pytest.mark.parametrize("noise", [{}, OU], ids=["gaussian", "ou"])
def test_a_loaded_agent_resumes_weights_optimizers_and_counters(tmp_path, noise):
    """The checkpoint alone resumes training: networks, optimizers, counters, replay buffer and
    generators; the OU state needs nothing saved, as end_episode has just reset it."""
    agent = _agent(learning_starts=2, **noise)
    for _ in range(3):
        agent.select_action(OBS, training=True)
    agent.update(_batch(seed=1))
    agent.end_episode()
    path = str(tmp_path / "ddpg.pt")
    agent.save(path)

    resumed = _agent(learning_starts=2, seed=99, **noise)
    resumed.load(path)  # the checkpoint holds the replay buffer and every generator
    assert counters(resumed) == {"training_actions": 3, "gradient_steps": 6}

    actions = [
        [run.select_action(OBS, training=True) for _ in range(3)] for run in (agent, resumed)
    ]
    np.testing.assert_array_equal(actions[1], actions[0], err_msg="the next training actions")
    for run in (agent, resumed):
        run.update(_batch(seed=5))
    assert_same_weights(resumed, agent, rtol=0.0)


def test_a_checkpoint_with_other_hidden_sizes_is_rejected(tmp_path):
    path = str(tmp_path / "ddpg.pt")
    _agent().save(path)
    with pytest.raises(ValueError, match="hidden_sizes, observation_dim or action_dim"):
        _agent(hidden_sizes=[4, 3]).load(path)
