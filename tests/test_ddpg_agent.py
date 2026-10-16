"""DDPG: deterministic actor, Q critic on [s, a], Polyak targets (SB3 DDPG as rl-zoo configures it).

Notation: actor u = tanh(mlp(s)), mu(s) = bias + scale * u with scale = (high - low) / 2 and
bias = (high + low) / 2; critic Q(s, a) = mlp([s, a]); target y = r + gamma (1 - terminated) Q'(s', mu'(s')).
Critic loss mean ½ (Q - y)²; actor loss -mean Q(s, mu(s)) with the critic held constant.
"""

import copy

import numpy as np
import pytest

from rl_core.agents import DDPGAgent
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
    _, critic_cache = agent._critic_forward(obs, actions, agent._critic)
    mu, u, actor_cache = agent._actor_forward(obs, agent._actor)
    _, mu_cache = agent._critic_forward(obs, mu, agent._critic)
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
    _, grads = agent._critic_loss_and_grads(obs, actions, targets)
    assert_gradients_match(
        lambda: agent._critic_loss_and_grads(obs, actions, targets)[0]["critic_loss"],
        agent._critic,
        grads,
    )


def test_actor_gradients_flow_through_the_critic_and_match_finite_differences():
    agent = _agent()
    obs, actions, _ = _data()
    _assert_off_kinks(agent, obs, actions)
    _, grads = agent._actor_loss_and_grads(obs)
    assert_gradients_match(
        lambda: agent._actor_loss_and_grads(obs)[0]["actor_loss"], agent._actor, grads
    )


def test_computing_losses_changes_no_weights():
    agent = _agent()
    obs, actions, targets = _data()
    before = copy.deepcopy([agent._actor, agent._critic, agent._target_actor, agent._target_critic])
    agent._critic_loss_and_grads(obs, actions, targets)
    agent._actor_loss_and_grads(obs)
    after = [agent._actor, agent._critic, agent._target_actor, agent._target_critic]
    for old, new in zip(before, after, strict=True):
        for name in old:
            np.testing.assert_array_equal(new[name], old[name])


def test_losses_match_a_hand_computation_with_a_constant_critic():
    agent = _agent()
    _constant(agent._critic, 0.5)
    obs, actions, _ = _data(n=4)
    components, _ = agent._critic_loss_and_grads(obs, actions, np.array([1.0, 2.0, 3.0, 4.0]))
    # ½ mean((0.5 - y)²) = ½ (0.25 + 2.25 + 6.25 + 12.25) / 4 = 2.625
    assert components["critic_loss"] == pytest.approx(2.625)
    assert components["mean_q"] == pytest.approx(0.5)
    assert agent._actor_loss_and_grads(obs)[0]["actor_loss"] == pytest.approx(-0.5)


def test_actor_output_is_tanh_scaled_to_the_bounds():
    agent = _agent()
    _constant(agent._actor, 0.0)
    agent._actor[f"b{len(agent.hidden_sizes)}"][...] = [np.arctanh(0.5), np.arctanh(-0.5)]
    mu, u, _ = agent._actor_forward(np.zeros((1, OBS_DIM)), agent._actor)
    np.testing.assert_allclose(u, [[0.5, -0.5]])
    np.testing.assert_allclose(
        mu, [[-0.5 + 1.5 * 0.5, 1.0 - 2.0 * 0.5]], atol=1e-12
    )  # bias + scale·u


def test_bellman_targets_use_the_target_networks_and_bootstrap_unless_terminated():
    agent = _agent(gamma=0.9)
    _constant(agent._critic, 5.0)  # the online critic must not be used
    _constant(agent._target_critic, 2.0)
    targets = agent._bellman_targets(
        np.array([1.0, -1.0]), np.zeros((2, OBS_DIM)), np.array([False, True])
    )
    np.testing.assert_allclose(targets, [1.0 + 0.9 * 2.0, -1.0])


def test_polyak_moves_each_target_a_fraction_tau_towards_the_online_network():
    agent = _agent(tau=0.3)
    for online, target in (
        (agent._actor, agent._target_actor),
        (agent._critic, agent._target_critic),
    ):
        for name in online:
            online[name][...] = 1.0
            target[name][...] = -1.0
    agent._polyak()
    for target in (agent._target_actor, agent._target_critic):
        for name in target:
            np.testing.assert_allclose(target[name], 0.3 * 1.0 + 0.7 * -1.0)
    assert not np.shares_memory(agent._target_actor["W0"], agent._actor["W0"])


def test_one_actor_step_raises_the_critic_value_of_the_policy():
    agent = _agent()
    obs, _, _ = _data()
    before, grads = agent._actor_loss_and_grads(obs)
    for name in agent._actor:
        agent._actor[name] -= 1e-3 * grads[name]
    after, _ = agent._actor_loss_and_grads(obs)
    assert after["actor_loss"] < before["actor_loss"], "Q(s, mu(s)) should rise"


def test_the_critic_can_fit_a_fixed_batch():
    agent = _agent()
    obs, actions, targets = _data()
    adam = ReferenceAdam(0.01)
    first = agent._critic_loss_and_grads(obs, actions, targets)[0]["critic_loss"]
    for _ in range(500):
        _, grads = agent._critic_loss_and_grads(obs, actions, targets)
        for name, step in adam.step(grads).items():
            agent._critic[name] -= step
    last = agent._critic_loss_and_grads(obs, actions, targets)[0]["critic_loss"]
    assert last < 0.05 * first


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------

OBS = np.array([0.2, -0.4, 0.1])


def _mu(agent: DDPGAgent) -> np.ndarray:
    return agent._actor_forward(OBS[None, :], agent._actor)[0][0]


def test_evaluation_acts_with_mu_and_uses_up_no_exploration():
    agent = _agent(learning_starts=5)
    action = agent.select_action(OBS, training=False)
    assert action.dtype == np.float32 and action.shape == (ACT_DIM,)
    np.testing.assert_allclose(action, _mu(agent), rtol=1e-6)
    assert agent._training_actions == 0


def test_warm_up_actions_are_uniform_in_the_bounds_and_then_mu_plus_noise():
    agent = _agent(learning_starts=3, noise_std=0.0)
    warm = [agent.select_action(OBS, training=True) for _ in range(3)]
    for action in warm:
        assert np.all(action >= LOW) and np.all(action <= HIGH)
        assert not np.allclose(action, _mu(agent)), "warm-up must not act from the policy"
    np.testing.assert_allclose(agent.select_action(OBS, training=True), _mu(agent), rtol=1e-6)
    assert agent._training_actions == 4


def test_warm_up_covers_the_whole_action_range():
    agent = _agent(learning_starts=2000)
    actions = np.array([agent.select_action(OBS, training=True) for _ in range(2000)])
    np.testing.assert_allclose(actions.min(axis=0), LOW, atol=0.02)
    np.testing.assert_allclose(actions.max(axis=0), HIGH, atol=0.02)


def test_noise_is_gaussian_with_noise_std_in_the_scaled_action_space():
    agent = _agent(noise_std=0.1)
    _constant(agent._actor, 0.0)  # mu = bias, far from both bounds
    actions = np.array([agent.select_action(OBS, training=True) for _ in range(4000)])
    np.testing.assert_allclose(actions.mean(axis=0), [-0.5, 1.0], atol=0.02)
    np.testing.assert_allclose(actions.std(axis=0), 0.1 * np.array([1.5, 2.0]), rtol=0.08)


def test_training_actions_stay_in_the_bounds_when_the_actor_is_saturated():
    agent = _agent(noise_std=0.5)
    _constant(agent._actor, 0.0)
    agent._actor[f"b{len(agent.hidden_sizes)}"][...] = [50.0, -50.0]  # mu at high[0] and low[1]
    actions = np.array([agent.select_action(OBS, training=True) for _ in range(200)])
    assert np.all(actions >= LOW.astype(np.float32)) and np.all(actions <= HIGH.astype(np.float32))
    assert (actions[:, 0] == np.float32(HIGH[0])).any(), "noise past the bound should be clipped"


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


def test_during_warm_up_update_stores_transitions_and_trains_nothing():
    agent = _agent(learning_starts=10)
    weights = copy.deepcopy(agent._actor)
    assert agent.update(_batch()) == {}
    assert len(agent.replay_buffer) == 6
    for name in weights:
        np.testing.assert_array_equal(agent._actor[name], weights[name])


@pytest.mark.parametrize("batch_change", ["empty", "no_next_observations"])
def test_a_batch_that_stores_nothing_trains_nothing(batch_change):
    agent = _agent()
    agent.update(_batch())  # warm buffer, past learning_starts
    weights = copy.deepcopy(agent._actor)
    batch = _batch()
    if batch_change == "empty":
        batch = {key: value[:0] for key, value in batch.items()}
    else:
        del batch["next_observations"]
    assert agent.update(batch) == {}
    for name in weights:
        np.testing.assert_array_equal(agent._actor[name], weights[name])


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
    reference.replay_buffer.add_batch(
        batch["observations"], batch["actions"], batch["rewards"],
        batch["next_observations"], batch["terminated"],
    )  # fmt: skip
    critic_adam = ReferenceAdam(reference.learning_rate)
    actor_adam = ReferenceAdam(reference.learning_rate)
    for _ in range(updates):
        s = reference.replay_buffer.sample(reference.batch_size)
        y = reference._bellman_targets(s["rewards"], s["next_observations"], s["dones"])
        _, critic_grads = reference._critic_loss_and_grads(s["observations"], s["actions"], y)
        for name, step in critic_adam.step(critic_grads).items():
            reference._critic[name] = reference._critic[name] - step
        _, actor_grads = reference._actor_loss_and_grads(s["observations"])
        for name, step in actor_adam.step(actor_grads).items():
            reference._actor[name] = reference._actor[name] - step
        for online, target in (
            (reference._actor, reference._target_actor),
            (reference._critic, reference._target_critic),
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
    for name in agent._actor:
        np.testing.assert_allclose(agent._actor[name], reference._actor[name], rtol=1e-10)
        np.testing.assert_allclose(
            agent._target_actor[name], reference._target_actor[name], rtol=1e-10
        )
    for name in agent._critic:
        np.testing.assert_allclose(agent._critic[name], reference._critic[name], rtol=1e-10)
        np.testing.assert_allclose(
            agent._target_critic[name], reference._target_critic[name], rtol=1e-10
        )
    assert agent._gradient_steps == expected_updates
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
    explore = copy.deepcopy(agent._explore_rng).standard_normal(4)
    replay = copy.deepcopy(agent.replay_buffer._rng).standard_normal(4)
    assert not np.allclose(explore, plain)
    assert not np.allclose(replay, plain)
    assert not np.allclose(replay, explore)


def test_two_agents_with_one_seed_act_and_learn_identically():
    first, second = _agent(seed=3), _agent(seed=3)
    for agent in (first, second):
        for _ in range(4):
            agent.select_action(OBS, training=True)
        agent.update(_batch())
    for name in first._actor:
        np.testing.assert_array_equal(first._actor[name], second._actor[name])


def test_a_loaded_agent_resumes_weights_optimizers_and_counters(tmp_path):
    """The replay buffer is not saved: the test hands the resumed agent the same one."""
    agent = _agent(learning_starts=2)
    for _ in range(3):
        agent.select_action(OBS, training=True)
    agent.update(_batch(seed=1))
    path = str(tmp_path / "ddpg.pt")
    agent.save(path)

    resumed = _agent(learning_starts=2)
    resumed.load(path)
    resumed.replay_buffer = copy.deepcopy(agent.replay_buffer)
    resumed._explore_rng = copy.deepcopy(agent._explore_rng)
    assert (resumed._training_actions, resumed._gradient_steps) == (3, 6)

    for run in (agent, resumed):
        run.select_action(OBS, training=True)
        run.update(_batch(seed=5))
    for params in ("_actor", "_critic", "_target_actor", "_target_critic"):
        for name, value in getattr(agent, params).items():
            np.testing.assert_array_equal(getattr(resumed, params)[name], value)


def test_a_checkpoint_with_other_hidden_sizes_is_rejected(tmp_path):
    path = str(tmp_path / "ddpg.pt")
    _agent().save(path)
    with pytest.raises(ValueError, match="hidden_sizes, observation_dim or action_dim"):
        _agent(hidden_sizes=[4, 3]).load(path)
