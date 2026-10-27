"""The committed CartPole configs carry the settings that were measured to solve CartPole-v1.

Each value pinned here was chosen from a five-seed measurement recorded in CHANGELOG.md
(2026-10-15 onwards). Changing one should come with a new
measurement, and this test makes that change visible in review.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_cartpole.benchmark import CartPoleChallenge  # noqa: E402
from rl_core.agents.optim import Adam  # noqa: E402
from rl_core.challenges import discover_challenges  # noqa: E402
from rl_core.utils.config import load_config  # noqa: E402
from tests.agent_internals import optimizer, weights
from tests.helpers import needs_torch

# Step budgets (CHANGELOG.md, 2026-10-16), from the
# rl-baselines3-zoo references where one exists; CartPole PPO gets twice its reference.
STEP_BUDGETS = {
    ("acrobot", "dqn"): 100_000,
    ("acrobot", "ppo"): 500_000,
    ("acrobot", "random"): 50_000,
    ("mountaincar", "dqn"): 200_000,
    ("mountaincar", "random"): 40_000,
    ("mountaincar_continuous", "ddpg"): 300_000,
    ("mountaincar_continuous", "random"): 200_000,
    ("mountaincar_continuous", "td3"): 300_000,
    ("cartpole", "reinforce"): 500_000,
    ("cartpole", "actor_critic"): 500_000,
    ("cartpole", "ppo"): 200_000,
    ("cartpole", "dqn"): 300_000,
    ("cartpole", "random"): 50_000,
    ("pendulum", "ppo"): 200_000,
    ("pendulum", "ddpg"): 40_000,
    ("pendulum", "td3"): 40_000,
    ("pendulum", "actor_critic"): 1_000_000,
    ("pendulum", "random"): 40_000,
}


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _uses_torch(challenge, agent_type):
    """Whether the committed config sets backend: torch, so builds only where PyTorch is installed."""
    name = discover_challenges()[challenge].CONFIG_MAP[agent_type]
    return load_config(os.path.join(CONFIGS_DIR, name))["agent"]["config"].get("backend") == "torch"


@pytest.mark.parametrize(
    "challenge,agent_type",
    [
        pytest.param(*key, marks=[needs_torch] if _uses_torch(*key) else [])
        for key in sorted(STEP_BUDGETS)
    ],
)
def test_every_agent_trains_for_its_step_budget(challenge, agent_type):
    env, agent, train_cfg = discover_challenges()[challenge]().setup_run(
        agent_type, seed=1, episodes=None
    )
    env.close()
    assert train_cfg["total_steps"] == STEP_BUDGETS[(challenge, agent_type)]
    anneal = getattr(agent, "anneal_steps", None)
    assert anneal is None or anneal == train_cfg["total_steps"], "Anneal over the whole budget."


@pytest.mark.parametrize("agent_type", ["reinforce", "actor_critic", "ppo"])
def test_cartpole_policy_gradient_agents_use_adam(agent_type):
    env, agent, _ = CartPoleChallenge().setup_run(agent_type, seed=1, episodes=None)
    env.close()
    assert isinstance(optimizer(agent), Adam)


def test_dqn_uses_the_measured_rl_zoo_like_settings():
    """Adam, a linear per-step exploration schedule and a slower target sync: 5 of 5 seeds solved
    within 300k steps (CHANGELOG.md, 2026-10-18)."""
    env, agent, _ = CartPoleChallenge().setup_run("dqn", seed=1, episodes=None)
    env.close()
    assert isinstance(optimizer(agent), Adam)
    assert optimizer(agent).max_grad_norm == pytest.approx(10.0)
    assert agent.learning_rate == pytest.approx(0.001)
    assert agent.target_update_frequency == 128
    assert (agent.exploration_steps, agent.epsilon_end) == (8000, pytest.approx(0.04))


def test_ppo_uses_standard_rollouts_with_an_annealed_learning_rate():
    """rl-zoo CartPole settings with linear annealing over the whole 200k-step budget (CHANGELOG.md, 2026-10-15)."""
    env, agent, _ = CartPoleChallenge().setup_run("ppo", seed=1, episodes=None)
    env.close()
    assert (agent.rollout_steps, agent.minibatch_size, agent.ppo_epochs) == (256, 256, 20)
    assert agent.anneal_steps == 200_000
    assert (agent.gae_lambda, agent.gamma, agent.entropy_coef) == (0.8, 0.98, 0.0)
    assert agent.learning_rate == pytest.approx(0.001)
    assert optimizer(agent).max_grad_norm == pytest.approx(0.5)
    assert (agent.clip_epsilon, agent.value_coef) == (0.2, 0.5)


def test_pendulum_ppo_uses_the_measured_settings():
    """γ 0.9, lr 0.001, reward scale 0.1, rollouts of 2048 transitions over 10 epochs, annealing over the
    200,000-step budget: 9 of seeds 1-10 solved under the 100-episode rule (2026-10-18)."""
    from rl_pendulum.benchmark import PendulumChallenge

    env, agent, train_cfg = PendulumChallenge().setup_run("ppo", seed=1, episodes=None)
    env.close()
    assert (agent.gamma, agent.learning_rate, agent.reward_scale) == pytest.approx(
        (0.9, 0.001, 0.1)
    )
    assert (agent.anneal_steps, agent.clip_ratio, agent.mean_init_scale) == (200_000, 0.2, 0.01)
    assert (agent.rollout_steps, agent.ppo_epochs) == (2048, 10)


def test_pendulum_td3_uses_rl_zoo_settings_with_a_smaller_network():
    """rl-zoo's TD3 for Pendulum-v1 with SB3's TD3 defaults and pendulum_ddpg.yaml's two departures
    (64x64, 40,000 steps): 5 of 5 seeds solved and held (CHANGELOG.md, 2026-10-25)."""
    from rl_pendulum.benchmark import PendulumChallenge

    env, agent, _ = PendulumChallenge().setup_run("td3", seed=1, episodes=None)
    env.close()
    assert agent.hidden_sizes == [64, 64]
    assert (agent.learning_rate, agent.gamma, agent.tau) == pytest.approx((0.001, 0.98, 0.005))
    assert (agent.batch_size, agent.learning_starts, agent.buffer_capacity) == (
        256,
        10_000,
        200_000,
    )
    assert (agent.noise_std, agent.updates_per_step) == pytest.approx((0.1, 1.0))
    assert agent.policy_delay == 2
    assert (agent.target_policy_noise, agent.target_noise_clip) == pytest.approx((0.2, 0.5))


def test_mountaincar_continuous_td3_uses_rl_zoo_settings_on_the_torch_backend():
    """rl-zoo's TD3 for MountainCarContinuous-v0 at its 400x300, with SB3's TD3 defaults and the OU
    settings of mountaincar_continuous_ddpg.yaml. Read from the file, so CI checks it without
    PyTorch, which the config's backend needs."""
    config = load_config(
        os.path.join(os.path.dirname(__file__), "..", "configs", "mountaincar_continuous_td3.yaml")
    )
    agent = config["agent"]["config"]
    assert config["agent"]["type"] == "td3" and agent["backend"] == "torch"
    assert agent["hidden_sizes"] == [400, 300]
    assert (agent["learning_rate"], agent["gamma"], agent["tau"]) == pytest.approx(
        (0.001, 0.99, 0.005)
    )
    assert (agent["batch_size"], agent["learning_starts"], agent["buffer_capacity"]) == (
        256,
        100,
        300_000,
    )
    assert (agent["noise_type"], agent["noise_std"]) == ("ornstein_uhlenbeck", 0.5)
    assert (agent["ou_theta"], agent["ou_dt"], agent["updates_per_step"]) == pytest.approx(
        (0.15, 0.01, 1.0)
    )
    assert agent["policy_delay"] == 2
    assert (agent["target_policy_noise"], agent["target_noise_clip"]) == pytest.approx((0.2, 0.5))
    assert config["training"]["total_steps"] == 300_000


def test_pendulum_ddpg_uses_rl_zoo_settings_with_a_smaller_network():
    """rl-zoo's DDPG for Pendulum-v1 with 64x64 networks: 5 of 5 seeds solved within 40,000 steps
    (CHANGELOG.md, 2026-10-18)."""
    from rl_pendulum.benchmark import PendulumChallenge

    env, agent, _ = PendulumChallenge().setup_run("ddpg", seed=1, episodes=None)
    env.close()
    assert agent.hidden_sizes == [64, 64]
    assert (agent.learning_rate, agent.gamma, agent.tau) == pytest.approx((0.001, 0.98, 0.005))
    assert (agent.batch_size, agent.learning_starts, agent.buffer_capacity) == (
        256,
        10_000,
        200_000,
    )
    assert (agent.noise_std, agent.updates_per_step) == pytest.approx((0.1, 1.0))
    np.testing.assert_allclose(agent.action_low, [-2.0])
    np.testing.assert_allclose(agent.action_high, [2.0])


def test_pendulum_actor_critic_uses_the_measured_settings():
    """Bounded mean, 8-step chunked updates, entropy 0.01 and rewards x 0.05: evaluation -152 +/- 45 on
    seeds 1-5, beating the random agent on 5 of 5 (CHANGELOG.md, 2026-10-18)."""
    from rl_pendulum.benchmark import PendulumChallenge

    env, agent, _ = PendulumChallenge().setup_run("actor_critic", seed=1, episodes=None)
    env.close()
    assert agent.bounded_mean and agent.n_steps == 8
    assert (agent.gae_lambda, agent.gamma, agent.learning_rate) == pytest.approx((0.9, 0.9, 0.0007))
    assert (agent.anneal_steps, agent.normalize_advantages) == (1_000_000, False)
    assert (agent.value_coef, agent.entropy_coef, agent.reward_scale) == pytest.approx(
        (0.4, 0.01, 0.05)
    )
    np.testing.assert_allclose(weights(agent)["log_std"], [-2.0])


def test_acrobot_dqn_uses_rl_zoo_settings_with_a_lower_exploration_floor():
    """rl-zoo's DQN for Acrobot-v1 (one hidden layer of 256 instead of two): one gradient step per
    transition over 100,000 steps. Epsilon falls to 0.02 rather than rl-zoo's 0.1 over the first 12,000:
    held -100 on 5 of 5 seeds against 1 of 5 (CHANGELOG.md, 2026-10-22)."""
    env, agent, _ = discover_challenges()["acrobot"]().setup_run("dqn", seed=1, episodes=None)
    env.close()
    assert isinstance(optimizer(agent), Adam)
    assert (agent.learning_rate, agent.gamma) == pytest.approx((6.3e-4, 0.99))
    assert (agent.hidden_dim, agent.batch_size, agent.buffer_capacity) == (256, 128, 50_000)
    assert agent.target_update_frequency == 250
    assert (agent.exploration_steps, agent.epsilon_end) == (12_000, pytest.approx(0.02))
    assert (agent.updates_per_step, agent.max_updates_per_call) == (1.0, 500)


def test_acrobot_ppo_starts_from_rl_zoo_settings():
    """rl-zoo's PPO for Acrobot-v1 without normalisation, on one environment: 4096-step rollouts,
    4 epochs of 64-transition minibatches, annealed over the 500,000-step budget."""
    env, agent, _ = discover_challenges()["acrobot"]().setup_run("ppo", seed=1, episodes=None)
    env.close()
    assert (agent.rollout_steps, agent.minibatch_size, agent.ppo_epochs) == (4096, 64, 4)
    assert (agent.gamma, agent.gae_lambda, agent.entropy_coef) == pytest.approx((0.99, 0.94, 0.0))
    assert agent.learning_rate == pytest.approx(3e-4)
    assert (agent.clip_epsilon, agent.value_coef) == (0.2, 0.5)
    assert optimizer(agent).max_grad_norm == pytest.approx(0.5)


def test_mountaincar_dqn_uses_rl_zoo_settings_with_a_decaying_learning_rate():
    """rl-zoo's DQN for MountainCar-v0 (one hidden layer of 256 instead of two): half a gradient step per
    transition. Its target sync every 600 environment steps is 300 gradient steps here, epsilon falls to
    0.02 rather than 0.07, and the learning rate decays to zero over a 200,000-step budget instead of
    staying fixed over 120,000 (CHANGELOG.md, 2026-10-23). The replay buffer holds 100,000 transitions,
    not rl-zoo's 10,000, so a good policy found mid-run is forgotten less often (CHANGELOG.md, 2026-10-25)."""
    env, agent, _ = discover_challenges()["mountaincar"]().setup_run("dqn", seed=1, episodes=None)
    env.close()
    assert isinstance(optimizer(agent), Adam)
    assert (agent.learning_rate, agent.gamma) == pytest.approx((0.004, 0.98))
    assert (agent.hidden_dim, agent.batch_size, agent.buffer_capacity) == (256, 128, 100_000)
    assert (agent.min_buffer_size, agent.target_update_frequency) == (1_000, 300)
    assert (agent.exploration_steps, agent.epsilon_end) == (24_000, pytest.approx(0.02))
    assert (agent.updates_per_step, agent.max_updates_per_call) == (0.5, 100)
    assert agent.anneal_steps == 200_000, "the rate decays over the whole budget"
