"""The committed CartPole configs carry the settings that were measured to solve CartPole-v1.

Each value pinned here was chosen from a five-seed measurement recorded in docs/roadmap.md
("CartPole Agents Learn Within Their Budgets"). Changing one should come with a new
measurement, and this test makes that change visible in review.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_cartpole.benchmark import CartPoleChallenge  # noqa: E402
from rl_core.agents.optim import Adam  # noqa: E402

# Step budgets (roadmap: Benchmark Budgets and Solve Rule in Environment Steps), from the
# rl-baselines3-zoo references where one exists; CartPole PPO gets twice its reference.
STEP_BUDGETS = {
    ("cartpole", "reinforce"): 500_000,
    ("cartpole", "actor_critic"): 500_000,
    ("cartpole", "ppo"): 200_000,
    ("cartpole", "dqn"): 300_000,
    ("cartpole", "random"): 50_000,
    ("pendulum", "ppo"): 100_000,
    ("pendulum", "ddpg"): 40_000,
    ("pendulum", "actor_critic"): 1_000_000,
    ("pendulum", "random"): 40_000,
}


@pytest.mark.parametrize("challenge,agent_type", sorted(STEP_BUDGETS))
def test_every_agent_trains_for_its_step_budget(challenge, agent_type):
    from rl_pendulum.benchmark import PendulumChallenge

    cls = {"cartpole": CartPoleChallenge, "pendulum": PendulumChallenge}[challenge]
    env, agent, train_cfg = cls().setup_run(agent_type, seed=1, episodes=None)
    env.close()
    assert train_cfg["total_steps"] == STEP_BUDGETS[(challenge, agent_type)]
    anneal = getattr(agent, "anneal_steps", None)
    assert anneal is None or anneal == train_cfg["total_steps"], "Anneal over the whole budget."


@pytest.mark.parametrize("agent_type", ["reinforce", "actor_critic", "ppo"])
def test_cartpole_policy_gradient_agents_use_adam(agent_type):
    env, agent, _ = CartPoleChallenge().setup_run(agent_type, seed=1, episodes=None)
    env.close()
    assert isinstance(agent._optimizer, Adam)


def test_dqn_uses_the_measured_rl_zoo_like_settings():
    """Adam, a linear per-step exploration schedule and a slower target sync: 5 of 5 seeds solved
    within 300k steps (roadmap: "DQN Reaches the rl-zoo Reference")."""
    env, agent, _ = CartPoleChallenge().setup_run("dqn", seed=1, episodes=None)
    env.close()
    assert isinstance(agent._optimizer, Adam)
    assert agent._optimizer.max_grad_norm == pytest.approx(10.0)
    assert agent.learning_rate == pytest.approx(0.001)
    assert agent.target_update_frequency == 128
    assert (agent.exploration_steps, agent.epsilon_end) == (8000, pytest.approx(0.04))


def test_ppo_uses_standard_rollouts_with_an_annealed_learning_rate():
    """rl-zoo CartPole settings with linear annealing over the whole 200k-step budget (roadmap)."""
    env, agent, _ = CartPoleChallenge().setup_run("ppo", seed=1, episodes=None)
    env.close()
    assert (agent.rollout_steps, agent.minibatch_size, agent.ppo_epochs) == (256, 256, 20)
    assert agent.anneal_steps == 200_000
    assert (agent.gae_lambda, agent.gamma, agent.entropy_coef) == (0.8, 0.98, 0.0)
    assert agent.learning_rate == pytest.approx(0.001)
    assert agent._optimizer.max_grad_norm == pytest.approx(0.5)
    assert (agent.clip_epsilon, agent.value_coef) == (0.2, 0.5)


def test_pendulum_ppo_uses_the_measured_settings():
    """γ 0.9, lr 0.001, reward scale 0.1, annealing over 100,000 transitions: the settings whose 10-episode
    average touched -200 on 5 of 5 seeds (2026-10-15); 0 of 5 hold it under the 100-episode rule."""
    from rl_pendulum.benchmark import PendulumChallenge

    env, agent, train_cfg = PendulumChallenge().setup_run("ppo", seed=1, episodes=None)
    env.close()
    assert (agent.gamma, agent.learning_rate, agent.reward_scale) == pytest.approx(
        (0.9, 0.001, 0.1)
    )
    assert (agent.anneal_steps, agent.clip_ratio, agent.mean_init_scale) == (100_000, 0.2, 0.01)


def test_pendulum_ddpg_uses_rl_zoo_settings_with_a_smaller_network():
    """rl-zoo's DDPG for Pendulum-v1 with 64x64 networks: 5 of 5 seeds solved within 40,000 steps
    (roadmap: "Continuous Off-Policy Learning & DDPG")."""
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
