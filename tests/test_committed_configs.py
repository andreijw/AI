"""The committed CartPole configs carry the settings that were measured to solve CartPole-v1.

Each value pinned here was chosen from a five-seed measurement recorded in docs/roadmap.md
("CartPole Agents Learn Within Their Budgets"). Changing one should come with a new
measurement, and this test makes that change visible in review.
"""

import os
import sys

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
    ("cartpole", "dqn"): 50_000,
    ("cartpole", "random"): 50_000,
    ("pendulum", "ppo"): 100_000,
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


def test_dqn_keeps_its_epsilon_schedule():
    env, agent, _ = CartPoleChallenge().setup_run("dqn", seed=1, episodes=None)
    env.close()
    # A faster decay (0.99) solved fewer seeds than this one in 500 episodes.
    assert agent.epsilon_decay == pytest.approx(0.995)


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
