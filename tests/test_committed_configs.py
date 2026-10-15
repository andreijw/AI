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

# agent type -> (optimizer, training budget)
MEASURED = {
    "reinforce": ("adam", 2000),
    "actor_critic": ("adam", 1500),
    "ppo": ("adam", 1000),
}


@pytest.mark.parametrize("agent_type", sorted(MEASURED))
def test_policy_gradient_agents_use_the_measured_optimizer_and_budget(agent_type):
    optimizer, budget = MEASURED[agent_type]
    env, agent, train_cfg = CartPoleChallenge().setup_run(agent_type, seed=1, episodes=None)
    env.close()
    assert isinstance(agent._optimizer, Adam) == (optimizer == "adam")
    assert train_cfg["num_episodes"] == budget


def test_dqn_trains_for_the_measured_budget_and_keeps_its_epsilon_schedule():
    env, agent, train_cfg = CartPoleChallenge().setup_run("dqn", seed=1, episodes=None)
    env.close()
    assert train_cfg["num_episodes"] == 750
    # A faster decay (0.99) solved fewer seeds than this one in 500 episodes.
    assert agent.epsilon_decay == pytest.approx(0.995)


def test_ppo_uses_standard_rollouts_with_an_annealed_learning_rate():
    """rl-zoo CartPole settings with linear annealing: 5 of 5 seeds by episode 427 (roadmap)."""
    env, agent, _ = CartPoleChallenge().setup_run("ppo", seed=1, episodes=None)
    env.close()
    assert (agent.rollout_steps, agent.minibatch_size, agent.ppo_epochs) == (256, 256, 20)
    assert agent.anneal_steps == 200_000
    assert (agent.gae_lambda, agent.gamma, agent.entropy_coef) == (0.8, 0.98, 0.0)
    assert agent.learning_rate == pytest.approx(0.001)
    assert agent._optimizer.max_grad_norm == pytest.approx(0.5)
    assert (agent.clip_epsilon, agent.value_coef) == (0.2, 0.5)
