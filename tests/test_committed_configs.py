"""The committed CartPole configs carry the settings that were measured to solve CartPole-v1.

Each value pinned here was chosen from a five-seed measurement recorded in CHANGELOG.md
(2026-10-15 onwards). Changing one should come with a new
measurement, and this test makes that change visible in review.
"""

import os

import numpy as np
import pytest

from rl_cartpole.benchmark import CartPoleChallenge
from rl_core.agents import TD3Agent
from rl_core.agents.optim import Adam
from rl_core.challenges import discover_challenges
from rl_core.utils.config import load_config
from tests.agent_internals import optimizer, weights
from tests.helpers import env_marks, needs_torch

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
    ("lunarlander", "dqn"): 300_000,
    ("lunarlander", "random"): 50_000,
    ("lunarlander_continuous", "random"): 50_000,
    ("lunarlander_continuous", "td3"): 400_000,
    ("lunarlander_continuous", "sac"): 500_000,
    ("navigation", "dqn"): 300_000,
    ("navigation", "ppo"): 200_000,
    ("navigation", "random"): 40_000,
    ("navigation_continuous", "ppo"): 200_000,
    ("navigation_continuous", "random"): 40_000,
    ("navigation_continuous", "td3"): 100_000,
    ("bipedalwalker", "random"): 50_000,
    ("bipedalwalker", "sac"): 500_000,
    ("bipedalwalker", "td3"): 500_000,
    ("cartpole", "reinforce"): 500_000,
    ("cartpole", "actor_critic"): 500_000,
    ("cartpole", "ppo"): 200_000,
    ("cartpole", "dqn"): 300_000,
    ("cartpole", "random"): 50_000,
    ("pendulum", "ppo"): 200_000,
    ("pendulum", "ddpg"): 40_000,
    ("pendulum", "td3"): 40_000,
    ("pendulum", "sac"): 40_000,
    ("pendulum", "actor_critic"): 1_000_000,
    ("pendulum", "random"): 40_000,
}


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _marks(challenge, agent_type):
    """What building the committed run needs: PyTorch when the config sets backend: torch, and
    whatever the challenge's environment needs (Box2D for LunarLander)."""
    challenge_cls = discover_challenges()[challenge]
    config = load_config(os.path.join(CONFIGS_DIR, challenge_cls.CONFIG_MAP[agent_type]))
    torch = [needs_torch] if config["agent"]["config"].get("backend") == "torch" else []
    return torch + env_marks(challenge_cls().name)


@pytest.mark.parametrize(
    "challenge,agent_type",
    [pytest.param(*key, marks=_marks(*key)) for key in sorted(STEP_BUDGETS)],
)
def test_every_agent_trains_for_its_step_budget(challenge, agent_type):
    env, agent, train_cfg = discover_challenges()[challenge]().setup_run(
        agent_type, seed=1, episodes=None
    )
    env.close()
    assert train_cfg["total_steps"] == STEP_BUDGETS[(challenge, agent_type)]
    anneal = getattr(agent, "anneal_steps", None)
    assert anneal is None or anneal == train_cfg["total_steps"], "Anneal over the whole budget."


@pytest.mark.parametrize("challenge", sorted(discover_challenges()))
def test_every_registered_agent_has_a_committed_config(challenge):
    """An agent added to a challenge's AGENT_CLASSES without a config in CONFIG_MAP (or the other
    way round) fails here, not when someone first asks the benchmark for it."""
    challenge_cls = discover_challenges()[challenge]
    assert set(challenge_cls.CONFIG_MAP) == set(challenge_cls.AGENT_CLASSES), (
        f"{challenge}: CONFIG_MAP and AGENT_CLASSES should name the same agents"
    )
    for config_file in challenge_cls.CONFIG_MAP.values():
        assert os.path.isfile(os.path.join(CONFIGS_DIR, config_file)), config_file


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


def test_lunarlander_dqn_translates_rl_zoo_for_a_trainer_that_updates_after_each_episode():
    """rl-zoo's DQN for LunarLander-v3 (the same entry as Acrobot-v1's), one hidden layer of 256
    instead of two: one gradient step per transition (train_freq 4, gradient_steps -1), so its
    250-step target interval is 250 gradient steps; epsilon falls over the first 12,000 training
    actions. The update cap, the trainer's episode cut and the environment's limit are all 1000: the
    Trainer would otherwise cut episodes at its default of 500. rl-zoo's exploration floor of 0.1 held
    200 on 5 of 5 seeds against 3 of 5 for 0.02 (CHANGELOG.md, 2026-10-27). Read from the file, so CI
    checks it without Box2D."""
    config = load_config(os.path.join(CONFIGS_DIR, "lunarlander_dqn.yaml"))
    agent = config["agent"]["config"]
    assert config["agent"]["type"] == "dqn" and agent["optimizer"] == "adam"
    assert (agent["learning_rate"], agent["gamma"], agent["max_grad_norm"]) == pytest.approx(
        (6.3e-4, 0.99, 10.0)
    )
    assert (agent["hidden_dim"], agent["batch_size"], agent["buffer_capacity"]) == (
        256,
        128,
        50_000,
    )
    assert (agent["min_buffer_size"], agent["target_update_frequency"]) == (128, 250)
    assert (agent["exploration_steps"], agent["updates_per_step"]) == (12_000, 1.0)
    assert agent["epsilon_end"] == pytest.approx(0.1)
    limits = (
        agent["max_updates_per_call"],
        config["training"]["max_steps_per_episode"],
        config["environment"]["max_episode_steps"],
    )
    assert limits == (1000, 1000, 1000)
    assert config["training"]["total_steps"] == 300_000


def test_lunarlander_continuous_td3_uses_rl_zoo_settings_on_the_torch_backend():
    """rl-zoo's TD3 for LunarLanderContinuous-v3 at its 400x300, with SB3's TD3 defaults: one gradient
    step per transition after 10,000 uniform warm-up actions, Gaussian noise 0.1 in the [-1, 1]
    action space. The trainer's episode cut matches the 1000-step limit (its default is 500), the
    agent updates after every environment step as rl-zoo does (the maintainer's choice, 2026-11-01, where
    5 of 5 seeds held 200), and the budget is the maintainer's 400,000 steps. Read from the file, so CI
    checks it without PyTorch or Box2D, which the config needs."""
    config = load_config(os.path.join(CONFIGS_DIR, "lunarlander_continuous_td3.yaml"))
    agent = config["agent"]["config"]
    assert config["agent"]["type"] == "td3" and agent["backend"] == "torch"
    assert agent["hidden_sizes"] == [400, 300]
    assert (agent["learning_rate"], agent["gamma"], agent["tau"]) == pytest.approx(
        (0.001, 0.98, 0.005)
    )
    assert (agent["batch_size"], agent["learning_starts"], agent["buffer_capacity"]) == (
        256,
        10_000,
        200_000,
    )
    assert (agent["noise_type"], agent["noise_std"]) == ("gaussian", pytest.approx(0.1))
    assert agent["updates_per_step"] == 1.0 and "max_updates_per_call" not in agent
    assert agent["policy_delay"] == 2
    assert (agent["target_policy_noise"], agent["target_noise_clip"]) == pytest.approx((0.2, 0.5))
    limits = (
        config["training"]["max_steps_per_episode"],
        config["environment"]["max_episode_steps"],
    )
    assert limits == (1000, 1000)
    assert config["training"]["total_steps"] == 400_000
    assert config["training"].get("update_every") == 1, (
        "the published LunarLanderContinuous TD3 results update after every step"
    )


def test_lunarlander_continuous_sac_uses_rl_zoo_settings_on_the_torch_backend():
    """rl-zoo's SAC for LunarLanderContinuous-v3 at its 400x300 and its 500,000 steps, with SB3's
    SAC defaults for the rest: one gradient step per transition after 10,000 uniform warm-up
    actions, a temperature learned from 1 towards -2 nats. rl-zoo decays its learning rate
    linearly from 7.3e-4; the agents here have no schedule, so it stays at 7.3e-4. The trainer's
    episode cut matches the 1000-step limit and it updates after every step, as
    lunarlander_continuous_td3.yaml. Read from the file, so CI checks it without PyTorch or Box2D,
    which the config needs."""
    config = load_config(os.path.join(CONFIGS_DIR, "lunarlander_continuous_sac.yaml"))
    agent = config["agent"]["config"]
    assert config["agent"]["type"] == "sac" and agent["backend"] == "torch"
    assert agent["hidden_sizes"] == [400, 300]
    assert (agent["learning_rate"], agent["gamma"], agent["tau"]) == pytest.approx(
        (7.3e-4, 0.99, 0.01)
    )
    assert (agent["batch_size"], agent["learning_starts"], agent["buffer_capacity"]) == (
        256,
        10_000,
        1_000_000,
    )
    assert agent["updates_per_step"] == 1.0 and "max_updates_per_call" not in agent
    assert (agent["ent_coef"], agent["target_entropy"]) == ("auto", "auto")
    assert "max_grad_norm" not in agent, "SB3's SAC clips no gradient"
    limits = (
        config["training"]["max_steps_per_episode"],
        config["environment"]["max_episode_steps"],
    )
    assert limits == (1000, 1000)
    assert config["training"]["total_steps"] == 500_000
    assert config["training"].get("update_every") == 1, "SB3 updates after every step"


def _bipedalwalker_config(agent_type: str) -> dict:
    """The committed BipedalWalker config for ``agent_type``, read from the file: CI checks it
    without PyTorch or Box2D, which it needs."""
    config = load_config(os.path.join(CONFIGS_DIR, f"bipedalwalker_{agent_type}.yaml"))
    assert config["agent"]["type"] == agent_type
    assert config["environment"]["name"] == "BipedalWalker-v3"
    return config


@pytest.mark.parametrize("agent_type", ["default", "sac", "td3"])
def test_bipedalwalker_episodes_run_to_gymnasiums_1600_step_limit(agent_type):
    """BipedalWalker-v3 cuts an episode at 1600 steps. The Trainer's own cut defaults to 500,
    which would end every episode of a walker that does not fall at a third of its length (and
    evaluation episodes too), so each config sets both limits to 1600."""
    config = load_config(os.path.join(CONFIGS_DIR, f"bipedalwalker_{agent_type}.yaml"))
    limits = (
        config["training"]["max_steps_per_episode"],
        config["environment"]["max_episode_steps"],
    )
    assert limits == (1600, 1600), "the trainer's and the environment's limits should be 1600"


def test_bipedalwalker_sac_uses_rl_zoo_settings_without_gsde():
    """rl-zoo's tuned SAC for BipedalWalker-v3 (hyperparams/sac.yml) on the PyTorch backend: lr
    7.3e-4 (constant, as rl-zoo), buffer 300,000, γ 0.98, τ 0.02, 64 gradient steps every 64
    environment steps (train_freq 64, gradient_steps 64), 10,000 warm-up steps, 500,000 steps.
    rl-zoo's gSDE (use_sde, log_std_init -3, use_expln) has no counterpart here: the maintainer
    dropped it (2026-11-05)."""
    config = _bipedalwalker_config("sac")
    agent = config["agent"]["config"]
    assert agent["backend"] == "torch"
    assert agent["hidden_sizes"] == [400, 300]
    assert (agent["learning_rate"], agent["gamma"], agent["tau"]) == pytest.approx(
        (7.3e-4, 0.98, 0.02)
    )
    assert (agent["batch_size"], agent["learning_starts"], agent["buffer_capacity"]) == (
        256,
        10_000,
        300_000,
    )
    assert agent["updates_per_step"] == 1.0 and "max_updates_per_call" not in agent
    assert (agent["ent_coef"], agent["target_entropy"]) == ("auto", "auto")
    assert "max_grad_norm" not in agent, "SB3's SAC clips no gradient"
    assert config["training"]["update_every"] == 64, (
        "rl-zoo trains every 64 steps with 64 gradient steps (train_freq 64, gradient_steps 64)"
    )
    assert config["training"]["total_steps"] == 500_000
    assert config["training"]["eval_frequency"] == 100


def test_bipedalwalker_td3_uses_rl_zoo_settings_on_the_sac_budget():
    """rl-zoo's TD3 for BipedalWalker-v3 (hyperparams/td3.yml) with SB3's TD3 defaults, on the
    PyTorch backend, one gradient step after every environment step; 500,000 steps, the SAC
    budget, where rl-zoo trains 1e6 (the maintainer's decision, 2026-11-05)."""
    config = _bipedalwalker_config("td3")
    agent = config["agent"]["config"]
    assert agent["backend"] == "torch"
    assert agent["hidden_sizes"] == [400, 300]
    assert (agent["learning_rate"], agent["gamma"], agent["tau"]) == pytest.approx(
        (0.001, 0.98, 0.005)
    )
    assert (agent["batch_size"], agent["learning_starts"], agent["buffer_capacity"]) == (
        256,
        10_000,
        200_000,
    )
    assert (agent["noise_type"], agent["noise_std"]) == ("gaussian", pytest.approx(0.1))
    assert agent["updates_per_step"] == 1.0 and "max_updates_per_call" not in agent
    assert agent["policy_delay"] == 2
    assert (agent["target_policy_noise"], agent["target_noise_clip"]) == pytest.approx((0.2, 0.5))
    assert config["training"]["update_every"] == 1, "rl-zoo trains after every step (train_freq 1)"
    assert config["training"]["total_steps"] == 500_000
    assert config["training"]["eval_frequency"] == 100


def test_pendulum_sac_uses_rl_zoos_settings_for_40000_steps():
    """rl-zoo's SAC for Pendulum-v1 (learning rate 1e-3, SB3's SAC defaults otherwise), updating
    after every step, with pendulum_td3.yaml's 40,000-step budget and its 64x64 networks (the
    maintainer's decision, 2026-11-04: seeds 1 to 5 solve and hold at 64x64)."""
    from rl_pendulum.benchmark import PendulumChallenge

    env, agent, train_cfg = PendulumChallenge().setup_run("sac", seed=1, episodes=None)
    env.close()
    assert agent.hidden_sizes == [64, 64]
    assert (agent.learning_rate, agent.gamma, agent.tau) == pytest.approx((0.001, 0.99, 0.005))
    assert (agent.batch_size, agent.learning_starts, agent.buffer_capacity) == (
        256,
        100,
        1_000_000,
    )
    assert agent.updates_per_step == pytest.approx(1.0)
    assert agent.learns_ent_coef and agent.ent_coef == pytest.approx(1.0)
    assert agent.target_entropy == -1.0
    assert train_cfg["update_every"] == 1


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
    """Bounded mean, entropy 0.01 and rewards x 0.05: evaluation -152 +/- 45 on seeds 1-5, beating the
    random agent on 5 of 5 (CHANGELOG.md, 2026-10-18). Observation normalisation raised the final training
    average on 5 of seeds 1-5 and 4 of 11-15 (CHANGELOG.md, 2026-11-04). Since 2026-11-12 rl-zoo's A2C
    layout, 8 environments x 8 steps (num_envs 8, update_every 8 = n_steps: one Adam step over 64
    transitions): final training average -244.1 against -282.7 with one environment, better on all 5 of
    seeds 1-5, none holding -200, greedy evaluation worse (CHANGELOG.md, 2026-11-10 and 2026-11-12)."""
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
    assert agent.observation_normalizer is not None, "the config must normalise observations"
    training = load_config(os.path.join(CONFIGS_DIR, "pendulum_actor_critic.yaml"))["training"]
    assert (training.get("num_envs"), training.get("update_every")) == (8, 8), (
        "8 environments x 8 steps per update, as rl-zoo's A2C (CHANGELOG.md, 2026-11-12)"
    )


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
    not rl-zoo's 10,000, so a good policy found mid-run is forgotten less often (CHANGELOG.md, 2026-10-25).
    Observations are normalised at sampling time, which reached -110 on 10 of seeds 1-5 and 11-15 against 3
    (CHANGELOG.md, 2026-11-04)."""
    env, agent, _ = discover_challenges()["mountaincar"]().setup_run("dqn", seed=1, episodes=None)
    env.close()
    assert isinstance(optimizer(agent), Adam)
    assert (agent.learning_rate, agent.gamma) == pytest.approx((0.004, 0.98))
    assert (agent.hidden_dim, agent.batch_size, agent.buffer_capacity) == (256, 128, 100_000)
    assert (agent.min_buffer_size, agent.target_update_frequency) == (1_000, 300)
    assert (agent.exploration_steps, agent.epsilon_end) == (24_000, pytest.approx(0.02))
    assert (agent.updates_per_step, agent.max_updates_per_call) == (0.5, 100)
    assert agent.anneal_steps == 200_000, "the rate decays over the whole budget"
    assert agent.observation_normalizer is not None, "the config must normalise observations"


def _navigation_run(challenge, agent_type):
    """The committed navigation run for agent_type, and its config's environment section beside the
    random baseline's: a learning config must train on the task the baseline was measured on."""
    challenge_cls = discover_challenges()[challenge]
    env, agent, train_cfg = challenge_cls().setup_run(agent_type, seed=1, episodes=None)
    env.close()

    def environment(agent_key):
        path = os.path.join(CONFIGS_DIR, challenge_cls.CONFIG_MAP[agent_key])
        return load_config(path)["environment"]

    assert environment(agent_type) == environment("random"), "the random baseline's task"
    assert train_cfg["max_steps_per_episode"] == 400, "the environment's 400-step limit"
    return agent, train_cfg


def test_navigation_ppo_uses_sb3_defaults_with_an_entropy_bonus():
    """SB3's PPO defaults (Adam 3e-4, clip 0.5 norm, 2048-step rollouts, 10 epochs of 64, gamma
    0.99, GAE 0.95, clip 0.2) with entropy 0.01 for exploration over five discrete actions, annealed
    over the whole 200,000-step budget."""
    agent, train_cfg = _navigation_run("navigation", "ppo")
    assert isinstance(optimizer(agent), Adam)
    assert (agent.learning_rate, optimizer(agent).max_grad_norm) == pytest.approx((3e-4, 0.5))
    assert (agent.rollout_steps, agent.minibatch_size, agent.ppo_epochs) == (2048, 64, 10)
    assert (agent.gamma, agent.gae_lambda, agent.entropy_coef) == pytest.approx((0.99, 0.95, 0.01))
    assert (agent.clip_epsilon, agent.value_coef, agent.hidden_dim) == (0.2, 0.5, 128)
    assert agent.anneal_steps == train_cfg["total_steps"] == 200_000


def test_navigation_continuous_ppo_uses_sb3_defaults_and_the_agents_own():
    """The same SB3 numbers as navigation_ppo.yaml with SB3's entropy 0.0 and rewards unscaled;
    ContinuousPPOAgent's defaults for the rest (max_grad_norm 0.5, log std -0.5, mean scale 0.01),
    and the environment's [-1, 1] action bounds from the challenge."""
    agent, train_cfg = _navigation_run("navigation_continuous", "ppo")
    assert (agent.learning_rate, agent.max_grad_norm) == pytest.approx((3e-4, 0.5))
    assert (agent.rollout_steps, agent.batch_size, agent.ppo_epochs) == (2048, 64, 10)
    assert (agent.gamma, agent.gae_lambda, agent.entropy_coef) == pytest.approx((0.99, 0.95, 0.0))
    assert (agent.clip_ratio, agent.value_coef, agent.reward_scale) == (0.2, 0.5, 1.0)
    assert (agent.hidden_dim, agent.mean_init_scale) == (128, 0.01)
    np.testing.assert_array_equal(weights(agent)["log_std"], [-0.5, -0.5])
    np.testing.assert_array_equal(agent.action_low, [-1.0, -1.0])
    np.testing.assert_array_equal(agent.action_high, [1.0, 1.0])
    assert agent.anneal_steps == train_cfg["total_steps"] == 200_000


def test_navigation_dqn_uses_lunarlander_dqn_settings_for_400_step_episodes():
    """lunarlander_dqn.yaml's rl-zoo settings, its measured exploration floor of 0.1 included, with
    epsilon falling over the first 30,000 actions, a warm-up of 1,000 transitions (about 15 random
    episodes, not one), the update cap at the 400-step episode limit and LunarLander's 300,000
    steps; and the fix for the training-doctor diagnosis of 2026-11-06: rl-zoo's Huber TD loss
    (SB3's smooth-L1, delta 1) and a learning rate decaying to zero over the whole budget."""
    agent, train_cfg = _navigation_run("navigation", "dqn")
    assert isinstance(optimizer(agent), Adam)
    assert (agent.learning_rate, agent.gamma) == pytest.approx((6.3e-4, 0.99))
    assert optimizer(agent).max_grad_norm == pytest.approx(10.0)
    assert (agent.hidden_dim, agent.batch_size, agent.buffer_capacity) == (256, 128, 50_000)
    assert (agent.min_buffer_size, agent.target_update_frequency) == (1_000, 250)
    assert (agent.epsilon_start, agent.epsilon_end) == pytest.approx((1.0, 0.1))
    assert agent.exploration_steps == 30_000
    assert (agent.updates_per_step, agent.max_updates_per_call) == (1.0, 400)
    assert (agent.td_loss, agent.huber_delta) == ("huber", 1.0)
    assert agent.anneal_steps == train_cfg["total_steps"] == 300_000


def test_navigation_continuous_td3_uses_pendulum_td3_settings_in_numpy():
    """pendulum_td3.yaml's rl-zoo and SB3 settings on the NumPy backend at 64x64, with gamma 0.99
    for the longer episodes and an update after every step, for 100,000 steps."""
    agent, train_cfg = _navigation_run("navigation_continuous", "td3")
    assert type(agent) is TD3Agent, "the NumPy backend: CI builds it without PyTorch"
    assert agent.hidden_sizes == [64, 64]
    assert (agent.learning_rate, agent.gamma, agent.tau) == pytest.approx((0.001, 0.99, 0.005))
    assert (agent.batch_size, agent.learning_starts, agent.buffer_capacity) == (
        256,
        10_000,
        200_000,
    )
    assert (agent.noise_std, agent.updates_per_step) == pytest.approx((0.1, 1.0))
    assert agent.policy_delay == 2
    assert (agent.target_policy_noise, agent.target_noise_clip) == pytest.approx((0.2, 0.5))
    assert (train_cfg["update_every"], train_cfg["total_steps"]) == (1, 100_000)
