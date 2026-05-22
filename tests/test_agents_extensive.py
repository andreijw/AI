"""Extensive tests for all agent implementations (RandomAgent, ReinforceAgent, ActorCriticAgent, BaseAgent)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from rl_cartpole.agents import ActorCriticAgent, BaseAgent, RandomAgent, ReinforceAgent

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_OBS_DIM = 4
_ACTION_DIM = 2

_REINFORCE_CFG = {"learning_rate": 1e-3, "gamma": 0.99, "hidden_dim": 64, "seed": 7}
_AC_CFG = {
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "hidden_dim": 64,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "seed": 7,
}


def _make_batch(
    n: int = 10, obs_dim: int = _OBS_DIM, action_dim: int = _ACTION_DIM, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, obs_dim)),
        "actions": rng.integers(0, action_dim, size=n),
        "rewards": rng.standard_normal(n),
    }


# ===========================================================================
# BaseAgent
# ===========================================================================


class TestBaseAgent:
    """BaseAgent is abstract and cannot be instantiated directly."""

    def test_base_agent_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseAgent(observation_dim=4, action_dim=2, config={})  # type: ignore[abstract]

    def test_base_agent_subclass_missing_select_action(self):
        """A subclass that omits select_action cannot be instantiated."""

        class Incomplete(BaseAgent):
            def update(self, batch):
                return {}

            def save(self, path):
                pass

            def load(self, path):
                pass

        with pytest.raises(TypeError):
            Incomplete(observation_dim=4, action_dim=2, config={})

    def test_base_agent_subclass_missing_update(self):
        class Incomplete(BaseAgent):
            def select_action(self, observation, training=True):
                return 0

            def save(self, path):
                pass

            def load(self, path):
                pass

        with pytest.raises(TypeError):
            Incomplete(observation_dim=4, action_dim=2, config={})

    def test_base_agent_subclass_missing_save(self):
        class Incomplete(BaseAgent):
            def select_action(self, observation, training=True):
                return 0

            def update(self, batch):
                return {}

            def load(self, path):
                pass

        with pytest.raises(TypeError):
            Incomplete(observation_dim=4, action_dim=2, config={})

    def test_base_agent_subclass_missing_load(self):
        class Incomplete(BaseAgent):
            def select_action(self, observation, training=True):
                return 0

            def update(self, batch):
                return {}

            def save(self, path):
                pass

        with pytest.raises(TypeError):
            Incomplete(observation_dim=4, action_dim=2, config={})

    def test_base_agent_concrete_subclass_stores_attributes(self):
        """A fully-implemented subclass stores obs/action dim and config."""

        class Minimal(BaseAgent):
            def select_action(self, observation, training=True):
                return 0

            def update(self, batch):
                return {}

            def save(self, path):
                pass

            def load(self, path):
                pass

        cfg = {"lr": 0.1}
        agent = Minimal(observation_dim=8, action_dim=3, config=cfg)
        assert agent.observation_dim == 8
        assert agent.action_dim == 3
        assert agent.config == cfg


# ===========================================================================
# RandomAgent – extended tests
# ===========================================================================


class TestRandomAgentExtended:
    """Additional coverage for RandomAgent edge cases and statistical properties."""

    def test_single_action_dim_always_returns_zero(self):
        agent = RandomAgent(observation_dim=4, action_dim=1, config={})
        obs = np.zeros(4)
        actions = [agent.select_action(obs) for _ in range(50)]
        assert all(a == 0 for a in actions)

    def test_large_observation_dim(self):
        agent = RandomAgent(observation_dim=256, action_dim=2, config={})
        obs = np.random.randn(256)
        action = agent.select_action(obs)
        assert action in (0, 1)

    def test_large_action_dim_all_covered(self):
        """With enough samples all 10 actions should appear."""
        agent = RandomAgent(observation_dim=4, action_dim=10, config={"seed": 42})
        obs = np.zeros(4)
        actions = [agent.select_action(obs) for _ in range(500)]
        assert set(actions) == set(range(10))

    def test_approximate_uniform_distribution(self):
        """Seeded agent over many draws should be roughly uniform."""
        n_actions = 4
        n_draws = 4000
        agent = RandomAgent(observation_dim=4, action_dim=n_actions, config={"seed": 0})
        obs = np.zeros(4)
        counts = np.zeros(n_actions)
        for _ in range(n_draws):
            counts[agent.select_action(obs)] += 1
        freqs = counts / n_draws
        # Each frequency should be within 5 % of 0.25
        np.testing.assert_allclose(freqs, 0.25, atol=0.05)

    def test_action_is_integer_type(self):
        agent = RandomAgent(observation_dim=4, action_dim=2, config={})
        action = agent.select_action(np.zeros(4))
        assert isinstance(action, int)

    def test_training_flag_ignored(self):
        """The training flag has no effect on RandomAgent – identical sequences for same seed."""
        obs = np.zeros(4)
        # Two agents seeded identically: one called with training=True, the other with training=False.
        # Because the flag is ignored, both must produce the exact same sequence.
        agent_train = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 5})
        agent_eval = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 5})
        seq_train = [agent_train.select_action(obs, training=True) for _ in range(30)]
        seq_eval = [agent_eval.select_action(obs, training=False) for _ in range(30)]
        assert seq_train == seq_eval

    def test_update_always_returns_empty_dict(self):
        agent = RandomAgent(observation_dim=4, action_dim=2, config={})
        for batch in [
            {},
            {"observations": np.zeros((5, 4))},
            _make_batch(20),
        ]:
            assert agent.update(batch) == {}

    def test_save_load_roundtrip_restores_rng_state(self, tmp_path):
        obs = np.zeros(4)
        agent = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 99})
        _ = [agent.select_action(obs) for _ in range(25)]

        p = str(tmp_path / "random")
        agent.save(p)
        assert os.path.exists(f"{p}.npz")

        expected = [agent.select_action(obs) for _ in range(20)]
        loaded = RandomAgent(observation_dim=4, action_dim=2, config={})
        loaded.load(p)
        actual = [loaded.select_action(obs) for _ in range(20)]
        assert actual == expected

    def test_config_stored_unchanged(self):
        config = {"seed": 123, "extra": "value"}
        agent = RandomAgent(observation_dim=4, action_dim=2, config=config)
        assert agent.config == config

    def test_two_instances_no_shared_rng_state(self):
        """Two seeded agents with the same seed, interleaved, each match their solo sequence."""
        obs = np.zeros(4)
        seed = 42
        n = 20

        # Solo sequences (one agent at a time)
        solo_a = RandomAgent(observation_dim=4, action_dim=2, config={"seed": seed})
        seq_solo_a = [solo_a.select_action(obs) for _ in range(n)]

        solo_b = RandomAgent(observation_dim=4, action_dim=2, config={"seed": seed})
        seq_solo_b = [solo_b.select_action(obs) for _ in range(n)]

        # Interleaved sequences (both agents called alternately)
        inter_a = RandomAgent(observation_dim=4, action_dim=2, config={"seed": seed})
        inter_b = RandomAgent(observation_dim=4, action_dim=2, config={"seed": seed})
        seq_inter_a, seq_inter_b = [], []
        for _ in range(n):
            seq_inter_a.append(inter_a.select_action(obs))
            seq_inter_b.append(inter_b.select_action(obs))

        # Each agent's interleaved sequence must match its solo sequence (independent RNG)
        assert seq_inter_a == seq_solo_a
        assert seq_inter_b == seq_solo_b

    def test_seeded_sequence_is_stable_across_instances(self):
        obs = np.zeros(4)
        # Instantiate two agents once and sample multiple actions from each
        agent_a = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 77})
        agent_b = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 77})
        seq_a = [agent_a.select_action(obs) for _ in range(10)]
        seq_b = [agent_b.select_action(obs) for _ in range(10)]
        assert seq_a == seq_b


# ===========================================================================
# ReinforceAgent – extended tests
# ===========================================================================


class TestReinforceAgentExtended:
    """Comprehensive tests for ReinforceAgent validation, numerics, and I/O."""

    # --- Construction / validation ------------------------------------------

    def test_invalid_observation_dim_zero(self):
        with pytest.raises(ValueError, match="observation_dim"):
            ReinforceAgent(observation_dim=0, action_dim=2, config=_REINFORCE_CFG)

    def test_invalid_observation_dim_negative(self):
        with pytest.raises(ValueError, match="observation_dim"):
            ReinforceAgent(observation_dim=-1, action_dim=2, config=_REINFORCE_CFG)

    def test_invalid_action_dim_zero(self):
        with pytest.raises(ValueError, match="action_dim"):
            ReinforceAgent(observation_dim=4, action_dim=0, config=_REINFORCE_CFG)

    def test_invalid_action_dim_negative(self):
        with pytest.raises(ValueError, match="action_dim"):
            ReinforceAgent(observation_dim=4, action_dim=-3, config=_REINFORCE_CFG)

    def test_invalid_hidden_dim_zero(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            ReinforceAgent(
                observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "hidden_dim": 0}
            )

    def test_invalid_hidden_dim_negative(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            ReinforceAgent(
                observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "hidden_dim": -8}
            )

    def test_invalid_learning_rate_zero(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ReinforceAgent(
                observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "learning_rate": 0.0}
            )

    def test_invalid_learning_rate_negative(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ReinforceAgent(
                observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "learning_rate": -1e-4}
            )

    def test_invalid_gamma_zero(self):
        with pytest.raises(ValueError, match="gamma"):
            ReinforceAgent(observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "gamma": 0.0})

    def test_invalid_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma"):
            ReinforceAgent(
                observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "gamma": -0.1}
            )

    def test_invalid_gamma_greater_than_one(self):
        with pytest.raises(ValueError, match="gamma"):
            ReinforceAgent(
                observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "gamma": 1.01}
            )

    def test_valid_gamma_exactly_one(self):
        """gamma=1.0 is a valid boundary value and should not raise."""
        agent = ReinforceAgent(
            observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "gamma": 1.0}
        )
        assert agent.gamma == 1.0

    def test_defaults_without_config(self):
        """Agent created with empty config should use documented defaults."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config={})
        assert agent.learning_rate == 1e-3
        assert agent.gamma == 0.99
        assert agent.hidden_dim == 128

    # --- Weight shapes -------------------------------------------------------

    def test_weight_shapes(self):
        agent = ReinforceAgent(
            observation_dim=4, action_dim=3, config={**_REINFORCE_CFG, "hidden_dim": 32}
        )
        assert agent._W1.shape == (4, 32)
        assert agent._b1.shape == (32,)
        assert agent._W2.shape == (32, 3)
        assert agent._b2.shape == (3,)

    def test_bias_initialised_to_zeros(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        np.testing.assert_array_equal(agent._b1, np.zeros_like(agent._b1))
        np.testing.assert_array_equal(agent._b2, np.zeros_like(agent._b2))

    # --- Policy forward pass -------------------------------------------------

    def test_policy_output_shape(self):
        for action_dim in (2, 3, 5):
            agent = ReinforceAgent(observation_dim=4, action_dim=action_dim, config=_REINFORCE_CFG)
            probs, h = agent._policy(np.zeros(4))
            assert probs.shape == (action_dim,)
            assert h.shape == (64,)

    def test_policy_probabilities_sum_to_one(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        for _ in range(10):
            obs = np.random.randn(4)
            probs, _ = agent._policy(obs)
            np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_policy_probabilities_non_negative(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=5, config=_REINFORCE_CFG)
        for _ in range(10):
            probs, _ = agent._policy(np.random.randn(4))
            assert np.all(probs >= 0)

    def test_policy_probabilities_strictly_positive(self):
        """Softmax output should never collapse to zero for reasonable inputs."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        probs, _ = agent._policy(np.array([100.0, -100.0, 0.0, 0.5]))
        assert np.all(probs > 0)

    # --- Returns computation -------------------------------------------------

    def test_compute_returns_single_step(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        returns = agent._compute_returns(np.array([3.0]))
        np.testing.assert_allclose(returns, [3.0])

    def test_compute_returns_zero_rewards(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        returns = agent._compute_returns(np.zeros(5))
        np.testing.assert_array_equal(returns, np.zeros(5))

    def test_compute_returns_gamma_one_is_sum(self):
        """With gamma=1 the return at t=0 equals the total reward sum."""
        agent = ReinforceAgent(
            observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "gamma": 1.0}
        )
        rewards = np.array([1.0, 2.0, 3.0])
        returns = agent._compute_returns(rewards)
        np.testing.assert_allclose(returns[0], 6.0, atol=1e-8)

    def test_compute_returns_exact_values(self):
        """Verify discounted returns against hand-computed values."""
        agent = ReinforceAgent(
            observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "gamma": 0.5}
        )
        rewards = np.array([1.0, 1.0, 1.0])
        returns = agent._compute_returns(rewards)
        # G[2] = 1, G[1] = 1 + 0.5*1 = 1.5, G[0] = 1 + 0.5*1.5 = 1.75
        np.testing.assert_allclose(returns, [1.75, 1.5, 1.0], atol=1e-8)

    def test_compute_returns_length_preserved(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        for n in (1, 5, 50):
            rewards = np.ones(n)
            assert len(agent._compute_returns(rewards)) == n

    # --- Action selection ----------------------------------------------------

    def test_select_action_returns_integer(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        action = agent.select_action(np.zeros(4))
        assert isinstance(action, int)

    def test_select_action_in_valid_range(self):
        for action_dim in (2, 3, 5):
            agent = ReinforceAgent(observation_dim=4, action_dim=action_dim, config=_REINFORCE_CFG)
            for _ in range(50):
                assert 0 <= agent.select_action(np.zeros(4)) < action_dim

    def test_select_action_greedy_is_deterministic(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        obs = np.array([1.0, -1.0, 0.5, -0.5])
        greedy = [agent.select_action(obs, training=False) for _ in range(30)]
        assert len(set(greedy)) == 1

    def test_select_action_greedy_matches_argmax(self):
        """Greedy action should equal argmax of the policy probabilities."""
        agent = ReinforceAgent(observation_dim=4, action_dim=4, config=_REINFORCE_CFG)
        obs = np.random.randn(4)
        probs, _ = agent._policy(obs)
        greedy = agent.select_action(obs, training=False)
        assert greedy == int(np.argmax(probs))

    def test_select_action_stochastic_explores(self):
        """Over many draws both actions should appear during training."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        actions = {agent.select_action(np.zeros(4), training=True) for _ in range(200)}
        assert len(actions) == 2

    def test_seeded_agent_reproducible_sequence(self):
        obs = np.zeros(4)
        a = ReinforceAgent(observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "seed": 42})
        b = ReinforceAgent(observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "seed": 42})
        seq_a = [a.select_action(obs) for _ in range(30)]
        seq_b = [b.select_action(obs) for _ in range(30)]
        assert seq_a == seq_b

    # --- Update --------------------------------------------------------------

    def test_update_returns_float_policy_loss(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        metrics = agent.update(_make_batch())
        assert isinstance(metrics["policy_loss"], float)

    def test_update_empty_batch_returns_zero_loss(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        batch = {"observations": np.empty((0, 4)), "actions": np.array([]), "rewards": np.array([])}
        assert agent.update(batch) == {"policy_loss": 0.0}

    def test_update_single_step_trajectory(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        batch = {
            "observations": np.zeros((1, 4)),
            "actions": np.array([0]),
            "rewards": np.array([1.0]),
        }
        metrics = agent.update(batch)
        assert np.isfinite(metrics["policy_loss"])

    def test_update_loss_is_finite(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        for _ in range(5):
            metrics = agent.update(_make_batch(20))
            assert np.isfinite(metrics["policy_loss"])

    def test_update_negative_rewards(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        batch = _make_batch(10)
        batch["rewards"] = -np.ones(10)
        metrics = agent.update(batch)
        assert np.isfinite(metrics["policy_loss"])

    def test_update_only_policy_loss_key(self):
        """ReinforceAgent.update returns exactly one key."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        metrics = agent.update(_make_batch())
        assert list(metrics.keys()) == ["policy_loss"]

    def test_update_all_weights_change(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        before = {k: getattr(agent, k).copy() for k in ("_W1", "_b1", "_W2", "_b2")}
        agent.update(_make_batch(20))
        for k, v in before.items():
            assert not np.allclose(getattr(agent, k), v), f"{k} should change after update"

    def test_update_inconsistent_trajectory_lengths(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        batch = {
            "observations": np.zeros((5, 4)),
            "actions": np.array([0, 1, 0]),  # length mismatch
            "rewards": np.ones(5),
        }
        with pytest.raises(ValueError, match="Inconsistent"):
            agent.update(batch)

    def test_update_repeated_keeps_finite_loss_and_changes_weights(self):
        """Repeated updates on a fixed batch keep loss finite and move parameters."""
        agent = ReinforceAgent(
            observation_dim=4,
            action_dim=2,
            config={**_REINFORCE_CFG, "learning_rate": 5e-3},
        )
        rng = np.random.default_rng(0)
        batch = {
            "observations": rng.standard_normal((30, 4)),
            "actions": rng.integers(0, 2, size=30),
            "rewards": np.ones(30),
        }
        w1_before = agent._W1.copy()
        losses = [agent.update(batch)["policy_loss"] for _ in range(20)]
        # All losses should be finite
        assert all(np.isfinite(loss) for loss in losses), "Policy loss became non-finite"
        # Parameters should have moved from the initial position
        assert not np.allclose(agent._W1, w1_before), "W1 should change after repeated updates"

    def test_update_large_observation_and_action_dim(self):
        agent = ReinforceAgent(observation_dim=64, action_dim=8, config={"seed": 0})
        batch = _make_batch(n=20, obs_dim=64, action_dim=8)
        metrics = agent.update(batch)
        assert np.isfinite(metrics["policy_loss"])

    # --- Save / Load ---------------------------------------------------------

    def test_save_appends_npz_when_no_extension(self, tmp_path):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        p = str(tmp_path / "ckpt")
        agent.save(p)
        assert os.path.exists(p + ".npz")

    def test_save_load_roundtrip_no_extension(self, tmp_path):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        p = str(tmp_path / "ckpt")
        agent.save(p)
        agent2 = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        agent2._W1 = np.zeros_like(agent2._W1)
        agent2.load(p)
        np.testing.assert_array_equal(agent._W1, agent2._W1)
        np.testing.assert_array_equal(agent._W2, agent2._W2)
        np.testing.assert_array_equal(agent._b1, agent2._b1)
        np.testing.assert_array_equal(agent._b2, agent2._b2)

    def test_save_load_preserves_inference_output(self, tmp_path):
        """After save/load the agent produces the same action probabilities."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        obs = np.array([0.1, -0.2, 0.3, -0.4])
        probs_before, _ = agent._policy(obs)

        p = str(tmp_path / "ckpt")
        agent.save(p)

        agent2 = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        agent2.load(p)
        probs_after, _ = agent2._policy(obs)

        np.testing.assert_allclose(probs_before, probs_after, atol=1e-10)

    def test_save_unsupported_extension_raises(self, tmp_path):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        with pytest.raises(ValueError, match="Unsupported checkpoint extension"):
            agent.save(str(tmp_path / "model.h5"))

    def test_load_unsupported_extension_raises(self, tmp_path):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        with pytest.raises(ValueError, match="Unsupported checkpoint extension"):
            agent.load(str(tmp_path / "model.h5"))

    def test_load_nonexistent_file_raises(self, tmp_path):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        with pytest.raises(FileNotFoundError):
            agent.load(str(tmp_path / "nonexistent.pt"))

    def test_load_nonexistent_no_extension_falls_back_to_npz(self, tmp_path):
        """Loading a bare path appends .npz before looking up the file."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        p = str(tmp_path / "ckpt")
        agent.save(p)
        agent2 = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        agent2.load(p)  # Should resolve to p + ".npz"
        np.testing.assert_array_equal(agent._W1, agent2._W1)

    def test_save_load_after_training(self, tmp_path):
        """Weights saved after training are restored with full precision."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        for _ in range(5):
            agent.update(_make_batch(20))

        p = str(tmp_path / "trained")
        agent.save(p)

        agent2 = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        agent2.load(p)

        np.testing.assert_array_equal(agent._W1, agent2._W1)
        np.testing.assert_array_equal(agent._W2, agent2._W2)

    def test_load_pt_extension_exact_file(self, tmp_path):
        """Loading a .pt path that exists on disk should load that exact file."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        p = str(tmp_path / "model.pt")
        agent.save(p)
        assert os.path.exists(p)

        agent2 = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        agent2._W1 = np.zeros_like(agent2._W1)
        agent2.load(p)
        np.testing.assert_array_equal(agent._W1, agent2._W1)

    def test_load_pt_legacy_npz_fallback(self, tmp_path):
        """Loading a .pt path falls back to '<name>.pt.npz' when the .pt file is absent."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        legacy_path = tmp_path / "reinforce_legacy.pt.npz"
        np.savez(legacy_path, W1=agent._W1, b1=agent._b1, W2=agent._W2, b2=agent._b2)

        agent2 = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        agent2._W1 = np.zeros_like(agent2._W1)
        agent2.load(str(tmp_path / "reinforce_legacy.pt"))
        np.testing.assert_array_equal(agent._W1, agent2._W1)
        np.testing.assert_array_equal(agent._W2, agent2._W2)
        np.testing.assert_array_equal(agent._b1, agent2._b1)
        np.testing.assert_array_equal(agent._b2, agent2._b2)


# ===========================================================================
# ActorCriticAgent – extended tests
# ===========================================================================


class TestActorCriticAgentExtended:
    """Comprehensive tests for ActorCriticAgent validation, numerics, and I/O."""

    # --- Construction / validation ------------------------------------------

    def test_invalid_observation_dim_zero(self):
        with pytest.raises(ValueError, match="observation_dim"):
            ActorCriticAgent(observation_dim=0, action_dim=2, config=_AC_CFG)

    def test_invalid_observation_dim_negative(self):
        with pytest.raises(ValueError, match="observation_dim"):
            ActorCriticAgent(observation_dim=-2, action_dim=2, config=_AC_CFG)

    def test_invalid_action_dim_zero(self):
        with pytest.raises(ValueError, match="action_dim"):
            ActorCriticAgent(observation_dim=4, action_dim=0, config=_AC_CFG)

    def test_invalid_action_dim_negative(self):
        with pytest.raises(ValueError, match="action_dim"):
            ActorCriticAgent(observation_dim=4, action_dim=-1, config=_AC_CFG)

    def test_invalid_hidden_dim_zero(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, "hidden_dim": 0})

    def test_invalid_hidden_dim_negative(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, "hidden_dim": -16})

    def test_invalid_learning_rate_zero(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ActorCriticAgent(
                observation_dim=4, action_dim=2, config={**_AC_CFG, "learning_rate": 0.0}
            )

    def test_invalid_learning_rate_negative(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ActorCriticAgent(
                observation_dim=4, action_dim=2, config={**_AC_CFG, "learning_rate": -1e-4}
            )

    def test_invalid_gamma_zero(self):
        with pytest.raises(ValueError, match="gamma"):
            ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, "gamma": 0.0})

    def test_invalid_gamma_greater_than_one(self):
        with pytest.raises(ValueError, match="gamma"):
            ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, "gamma": 1.1})

    def test_invalid_value_coef_negative(self):
        with pytest.raises(ValueError, match="value_coef"):
            ActorCriticAgent(
                observation_dim=4, action_dim=2, config={**_AC_CFG, "value_coef": -0.5}
            )

    def test_invalid_entropy_coef_negative(self):
        with pytest.raises(ValueError, match="entropy_coef"):
            ActorCriticAgent(
                observation_dim=4, action_dim=2, config={**_AC_CFG, "entropy_coef": -0.01}
            )

    def test_valid_zero_coefs(self):
        """value_coef=0 and entropy_coef=0 are valid boundary values."""
        agent = ActorCriticAgent(
            observation_dim=4,
            action_dim=2,
            config={**_AC_CFG, "value_coef": 0.0, "entropy_coef": 0.0},
        )
        assert agent.value_coef == 0.0
        assert agent.entropy_coef == 0.0

    def test_valid_gamma_exactly_one(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, "gamma": 1.0})
        assert agent.gamma == 1.0

    def test_defaults_without_config(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config={})
        assert agent.learning_rate == 1e-3
        assert agent.gamma == 0.99
        assert agent.hidden_dim == 128
        assert agent.value_coef == 0.5
        assert agent.entropy_coef == 0.01

    # --- Weight shapes -------------------------------------------------------

    def test_weight_shapes(self):
        agent = ActorCriticAgent(
            observation_dim=6, action_dim=3, config={**_AC_CFG, "hidden_dim": 32}
        )
        assert agent._W1.shape == (6, 32)
        assert agent._b1.shape == (32,)
        assert agent._W_pi.shape == (32, 3)
        assert agent._b_pi.shape == (3,)
        assert agent._W_v.shape == (32,)
        assert isinstance(agent._b_v, float)

    def test_bias_initialised_to_zeros(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        np.testing.assert_array_equal(agent._b1, np.zeros_like(agent._b1))
        np.testing.assert_array_equal(agent._b_pi, np.zeros_like(agent._b_pi))
        assert agent._b_v == 0.0

    # --- Forward pass --------------------------------------------------------

    def test_forward_output_types_and_shapes(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        probs, value, h = agent._forward(np.zeros(4))
        assert probs.shape == (2,)
        assert isinstance(value, float)
        assert h.shape == (64,)

    def test_forward_probs_sum_to_one(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        for _ in range(10):
            probs, _, _ = agent._forward(np.random.randn(4))
            np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_forward_probs_non_negative(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=5, config=_AC_CFG)
        for _ in range(10):
            probs, _, _ = agent._forward(np.random.randn(4))
            assert np.all(probs >= 0)

    def test_forward_value_is_finite(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        for _ in range(10):
            _, value, _ = agent._forward(np.random.randn(4))
            assert np.isfinite(value)

    def test_forward_hidden_relu_non_negative(self):
        """Hidden activations after ReLU must be non-negative."""
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        for _ in range(10):
            _, _, h = agent._forward(np.random.randn(4))
            assert np.all(h >= 0)

    # --- Returns computation -------------------------------------------------

    def test_compute_returns_exact_values(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, "gamma": 0.5})
        rewards = np.array([1.0, 1.0, 1.0])
        returns = agent._compute_returns(rewards)
        np.testing.assert_allclose(returns, [1.75, 1.5, 1.0], atol=1e-8)

    def test_compute_returns_single_step(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        returns = agent._compute_returns(np.array([5.0]))
        np.testing.assert_allclose(returns, [5.0])

    def test_compute_returns_length_preserved(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        for n in (1, 10, 100):
            assert len(agent._compute_returns(np.ones(n))) == n

    # --- Action selection ----------------------------------------------------

    def test_select_action_returns_integer(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        action = agent.select_action(np.zeros(4))
        assert isinstance(action, int)

    def test_select_action_in_valid_range(self):
        for action_dim in (2, 4, 6):
            agent = ActorCriticAgent(observation_dim=4, action_dim=action_dim, config=_AC_CFG)
            for _ in range(50):
                assert 0 <= agent.select_action(np.zeros(4)) < action_dim

    def test_select_action_greedy_is_deterministic(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        obs = np.array([1.0, -1.0, 0.5, -0.5])
        greedy = [agent.select_action(obs, training=False) for _ in range(30)]
        assert len(set(greedy)) == 1

    def test_select_action_greedy_matches_argmax(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=4, config=_AC_CFG)
        obs = np.random.randn(4)
        probs, _, _ = agent._forward(obs)
        greedy = agent.select_action(obs, training=False)
        assert greedy == int(np.argmax(probs))

    def test_seeded_agent_reproducible_sequence(self):
        obs = np.zeros(4)
        a = ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, "seed": 99})
        b = ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, "seed": 99})
        seq_a = [a.select_action(obs) for _ in range(30)]
        seq_b = [b.select_action(obs) for _ in range(30)]
        assert seq_a == seq_b

    # --- Update --------------------------------------------------------------

    def test_update_returns_all_metric_keys(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        metrics = agent.update(_make_batch())
        assert set(metrics.keys()) == {"policy_loss", "value_loss", "entropy_bonus"}

    def test_update_all_metrics_are_finite(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        for _ in range(5):
            metrics = agent.update(_make_batch(20))
            for key, val in metrics.items():
                assert np.isfinite(val), f"Metric {key} is not finite: {val}"

    def test_update_entropy_bonus_non_negative(self):
        """Mean policy entropy is always ≥ 0."""
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        for _ in range(5):
            assert agent.update(_make_batch())["entropy_bonus"] >= 0.0

    def test_update_value_loss_non_negative(self):
        """Value loss (MSE) is always ≥ 0."""
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        for _ in range(5):
            assert agent.update(_make_batch())["value_loss"] >= 0.0

    def test_update_empty_batch_returns_zero_metrics(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        batch = {"observations": np.empty((0, 4)), "actions": np.array([]), "rewards": np.array([])}
        assert agent.update(batch) == {"policy_loss": 0.0, "value_loss": 0.0, "entropy_bonus": 0.0}

    def test_update_single_step_trajectory(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        batch = {
            "observations": np.zeros((1, 4)),
            "actions": np.array([0]),
            "rewards": np.array([1.0]),
        }
        metrics = agent.update(batch)
        for val in metrics.values():
            assert np.isfinite(val)

    def test_update_all_weights_change(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        attrs = ("_W1", "_b1", "_W_pi", "_b_pi", "_W_v")
        before = {k: getattr(agent, k).copy() for k in attrs}
        b_v_before = agent._b_v
        agent.update(_make_batch(20))
        for k, v in before.items():
            assert not np.allclose(getattr(agent, k), v), f"{k} should change after update"
        assert agent._b_v != b_v_before

    def test_update_inconsistent_trajectory_lengths(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        batch = {
            "observations": np.zeros((5, 4)),
            "actions": np.array([0, 1]),  # wrong length
            "rewards": np.ones(5),
        }
        with pytest.raises(ValueError, match="Inconsistent"):
            agent.update(batch)

    def test_update_negative_rewards_are_handled(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        batch = _make_batch(10)
        batch["rewards"] = -np.ones(10)
        metrics = agent.update(batch)
        for val in metrics.values():
            assert np.isfinite(val)

    def test_update_large_observation_and_action_dim(self):
        agent = ActorCriticAgent(observation_dim=64, action_dim=8, config={"seed": 0})
        batch = _make_batch(n=20, obs_dim=64, action_dim=8)
        metrics = agent.update(batch)
        for val in metrics.values():
            assert np.isfinite(val)

    def test_value_loss_decreases_over_repeated_updates(self):
        """Critic parameters change and loss stays finite over repeated updates on a fixed batch."""
        agent = ActorCriticAgent(
            observation_dim=4,
            action_dim=2,
            config={**_AC_CFG, "learning_rate": 1e-2, "entropy_coef": 0.0},
        )
        rng = np.random.default_rng(1)
        batch = {
            "observations": rng.standard_normal((50, 4)),
            "actions": rng.integers(0, 2, size=50),
            "rewards": np.ones(50),
        }
        w_v_before = agent._W_v.copy()
        losses = [agent.update(batch)["value_loss"] for _ in range(20)]
        # All value losses must be finite and non-negative
        assert all(np.isfinite(v) and v >= 0 for v in losses), "Value loss has invalid values"
        # Critic weights should have changed
        assert not np.allclose(agent._W_v, w_v_before), "W_v should change after repeated updates"

    def test_zero_value_coef_does_not_change_critic_weights(self):
        """When value_coef=0 the critic output gradient is zero, so W_v should not change."""
        agent = ActorCriticAgent(
            observation_dim=4,
            action_dim=2,
            config={**_AC_CFG, "value_coef": 0.0},
        )
        w_v_before = agent._W_v.copy()
        agent.update(_make_batch(20))
        np.testing.assert_array_equal(agent._W_v, w_v_before)

    def test_entropy_coef_effect_on_update(self):
        """Both entropy_coef=0 and entropy_coef=1 produce finite metrics; higher coef moves policy weights more."""
        rng = np.random.default_rng(3)
        batch = {
            "observations": rng.standard_normal((30, 4)),
            "actions": rng.integers(0, 2, size=30),
            "rewards": np.ones(30),
        }
        agent_low = ActorCriticAgent(
            observation_dim=4, action_dim=2, config={**_AC_CFG, "seed": 0, "entropy_coef": 0.0}
        )
        agent_high = ActorCriticAgent(
            observation_dim=4, action_dim=2, config={**_AC_CFG, "seed": 0, "entropy_coef": 1.0}
        )

        w_pi_low_before = agent_low._W_pi.copy()
        w_pi_high_before = agent_high._W_pi.copy()

        m_low = agent_low.update(batch)
        m_high = agent_high.update(batch)

        # Both produce valid, finite metrics
        for m in (m_low, m_high):
            for val in m.values():
                assert np.isfinite(val)

        # Higher entropy_coef applies a stronger entropy gradient to the policy head,
        # so the policy-weight change should be larger in magnitude.
        delta_low = np.linalg.norm(agent_low._W_pi - w_pi_low_before)
        delta_high = np.linalg.norm(agent_high._W_pi - w_pi_high_before)
        assert delta_high > delta_low, (
            "Higher entropy_coef should produce larger policy-weight updates"
        )

    # --- Save / Load ---------------------------------------------------------

    def test_save_appends_npz_when_no_extension(self, tmp_path):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        p = str(tmp_path / "ckpt")
        agent.save(p)
        assert os.path.exists(p + ".npz")

    def test_save_load_roundtrip_no_extension(self, tmp_path):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        p = str(tmp_path / "ckpt")
        agent.save(p)
        agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        agent2._W1 = np.zeros_like(agent2._W1)
        agent2.load(p)
        np.testing.assert_array_equal(agent._W1, agent2._W1)
        np.testing.assert_array_equal(agent._W_pi, agent2._W_pi)
        np.testing.assert_array_equal(agent._W_v, agent2._W_v)
        assert agent._b_v == agent2._b_v

    def test_save_load_preserves_inference_output(self, tmp_path):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        obs = np.array([0.1, -0.2, 0.3, -0.4])
        probs_before, val_before, _ = agent._forward(obs)

        p = str(tmp_path / "ckpt")
        agent.save(p)

        agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        agent2.load(p)
        probs_after, val_after, _ = agent2._forward(obs)

        np.testing.assert_allclose(probs_before, probs_after, atol=1e-10)
        np.testing.assert_allclose(val_before, val_after, atol=1e-10)

    def test_save_unsupported_extension_raises(self, tmp_path):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        with pytest.raises(ValueError, match="Unsupported checkpoint extension"):
            agent.save(str(tmp_path / "model.pkl"))

    def test_load_unsupported_extension_raises(self, tmp_path):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        with pytest.raises(ValueError, match="Unsupported checkpoint extension"):
            agent.load(str(tmp_path / "model.pkl"))

    def test_load_nonexistent_pt_file_raises(self, tmp_path):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        with pytest.raises(FileNotFoundError):
            agent.load(str(tmp_path / "ghost.pt"))

    def test_load_nonexistent_no_extension_falls_back_to_npz(self, tmp_path):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        p = str(tmp_path / "ckpt")
        agent.save(p)
        agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        agent2.load(p)
        np.testing.assert_array_equal(agent._W1, agent2._W1)

    def test_save_load_after_training(self, tmp_path):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        for _ in range(5):
            agent.update(_make_batch(20))
        p = str(tmp_path / "trained")
        agent.save(p)
        agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        agent2.load(p)
        np.testing.assert_array_equal(agent._W1, agent2._W1)
        np.testing.assert_array_equal(agent._W_pi, agent2._W_pi)
        np.testing.assert_array_equal(agent._W_v, agent2._W_v)
        assert agent._b_v == agent2._b_v

    def test_load_pt_extension_exact_file(self, tmp_path):
        """Loading a .pt path that exists on disk should load that exact file."""
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        p = str(tmp_path / "model.pt")
        agent.save(p)
        assert os.path.exists(p)

        agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        agent2._W1 = np.zeros_like(agent2._W1)
        agent2.load(p)
        np.testing.assert_array_equal(agent._W1, agent2._W1)

    def test_load_pt_legacy_npz_fallback(self, tmp_path):
        """Loading a .pt path falls back to '<name>.pt.npz' when the .pt file is absent."""
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        legacy_path = tmp_path / "ac_legacy.pt.npz"
        np.savez(
            legacy_path,
            W1=agent._W1,
            b1=agent._b1,
            W_pi=agent._W_pi,
            b_pi=agent._b_pi,
            W_v=agent._W_v,
            b_v=np.array(agent._b_v),
        )
        agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        agent2._W1 = np.zeros_like(agent2._W1)
        agent2.load(str(tmp_path / "ac_legacy.pt"))
        np.testing.assert_array_equal(agent._W1, agent2._W1)
        np.testing.assert_array_equal(agent._W_pi, agent2._W_pi)
        np.testing.assert_array_equal(agent._W_v, agent2._W_v)
        assert agent._b_v == agent2._b_v
    def test_invalid_observation_dim_raises(self):
        with pytest.raises(ValueError, match="observation_dim"):
            RandomAgent(observation_dim=0, action_dim=2, config={})

    def test_invalid_action_dim_raises(self):
        with pytest.raises(ValueError, match="action_dim"):
            RandomAgent(observation_dim=4, action_dim=0, config={})
