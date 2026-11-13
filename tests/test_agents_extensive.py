"""Extensive tests for all agent implementations (RandomAgent, ReinforceAgent, ActorCriticAgent, BaseAgent)."""

import os

import numpy as np
import pytest

from rl_core.agents import ActorCriticAgent, BaseAgent, RandomAgent, ReinforceAgent
from rl_core.seeding import AGENT_EXPLORATION_STREAM, derive_rng
from tests.agent_internals import (
    hidden_activations,
    loss_and_grads,
    probabilities,
    set_weights,
    state_value,
    weights,
)
from tests.helpers import assert_same_checkpoint

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

            def _checkpoint_arrays(self):
                return {}

            def _load_checkpoint_arrays(self, data):
                pass

        with pytest.raises(TypeError):
            Incomplete(observation_dim=4, action_dim=2, config={})

    def test_base_agent_subclass_missing_update(self):
        class Incomplete(BaseAgent):
            def select_action(self, observation, training=True):
                return 0

            def _checkpoint_arrays(self):
                return {}

            def _load_checkpoint_arrays(self, data):
                pass

        with pytest.raises(TypeError):
            Incomplete(observation_dim=4, action_dim=2, config={})

    def test_base_agent_subclass_missing_checkpoint_arrays(self):
        class Incomplete(BaseAgent):
            def select_action(self, observation, training=True):
                return 0

            def update(self, batch):
                return {}

            def _load_checkpoint_arrays(self, data):
                pass

        with pytest.raises(TypeError):
            Incomplete(observation_dim=4, action_dim=2, config={})

    def test_base_agent_subclass_missing_load_checkpoint_arrays(self):
        class Incomplete(BaseAgent):
            def select_action(self, observation, training=True):
                return 0

            def update(self, batch):
                return {}

            def _checkpoint_arrays(self):
                return {}

        with pytest.raises(TypeError):
            Incomplete(observation_dim=4, action_dim=2, config={})

    def test_base_agent_concrete_subclass_stores_attributes(self):
        """A fully-implemented subclass stores obs/action dim and config."""

        class Minimal(BaseAgent):
            def select_action(self, observation, training=True):
                return 0

            def update(self, batch):
                return {}

            def _checkpoint_arrays(self):
                return {}

            def _load_checkpoint_arrays(self, data):
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
        obs = np.random.default_rng(0).standard_normal(256)
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

    def test_actions_are_python_ints_from_the_exploration_stream(self):
        """Training actions are the seed's exploration stream, uniform over the actions, each
        returned as a Python int (not a NumPy integer)."""
        agent = RandomAgent(observation_dim=4, action_dim=3, config={"seed": 11})
        stream = derive_rng(11, AGENT_EXPLORATION_STREAM)
        actions = [agent.select_action(np.zeros(4)) for _ in range(20)]
        assert actions == [int(stream.integers(0, 3)) for _ in range(20)]
        assert all(type(action) is int for action in actions)

    @pytest.mark.parametrize("dims,match", [((0, 2), "observation_dim"), ((4, 0), "action_dim")])
    def test_invalid_dimension_raises(self, dims, match):
        with pytest.raises(ValueError, match=match):
            RandomAgent(observation_dim=dims[0], action_dim=dims[1], config={})

    def test_evaluation_draws_from_its_own_reproducible_generator(self):
        """Since 2026-10-18 evaluation actions come from a separate stream (CHANGELOG.md,
        2026-10-18): different from the training sequence, reproducible per seed."""
        obs = np.zeros(4)

        def actions(training):
            agent = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 5})
            return [agent.select_action(obs, training=training) for _ in range(30)]

        assert actions(False) == actions(False)
        assert actions(True) != actions(False)

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
# REINFORCE and Actor-Critic – what the two discrete policy-gradient agents share
# ===========================================================================

# (class, config, the name of the policy head's bias)
_POLICY_GRADIENT_AGENTS = [
    pytest.param(ReinforceAgent, _REINFORCE_CFG, "b2", id="reinforce"),
    pytest.param(ActorCriticAgent, _AC_CFG, "b_pi", id="actor_critic"),
]
_INVALID_SETTINGS = [
    ("hidden_dim", 0),
    ("hidden_dim", -8),
    ("learning_rate", 0.0),
    ("learning_rate", -1e-4),
    ("gamma", 0.0),
    ("gamma", -0.1),
    ("gamma", 1.01),
]


@pytest.mark.parametrize("agent_cls,cfg,head_bias", _POLICY_GRADIENT_AGENTS)
class TestPolicyGradientAgentsShared:
    """Validation, action selection, update edges and checkpoint files: the same for both."""

    # --- Construction / validation ------------------------------------------

    @pytest.mark.parametrize(
        "dims,match",
        [
            ((0, 2), "observation_dim"),
            ((-1, 2), "observation_dim"),
            ((4, 0), "action_dim"),
            ((4, -3), "action_dim"),
        ],
    )
    def test_invalid_dimension_raises(self, agent_cls, cfg, head_bias, dims, match):
        with pytest.raises(ValueError, match=match):
            agent_cls(observation_dim=dims[0], action_dim=dims[1], config=cfg)

    @pytest.mark.parametrize("key,value", _INVALID_SETTINGS)
    def test_invalid_setting_raises(self, agent_cls, cfg, head_bias, key, value):
        with pytest.raises(ValueError, match=key):
            agent_cls(observation_dim=4, action_dim=2, config={**cfg, key: value})

    def test_valid_gamma_exactly_one(self, agent_cls, cfg, head_bias):
        """gamma=1.0 is a valid boundary value and should not raise."""
        agent = agent_cls(observation_dim=4, action_dim=2, config={**cfg, "gamma": 1.0})
        assert agent.gamma == 1.0

    # --- Policy --------------------------------------------------------------

    def test_policy_probabilities_are_a_distribution(self, agent_cls, cfg, head_bias):
        agent = agent_cls(observation_dim=4, action_dim=5, config=cfg)
        rng = np.random.default_rng(1)
        for _ in range(10):
            probs = probabilities(agent, rng.standard_normal(4))
            assert np.all(probs >= 0)
            np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    # --- Action selection ----------------------------------------------------

    def test_select_action_samples_the_only_likely_action(self, agent_cls, cfg, head_bias):
        """With a zero observation the hidden layer is zero and the logits are the head's bias;
        a bias of (-50, 50, -50) leaves action 1 the only one with any probability, sampled or
        greedy, and it comes back as a Python int."""
        agent = agent_cls(observation_dim=4, action_dim=3, config=cfg)
        set_weights(agent, **{head_bias: np.array([-50.0, 50.0, -50.0])})
        for training in (True, False):
            actions = [agent.select_action(np.zeros(4), training=training) for _ in range(50)]
            assert actions == [1] * 50, f"Only action 1 is likely (training={training})"
            assert all(type(action) is int for action in actions), "Actions are Python ints"

    def test_select_action_in_valid_range(self, agent_cls, cfg, head_bias):
        for action_dim in (2, 3, 5):
            agent = agent_cls(observation_dim=4, action_dim=action_dim, config=cfg)
            for _ in range(50):
                assert 0 <= agent.select_action(np.zeros(4)) < action_dim

    def test_select_action_greedy_is_deterministic(self, agent_cls, cfg, head_bias):
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        obs = np.array([1.0, -1.0, 0.5, -0.5])
        greedy = [agent.select_action(obs, training=False) for _ in range(30)]
        assert len(set(greedy)) == 1

    def test_select_action_greedy_matches_argmax(self, agent_cls, cfg, head_bias):
        """Greedy action should equal argmax of the policy probabilities."""
        agent = agent_cls(observation_dim=4, action_dim=4, config=cfg)
        obs = np.random.default_rng(2).standard_normal(4)
        greedy = agent.select_action(obs, training=False)
        assert greedy == int(np.argmax(probabilities(agent, obs)))

    def test_seeded_agent_reproducible_sequence(self, agent_cls, cfg, head_bias):
        obs = np.zeros(4)
        a = agent_cls(observation_dim=4, action_dim=2, config={**cfg, "seed": 42})
        b = agent_cls(observation_dim=4, action_dim=2, config={**cfg, "seed": 42})
        assert [a.select_action(obs) for _ in range(30)] == [
            b.select_action(obs) for _ in range(30)
        ]

    # --- Update --------------------------------------------------------------

    def test_update_inconsistent_trajectory_lengths(self, agent_cls, cfg, head_bias):
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        batch = {
            "observations": np.zeros((5, 4)),
            "actions": np.array([0, 1, 0]),  # length mismatch
            "rewards": np.ones(5),
        }
        with pytest.raises(ValueError, match="Inconsistent"):
            agent.update(batch)

    @pytest.mark.parametrize(
        "batch",
        [
            pytest.param(
                {"observations": np.zeros((1, 4)), "actions": np.array([0]), "rewards": [1.0]},
                id="single-step",
            ),
            pytest.param({**_make_batch(10), "rewards": -np.ones(10)}, id="negative-rewards"),
        ],
    )
    def test_update_edge_batches_give_finite_metrics(self, agent_cls, cfg, head_bias, batch):
        metrics = agent_cls(observation_dim=4, action_dim=2, config=cfg).update(batch)
        assert metrics and all(np.isfinite(v) for v in metrics.values()), metrics

    def test_update_large_observation_and_action_dim(self, agent_cls, cfg, head_bias):
        agent = agent_cls(observation_dim=64, action_dim=8, config={"seed": 0})
        metrics = agent.update(_make_batch(n=20, obs_dim=64, action_dim=8))
        assert metrics and all(np.isfinite(v) for v in metrics.values()), metrics

    # --- Save / Load ---------------------------------------------------------

    def _restored(self, agent_cls, cfg, path):
        restored = agent_cls(observation_dim=4, action_dim=2, config={**cfg, "seed": 9})
        restored.load(path)
        return restored

    def test_save_appends_npz_when_no_extension(self, agent_cls, cfg, head_bias, tmp_path):
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        p = str(tmp_path / "ckpt")
        agent.save(p)
        assert os.path.exists(p + ".npz")

    def test_load_without_extension_falls_back_to_npz(self, agent_cls, cfg, head_bias, tmp_path):
        """Loading a bare path appends .npz before looking up the file."""
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        p = str(tmp_path / "ckpt")
        agent.save(p)
        assert_same_checkpoint(agent, self._restored(agent_cls, cfg, p), tmp_path)

    def test_save_load_preserves_inference_output(self, agent_cls, cfg, head_bias, tmp_path):
        """After save/load the agent produces the same action probabilities."""
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        obs = np.array([0.1, -0.2, 0.3, -0.4])
        p = str(tmp_path / "ckpt")
        agent.save(p)
        np.testing.assert_allclose(
            probabilities(self._restored(agent_cls, cfg, p), obs),
            probabilities(agent, obs),
            atol=1e-10,
        )

    @pytest.mark.parametrize("extension", [".h5", ".pkl"])
    def test_unsupported_extension_raises(self, agent_cls, cfg, head_bias, tmp_path, extension):
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        with pytest.raises(ValueError, match="Unsupported checkpoint extension"):
            agent.save(str(tmp_path / f"model{extension}"))
        with pytest.raises(ValueError, match="Unsupported checkpoint extension"):
            agent.load(str(tmp_path / f"model{extension}"))

    def test_load_nonexistent_file_raises(self, agent_cls, cfg, head_bias, tmp_path):
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        with pytest.raises(FileNotFoundError):
            agent.load(str(tmp_path / "nonexistent.pt"))

    def test_save_load_after_training(self, agent_cls, cfg, head_bias, tmp_path):
        """Weights saved after training are restored with full precision."""
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        for _ in range(5):
            agent.update(_make_batch(20))
        p = str(tmp_path / "trained")
        agent.save(p)
        assert_same_checkpoint(agent, self._restored(agent_cls, cfg, p), tmp_path)

    def test_load_pt_extension_exact_file(self, agent_cls, cfg, head_bias, tmp_path):
        """Loading a .pt path that exists on disk should load that exact file."""
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        p = str(tmp_path / "model.pt")
        agent.save(p)
        assert os.path.exists(p)
        assert_same_checkpoint(agent, self._restored(agent_cls, cfg, p), tmp_path)

    def test_load_pt_legacy_npz_fallback(self, agent_cls, cfg, head_bias, tmp_path):
        """Loading a .pt path falls back to '<name>.pt.npz' when the .pt file is absent."""
        agent = agent_cls(observation_dim=4, action_dim=2, config=cfg)
        agent.save(str(tmp_path / "legacy.pt.npz"))
        restored = self._restored(agent_cls, cfg, str(tmp_path / "legacy.pt"))
        assert_same_checkpoint(agent, restored, tmp_path)


# ===========================================================================
# ReinforceAgent – extended tests
# ===========================================================================


class TestReinforceAgentExtended:
    """What only REINFORCE has: its defaults, network, single loss and reported value."""

    def test_defaults_without_config(self):
        """Agent created with empty config should use documented defaults."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config={})
        assert agent.learning_rate == 1e-3
        assert agent.gamma == 0.99
        assert agent.hidden_dim == 128

    def test_weight_shapes(self):
        agent = ReinforceAgent(
            observation_dim=4, action_dim=3, config={**_REINFORCE_CFG, "hidden_dim": 32}
        )
        assert weights(agent)["W1"].shape == (4, 32)
        assert weights(agent)["b1"].shape == (32,)
        assert weights(agent)["W2"].shape == (32, 3)
        assert weights(agent)["b2"].shape == (3,)

    def test_bias_initialised_to_zeros(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        np.testing.assert_array_equal(weights(agent)["b1"], np.zeros_like(weights(agent)["b1"]))
        np.testing.assert_array_equal(weights(agent)["b2"], np.zeros_like(weights(agent)["b2"]))

    def test_policy_output_shape(self):
        for action_dim in (2, 3, 5):
            agent = ReinforceAgent(observation_dim=4, action_dim=action_dim, config=_REINFORCE_CFG)
            assert probabilities(agent, np.zeros(4)).shape == (action_dim,)
            assert hidden_activations(agent, np.zeros(4)).shape == (64,)

    def test_policy_probabilities_strictly_positive(self):
        """Softmax output should never collapse to zero for reasonable inputs."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        probs = probabilities(agent, np.array([100.0, -100.0, 0.0, 0.5]))
        assert np.all(probs > 0)

    def test_select_action_stochastic_explores(self):
        """Over many draws both actions should appear during training."""
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        actions = {agent.select_action(np.zeros(4), training=True) for _ in range(200)}
        assert len(actions) == 2

    def test_update_reports_the_mean_loss_over_the_episode(self):
        """The reported policy_loss is the summed loss over the episode's steps, weighted by
        the normalised discounted returns, divided by its length.

        Rewards (1, 0, 2) with gamma 0.5 give returns G2 = 2, G1 = 0 + 0.5 * 2 = 1 and
        G0 = 1 + 0.5 * 1 = 1.5; normalised, (G - 1.5) / std(G) with std(G) = sqrt(1/6)."""
        agent = ReinforceAgent(
            observation_dim=4, action_dim=2, config={**_REINFORCE_CFG, "gamma": 0.5}
        )
        batch = _make_batch(n=3)
        batch["rewards"] = np.array([1.0, 0.0, 2.0])
        returns = np.array([1.5, 1.0, 2.0])
        normalised = (returns - 1.5) / np.sqrt(1.0 / 6.0)
        summed, _ = loss_and_grads(agent, batch["observations"], batch["actions"], normalised)

        metrics = agent.update(batch)

        assert type(metrics["policy_loss"]) is float
        np.testing.assert_allclose(metrics["policy_loss"], summed / 3, rtol=1e-12)

    def test_update_empty_batch_returns_zero_loss(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        batch = {"observations": np.empty((0, 4)), "actions": np.array([]), "rewards": np.array([])}
        assert agent.update(batch) == {"policy_loss": 0.0}

    def test_update_loss_is_finite(self):
        agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CFG)
        for _ in range(5):
            assert np.isfinite(agent.update(_make_batch(20))["policy_loss"])

    def test_update_repeated_lowers_the_policy_loss(self):
        """Repeated updates on a fixed batch descend the loss they report; a gradient with the
        wrong sign or applied to the wrong weights would not."""
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
        losses = [agent.update(batch)["policy_loss"] for _ in range(20)]
        assert all(np.isfinite(loss) for loss in losses), "Policy loss became non-finite"
        assert np.mean(losses[-5:]) < np.mean(losses[:5]), (
            f"The reported policy loss must fall on a fixed batch: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


# ===========================================================================
# ActorCriticAgent – extended tests
# ===========================================================================


class TestActorCriticAgentExtended:
    """What only the Actor-Critic has: its coefficients, value head and three metrics."""

    @pytest.mark.parametrize("key", ["value_coef", "entropy_coef"])
    def test_invalid_coefficient_negative(self, key):
        with pytest.raises(ValueError, match=key):
            ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, key: -0.5})

    def test_valid_zero_coefs(self):
        """value_coef=0 and entropy_coef=0 are valid boundary values."""
        agent = ActorCriticAgent(
            observation_dim=4,
            action_dim=2,
            config={**_AC_CFG, "value_coef": 0.0, "entropy_coef": 0.0},
        )
        assert agent.value_coef == 0.0
        assert agent.entropy_coef == 0.0

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
        assert weights(agent)["W1"].shape == (6, 32)
        assert weights(agent)["b1"].shape == (32,)
        assert weights(agent)["W_pi"].shape == (32, 3)
        assert weights(agent)["b_pi"].shape == (3,)
        assert weights(agent)["W_v"].shape == (32,)

    def test_bias_initialised_to_zeros(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        np.testing.assert_array_equal(weights(agent)["b1"], np.zeros_like(weights(agent)["b1"]))
        np.testing.assert_array_equal(weights(agent)["b_pi"], np.zeros_like(weights(agent)["b_pi"]))
        assert weights(agent)["b_v"] == 0.0

    # --- Forward pass --------------------------------------------------------

    def test_forward_value_is_the_value_head_of_the_hidden_layer(self):
        """V(s) = h · W_v + b_v, a Python float; b_v is set away from its zero start so a value
        that drops it, or reads the wrong head, fails."""
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        set_weights(agent, b_v=0.7)
        obs = np.array([0.5, -1.0, 0.25, 2.0])
        h = hidden_activations(agent, obs)

        value = state_value(agent, obs)

        assert type(value) is float
        assert h.shape == (64,) and probabilities(agent, obs).shape == (2,)
        np.testing.assert_allclose(value, h @ weights(agent)["W_v"] + 0.7, rtol=1e-12)

    def test_forward_value_is_finite(self):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        rng = np.random.default_rng(3)
        for _ in range(10):
            assert np.isfinite(state_value(agent, rng.standard_normal(4)))

    def test_forward_hidden_relu_non_negative(self):
        """Hidden activations after ReLU must be non-negative."""
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        rng = np.random.default_rng(4)
        for _ in range(10):
            assert np.all(hidden_activations(agent, rng.standard_normal(4)) >= 0)

    # --- Update --------------------------------------------------------------

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

    def test_zero_value_coef_does_not_change_critic_weights(self):
        """When value_coef=0 the critic output gradient is zero, so W_v should not change."""
        agent = ActorCriticAgent(
            observation_dim=4,
            action_dim=2,
            config={**_AC_CFG, "value_coef": 0.0},
        )
        w_v_before = weights(agent)["W_v"]
        agent.update(_make_batch(20))
        np.testing.assert_array_equal(weights(agent)["W_v"], w_v_before)

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

        w_pi_low_before = weights(agent_low)["W_pi"]
        w_pi_high_before = weights(agent_high)["W_pi"]

        m_low = agent_low.update(batch)
        m_high = agent_high.update(batch)

        # Both produce valid, finite metrics
        for m in (m_low, m_high):
            for val in m.values():
                assert np.isfinite(val)

        # Higher entropy_coef applies a stronger entropy gradient to the policy head,
        # so the policy-weight change should be larger in magnitude.
        delta_low = np.linalg.norm(weights(agent_low)["W_pi"] - w_pi_low_before)
        delta_high = np.linalg.norm(weights(agent_high)["W_pi"] - w_pi_high_before)
        assert delta_high > delta_low, (
            "Higher entropy_coef should produce larger policy-weight updates"
        )

    def test_save_load_preserves_state_value(self, tmp_path):
        agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CFG)
        set_weights(agent, b_v=0.3)
        obs = np.array([0.1, -0.2, 0.3, -0.4])
        p = str(tmp_path / "ckpt")
        agent.save(p)
        restored = ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CFG, "seed": 9})
        restored.load(p)
        np.testing.assert_allclose(state_value(restored, obs), state_value(agent, obs), atol=1e-10)
