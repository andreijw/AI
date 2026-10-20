"""Options that let the continuous Actor-Critic learn Pendulum-v1 (roadmap: "Continuous Actor-Critic Learns
Pendulum-v1").

* ``bounded_mean`` (both continuous agents): mu = bias + scale * tanh(z), so the policy mean stays inside the
  action bounds; without it the mean drifts past the bounds, where every executed torque is the same.
* ``n_steps`` (Actor-Critic): the episode is replayed in consecutive chunks of n steps, each with its own
  GAE targets from the current critic, bootstrapped from the chunk's next state unless the episode truly
  terminated, and one optimizer step per chunk.
* ``anneal_steps`` (Actor-Critic): the learning rate falls linearly to zero, its counter checkpointed.
"""

import numpy as np
import pytest

from rl_core.agents import ContinuousActorCriticAgent, ContinuousPPOAgent
from rl_core.agents.returns import gae

GAMMA, LAMBDA = 0.9, 0.9
BOUNDS = {"action_low": -2.0, "action_high": 3.0}  # scale 2.5, bias 0.5


def _ac(**config) -> ContinuousActorCriticAgent:
    settings = {
        "seed": 4,
        "hidden_dim": 8,
        "gamma": GAMMA,
        "gae_lambda": LAMBDA,
        "reward_scale": 0.05,
        **BOUNDS,
    }
    return ContinuousActorCriticAgent(3, 1, config={**settings, **config})


def _episode(length: int = 20, terminated: bool = False, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((length, 3)),
        "actions": rng.uniform(-2.0, 3.0, (length, 1)),
        "rewards": -rng.uniform(0.0, 16.0, length),
        "next_observations": rng.standard_normal((length, 3)),
        "dones": np.array([False] * (length - 1) + [True]),
        "terminated": np.array([False] * (length - 1) + [terminated]),
    }


def _constant_critic(agent, value: float) -> None:
    agent._W_v[...] = 0.0
    agent._b_v = value


# ---------------------------------------------------------------------------
# bounded_mean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_cls", [ContinuousActorCriticAgent, ContinuousPPOAgent])
def test_a_bounded_mean_stays_inside_the_bounds_whatever_the_weights(agent_cls):
    agent = agent_cls(3, 1, config={"seed": 1, "bounded_mean": True, **BOUNDS})
    agent._b_mu[...] = 50.0
    mu, _, _, _ = agent._forward(np.zeros((4, 3)))
    np.testing.assert_allclose(mu, 3.0)
    agent._b_mu[...] = -50.0
    np.testing.assert_allclose(agent._forward(np.zeros(3))[0], [-2.0])


def test_a_bounded_mean_is_tanh_scaled_to_the_bounds():
    agent = _ac(bounded_mean=True)
    agent._W_mu[...] = 0.0
    agent._b_mu[...] = np.arctanh(0.4)
    np.testing.assert_allclose(agent._forward(np.zeros(3))[0], [0.5 + 2.5 * 0.4])


@pytest.mark.parametrize("agent_cls", [ContinuousActorCriticAgent, ContinuousPPOAgent])
def test_without_bounded_mean_the_mean_is_the_linear_head_as_before(agent_cls):
    agent = agent_cls(3, 1, config={"seed": 1, **BOUNDS})
    agent._b_mu[...] = 50.0
    np.testing.assert_allclose(agent._forward(np.zeros(3))[0], [50.0], atol=1.0)


@pytest.mark.parametrize("bad", [{"action_high": np.inf}, {"action_low": 3.0}])
def test_a_bounded_mean_needs_finite_bounds_with_room(bad):
    with pytest.raises(ValueError, match="action_low|action_high"):
        _ac(bounded_mean=True, **bad)


def test_a_checkpoint_keeps_its_bounded_mean_setting(tmp_path):
    path = str(tmp_path / "ac.pt")
    _ac(bounded_mean=True).save(path)
    with pytest.raises(ValueError, match="bounded_mean"):
        _ac().load(path)
    _ac(bounded_mean=True).load(path)  # same setting loads


def test_an_old_checkpoint_counts_as_unbounded(tmp_path):
    path = str(tmp_path / "ac.pt")
    agent = _ac()
    agent.save(path)
    with np.load(path) as data:
        old = {k: data[k] for k in data.files if k != "bounded_mean"}
    with open(path, "wb") as f:
        np.savez(f, **old)
    _ac().load(path)
    with pytest.raises(ValueError, match="bounded_mean"):
        _ac(bounded_mean=True).load(path)


def test_update_reports_how_often_a_bounded_mean_saturates():
    agent = _ac(bounded_mean=True)
    agent._b_mu[...] = 50.0
    assert agent.update(_episode())["saturation"] == pytest.approx(1.0)
    assert "saturation" not in _ac().update(_episode())


# ---------------------------------------------------------------------------
# n_steps: chunked updates with bootstrapped GAE
# ---------------------------------------------------------------------------


def test_chunk_targets_match_a_hand_computation():
    agent = _ac()
    _constant_critic(agent, 2.0)
    rewards = np.array([-1.0, -3.0])
    advantages, returns = agent._chunk_targets(
        np.zeros((2, 3)), rewards, np.zeros(3), terminal=False
    )
    # V = 2 everywhere; bootstrap V(s_next) = 2: d1 = -3 + 0.9 * 2 - 2 = -3.2, d0 = -1 + 0.9 * 2 - 2 = -1.2,
    # A1 = d1, A0 = d0 + 0.9 * 0.9 * d1
    np.testing.assert_allclose(advantages, [-1.2 + GAMMA * LAMBDA * -3.2, -3.2])
    np.testing.assert_allclose(returns, advantages + 2.0)


def test_a_terminal_chunk_does_not_bootstrap():
    agent = _ac()
    _constant_critic(agent, 2.0)
    advantages, _ = agent._chunk_targets(
        np.zeros((1, 3)), np.array([-3.0]), np.zeros(3), terminal=True
    )
    np.testing.assert_allclose(advantages, [-3.0 - 2.0])


def _chunk_calls(agent, episode, monkeypatch):
    calls = []
    original = agent._chunk_targets

    def record(states, rewards, bootstrap_obs, terminal):
        calls.append(
            (len(rewards), terminal, None if bootstrap_obs is None else bootstrap_obs.copy())
        )
        return original(states, rewards, bootstrap_obs, terminal)

    monkeypatch.setattr(agent, "_chunk_targets", record)
    agent.update(episode)
    return calls


def test_an_episode_is_trained_in_chunks_of_n_steps_with_a_short_last_one(monkeypatch):
    agent = _ac(n_steps=8)
    episode = _episode(length=20)
    calls = _chunk_calls(agent, episode, monkeypatch)
    assert [length for length, _, _ in calls] == [8, 8, 4]
    assert agent._optimizer.t == 3, "one optimizer step per chunk"
    # each chunk bootstraps from the state after its last step
    np.testing.assert_array_equal(calls[0][2], episode["next_observations"][7])
    np.testing.assert_array_equal(calls[2][2], episode["next_observations"][19])


@pytest.mark.parametrize(("terminated", "expected"), [(False, False), (True, True)])
def test_only_a_true_termination_ends_the_last_chunks_bootstrap(monkeypatch, terminated, expected):
    calls = _chunk_calls(_ac(n_steps=8), _episode(length=20, terminated=terminated), monkeypatch)
    assert [terminal for _, terminal, _ in calls] == [False, False, expected]


def test_chunk_rewards_are_scaled():
    agent = _ac(n_steps=4, reward_scale=0.05)
    seen = []
    original = agent._chunk_targets
    agent._chunk_targets = lambda s, r, b, terminal: (
        seen.append(r.copy()) or original(s, r, b, terminal)
    )
    episode = _episode(length=4)
    agent.update(episode)
    np.testing.assert_allclose(seen[0], episode["rewards"] * 0.05)


def test_advantages_are_not_normalised_when_asked(monkeypatch):
    agent = _ac(n_steps=4, normalize_advantages=False)
    _constant_critic(agent, 2.0)
    received = []
    original = agent._loss_and_grads

    def record(states, actions, returns, advantages=None):
        received.append(advantages.copy())
        return original(states, actions, returns, advantages)

    monkeypatch.setattr(agent, "_loss_and_grads", record)
    episode = _episode(length=4)
    agent.update(episode)
    values = np.full(4, 2.0)
    targets = gae(
        episode["rewards"] * 0.05, values, np.zeros(4, bool), GAMMA, LAMBDA, last_value=2.0
    )
    np.testing.assert_allclose(received[0], targets)


@pytest.mark.parametrize(
    "bad", [{"n_steps": -1}, {"n_steps": 2.5}, {"gae_lambda": 1.5}, {"gae_lambda": -0.1}]
)
def test_invalid_chunk_settings_are_rejected(bad):
    with pytest.raises(ValueError, match="n_steps|gae_lambda"):
        _ac(**bad)


# ---------------------------------------------------------------------------
# anneal_steps and determinism
# ---------------------------------------------------------------------------


def test_the_learning_rate_anneals_per_trained_transition(monkeypatch):
    agent = _ac(n_steps=8, anneal_steps=40, learning_rate=0.01)
    rates = []
    original = agent._adam_step
    monkeypatch.setattr(
        agent, "_adam_step", lambda grads, lr: rates.append(lr) or original(grads, lr)
    )
    agent.update(_episode(length=20))
    # chunks start after 0, 8 and 16 trained transitions of 40
    np.testing.assert_allclose(rates, [0.01, 0.01 * 32 / 40, 0.01 * 24 / 40])


def test_annealing_progress_survives_a_checkpoint(tmp_path):
    agent = _ac(n_steps=8, anneal_steps=100)
    agent.update(_episode(length=20))
    path = str(tmp_path / "ac.pt")
    agent.save(path)
    resumed = _ac(n_steps=8, anneal_steps=100)
    resumed.load(path)
    assert resumed._steps_trained == 20


def test_a_resumed_agent_continues_exactly_like_one_that_never_stopped(tmp_path):
    """Weights, Adam moments and the annealing counter all come back from the checkpoint."""
    settings = {"n_steps": 8, "bounded_mean": True, "anneal_steps": 200, "learning_rate": 0.01}
    agent = _ac(**settings)
    agent.update(_episode(seed=1))
    path = str(tmp_path / "ac.pt")
    agent.save(path)
    resumed = _ac(**settings, seed=99)  # other initial weights: everything must come from the file
    resumed.load(path)
    for run in (agent, resumed):
        run.update(_episode(seed=2))
    for name in ("_W1", "_b1", "_W_mu", "_b_mu", "_log_std", "_W_v"):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(agent, name))
    assert resumed._b_v == agent._b_v


def test_the_annealed_fraction_is_reported():
    agent = _ac(n_steps=8, anneal_steps=40)
    assert agent.update(_episode(length=20))["anneal_fraction"] == 1.0
    assert agent.update(_episode(length=20))["anneal_fraction"] == pytest.approx(0.5)
    assert "anneal_fraction" not in _ac(n_steps=8).update(_episode())


def _as_list(episode, keep):
    return [
        {
            "state": episode["observations"][i], "action": episode["actions"][i],
            "reward": episode["rewards"][i], "next_state": episode["next_observations"][i],
            **{key: bool(episode[source][i]) for key, source in keep},
        }
        for i in range(len(episode["rewards"]))
    ]  # fmt: skip


@pytest.mark.parametrize("form", ["dict", "list"])
@pytest.mark.parametrize(
    ("flags", "expected_last"),
    [
        ("terminated_and_done", False),  # a cut: done but not terminated -> bootstrap
        ("done_only", True),  # no terminated flag: done counts as the end, as before
    ],
)
def test_terminated_wins_over_done_and_done_alone_ends_the_episode(
    monkeypatch, form, flags, expected_last
):
    episode = _episode(length=20)
    keep = (
        [("done", "dones"), ("terminated", "terminated")]
        if flags == "terminated_and_done"
        else [("done", "dones")]
    )
    if form == "dict":
        trajectory = dict(episode)
        if flags == "done_only":
            del trajectory["terminated"]
    else:
        trajectory = _as_list(episode, keep)
    agent = _ac(n_steps=8)
    calls = []
    original = agent._chunk_targets
    monkeypatch.setattr(
        agent, "_chunk_targets",
        lambda s, r, b, terminal: calls.append(terminal) or original(s, r, b, terminal),
    )  # fmt: skip
    agent.update(trajectory)
    assert calls == [False, False, expected_last]


def test_the_default_update_reproduces_the_previous_weights():
    """bounded_mean off and n_steps 0 train as before the options existed (bit for bit when they were added,
    2026-10-18). The values were re-recorded when agents moved to named random streams (roadmap: "Separate
    Random Streams for Agents and Buffers"), which legitimately changed every seeded weight. Other CPUs' BLAS
    can round the last bit differently (CI on x86 differs by 2e-16 relative), so the comparison allows 1e-12,
    far below what any change to the update would move them."""
    from rl_core.agents import ContinuousPPOAgent

    rng = np.random.default_rng(0)
    episode = {
        "observations": rng.standard_normal((6, 3)), "actions": rng.uniform(-2, 2, (6, 1)),
        "rewards": -rng.uniform(0, 16, 6), "next_observations": rng.standard_normal((6, 3)),
        "dones": np.array([0, 0, 0, 0, 0, 1], bool), "terminated": np.zeros(6, bool),
    }  # fmt: skip
    expected = {
        ContinuousActorCriticAgent: (
            [
                0.02018477752485912,
                0.005797121972952909,
                -0.0011780786197601398,
                0.029668696793881844,
            ],
            -0.009999999667289254,
        ),
        ContinuousPPOAgent: (
            [0.035114459622755764, 0.033382242554407346, 0.015701151162615997, 0.0447953456592937],
            -0.039761046476797166,
        ),
    }
    for cls, (w_mu, b_v) in expected.items():
        agent = cls(
            3,
            1,
            {"seed": 3, "hidden_dim": 4, "learning_rate": 0.01, "batch_size": 3, "ppo_epochs": 2},
        )
        agent.update(episode)
        np.testing.assert_allclose(agent._W_mu.ravel(), w_mu, rtol=1e-12, atol=0)
        assert agent._b_v == pytest.approx(b_v, rel=1e-12, abs=0)


def test_without_next_observations_chunks_bootstrap_from_the_next_state_and_the_last_is_terminal(
    monkeypatch,
):
    episode = _episode(length=20)
    del episode["next_observations"]
    calls = _chunk_calls(_ac(n_steps=8), episode, monkeypatch)
    np.testing.assert_array_equal(calls[0][2], episode["observations"][8])
    assert [terminal for _, terminal, _ in calls] == [False, False, True]


def test_a_monte_carlo_update_can_skip_advantage_normalisation(monkeypatch):
    agent = _ac(normalize_advantages=False)
    _constant_critic(agent, 2.0)
    received = []
    original = agent._loss_and_grads

    def record(states, actions, returns, advantages=None):
        received.append((returns.copy(), advantages.copy()))
        return original(states, actions, returns, advantages)

    monkeypatch.setattr(agent, "_loss_and_grads", record)
    agent.update(_episode(length=5))
    returns, advantages = received[0]
    np.testing.assert_allclose(advantages, returns - 2.0)


@pytest.mark.parametrize("trajectory", [[], {"observations": []}, "not a batch"])
def test_an_empty_or_unrecognised_episode_trains_nothing(trajectory):
    agent = _ac(n_steps=8)
    assert agent.update(trajectory) == {}
    assert agent._optimizer.t == 0


@pytest.mark.parametrize("bad", [0, 2.5])
def test_anneal_steps_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="anneal_steps must be a positive integer"):
        _ac(anneal_steps=bad)
