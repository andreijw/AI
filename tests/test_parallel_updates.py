"""``update_every`` on one clock across several training environments (parallel environments, PR 2).

With ``training.num_envs`` above 1 and ``update_every`` k, the trainer hands the agent, every k vector
steps, every environment's steps of that window in one ``update_segments`` call: a segment per
environment and episode, cut at the window's end or the episode's. The continuous Actor-Critic makes
one Adam step over all of them (A2C with N environments); PPO buffers them and trains only at the end
of a call, so every rollout is data its current weights collected.
"""

import numpy as np
import pytest

from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_core.agents import (
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
)
from rl_core.training import Trainer
from rl_pendulum.environments.pendulum_env import PendulumEnv
from tests.agent_internals import (
    forward_batch,
    gaussian_forward,
    log_prob,
    ppo_rollout_episode,
    set_constant_critic,
    spy_on_adam_step,
    spy_on_chunk_targets,
    spy_on_loss_and_grads,
    spy_on_ppo_minibatches,
    spy_on_ppo_rollout_training,
    weights,
)
from tests.test_parallel_envs import EVAL_SEED, MessageLogger, ScriptedEnv

# ----------------------------------------------------------------------------------------------
# The clock
# ----------------------------------------------------------------------------------------------


class _SegmentRecorder(RandomAgent):
    """A random agent that acts 0 and keeps every ``update_segments`` call."""

    def __init__(self) -> None:
        super().__init__(3, 2, {"seed": 0})
        self.calls: list[list[dict]] = []
        # "update" for each update_segments call, "end" for each end_episode call, in order.
        self.events: list[str] = []

    def select_action(self, observation, training=True):
        return 0

    def update_segments(self, segments):
        self.calls.append(list(segments))
        self.events.append("update")
        return [{} for _ in segments]

    def end_episode(self):
        self.events.append("end")


def _clocked(lengths_per_env, update_every, agent=None, **training):
    envs = [ScriptedEnv(i, lengths) for i, lengths in enumerate(lengths_per_env)]
    config = {"eval_frequency": 1000, "save_frequency": 1000, "num_envs": len(envs),
              "update_every": update_every, **training}  # fmt: skip
    trainer = Trainer(
        env=envs[0],
        agent=agent or _SegmentRecorder(),
        config=config,
        logger=MessageLogger(),
        eval_env=ScriptedEnv(99, [2]),
        eval_seed=EVAL_SEED,
        extra_envs=envs[1:],
    )
    return trainer, envs


def _describe(segment):
    obs = segment["observations"]
    return int(obs[0, 0]), int(obs[0, 1]), int(obs[0, 2]), len(obs)  # env, episode, first step, n


def test_every_window_hands_over_each_environments_steps_in_one_call():
    trainer, envs = _clocked([[4], [7]], update_every=3, num_episodes=3)
    trainer.train()
    calls = [[_describe(s) for s in call] for call in trainer.agent.calls]
    # Window 1 (vector steps 1-3): both open. Window 2 (4-6): env 0 ends at step 4 and restarts
    # at 5, so it gives two segments. Window 3 (7-8): env 1 ends at 7 and stays idle (the third
    # episode is in flight), env 0 ends at 8; training ends, and the window is handed over.
    assert calls == [
        [(0, 0, 0, 3), (1, 0, 0, 3)],
        [(0, 0, 3, 1), (0, 1, 0, 2), (1, 0, 3, 3)],
        [(0, 1, 2, 2), (1, 0, 6, 1)],
    ]
    assert sum(env.steps for env in envs) == sum(n for call in calls for *_, n in call)


def test_end_episode_follows_the_window_that_trained_each_episodes_last_steps():
    trainer, _ = _clocked([[4], [7]], update_every=3, num_episodes=3)
    trainer.train()
    # Env 0's first episode ends in window 2, env 1's and env 0's second in window 3 (the last).
    assert trainer.agent.events == ["update", "update", "end", "update", "end", "end"], (
        "end_episode once per ended episode, after the call carrying that episode's last steps"
    )


def test_a_boundary_cut_bootstraps_from_the_true_next_step_and_an_end_does_not():
    trainer, _ = _clocked([[4], [7]], update_every=3, num_episodes=3)
    trainer.train()
    for call in trainer.agent.calls:
        for segment in call:
            env, episode, first, n = _describe(segment)
            last_next = segment["next_observations"][-1]
            np.testing.assert_array_equal(last_next, [env, episode, first + n])
            ends_episode = first + n == (4 if env == 0 else 7)
            assert bool(segment["terminated"][-1]) == ends_episode, (
                "only the step that ends an episode is a termination; a window cut bootstraps"
            )


def test_update_every_with_several_environments_needs_an_agent_that_takes_it():
    with pytest.raises(ValueError, match="update_every"):
        envs = [CartPoleEnv(seed=1), CartPoleEnv(seed=2)]
        Trainer(env=envs[0], agent=ReinforceAgent(4, 2, {"seed": 1}),
                config={"num_envs": 2, "update_every": 4}, eval_env=CartPoleEnv(seed=3),
                eval_seed=EVAL_SEED, extra_envs=envs[1:])  # fmt: skip


# ----------------------------------------------------------------------------------------------
# The continuous Actor-Critic: one Adam step over every environment's window
# ----------------------------------------------------------------------------------------------

AC = {"seed": 3, "hidden_dim": 6, "n_steps": 3, "action_low": -2.0, "action_high": 2.0,
      "learning_rate": 0.01, "normalize_advantages": False}  # fmt: skip


def _segment(n, terminated_last, seed, first_reward=None):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n + 1, 3))
    terminated = np.zeros(n, dtype=bool)
    terminated[-1] = terminated_last
    rewards = rng.normal(size=n) if first_reward is None else np.asarray(first_reward, float)
    return {
        "observations": obs[:-1],
        "actions": rng.normal(size=(n, 1)),
        "rewards": rewards,
        "next_observations": obs[1:],
        "dones": terminated.copy(),
        "terminated": terminated,
    }


@pytest.mark.parametrize("normalize_advantages", [False, True])
@pytest.mark.parametrize("bounded_mean", [False, True])
def test_one_segment_trains_exactly_as_update_does(normalize_advantages, bounded_mean):
    config = {**AC, "normalize_advantages": normalize_advantages, "bounded_mean": bounded_mean}
    one, other = (
        ContinuousActorCriticAgent(3, 1, dict(config)),
        ContinuousActorCriticAgent(3, 1, dict(config)),
    )
    segment = _segment(3, False, 5)
    assert one.update_segments([segment]) == [other.update(segment)]
    for name, value in weights(one).items():
        np.testing.assert_array_equal(value, weights(other)[name], err_msg=name)


def test_every_target_is_computed_before_the_single_adam_step(monkeypatch):
    agent = ContinuousActorCriticAgent(3, 1, dict(AC))
    events: list[str] = []
    spy_on_chunk_targets(monkeypatch, agent, lambda f, *a: (events.append("targets"), f(*a))[1])
    spy_on_adam_step(monkeypatch, agent, lambda f, *a: (events.append("adam"), f(*a))[1])
    agent.update_segments([_segment(3, False, 1), _segment(2, True, 2), _segment(3, False, 3)])
    assert events == ["targets", "targets", "targets", "adam"], (
        "targets from one set of weights, then one optimizer step for the whole window"
    )


def test_each_segment_bootstraps_by_its_own_end(monkeypatch):
    agent = ContinuousActorCriticAgent(3, 1, {**AC, "gamma": 0.5, "gae_lambda": 1.0})
    set_constant_critic(agent, 1.0)
    seen = {}
    spy_on_loss_and_grads(
        monkeypatch,
        agent,
        lambda f, s, a, r, adv=None: (seen.update(adv=adv, ret=r), f(s, a, r, adv))[1],
    )
    agent.update_segments([
        _segment(2, True, 1, first_reward=[1.0, 2.0]),   # terminated: delta_1 = 2 - 1
        _segment(2, False, 2, first_reward=[1.0, 1.0]),  # cut: delta_1 = 1 + 0.5 - 1
        _segment(1, False, 3, first_reward=[3.0]),       # window cut: 3 + 0.5 - 1
    ])  # fmt: skip
    # V = 1 everywhere, gamma 0.5, lambda 1: A = delta_t + 0.5 * A_{t+1} within each segment.
    # Terminated [1, 2]: A_1 = 1, A_0 = (1 + 0.5 - 1) + 0.5 = 1. Cut [1, 1]: A_1 = 0.5,
    # A_0 = 0.5 + 0.25 = 0.75. Window cut [3]: 2.5.
    np.testing.assert_allclose(seen["adv"], [1.0, 1.0, 0.75, 0.5, 2.5], atol=1e-12)
    np.testing.assert_allclose(seen["ret"], np.asarray(seen["adv"]) + 1.0, atol=1e-12)


def test_advantages_are_normalised_over_the_whole_window(monkeypatch):
    agent = ContinuousActorCriticAgent(3, 1, {**AC, "normalize_advantages": True})
    seen = {}
    spy_on_loss_and_grads(
        monkeypatch, agent, lambda f, s, a, r, adv=None: (seen.update(adv=adv), f(s, a, r, adv))[1]
    )
    agent.update_segments([_segment(3, False, 1), _segment(2, True, 2)])
    assert len(seen["adv"]) == 5
    assert abs(np.mean(seen["adv"])) < 1e-12 and abs(np.std(seen["adv"]) - 1.0) < 1e-6


def test_the_anneal_counts_every_transition_of_the_window():
    agent = ContinuousActorCriticAgent(3, 1, {**AC, "anneal_steps": 10})
    first = agent.update_segments([_segment(3, False, 1), _segment(2, True, 2)])
    second = agent.update_segments([_segment(1, False, 3)])
    assert [m["anneal_fraction"] for m in first + second] == [1.0, 0.5]


# ----------------------------------------------------------------------------------------------
# PPO: only with the window layout, trained at the end of a call
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "obs", "act", "extra"),
    [(PPOAgent, 4, 2, {"minibatch_size": 8}), (ContinuousPPOAgent, 3, 1, {"batch_size": 8})],
    ids=["PPOAgent", "ContinuousPPOAgent"],
)
def test_ppo_takes_several_environments_only_with_whole_windows_per_rollout(cls, obs, act, extra):
    def agent(rollout_steps=32):
        return cls(obs, act, {"seed": 1, "rollout_steps": rollout_steps, **extra})

    with pytest.raises(ValueError, match="update_every"):
        agent().enable_parallel_envs(4, None)
    with pytest.raises(ValueError, match="rollout_steps"):
        agent(rollout_steps=30).enable_parallel_envs(4, 8)
    with pytest.raises(ValueError, match="num_envs"):
        agent().enable_mid_episode_updates(8)  # one environment: as before, whole episodes
    accepting = agent(rollout_steps=64)
    accepting.enable_parallel_envs(4, 8)
    accepting.enable_mid_episode_updates(8)


def _discrete_segment(n, terminated_last, seed):
    segment = _segment(n, terminated_last, seed)
    rng = np.random.default_rng(seed + 100)
    segment["observations"] = rng.normal(size=(n, 4))
    segment["next_observations"] = rng.normal(size=(n, 4))
    segment["actions"] = rng.integers(0, 2, size=n)
    return segment


def test_ppo_trains_once_at_the_end_of_a_call_on_every_segment(monkeypatch):
    agent = PPOAgent(4, 2, {"seed": 1, "rollout_steps": 8, "minibatch_size": 4, "ppo_epochs": 1})
    trained = []
    spy_on_ppo_rollout_training(
        monkeypatch, agent, lambda f, o, a, r, d: (trained.append(len(r)), f(o, a, r, d))[1]
    )
    metrics = agent.update_segments([_discrete_segment(8, True, 1), _discrete_segment(4, False, 2)])
    assert trained == [12], "one rollout of both segments, trained after the last was added"
    assert [m.get("rollout_steps") for m in metrics if m] == [12.0]


def test_ppo_bootstraps_every_segment_from_the_value_before_the_update(monkeypatch):
    agent = PPOAgent(4, 2, {"seed": 1, "rollout_steps": 8, "minibatch_size": 4, "gamma": 0.9})
    twin = PPOAgent(4, 2, {"seed": 1, "rollout_steps": 8, "minibatch_size": 4, "gamma": 0.9})
    first, second = _discrete_segment(8, True, 1), _discrete_segment(3, False, 2)
    expected = ppo_rollout_episode(twin, second)
    assert expected is not None
    seen = {}
    spy_on_ppo_rollout_training(
        monkeypatch, agent, lambda f, o, a, r, d: (seen.update(rewards=r.copy()), f(o, a, r, d))[1]
    )
    agent.update_segments([first, second])
    np.testing.assert_array_equal(seen["rewards"][-3:], expected.rewards)


# ----------------------------------------------------------------------------------------------
# PPO through the trainer: the ratio is 1 on every update (as with one environment, #201)
# ----------------------------------------------------------------------------------------------


def _policy_log_probs(agent, observations, actions):
    if isinstance(agent, ContinuousPPOAgent):
        mu, log_std, _, _ = gaussian_forward(agent, observations)
        return log_prob(agent, actions, mu, log_std)
    probs, _, _ = forward_batch(agent, observations)
    return np.log(np.clip(probs[np.arange(len(actions)), actions], 1e-8, 1.0))


RATIO_CASES = [
    pytest.param(
        PPOAgent, {"minibatch_size": 16}, lambda s: CartPoleEnv(seed=s), (4, 2), id="PPOAgent"
    ),
    pytest.param(
        ContinuousPPOAgent,
        {"batch_size": 16, "action_low": -2.0, "action_high": 2.0},
        lambda s: PendulumEnv(seed=s, max_episode_steps=20),
        (3, 1),
        id="ContinuousPPOAgent",
    ),
]


@pytest.mark.parametrize(("cls", "config", "make_env", "dims"), RATIO_CASES)
def test_every_ppo_update_trains_on_data_its_own_weights_collected(
    cls, config, make_env, dims, monkeypatch
):
    num_envs, update_every = 4, 8
    rollout = num_envs * update_every
    agent = cls(*dims, {"seed": 3, "hidden_dim": 8, "ppo_epochs": 2, "rollout_steps": rollout,
                        "normalize_observations": True, **config})  # fmt: skip
    collected = {}
    select_action = agent.select_action

    def acting(observation, training=True):
        action = select_action(observation, training=training)
        if training:
            row, acts = np.asarray(observation, float)[None], np.asarray(action)[None]
            collected[row.tobytes() + acts.tobytes()] = float(
                _policy_log_probs(agent, row, acts)[0]
            )
        return action

    monkeypatch.setattr(agent, "select_action", acting)
    state = {"pending": False, "checked": 0, "sizes": []}
    update_segments = agent.update_segments

    def updating(segments):
        state["pending"] = True
        metrics = update_segments(segments)
        state["sizes"] += [m["rollout_steps"] for m in metrics if m]
        return metrics

    monkeypatch.setattr(agent, "update_segments", updating)

    def minibatch(original, observations, actions, returns, advantages, old_log_probs, *a, **kw):
        for row, act, old in zip(observations, actions, old_log_probs, strict=True):
            key = row[None].tobytes() + np.asarray(act)[None].tobytes()
            assert abs(old - collected[key]) <= 1e-12, "old log-probability is not collection's"
        if state.pop("pending", False):
            ratio = np.exp(_policy_log_probs(agent, observations, actions) - old_log_probs)
            assert np.max(np.abs(ratio - 1.0)) <= 1e-12, "the first mini-batch's ratio must be 1"
            state["checked"] += 1
        return original(observations, actions, returns, advantages, old_log_probs, *a, **kw)

    spy_on_ppo_minibatches(monkeypatch, agent, minibatch)
    envs = [make_env(10 + i) for i in range(num_envs)]
    eval_env = make_env(99)
    try:
        Trainer(
            env=envs[0], agent=agent, logger=MessageLogger(), eval_env=eval_env, eval_seed=EVAL_SEED,
            config={"num_envs": num_envs, "update_every": update_every, "total_steps": 400,
                    "max_steps_per_episode": 20, "eval_frequency": 1000, "save_frequency": 1000},
            extra_envs=envs[1:],
        ).train()  # fmt: skip
    finally:
        for env in (*envs, eval_env):
            env.close()
    assert state["checked"] >= 3, "several rollouts should be trained on"
    assert all(size == rollout for size in state["sizes"][:-1]), (
        f"every rollout but the last should be exactly {num_envs} x {update_every} transitions: "
        f"{state['sizes']}"
    )
