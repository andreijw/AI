"""Off-policy updates during the episode: ``training.update_every``.

With it set, the trainer hands the agent the transitions gathered so far every ``update_every``
steps, and the rest with the step that ends the episode; ``end_episode`` still comes once, after
the last chunk. Unset, the agent gets the whole episode in one call, as before. Only agents that
can learn from part of an episode accept it. DQN, DDPG and TD3 then count gradient steps over all
the transitions they have trained on, so ``updates_per_step: 0.5`` means one step every two
transitions however the episode is cut up. The continuous Actor-Critic with ``n_steps`` equal to it
becomes true A2C: each chunk is trained on as it is collected, so the next one is sampled by the
updated policy.
"""

import numpy as np
import pytest

from rl_core.agents import (
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DDPGAgent,
    DQNAgent,
    RandomAgent,
    ReinforceAgent,
    TD3Agent,
)
from rl_core.agents.base_agent import BaseAgent
from rl_core.training import Trainer
from tests.agent_internals import counters, spy_on_chunk_targets

OBS_DIM, ACT_DIM = 3, 1
BOUNDS = {"action_low": [-2.0], "action_high": [2.0]}


# ---------------------------------------------------------------------------
# The trainer: how an episode is cut up
# ---------------------------------------------------------------------------


class _Env:
    """Episodes of ``length`` steps that terminate, or are truncated at step ``truncate_at``; step i
    (from 1) pays reward i and observes i in every coordinate."""

    def __init__(self, length, truncate_at=None):
        self.length = length
        self.truncate_at = truncate_at
        self.actions: list = []  # every action stepped, in order
        self._t = 0

    def reset(self, **_kwargs):
        self._t = 0
        return np.zeros(OBS_DIM), {}

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        self._t += 1
        truncated = self._t == self.truncate_at
        return np.full(OBS_DIM, self._t), float(self._t), self._t == self.length, truncated, {}

    def close(self):
        pass


class _ChunkRecorder(BaseAgent):
    """Records what the trainer passes to update() and when it ends an episode."""

    def __init__(self):
        super().__init__(observation_dim=OBS_DIM, action_dim=2, config={})
        self.events: list = []
        self.enabled = False

    def enable_mid_episode_updates(self, update_every):
        self.enabled = update_every

    def select_action(self, observation, training=True):
        return 0

    def update(self, batch):
        self.events.append(
            (list(batch["rewards"]), list(batch["terminated"]), list(batch["dones"]))
        )
        return {}

    def end_episode(self):
        self.events.append("end")

    def _checkpoint_arrays(self):
        return {}

    def _load_checkpoint_arrays(self, data):
        pass


def _train(tmp_path, agent, length=7, episodes=1, truncate_at=None, **training):
    config = {
        "num_episodes": episodes,
        "max_steps_per_episode": 50,
        "eval_frequency": episodes + 1,
        "save_frequency": episodes + 1,
        "checkpoint_dir": str(tmp_path / "ckpt"),
        **training,
    }
    trainer = Trainer(env=_Env(length, truncate_at), agent=agent, config=config)
    trainer.train()
    return trainer


def _sizes(events):
    return [len(e[0]) if e != "end" else "end" for e in events]


def test_without_update_every_the_agent_gets_each_whole_episode_once(tmp_path):
    agent = _ChunkRecorder()
    _train(tmp_path, agent, episodes=2)
    assert not agent.enabled, "Without update_every the agent must not be switched to chunks."
    assert _sizes(agent.events) == [7, "end", 7, "end"], "One update per episode, then its end."


def test_update_every_cuts_the_episode_into_chunks_in_order(tmp_path):
    agent = _ChunkRecorder()
    _train(tmp_path, agent, update_every=3)
    assert agent.enabled == 3, "The agent must be told the chunk size."
    assert _sizes(agent.events) == [3, 3, 1, "end"], "7 steps by 3: chunks of 3, 3, 1, then end."
    rewards = [r for e in agent.events[:-1] for r in e[0]]
    assert rewards == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "Every step once, in order."
    terminated = [t for e in agent.events[:-1] for t in e[1]]
    dones = [d for e in agent.events[:-1] for d in e[2]]
    assert terminated == dones == [False] * 6 + [True], "Only the real last step ends the episode."


def test_a_chunk_boundary_on_the_last_step_sends_no_empty_update(tmp_path):
    agent = _ChunkRecorder()
    _train(tmp_path, agent, length=6, update_every=3)
    assert _sizes(agent.events) == [3, 3, "end"]


@pytest.mark.parametrize(
    ("limit", "expected"),
    [(10, [4, 4, 2, "end"]), (8, [4, 4, "end"])],  # 8: the cut falls on a chunk boundary
)
def test_the_step_limit_ends_the_last_chunk(tmp_path, limit, expected):
    agent = _ChunkRecorder()
    _train(tmp_path, agent, length=20, update_every=4, max_steps_per_episode=limit)
    assert _sizes(agent.events) == expected, f"A {limit}-step cut by 4: {expected}."
    last_chunk = agent.events[-2]
    assert not any(last_chunk[1]), "A cut at the step limit is not a termination."


def test_update_every_one_updates_after_every_step(tmp_path):
    agent = _ChunkRecorder()
    _train(tmp_path, agent, length=3, update_every=1)
    assert _sizes(agent.events) == [1, 1, 1, "end"]


def test_update_every_longer_than_the_episode_is_one_update(tmp_path):
    agent = _ChunkRecorder()
    _train(tmp_path, agent, update_every=100)
    assert _sizes(agent.events) == [7, "end"]


def test_evaluation_episodes_are_not_trained_on(tmp_path):
    agent = _ChunkRecorder()
    trainer = Trainer(
        env=_Env(5),
        agent=agent,
        config={"update_every": 2, "checkpoint_dir": str(tmp_path / "ckpt")},
    )
    trainer.evaluate(num_episodes=2)
    assert agent.events == [], "Evaluation must not call update() or end_episode()."


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "3"])
def test_update_every_must_be_a_positive_integer(tmp_path, bad):
    with pytest.raises(ValueError, match="update_every must be a positive integer"):
        Trainer(env=_Env(3), agent=_ChunkRecorder(), config={"update_every": bad})


def test_an_agent_that_trains_on_whole_episodes_rejects_update_every():
    agent = ReinforceAgent(OBS_DIM, 2, config={"seed": 1})
    with pytest.raises(ValueError, match="trains on whole episodes"):
        Trainer(env=_Env(3), agent=agent, config={"update_every": 1})


@pytest.mark.parametrize(
    "make",
    [
        lambda: RandomAgent(OBS_DIM, 2, config={"seed": 1}),
        lambda: ContinuousRandomAgent(OBS_DIM, ACT_DIM, config={"seed": 1, **BOUNDS}),
    ],
    ids=["discrete", "continuous"],
)
def test_the_random_agents_accept_update_every(tmp_path, make):
    """They learn nothing, so a benchmark floor can share a config that sets update_every."""
    trainer = _train(tmp_path, make(), update_every=2)
    assert trainer.episode_rewards == [28.0]


def test_a_real_agent_trains_once_per_transition_through_the_trainer(tmp_path):
    """DDPG at one step per transition, updated after every step: two 7-step episodes give 14
    transitions; the first 3 calls find fewer than a batch (4) in the buffer and train nothing."""
    agent = DDPGAgent(OBS_DIM, ACT_DIM, config=SMALL_DDPG)
    _train(tmp_path, agent, episodes=2, update_every=1)
    assert counters(agent)["gradient_steps"] == 11, "14 transitions - 3 before a full batch."


# ---------------------------------------------------------------------------
# The agents: gradient steps counted over every transition trained on
# ---------------------------------------------------------------------------


def _transitions(n, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OBS_DIM)),
        "actions": rng.uniform(-2.0, 2.0, (n, ACT_DIM)),
        "rewards": rng.standard_normal(n),
        "next_observations": rng.standard_normal((n, OBS_DIM)),
        "dones": np.zeros(n, dtype=bool),
        "terminated": np.zeros(n, dtype=bool),
    }


def _one_at_a_time(agent, n, seed=0):
    """``n`` transitions, one per update() call, as update_every: 1 hands them over."""
    batch = _transitions(n, seed)
    if isinstance(agent, DQNAgent):
        batch["actions"] = np.zeros(n, dtype=int)
    for i in range(n):
        agent.update({key: value[i : i + 1] for key, value in batch.items()})


SMALL_DDPG = {"seed": 7, "hidden_sizes": [6, 5], "batch_size": 4, "learning_starts": 0, **BOUNDS}
SMALL_DQN = {
    "seed": 5,
    "hidden_dim": 8,
    "batch_size": 4,
    "min_buffer_size": 4,
    "exploration_steps": 100,
    "updates_per_step": 1.0,
}


def _ddpg(cls=DDPGAgent, **config):
    agent = cls(OBS_DIM, ACT_DIM, config={**SMALL_DDPG, **config})
    agent.update(_transitions(4, seed=9))  # a batch's worth in the buffer
    return agent


def _dqn(**config):
    agent = DQNAgent(OBS_DIM, 2, config={**SMALL_DQN, **config})
    agent.update(_transitions(4, seed=9) | {"actions": np.zeros(4, dtype=int)})  # fill to 4
    return agent


def _steps(agent):
    c = counters(agent)
    return c["gradient_steps"] if "gradient_steps" in c else c["update_counter"]


@pytest.mark.parametrize("cls", [DDPGAgent, TD3Agent])
def test_ddpg_half_an_update_per_transition_holds_across_one_transition_chunks(cls):
    agent = _ddpg(cls, updates_per_step=0.5)
    agent.enable_mid_episode_updates(1)
    start = _steps(agent)
    _one_at_a_time(agent, 6)
    assert _steps(agent) - start == 3, "6 transitions at 0.5 must give 3 gradient steps, not 6."


@pytest.mark.parametrize("cls", [DDPGAgent, TD3Agent])
def test_ddpg_without_mid_episode_updates_rounds_each_call_up_as_before(cls):
    agent = _ddpg(cls, updates_per_step=0.5)
    start = _steps(agent)
    _one_at_a_time(agent, 6)
    assert _steps(agent) - start == 6, "Per-call rounding (max 1 step a call) must be unchanged."


def test_ddpg_one_update_per_transition_is_one_step_per_one_transition_chunk():
    agent = _ddpg(updates_per_step=1.0)
    agent.enable_mid_episode_updates(1)
    start = _steps(agent)
    _one_at_a_time(agent, 5)
    assert _steps(agent) - start == 5


def test_ddpg_warm_up_transitions_earn_no_gradient_steps():
    agent = _ddpg(updates_per_step=0.5, learning_starts=2)  # its 4 transitions: warm-up
    agent.enable_mid_episode_updates(1)
    _one_at_a_time(agent, 3)  # 0 of 2 warm-up actions taken: none of these count
    for _ in range(2):
        agent.select_action(np.zeros(OBS_DIM), training=True)
    _one_at_a_time(agent, 2)
    assert _steps(agent) == 1, "Only the 2 transitions after warm-up count: 2 * 0.5 = 1 step."


def test_dqn_half_an_update_per_transition_holds_across_one_transition_chunks():
    agent = _dqn(updates_per_step=0.5)
    agent.enable_mid_episode_updates(1)
    start = _steps(agent)
    _one_at_a_time(agent, 6)
    assert _steps(agent) - start == 3, "6 transitions at 0.5 must give 3 gradient steps, not 6."


def test_dqn_without_mid_episode_updates_rounds_each_call_up_as_before():
    agent = _dqn(updates_per_step=0.5)
    start = _steps(agent)
    _one_at_a_time(agent, 6)
    assert _steps(agent) - start == 6


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        ({"updates_per_step": None}, "updates_per_step"),
        ({"exploration_steps": None}, "epsilon_decay"),
    ],
)
def test_dqn_rejects_mid_episode_updates_with_per_call_rules(config, reason):
    agent = DQNAgent(OBS_DIM, 2, config={**SMALL_DQN, **config})
    with pytest.raises(ValueError, match=reason):
        agent.enable_mid_episode_updates(1)


@pytest.mark.parametrize(
    "make",
    [lambda: _ddpg(updates_per_step=0.5), lambda: _dqn(updates_per_step=0.5)],
    ids=["ddpg", "dqn"],
)
def test_the_transition_count_survives_a_checkpoint(tmp_path, make):
    """3 transitions at 0.5 is 1 step with half a step owed; after a reload the 4th pays it."""
    agent = make()
    agent.enable_mid_episode_updates(1)
    _one_at_a_time(agent, 3)
    agent.save(str(tmp_path / "a.pt"))

    resumed = make()
    resumed.enable_mid_episode_updates(1)
    resumed.load(str(tmp_path / "a.pt"))
    before = _steps(resumed)
    _one_at_a_time(resumed, 1)
    assert _steps(resumed) - before == 1, "The 4th transition completes the second step."


A2C = {"seed": 5, "hidden_dim": 6, "gamma": 0.9, "gae_lambda": 0.9, "n_steps": 2, **BOUNDS}


@pytest.mark.parametrize(
    ("make", "extra"),
    [
        (lambda: _ddpg(), ["update_transitions"]),
        (lambda: _dqn(), ["update_transitions"]),
        (lambda: ContinuousActorCriticAgent(OBS_DIM, ACT_DIM, config=A2C), []),
    ],
    ids=["ddpg", "dqn", "a2c"],
)
def test_mid_episode_updates_add_only_the_transition_count_to_a_checkpoint(tmp_path, make, extra):
    """One episode trained each way; A2C, which counts nothing per call, adds no key at all."""
    plain, chunked = make(), make()
    _train(tmp_path / "plain", plain)
    _train(tmp_path / "chunked", chunked, update_every=2)
    plain.save(str(tmp_path / "plain.pt"))
    chunked.save(str(tmp_path / "chunked.pt"))
    with np.load(tmp_path / "plain.pt") as p, np.load(tmp_path / "chunked.pt") as c:
        assert sorted(c.files) == sorted(set(p.files) | set(extra)), (
            f"Updating mid-episode must add exactly {extra} to the checkpoint."
        )


# ---------------------------------------------------------------------------
# The continuous Actor-Critic: true A2C, one chunk of n_steps per call
# ---------------------------------------------------------------------------


def _a2c_actions(tmp_path, **training):
    trainer = _train(tmp_path, ContinuousActorCriticAgent(OBS_DIM, ACT_DIM, config=A2C), **training)
    return np.array(trainer.env.actions)


def test_a2c_samples_each_chunk_with_the_policy_updated_on_the_one_before(tmp_path):
    """Same seed, so the first chunk's actions agree; replaying the finished episode samples the
    second chunk with the untrained policy, true A2C with the policy one update on."""
    replayed = _a2c_actions(tmp_path / "replay")
    live = _a2c_actions(tmp_path / "live", update_every=2)
    np.testing.assert_array_equal(live[:2], replayed[:2], err_msg="No update before step 3.")
    assert not np.allclose(live[2], replayed[2]), (
        "Step 3 must be sampled after the first chunk's update, not by the initial policy."
    )


def _record_chunks(monkeypatch, agent):
    """Each chunk's targets as computed: (steps, terminal, first coordinate of its bootstrap state)."""
    calls = []

    def record(original, states, rewards, bootstrap_obs, terminal):
        calls.append((len(rewards), terminal, None if bootstrap_obs is None else bootstrap_obs[0]))
        return original(states, rewards, bootstrap_obs, terminal)

    spy_on_chunk_targets(monkeypatch, agent, record)
    return calls


def test_a2c_trains_each_chunk_once_bootstrapping_all_but_the_terminal_one(tmp_path, monkeypatch):
    agent = ContinuousActorCriticAgent(OBS_DIM, ACT_DIM, config=A2C)
    calls = _record_chunks(monkeypatch, agent)
    _train(tmp_path, agent, update_every=2)
    # The env's observation after step i is i everywhere: chunk (1, 2) bootstraps from V(2), etc.
    assert calls == [(2, False, 2.0), (2, False, 4.0), (2, False, 6.0), (1, True, 7.0)], (
        "Chunks of 2, 2, 2, 1, each bootstrapped from the state after it but the terminal last."
    )


@pytest.mark.parametrize(
    "cut",
    [{"max_steps_per_episode": 5}, {"truncate_at": 5}],
    ids=["trainer-step-limit", "env-truncation"],
)
def test_a2c_cut_short_bootstraps_its_last_chunk(tmp_path, monkeypatch, cut):
    agent = ContinuousActorCriticAgent(OBS_DIM, ACT_DIM, config=A2C)
    calls = _record_chunks(monkeypatch, agent)
    _train(tmp_path, agent, length=20, update_every=2, **cut)
    assert calls == [(2, False, 2.0), (2, False, 4.0), (1, False, 5.0)], (
        "A cut at step 5 is no termination: the last chunk bootstraps from V(state 5)."
    )


@pytest.mark.parametrize(
    ("n_steps", "update_every", "message"),
    [
        (0, 2, "update_every needs n_steps"),
        (2, 1, r"training.update_every \(1\) must equal n_steps \(2\)"),
        (2, 4, r"training.update_every \(4\) must equal n_steps \(2\)"),
    ],
)
def test_a2c_refuses_chunks_that_are_not_its_n_steps(n_steps, update_every, message):
    agent = ContinuousActorCriticAgent(OBS_DIM, ACT_DIM, config={**A2C, "n_steps": n_steps})
    with pytest.raises(ValueError, match=message):
        Trainer(env=_Env(3), agent=agent, config={"update_every": update_every})


@pytest.mark.parametrize(
    "make",
    [
        lambda: ContinuousPPOAgent(OBS_DIM, ACT_DIM, config={"seed": 1, **BOUNDS}),
        lambda: ActorCriticAgent(OBS_DIM, 2, config={"seed": 1}),
    ],
    ids=["continuous-ppo", "discrete-actor-critic"],
)
def test_the_other_actor_critics_still_train_on_whole_episodes(make):
    with pytest.raises(ValueError, match="trains on whole episodes"):
        Trainer(env=_Env(3), agent=make(), config={"update_every": 2})
