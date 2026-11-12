"""A checkpoint that does not fit the agent is refused whole, before anything is assigned.

Loading a checkpoint saved with another ``hidden_dim`` (or ``hidden_sizes``, ``observation_dim``,
``action_dim``) raises a ValueError naming the entry, the checkpoint's shape and the agent's, and
leaves the agent exactly as it was: weights, optimizer moments, replay buffer, counters and
generators (compared through ``save``). Without the check, plain SGD loaded the wrong-sized weights
silently and Adam assigned them before its moments raised.
"""

import copy
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_core.agents import (  # noqa: E402
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DDPGAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
    SACAgent,
    TD3Agent,
)
from tests.agent_internals import weights  # noqa: E402
from tests.helpers import assert_same_checkpoint  # noqa: E402

OBS, ACT = 4, 2
DISCRETE = (ReinforceAgent, ActorCriticAgent, PPOAgent, DQNAgent)
GAUSSIAN = (ContinuousActorCriticAgent, ContinuousPPOAgent)  # always Adam
OFF_POLICY = (DDPGAgent, TD3Agent, SACAgent)  # Adam per network

# (agent class, optimizer, the saved agent's config, the loading agent's config)
CASES = [
    *[
        (cls, opt, {"hidden_dim": 8, "optimizer": opt}, {"hidden_dim": 16, "optimizer": opt})
        for cls in DISCRETE
        for opt in ("sgd", "adam")
    ],
    *[(cls, "adam", {"hidden_dim": 8}, {"hidden_dim": 16}) for cls in GAUSSIAN],
    *[
        (cls, "adam", {"hidden_sizes": [8], "learning_starts": 2, "batch_size": 2},
         {"hidden_sizes": [16], "learning_starts": 2, "batch_size": 2})
        for cls in OFF_POLICY
    ],
]  # fmt: skip


def _ids(case):
    return f"{case[0].__name__}-{case[1]}"


def _transitions(agent, n=6, seed=3):
    """A batch of ``n`` transitions in the agent's action space."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n + 1, OBS))
    actions = np.array([agent.select_action(o, training=True) for o in obs[:-1]])
    dones = np.zeros(n, dtype=bool)
    dones[-1] = True
    return {
        "observations": obs[:-1],
        "actions": actions,
        "rewards": rng.standard_normal(n),
        "next_observations": obs[1:],
        "dones": dones,
        "terminated": dones,
    }


def _trained(cls, config, seed):
    """An agent that has learnt from one batch, so its moments and any buffer are not empty."""
    agent = cls(OBS, ACT, {**config, "seed": seed})
    agent.update(_transitions(agent, seed=seed))
    return agent


@pytest.mark.parametrize(("cls", "opt", "saved", "loading"), CASES, ids=map(_ids, CASES))
def test_a_checkpoint_with_other_layer_sizes_is_refused_and_changes_nothing(
    cls, opt, saved, loading, tmp_path
):
    path = str(tmp_path / "small.npz")
    _trained(cls, saved, seed=0).save(path)
    agent = _trained(cls, loading, seed=1)
    before, snapshot = weights(agent), copy.deepcopy(agent)

    with pytest.raises(ValueError, match=r"W.*\(4, 8\).*\(4, 16\)") as error:
        agent.load(path)

    assert path in str(error.value), "the error names the file"
    assert weights(agent).keys() == before.keys()
    for name, value in before.items():
        np.testing.assert_array_equal(weights(agent)[name], value, err_msg=name)
    assert_same_checkpoint(snapshot, agent, tmp_path)  # moments, buffer, counters, generators


def _rewritten(path, tmp_path, drop=(), **entries):
    """A copy of checkpoint ``path`` without the keys in ``drop`` and with ``entries`` replaced."""
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key not in drop}
    arrays.update(entries)
    out = str(tmp_path / "rewritten.npz")
    np.savez(out, **arrays)
    return out


@pytest.mark.parametrize(
    ("cls", "config", "key"),
    [
        (ReinforceAgent, {"optimizer": "adam", "hidden_dim": 8}, "adam_m_W1"),
        (ActorCriticAgent, {"optimizer": "adam", "hidden_dim": 8}, "adam_v_W_pi"),
        (PPOAgent, {"optimizer": "adam", "hidden_dim": 8}, "adam_m_b1"),
        (DQNAgent, {"optimizer": "adam", "hidden_dim": 8}, "adam_m_W2"),
        (ContinuousActorCriticAgent, {"hidden_dim": 8}, "adam_v_log_std"),
        (ContinuousPPOAgent, {"hidden_dim": 8}, "adam_m_W_mu"),
        (DDPGAgent, {"hidden_sizes": [8]}, "critic_adam_m_W0"),
        (TD3Agent, {"hidden_sizes": [8]}, "critic_2_adam_v_b1"),
        (SACAgent, {"hidden_sizes": [8]}, "actor_adam_m_W1"),
    ],
    ids=lambda v: v if isinstance(v, str) else getattr(v, "__name__", None),
)
@pytest.mark.parametrize(
    "wrong_shape",
    [lambda shape: (*shape, 2), lambda shape: tuple(d + 1 for d in shape)],
    ids=["extra-axis", "other-sizes"],
)
def test_a_wrong_sized_optimizer_moment_is_refused_before_the_weights_are_assigned(
    cls, config, key, wrong_shape, tmp_path
):
    """The weights fit and one moment does not: Adam used to raise after the weights were in."""
    source = cls(OBS, ACT, {**config, "seed": 0})
    path = str(tmp_path / "agent.npz")
    source.save(path)
    with np.load(path) as data:
        wrong = np.zeros(wrong_shape(data[key].shape))
    path = _rewritten(path, tmp_path, **{key: wrong})
    agent = cls(OBS, ACT, {**config, "seed": 1})
    snapshot = copy.deepcopy(agent)

    with pytest.raises(ValueError, match=re.escape(f"{key} has shape {wrong.shape}")):
        agent.load(path)

    assert_same_checkpoint(snapshot, agent, tmp_path)


def test_a_legacy_gaussian_checkpoint_without_moments_still_loads(tmp_path):
    """Continuous checkpoints saved before the moments were stored hold only ``adam_t``: they load,
    with zero moments, as they always did."""
    source = ContinuousActorCriticAgent(OBS, ACT, {"hidden_dim": 8, "seed": 0})
    path = str(tmp_path / "agent.npz")
    source.save(path)
    with np.load(path) as data:
        legacy = {k: data[k] for k in data.files if not k.startswith(("adam_m_", "adam_v_"))}
    np.savez(path, **legacy)
    agent = ContinuousActorCriticAgent(OBS, ACT, {"hidden_dim": 8, "seed": 1})
    agent.load(path)
    for name, value in weights(source).items():
        np.testing.assert_array_equal(weights(agent)[name], value, err_msg=name)


@pytest.mark.parametrize(
    ("saved", "loading", "key"),
    [
        ({"observation_dim": 3}, {}, "observation_dim"),
        ({"action_dim": 1}, {}, "action_dim"),
        ({"config": {"action_low": -2.0, "action_high": 2.0}}, {}, "action_low"),
        ({"config": {"action_high": 3.0}}, {}, "action_high"),
    ],
)
def test_a_continuous_random_checkpoint_for_another_space_is_refused(saved, loading, key, tmp_path):
    """The random agent's checkpoint describes its spaces: one for other spaces is refused, not
    taken over (it would act outside this environment's bounds)."""
    build = {"observation_dim": OBS, "action_dim": ACT, "config": {"seed": 0}}
    path = str(tmp_path / "random.npz")
    ContinuousRandomAgent(**{**build, **saved}).save(path)
    agent = ContinuousRandomAgent(**{**build, **loading, "config": {"seed": 1}})
    snapshot = copy.deepcopy(agent)

    with pytest.raises(ValueError, match=key):
        agent.load(path)

    assert_same_checkpoint(snapshot, agent, tmp_path)
    assert (agent.observation_dim, agent.action_dim) == (OBS, ACT)


@pytest.mark.parametrize("saved", [{"observation_dim": 3}, {"action_dim": 3}])
def test_a_discrete_random_checkpoint_for_another_space_is_refused(saved, tmp_path):
    build = {"observation_dim": OBS, "action_dim": ACT, "config": {"seed": 0}}
    path = str(tmp_path / "random.npz")
    RandomAgent(**{**build, **saved}).save(path)
    agent = RandomAgent(**{**build, "config": {"seed": 1}})
    snapshot = copy.deepcopy(agent)

    with pytest.raises(ValueError, match="dim"):
        agent.load(path)

    assert_same_checkpoint(snapshot, agent, tmp_path)


@pytest.mark.parametrize(
    ("cls", "config"),
    [(DQNAgent, {"hidden_dim": 8}), (DDPGAgent, {"hidden_sizes": [8], "learning_starts": 2})],
    ids=["DQNAgent", "DDPGAgent"],
)
def test_a_replay_buffer_of_another_fill_level_still_loads(cls, config, tmp_path):
    """The buffer's saved columns hold its filled part, so their length is not this agent's: a
    checkpoint with transitions loads into a fresh agent, buffer and all."""
    source = _trained(cls, config, seed=0)
    path = str(tmp_path / "agent.npz")
    source.save(path)
    agent = cls(OBS, ACT, {**config, "seed": 1})  # an empty buffer
    agent.load(path)
    assert_same_checkpoint(source, agent, tmp_path)


def test_a_lone_first_moment_of_another_shape_is_refused(tmp_path):
    """``adam_m_W1`` without its ``adam_v_W1``: Adam alone would reset both to zero, but an entry of
    another shape is refused, as any other entry of the agent's is."""
    config = {"optimizer": "adam", "hidden_dim": 8}
    path = str(tmp_path / "agent.npz")
    ReinforceAgent(OBS, ACT, {**config, "seed": 0}).save(path)
    path = _rewritten(path, tmp_path, drop=("adam_v_W1",), adam_m_W1=np.zeros((5, 9)))
    agent = ReinforceAgent(OBS, ACT, {**config, "seed": 1})
    snapshot = copy.deepcopy(agent)

    with pytest.raises(ValueError, match=re.escape("adam_m_W1 has shape (5, 9), expected (4, 8)")):
        agent.load(path)

    assert_same_checkpoint(snapshot, agent, tmp_path)


# (agent class, config, a key its load needs and the checkpoint lacks)
MISSING = [
    (ReinforceAgent, {"optimizer": "adam"}, "W1"),
    (ActorCriticAgent, {"optimizer": "adam"}, "b_v"),
    (PPOAgent, {}, "W_v"),
    (DQNAgent, {"optimizer": "adam"}, "target_W2"),
    (DQNAgent, {}, "W1"),
    (ContinuousActorCriticAgent, {}, "log_std"),
    (ContinuousPPOAgent, {}, "b_v"),
    (DDPGAgent, {"learning_starts": 2, "batch_size": 2}, "training_actions"),
    (SACAgent, {"learning_starts": 2, "batch_size": 2}, "gradient_steps"),
]


@pytest.mark.parametrize(
    ("cls", "config", "key"), MISSING, ids=[f"{c.__name__}-{k}" for c, _, k in MISSING]
)
def test_a_checkpoint_missing_an_entry_is_refused_and_changes_nothing(cls, config, key, tmp_path):
    """A missing weight or counter used to raise KeyError halfway, after the replay buffer or the
    first weights were replaced."""
    path = str(tmp_path / "agent.npz")
    _trained(cls, config, seed=0).save(path)
    path = _rewritten(path, tmp_path, drop=(key,))
    agent = cls(OBS, ACT, {**config, "seed": 1})  # an empty buffer, if it has one
    snapshot = copy.deepcopy(agent)

    with pytest.raises(ValueError, match=rf"has no {key}\b"):
        agent.load(path)

    assert_same_checkpoint(snapshot, agent, tmp_path)


@pytest.mark.parametrize(
    ("cls", "key"),
    [(RandomAgent, "action_dim"), (ContinuousRandomAgent, "action_low")],
    ids=["RandomAgent", "ContinuousRandomAgent"],
)
def test_a_random_checkpoint_missing_an_entry_is_refused(cls, key, tmp_path):
    path = str(tmp_path / "random.npz")
    cls(OBS, ACT, {"seed": 0}).save(path)
    path = _rewritten(path, tmp_path, drop=(key,))
    with pytest.raises(ValueError, match=rf"has no {key}\b"):
        cls(OBS, ACT, {"seed": 1}).load(path)


@pytest.mark.parametrize(
    ("cls", "config"),
    [
        (
            DQNAgent,
            {"hidden_dim": 8, "buffer_capacity": 100, "batch_size": 2, "min_buffer_size": 2},
        ),
        (DDPGAgent, {"hidden_sizes": [8], "learning_starts": 2, "buffer_capacity": 100}),
    ],
    ids=["DQNAgent", "DDPGAgent"],
)
def test_a_replay_buffer_of_another_capacity_is_refused_and_changes_nothing(cls, config, tmp_path):
    path = str(tmp_path / "agent.npz")
    _trained(cls, config, seed=0).save(path)
    agent = _trained(cls, {**config, "buffer_capacity": 50}, seed=1)
    snapshot = copy.deepcopy(agent)

    with pytest.raises(ValueError, match="replay_capacity is 100"):
        agent.load(path)

    assert_same_checkpoint(snapshot, agent, tmp_path)


@pytest.mark.parametrize("cls", [ReinforceAgent, ActorCriticAgent, PPOAgent, DQNAgent])
def test_an_adam_agent_loads_a_plain_sgd_checkpoint_with_fresh_moments(cls, tmp_path):
    """Switching a run to Adam: an SGD checkpoint holds no optimizer state, so its weights load and
    Adam starts again from zero moments and step 0."""
    sgd_path, adam_path = str(tmp_path / "sgd.npz"), str(tmp_path / "adam.npz")
    source = _trained(cls, {"optimizer": "sgd", "hidden_dim": 8}, seed=0)
    source.save(sgd_path)
    agent = _trained(cls, {"optimizer": "adam", "hidden_dim": 8}, seed=1)  # moments not zero

    agent.load(sgd_path)

    for name, value in weights(source).items():
        np.testing.assert_array_equal(weights(agent)[name], value, err_msg=name)
    agent.save(adam_path)
    with np.load(adam_path) as data:
        adam = [k for k in data.files if k.startswith("adam_")]
        assert "adam_t" in adam and int(data["adam_t"]) == 0
        for key in adam:
            np.testing.assert_array_equal(data[key], 0, err_msg=key)


@pytest.mark.parametrize("cls", [ReinforceAgent, ActorCriticAgent, PPOAgent, DQNAgent])
def test_an_sgd_agent_loads_an_adam_checkpoint_ignoring_its_moments(cls, tmp_path):
    path = str(tmp_path / "adam.npz")
    source = _trained(cls, {"optimizer": "adam", "hidden_dim": 8}, seed=0)
    source.save(path)
    agent = cls(OBS, ACT, {"optimizer": "sgd", "hidden_dim": 8, "seed": 1})

    agent.load(path)

    for name, value in weights(source).items():
        np.testing.assert_array_equal(weights(agent)[name], value, err_msg=name)


@pytest.mark.parametrize(
    ("cls", "config", "legacy_lacks"),
    [
        # DQN checkpoints from before the counters, Adam state and buffer were saved
        (
            DQNAgent,
            {"optimizer": "adam", "batch_size": 2, "min_buffer_size": 2},
            ("adam_", "epsilon", "exploration_step", "update_counter", "steps_trained", "replay_"),
        ),
        # off-policy checkpoints from before the replay buffer was saved
        (DDPGAgent, {"learning_starts": 2, "batch_size": 2}, ("replay_",)),
    ],
    ids=["DQNAgent", "DDPGAgent"],
)
def test_an_older_replay_checkpoint_still_loads(cls, config, legacy_lacks, tmp_path):
    source = _trained(cls, config, seed=0)
    path = str(tmp_path / "agent.npz")
    source.save(path)
    with np.load(path) as data:
        dropped = tuple(k for k in data.files if k.startswith(legacy_lacks))
    path = _rewritten(path, tmp_path, drop=dropped)
    agent = _trained(cls, config, seed=1)

    agent.load(path)

    for name, value in weights(source).items():
        np.testing.assert_array_equal(weights(agent)[name], value, err_msg=name)
