"""DQN options that bring it in line with rl-zoo's DQN.

* ``exploration_steps``: epsilon falls linearly per training action instead of once per update call;
* bootstrapping through time-limit cuts: the trainer's ``terminated`` flag, not ``dones``, ends the
  bootstrap;
* ``optimizer``: the shared SGD / Adam option, with ``max_grad_norm`` a global-norm clip for Adam and
  ``grad_clip_value`` DQN's element-wise clip for plain SGD;
* checkpoints that hold everything training needs to resume.
"""

import copy

import numpy as np
import pytest

from rl_core.agents import DQNAgent
from tests.agent_internals import (
    DQN_ONLINE_WEIGHTS,
    apply_gradients,
    assert_same_weights,
    bellman_targets,
    counters,
    loss_and_grads,
    share_generators,
    weights,
)
from tests.helpers import ReferenceAdam

OBS_DIM, ACTION_DIM = 4, 2


def _agent(**config) -> DQNAgent:
    return DQNAgent(OBS_DIM, ACTION_DIM, config={"seed": 5, "hidden_dim": 8, **config})


def _batch(n: int = 6, seed: int = 0, **extra) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = {
        "observations": rng.standard_normal((n, OBS_DIM)),
        "actions": rng.integers(0, ACTION_DIM, size=n),
        "rewards": np.ones(n),
        "next_observations": rng.standard_normal((n, OBS_DIM)),
        "dones": np.array([False] * (n - 1) + [True]),
    }
    batch.update(extra)
    return batch


def _act(agent: DQNAgent, times: int, training: bool = True) -> None:
    observation = np.zeros(OBS_DIM)
    for _ in range(times):
        agent.select_action(observation, training=training)


# ---------------------------------------------------------------------------
# exploration_steps
# ---------------------------------------------------------------------------

LINEAR = {"epsilon_start": 1.0, "epsilon_end": 0.04, "exploration_steps": 8}


@pytest.mark.parametrize(
    ("actions", "expected"),
    # epsilon = 0.04 + 0.96 * max(0, 1 - k / 8) after k training actions
    [(0, 1.0), (1, 0.88), (4, 0.52), (7, 0.16), (8, 0.04), (20, 0.04)],
)
def test_epsilon_falls_linearly_with_training_actions(actions, expected):
    agent = _agent(**LINEAR)
    _act(agent, actions)
    assert agent.epsilon == pytest.approx(expected, abs=1e-12)


def test_evaluation_actions_do_not_advance_exploration():
    agent = _agent(**LINEAR)
    _act(agent, 5, training=False)
    assert agent.epsilon == 1.0, "greedy evaluation and record.py must not use up exploration"


def test_update_does_not_decay_epsilon_when_exploration_is_by_step():
    agent = _agent(**LINEAR, min_buffer_size=2, batch_size=2)
    _act(agent, 4)
    metrics = agent.update(_batch())
    assert agent.epsilon == pytest.approx(0.52) and metrics["epsilon"] == pytest.approx(0.52)


def test_without_exploration_steps_epsilon_still_decays_once_per_update():
    agent = _agent(epsilon_decay=0.5, min_buffer_size=2, batch_size=2)
    _act(agent, 4)
    agent.update(_batch())
    assert agent.epsilon == pytest.approx(0.5)


@pytest.mark.parametrize("value", [0, -3, 2.5, "many", True])
def test_exploration_steps_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="exploration_steps"):
        _agent(exploration_steps=value)


def test_exploration_steps_and_epsilon_decay_are_exclusive():
    with pytest.raises(
        ValueError, match="epsilon_decay.*exploration_steps|exploration_steps.*epsilon_decay"
    ):
        _agent(exploration_steps=100, epsilon_decay=0.99)


# ---------------------------------------------------------------------------
# Bootstrapping through time-limit cuts
# ---------------------------------------------------------------------------


def test_a_time_limit_cut_is_stored_as_not_terminal():
    """The trainer marks a cut episode done but not terminated: its future value still counts."""
    agent = _agent()
    terminated = np.zeros(6, dtype=bool)  # the episode was cut at the time limit
    agent.update(_batch(terminated=terminated))
    stored = agent.replay_buffer.dones[: len(agent.replay_buffer)]
    np.testing.assert_array_equal(stored, terminated)


def test_a_real_termination_is_stored_as_terminal():
    agent = _agent()
    terminated = np.array([False] * 5 + [True])
    agent.update(_batch(terminated=terminated))
    assert agent.replay_buffer.dones[5] and not agent.replay_buffer.dones[:5].any()


def test_a_batch_without_terminated_falls_back_to_dones():
    agent = _agent()
    agent.update(_batch())
    assert agent.replay_buffer.dones[5] and len(agent.replay_buffer) == 6


def test_a_batch_with_terminated_and_no_dones_is_stored():
    batch = _batch(terminated=np.zeros(6, dtype=bool))
    del batch["dones"]
    agent = _agent()
    agent.update(batch)
    assert len(agent.replay_buffer) == 6


def test_terminated_of_the_wrong_length_is_an_error():
    agent = _agent()
    with pytest.raises(ValueError, match="terminated"):
        agent.update(_batch(terminated=np.zeros(5, dtype=bool)))


# ---------------------------------------------------------------------------
# optimizer, max_grad_norm and grad_clip_value
# ---------------------------------------------------------------------------

ONE_STEP = {"min_buffer_size": 6, "batch_size": 6, "learning_rate": 0.01, "gamma": 0.9}


def _expected_adam_weights(agent: DQNAgent, batch, adam: ReferenceAdam) -> dict[str, np.ndarray]:
    """One update written out: store, sample, Bellman targets, gradients, reference Adam step."""
    reference = copy.deepcopy(agent)
    reference.replay_buffer.add_batch(
        batch["observations"], batch["actions"], batch["rewards"],
        batch["next_observations"], batch["dones"],
    )  # fmt: skip
    sample = reference.replay_buffer.sample(reference.batch_size)
    targets = bellman_targets(
        reference, sample["rewards"], sample["next_observations"], sample["dones"]
    )
    _, grads = loss_and_grads(reference, sample["observations"], sample["actions"], targets)
    steps = adam.step(grads)
    current = weights(reference)
    return {name: current[name] - steps[name] for name in DQN_ONLINE_WEIGHTS}


@pytest.mark.parametrize("max_grad_norm", [None, 0.05])
def test_adam_takes_the_reference_adam_step(max_grad_norm):
    config = {**ONE_STEP, "optimizer": "adam"}
    if max_grad_norm is not None:
        config["max_grad_norm"] = max_grad_norm  # small enough that the global clip bites
    agent = _agent(**config)
    adam = ReferenceAdam(0.01, max_grad_norm)
    batch = _batch()

    for _ in range(2):  # two steps, so the moments carry over
        expected = _expected_adam_weights(agent, batch, adam)
        agent.update(batch)
        after = weights(agent)
        for name in DQN_ONLINE_WEIGHTS:
            np.testing.assert_allclose(after[name], expected[name], rtol=1e-10, err_msg=name)


def test_sgd_clips_each_gradient_element_at_grad_clip_value():
    agent = _agent(learning_rate=0.1, grad_clip_value=0.5)
    before = weights(agent)
    grads = {name: np.full(before[name].shape, 2.0) for name in DQN_ONLINE_WEIGHTS}
    grads["b2"] = np.full(ACTION_DIM, 0.25)
    apply_gradients(agent, grads)
    after = weights(agent)
    for name in DQN_ONLINE_WEIGHTS:
        step = 0.1 * np.clip(grads[name], -0.5, 0.5)
        np.testing.assert_allclose(after[name], before[name] - step, rtol=1e-12, err_msg=name)


def test_grad_clip_value_defaults_to_ten():
    assert _agent().grad_clip_value == 10.0


def test_max_grad_norm_with_plain_sgd_is_an_error():
    """As in every other agent: max_grad_norm is Adam's global-norm clip, not an SGD setting."""
    with pytest.raises(ValueError, match="max_grad_norm"):
        _agent(max_grad_norm=1.0)


def test_grad_clip_value_with_adam_is_an_error():
    with pytest.raises(ValueError, match="grad_clip_value"):
        _agent(optimizer="adam", grad_clip_value=1.0)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_grad_clip_value_must_be_positive(value):
    with pytest.raises(ValueError, match="grad_clip_value"):
        _agent(grad_clip_value=value)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

RESUMABLE = {
    **LINEAR, "exploration_steps": 50, "optimizer": "adam", "max_grad_norm": 10.0,
    "min_buffer_size": 4, "batch_size": 4, "updates_per_step": 0.5, "target_update_frequency": 3,
}  # fmt: skip


def _trained(config) -> DQNAgent:
    agent = _agent(**config)
    for episode in range(3):
        _act(agent, 6)
        agent.update(_batch(seed=episode, terminated=np.zeros(6, dtype=bool)))
    return agent


def test_a_loaded_agent_resumes_exactly(tmp_path):
    """Save after a few updates, load into a fresh agent, and both continue identically."""
    agent = _trained(RESUMABLE)
    path = str(tmp_path / "dqn.pt")
    agent.save(path)
    resumed = _agent(**RESUMABLE)
    resumed.load(path)
    # Training continues from the same replay data and generator state in both.
    resumed.replay_buffer = copy.deepcopy(agent.replay_buffer)
    share_generators(agent, resumed)

    for run in (agent, resumed):
        _act(run, 6)
        run.update(_batch(seed=9, terminated=np.zeros(6, dtype=bool)))

    assert_same_weights(resumed, agent, rtol=0.0)
    assert resumed.epsilon == agent.epsilon
    assert counters(resumed) == counters(agent)


def test_the_saved_exploration_step_is_restored_not_rebuilt(tmp_path):
    """An epsilon off the schedule's grid would rebuild to a different step than the saved one."""
    agent = _agent(**{**LINEAR, "exploration_steps": 50})
    _act(agent, 3)
    agent.epsilon = 0.9  # rebuilding from this would give step round(0.1 / 0.96 * 50) = 5, not 3
    path = str(tmp_path / "dqn.pt")
    agent.save(path)
    restored = _agent(**{**LINEAR, "exploration_steps": 50})
    restored.load(path)
    assert counters(restored)["exploration_step"] == 3


def test_the_trainer_time_limit_cut_reaches_the_buffer_as_not_terminal(tmp_path):
    """Through the real Trainer: episodes cut at 5 steps leave no terminal flag in the buffer."""
    from rl_cartpole.environments import CartPoleEnv
    from rl_core.training import Trainer

    env = CartPoleEnv(seed=0, max_episode_steps=5)  # far too short for the pole to fall
    agent = _agent(min_buffer_size=4, batch_size=4)
    config = {
        "num_episodes": 2,
        "max_steps_per_episode": 5,
        "eval_frequency": 3,
        "save_frequency": 3,
        "checkpoint_dir": str(tmp_path / "ckpt"),
    }
    try:
        Trainer(env=env, agent=agent, config=config).train()
    finally:
        env.close()
    assert len(agent.replay_buffer) == 10
    assert not agent.replay_buffer.dones[:10].any(), "a time-limit cut was stored as terminal"


def _old_checkpoint(path, agent: DQNAgent, epsilon: float) -> None:
    """A checkpoint in the format written before this change: weights and epsilon only."""
    arrays = weights(agent)  # keyed by checkpoint name: W1 … b2 and target_W1 … target_b2
    with open(path, "wb") as f:
        np.savez(f, **arrays, epsilon=np.array(epsilon))


def test_an_old_checkpoint_rebuilds_the_exploration_step_from_epsilon(tmp_path):
    """Without the saved step count, resuming must not restart exploration at epsilon_start."""
    path = str(tmp_path / "old.pt")
    _old_checkpoint(path, _agent(), epsilon=0.52)
    agent = _agent(**{**LINEAR, "exploration_steps": 8000})
    agent.load(path)
    assert agent.epsilon == pytest.approx(0.52)
    _act(agent, 1)  # 0.52 is 4000 of 8000 steps in; one more step lowers it by 0.96 / 8000
    assert agent.epsilon == pytest.approx(0.52 - 0.96 / 8000)


def test_an_old_checkpoint_loads_into_an_adam_agent_with_fresh_moments(tmp_path):
    path = str(tmp_path / "old.pt")
    source = _agent()
    _old_checkpoint(path, source, epsilon=0.3)
    agent = _agent(optimizer="adam")
    agent.load(path)
    assert_same_weights(agent, source, rtol=0.0)
    assert counters(agent)["update_counter"] == 0


def test_an_adam_checkpoint_loads_into_an_sgd_agent(tmp_path):
    path = str(tmp_path / "adam.pt")
    source = _trained(RESUMABLE)
    source.save(path)
    agent = _agent()
    agent.load(path)
    assert_same_weights(agent, source, rtol=0.0)
