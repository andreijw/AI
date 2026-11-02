"""Checkpoints record the observation settings (``OBSERVATION_DEFAULTS``'s keys) an agent was
built for, and ``load`` refuses a checkpoint saved with other settings before changing anything."""

import numpy as np
import pytest

from rl_cartpole.benchmark import CartPoleChallenge
from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import RandomAgent, ReinforceAgent
from rl_core.agents.base_agent import OBSERVATION_KEY_PREFIX
from rl_core.environments import OBSERVATION_DEFAULTS
from rl_core.training import Trainer
from tests.agent_internals import with_extra_checkpoint_entry
from tests.helpers import minimal_config, run_main
from tests.test_agent_harness import (
    ALL_AGENTS,
    ARCHITECTURE_KEYS,
    CONTINUOUS_AGENTS,
    agent_id,
)

DEFAULTS = {"frame_stack": 1, "mask_velocity": False}
MASKED = {"frame_stack": 1, "mask_velocity": True}
STACKED = {"frame_stack": 4, "mask_velocity": True}

# Every agent, with the input and output sizes of a CartPole (discrete) or Pendulum (continuous) run.
EVERY_AGENT = [pytest.param(c, cfg, 4, 2, id=agent_id(c, cfg)) for c, cfg in ALL_AGENTS] + [
    pytest.param(c, cfg, 3, 1, id=c.__name__) for c, cfg in CONTINUOUS_AGENTS
]


def _agent(agent_cls, config, obs_dim, action_dim, settings=None, defaults=DEFAULTS):
    agent = agent_cls(observation_dim=obs_dim, action_dim=action_dim, config=dict(config))
    if settings is not None:
        agent.record_observation_settings(settings, defaults)
    return agent


def _loader_config(config):
    """Another seed, the same architecture: a dueling DQN loads only into a dueling DQN."""
    return {"seed": 99, **{k: config[k] for k in ARCHITECTURE_KEYS if k in config}}


def _entries(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def _rewrite(path, change):
    """Rewrite the checkpoint at ``path`` with ``change`` applied to its entries."""
    entries = _entries(path)
    change(entries)
    with open(path, "wb") as f:
        np.savez_compressed(f, **entries)


def _assert_acts_alike(saved, loaded, obs_dim):
    obs = np.linspace(-0.3, 0.3, obs_dim)
    assert np.all(
        np.asarray(loaded.select_action(obs, training=False))
        == np.asarray(saved.select_action(obs, training=False))
    ), "the loaded agent acts differently"


def _assert_unchanged(agent, before, tmp_path):
    after = tmp_path / "after.npz"
    agent.save(str(after))
    got = _entries(after)
    assert sorted(got) == sorted(before), "a refused load changed which entries the agent saves"
    for key, value in before.items():
        np.testing.assert_array_equal(got[key], value, err_msg=f"a refused load changed {key}")


# --- Recording -----------------------------------------------------------------------------------


@pytest.mark.parametrize("settings", [DEFAULTS, MASKED, STACKED])
def test_make_agent_records_the_environments_observation_settings(settings):
    env = CartPoleEnv(seed=0, **settings)
    try:
        agent = CartPoleChallenge().make_agent("reinforce", {"seed": 1}, env)
        assert dict(agent.observation_settings) == {**OBSERVATION_DEFAULTS, **settings}, (
            "make_agent recorded other settings"
        )
    finally:
        env.close()


def test_the_environment_reports_every_observation_setting():
    env = CartPoleEnv(seed=0, frame_stack=3)
    try:
        assert env.observation_settings() == {**OBSERVATION_DEFAULTS, "frame_stack": 3}
        assert set(env.observation_settings()) == set(OBSERVATION_DEFAULTS)
    finally:
        env.close()


def test_an_agent_built_directly_records_nothing_and_saves_no_setting(tmp_path):
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config={"seed": 1})
    assert agent.observation_settings is None
    agent.save(str(tmp_path / "plain.npz"))
    saved = _entries(tmp_path / "plain.npz")
    assert not [k for k in saved if k.startswith(OBSERVATION_KEY_PREFIX)], (
        "an agent without settings must save the same entries as before"
    )


def test_the_record_is_a_copy_the_caller_cannot_change():
    settings = dict(MASKED)
    agent = _agent(RandomAgent, {"seed": 1}, 4, 2, settings)
    settings["mask_velocity"] = False
    assert agent.observation_settings["mask_velocity"] is True
    with pytest.raises(TypeError):
        agent.observation_settings["frame_stack"] = 2  # type: ignore[index]
    with pytest.raises(AttributeError):
        agent.observation_settings = {}  # type: ignore[misc]  # only through the validating setter


@pytest.mark.parametrize(
    "settings,defaults,match",
    [
        ({"frame_stack": [4]}, {"frame_stack": 1}, "frame_stack"),
        ({"frame_stack": 1.5}, {"frame_stack": 1}, "frame_stack"),
        ({"frame_stack": None}, {"frame_stack": 1}, "frame_stack"),
        ({"frame_stack": 1}, {"frame_stack": "1"}, "frame_stack"),
        ({3: 1}, {3: 1}, "3"),
        ({"stack": 1}, {"frame_stack": 1}, "stack"),
    ],
)
def test_record_observation_settings_refuses_a_malformed_record(settings, defaults, match):
    agent = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 1})
    with pytest.raises(ValueError, match=match):
        agent.record_observation_settings(settings, defaults)
    assert agent.observation_settings is None, "a refused record must leave none"


# --- Saving and loading, for every agent ----------------------------------------------------------


@pytest.mark.parametrize("agent_cls,config,obs_dim,action_dim", EVERY_AGENT)
def test_a_checkpoint_with_the_same_settings_loads(
    agent_cls, config, obs_dim, action_dim, tmp_path
):
    saved = _agent(agent_cls, config, obs_dim, action_dim, MASKED)
    path = str(tmp_path / "agent.pt")
    saved.save(path)
    entries = _entries(path)
    assert entries[f"{OBSERVATION_KEY_PREFIX}frame_stack"] == 1
    assert entries[f"{OBSERVATION_KEY_PREFIX}mask_velocity"] == np.True_

    loaded = _agent(agent_cls, _loader_config(config), obs_dim, action_dim, MASKED)
    loaded.load(path)
    _assert_acts_alike(saved, loaded, obs_dim)


@pytest.mark.parametrize("agent_cls,config,obs_dim,action_dim", EVERY_AGENT)
def test_a_masked_checkpoint_is_refused_by_an_unmasked_agent_of_the_same_size(
    agent_cls, config, obs_dim, action_dim, tmp_path
):
    """The silent case: masking keeps the observation's length, so nothing else would notice."""
    path = str(tmp_path / "agent.pt")
    _agent(agent_cls, config, obs_dim, action_dim, MASKED).save(path)
    loader = _agent(agent_cls, _loader_config(config), obs_dim, action_dim, DEFAULTS)
    loader.save(str(tmp_path / "before.npz"))
    before = _entries(tmp_path / "before.npz")

    with pytest.raises(ValueError, match=r"mask_velocity=true.*mask_velocity=false") as error:
        loader.load(path)
    assert "frame_stack" not in str(error.value), "only the differing setting is named"
    _assert_unchanged(loader, before, tmp_path)


@pytest.mark.parametrize("agent_cls,config,obs_dim,action_dim", EVERY_AGENT)
def test_a_stacked_checkpoint_is_refused_by_name_before_any_shape_check(
    agent_cls, config, obs_dim, action_dim, tmp_path
):
    """A stack of 4 has four times the inputs; the message names frame_stack, not a shape."""
    path = str(tmp_path / "agent.pt")
    _agent(agent_cls, config, obs_dim * 4, action_dim, STACKED).save(path)
    loader = _agent(agent_cls, _loader_config(config), obs_dim, action_dim, MASKED)
    with pytest.raises(ValueError, match=r"frame_stack=4.*frame_stack=1"):
        loader.load(path)


# --- Old checkpoints, agents without a record, defaults --------------------------------------------


@pytest.mark.parametrize("agent_cls", [ReinforceAgent, RandomAgent])
def test_a_checkpoint_without_a_record_loads_into_an_agent_with_one(agent_cls, tmp_path):
    """RandomAgent saves ``observation_dim``, which is not an observation setting."""
    path = str(tmp_path / "old.pt")
    saved = agent_cls(observation_dim=4, action_dim=2, config={"seed": 1})
    saved.save(path)
    loaded = _agent(agent_cls, {"seed": 2}, 4, 2, STACKED)
    loaded.load(path)  # nothing to compare the settings against
    _assert_acts_alike(saved, loaded, 4)


def test_an_agent_without_a_record_loads_any_checkpoint(tmp_path):
    path = str(tmp_path / "masked.pt")
    saved = _agent(ReinforceAgent, {"seed": 1}, 4, 2, MASKED)
    saved.save(path)
    loaded = ReinforceAgent(observation_dim=4, action_dim=2, config={"seed": 2})
    loaded.load(path)
    _assert_acts_alike(saved, loaded, 4)


def test_a_setting_the_checkpoint_lacks_counts_as_its_default(tmp_path):
    """A setting added after a checkpoint was saved must not refuse it while at its default."""
    path = str(tmp_path / "older.pt")
    _agent(ReinforceAgent, {"seed": 1}, 4, 2, {"frame_stack": 1}, {"frame_stack": 1}).save(path)

    _agent(ReinforceAgent, {"seed": 2}, 4, 2, DEFAULTS).load(path)
    with pytest.raises(ValueError, match=r"mask_velocity=false.*mask_velocity=true"):
        _agent(ReinforceAgent, {"seed": 2}, 4, 2, MASKED).load(path)


def test_a_setting_the_agent_does_not_know_is_refused(tmp_path):
    path = str(tmp_path / "newer.pt")
    newer = {**DEFAULTS, "normalize": True}
    _agent(ReinforceAgent, {"seed": 1}, 4, 2, newer, {**DEFAULTS, "normalize": False}).save(path)
    with pytest.raises(ValueError, match="normalize"):
        _agent(ReinforceAgent, {"seed": 2}, 4, 2, DEFAULTS).load(path)


def test_true_and_one_are_different_settings(tmp_path):
    path = str(tmp_path / "agent.pt")
    _agent(ReinforceAgent, {"seed": 1}, 4, 2, {"flag": 1}, {"flag": 0}).save(path)
    with pytest.raises(ValueError, match="flag"):
        _agent(ReinforceAgent, {"seed": 2}, 4, 2, {"flag": True}, {"flag": False}).load(path)


@pytest.mark.parametrize(
    "value",
    [np.array([1, 4]), np.array(1.0), np.array("1")],
    ids=["not one value", "float", "string"],
)
def test_a_malformed_saved_setting_is_refused_naming_its_key(value, tmp_path):
    path = str(tmp_path / "agent.npz")
    _agent(ReinforceAgent, {"seed": 1}, 4, 2, DEFAULTS).save(path)
    _rewrite(path, lambda entries: entries.update({f"{OBSERVATION_KEY_PREFIX}frame_stack": value}))
    with pytest.raises(ValueError, match=f"{OBSERVATION_KEY_PREFIX}frame_stack"):
        _agent(ReinforceAgent, {"seed": 2}, 4, 2, DEFAULTS).load(path)


def test_save_refuses_another_entry_under_the_settings_prefix(tmp_path):
    key = f"{OBSERVATION_KEY_PREFIX}mean"
    agent = with_extra_checkpoint_entry(ReinforceAgent, key)(
        observation_dim=4, action_dim=2, config={"seed": 1}
    )
    with pytest.raises(ValueError, match=key):
        agent.save(str(tmp_path / "agent.npz"))


# --- Resume and playback --------------------------------------------------------------------------


def test_resume_refuses_a_run_saved_with_other_settings(tmp_path):
    def trainer(settings, episodes):
        env, eval_env = CartPoleEnv(seed=3, **settings), CartPoleEnv(seed=3, **settings)
        agent = CartPoleChallenge().make_agent("reinforce", {"seed": 1}, env)
        training = {"num_episodes": episodes, "max_steps_per_episode": 20, "save_frequency": 2,
                    "eval_frequency": 100, "checkpoint_dir": str(tmp_path)}  # fmt: skip
        return Trainer(env=env, agent=agent, config=training, eval_env=eval_env, eval_seed=1000)

    trainer(MASKED, 2).train()
    unmasked = trainer(DEFAULTS, 4)
    with pytest.raises(ValueError, match="mask_velocity"):
        unmasked.resume(str(tmp_path / "agent_episode_2.pt"))
    assert unmasked.episode_rewards == [], "nothing restored"


@pytest.mark.usefixtures("clear_sdl")
def test_record_refuses_a_checkpoint_trained_with_other_settings(tmp_path, monkeypatch):
    pytest.importorskip("moviepy")
    agent = {"type": "reinforce", "config": {"seed": 1}}
    masked = minimal_config(
        tmp_path, environment={"mask_velocity": True}, training={"save_frequency": 2}, agent=agent
    )
    run_main(["--config", masked], monkeypatch)
    checkpoint = tmp_path / "checkpoints" / "cartpole" / "reinforce" / "agent_episode_4.pt"
    assert checkpoint.exists(), "train.py wrote no checkpoint"

    unmasked_dir = tmp_path / "unmasked"
    unmasked_dir.mkdir()
    unmasked = minimal_config(unmasked_dir, agent=agent)
    with pytest.raises(ValueError, match=r"mask_velocity=true.*mask_velocity=false"):
        run_main(
            ["--config", unmasked, "--checkpoint", str(checkpoint), "--episodes", "1"]
            + ["--video-dir", str(tmp_path / "videos")],
            monkeypatch,
            module="record",
        )


def test_a_checkpoint_with_frame_deltas_is_refused_without_them_and_an_older_one_loads(tmp_path):
    """frame_deltas joined the record after 2026-11-02's checkpoints, which lack it: they read as
    false, so they still load; a checkpoint trained with it loads only with it."""

    def built(**settings):
        env = CartPoleEnv(seed=0, frame_stack=4, mask_velocity=True, **settings)
        try:
            return CartPoleChallenge().make_agent("reinforce", {"seed": 1}, env)
        finally:
            env.close()

    path = str(tmp_path / "deltas.pt")
    built(frame_deltas=True).save(path)
    with pytest.raises(ValueError, match=r"frame_deltas=true.*frame_deltas=false"):
        built().load(path)
    built(frame_deltas=True).load(path)

    older = str(tmp_path / "older.npz")
    built().save(older)
    _rewrite(older, lambda e: e.pop(f"{OBSERVATION_KEY_PREFIX}frame_deltas"))
    built().load(older)
    with pytest.raises(ValueError, match="frame_deltas"):
        built(frame_deltas=True).load(older)
