"""Tests for frame stacking and velocity masking of observations."""

import gymnasium as gym
import numpy as np
import pytest

from rl_acrobot.environments.make_env import make_env_from_config as acrobot_from_config
from rl_cartpole.environments import CartPoleEnv
from rl_cartpole.environments import make_env_from_config as cartpole_from_config
from rl_core.agents import ReinforceAgent
from rl_core.environments.observation import (
    FrameStack,
    differenced_space,
    frame_differences,
    mask_entries,
    masked_space,
    stacked_space,
)
from rl_lunarlander.environments import (
    make_continuous_env_from_config as lunarlander_continuous_from_config,
)
from rl_lunarlander.environments import make_env_from_config as lunarlander_from_config
from rl_mountaincar.environments import (
    make_continuous_env_from_config as mountaincar_continuous_from_config,
)
from rl_mountaincar.environments import make_env_from_config as mountaincar_from_config
from rl_pendulum.environments import make_env_from_config as pendulum_from_config
from tests.helpers import challenge_env_factories, env_marks, minimal_config, run_main, vector_env


def _obs(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


# --- FrameStack ---------------------------------------------------------------------------------


def test_frame_stack_reset_repeats_the_first_observation():
    stack = FrameStack(3)
    out = stack.reset(_obs(1.0, 2.0))
    np.testing.assert_array_equal(
        out,
        _obs(1.0, 2.0, 1.0, 2.0, 1.0, 2.0),
        err_msg="reset should pad the stack with copies of the first observation, not zeros",
    )


def test_frame_stack_push_keeps_the_last_k_oldest_first():
    stack = FrameStack(3)
    stack.reset(_obs(0.0, 0.5))
    stack.push(_obs(1.0, 1.5))
    stack.push(_obs(2.0, 2.5))
    out = stack.push(_obs(3.0, 3.5))
    # After reset(o0) and three pushes, the window is o1, o2, o3; o0 has dropped out.
    np.testing.assert_array_equal(
        out,
        _obs(1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
        err_msg="the stack should hold the last 3 observations, oldest first",
    )


def test_frame_stack_reset_forgets_the_previous_episode():
    stack = FrameStack(2)
    stack.reset(_obs(9.0))
    stack.push(_obs(8.0))
    out = stack.reset(_obs(1.0))
    np.testing.assert_array_equal(
        out, _obs(1.0, 1.0), err_msg="no frame from the previous episode may survive a reset"
    )


def test_frame_stack_output_does_not_alias_the_ring():
    stack = FrameStack(2)
    first = stack.reset(_obs(1.0))
    kept = first.copy()
    second = stack.push(_obs(2.0))
    np.testing.assert_array_equal(
        first, kept, err_msg="a returned stack must not change when later frames are pushed"
    )
    second[:] = -1.0
    np.testing.assert_array_equal(
        stack.push(_obs(3.0)),
        _obs(2.0, 3.0),
        err_msg="writing into a returned stack must not change the stack's own frames",
    )


def test_frame_stack_does_not_alias_the_pushed_observation():
    stack = FrameStack(2)
    obs = _obs(1.0)
    stack.reset(obs)
    obs[0] = 5.0
    np.testing.assert_array_equal(
        stack.push(_obs(2.0)),
        _obs(1.0, 2.0),
        err_msg="the stack must copy an observation, not keep a reference the caller may change",
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_frame_stack_keeps_the_observation_dtype(dtype):
    stack = FrameStack(2)
    obs = np.array([1.0, 2.0], dtype=dtype)
    assert stack.reset(obs).dtype == dtype, "reset changed the observation's dtype"
    assert stack.push(obs).dtype == dtype, "push changed the observation's dtype"


def test_frame_stack_of_one_returns_the_observation_values():
    stack = FrameStack(1)
    np.testing.assert_array_equal(
        stack.reset(_obs(1.0, 2.0)), _obs(1.0, 2.0), err_msg="a stack of one is the observation"
    )
    np.testing.assert_array_equal(
        stack.push(_obs(3.0, 4.0)), _obs(3.0, 4.0), err_msg="a stack of one is the observation"
    )


def test_frame_stack_push_before_reset_is_refused():
    with pytest.raises(RuntimeError, match="reset"):
        FrameStack(2).push(_obs(1.0))


def test_frame_stack_refuses_a_frame_of_another_shape():
    stack = FrameStack(2)
    stack.reset(_obs(1.0, 2.0))
    with pytest.raises(ValueError, match="shape"):
        stack.push(_obs(1.0, 2.0, 3.0))


@pytest.mark.parametrize("size", [0, -1, 2.5, True, "4", None])
def test_frame_stack_refuses_a_size_that_is_not_a_positive_int(size):
    with pytest.raises(ValueError, match="frame_stack"):
        FrameStack(size)


# --- Spaces -------------------------------------------------------------------------------------


def _box() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        high=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        dtype=np.float32,
    )


def test_stacked_space_tiles_the_bounds():
    space = stacked_space(_box(), 2)
    assert gym.spaces.flatdim(space) == 6, "a stack of 2 three-entry frames should have 6 entries"
    np.testing.assert_array_equal(
        space.low,
        _obs(-1.0, -2.0, -3.0, -1.0, -2.0, -3.0),
        err_msg="lower bounds not tiled per frame",
    )
    np.testing.assert_array_equal(
        space.high, _obs(1.0, 2.0, 3.0, 1.0, 2.0, 3.0), err_msg="upper bounds not tiled per frame"
    )
    assert space.dtype == np.float32, "stacking should keep the space's dtype"


def test_stacked_space_holds_what_the_stack_returns():
    space = stacked_space(_box(), 3)
    stack = FrameStack(3)
    stack.reset(_obs(0.5, -1.5, 2.5))
    assert space.contains(stack.push(_obs(-0.5, 1.5, -2.5))), (
        "the stacked space must contain the stacked observation"
    )


def test_masked_space_pins_masked_entries_at_zero():
    space = masked_space(_box(), (1,))
    np.testing.assert_array_equal(space.low, _obs(-1.0, 0.0, -3.0))
    np.testing.assert_array_equal(space.high, _obs(1.0, 0.0, 3.0))
    assert space.shape == (3,), "masking should keep the observation's length"


def test_masked_space_leaves_the_original_unchanged():
    original = _box()
    masked_space(original, (0, 2))
    np.testing.assert_array_equal(original.low, _obs(-1.0, -2.0, -3.0))
    np.testing.assert_array_equal(original.high, _obs(1.0, 2.0, 3.0))


def test_mask_entries_zeroes_only_the_listed_entries_and_copies():
    obs = _obs(1.0, 2.0, 3.0, 4.0)
    out = mask_entries(obs, (1, 3))
    np.testing.assert_array_equal(out, _obs(1.0, 0.0, 3.0, 0.0))
    np.testing.assert_array_equal(
        obs, _obs(1.0, 2.0, 3.0, 4.0), err_msg="masking must not write into the caller's array"
    )
    assert out.dtype == np.float32, "masking should keep the observation's dtype"


@pytest.mark.parametrize("indices", [(), (3,), (-1,)])
def test_masked_space_refuses_indices_outside_the_observation(indices):
    with pytest.raises(ValueError, match="mask"):
        masked_space(_box(), indices)


def test_spaces_refuse_a_space_that_is_not_a_flat_box():
    with pytest.raises(ValueError, match="Box"):
        stacked_space(gym.spaces.Discrete(3), 2)
    with pytest.raises(ValueError, match="Box"):
        masked_space(gym.spaces.Box(low=0.0, high=1.0, shape=(2, 2)), (0,))


# --- Frame differences (frame_deltas) -------------------------------------------------------------


def test_frame_differences_put_the_newest_frame_first_then_the_differences_newest_first():
    # Frames, oldest first: (1, 10), (4, 20), (9, 25).
    stack = _obs(1.0, 10.0, 4.0, 20.0, 9.0, 25.0)
    out = frame_differences(stack, 3)
    # Newest frame (9, 25); then (9, 25) - (4, 20) = (5, 5); then (4, 20) - (1, 10) = (3, 10).
    np.testing.assert_array_equal(out, _obs(9.0, 25.0, 5.0, 5.0, 3.0, 10.0))
    assert out.dtype == stack.dtype, "the differences keep the stack's dtype"


def test_frame_differences_of_a_padded_stack_are_zero():
    stack = FrameStack(3).reset(_obs(0.5, -1.5))
    np.testing.assert_array_equal(frame_differences(stack, 3), _obs(0.5, -1.5, 0.0, 0.0, 0.0, 0.0))


def test_frame_differences_invert_back_to_the_stack():
    rng = np.random.default_rng(1)
    stack = rng.normal(size=12)
    out = frame_differences(stack, 4).reshape(4, 3)
    frames = [out[0]]
    for diff in out[1:]:
        frames.append(frames[-1] - diff)
    np.testing.assert_allclose(np.concatenate(frames[::-1]), stack, rtol=1e-12)


@pytest.mark.parametrize("size", [1, 0, 2.5])
def test_frame_differences_need_a_stack_of_two_or_more(size):
    with pytest.raises(ValueError, match="frame_deltas"):
        frame_differences(_obs(1.0, 2.0, 3.0, 4.0), size)
    with pytest.raises(ValueError, match="frame_deltas"):
        differenced_space(_box(), size)


def test_frame_differences_refuse_a_stack_that_does_not_split_into_frames():
    with pytest.raises(ValueError, match="5"):
        frame_differences(_obs(1.0, 2.0, 3.0, 4.0, 5.0), 2)


def test_differenced_space_keeps_the_newest_frames_bounds_and_widens_the_differences():
    one = gym.spaces.Box(
        low=np.array([-1.0, -np.inf, 0.0], dtype=np.float32),
        high=np.array([3.0, np.inf, 0.0], dtype=np.float32),
        dtype=np.float32,
    )
    space = differenced_space(one, 3)
    # Newest frame: the frame's own bounds; each difference: [low - high, high - low].
    np.testing.assert_array_equal(space.low, _obs(-1, -np.inf, 0, -4, -np.inf, 0, -4, -np.inf, 0))
    np.testing.assert_array_equal(space.high, _obs(3, np.inf, 0, 4, np.inf, 0, 4, np.inf, 0))
    assert space.dtype == np.float32


def test_differenced_space_holds_what_the_differences_return():
    space = differenced_space(_box(), 3)
    stack = FrameStack(3)
    stack.reset(_obs(1.0, -2.0, 3.0))
    stack.push(_obs(-1.0, 2.0, -3.0))
    assert space.contains(frame_differences(stack.push(_obs(1.0, -2.0, 3.0)), 3)), (
        "the differenced space must contain the extreme differences of in-bounds frames"
    )


# --- The environment wrappers -------------------------------------------------------------------

# Each challenge's config factory, environment id and velocity entries in Gymnasium's layout (a
# challenge whose wrapper lists none refuses mask_velocity, as BipedalWalker's does).
WRAPPER_CASES = [
    pytest.param(cartpole_from_config, "CartPole-v1", (1, 3), id="cartpole"),
    pytest.param(pendulum_from_config, "Pendulum-v1", (2,), id="pendulum"),
    pytest.param(acrobot_from_config, "Acrobot-v1", (4, 5), id="acrobot"),
    pytest.param(mountaincar_from_config, "MountainCar-v0", (1,), id="mountaincar"),
    pytest.param(
        mountaincar_continuous_from_config,
        "MountainCarContinuous-v0",
        (1,),
        id="mountaincar-continuous",
    ),
    pytest.param(
        lunarlander_from_config,
        "LunarLander-v3",
        (2, 3, 5),
        id="lunarlander",
        marks=env_marks("LunarLander-v3"),
    ),
    pytest.param(
        lunarlander_continuous_from_config,
        "LunarLanderContinuous-v3",
        (2, 3, 5),
        id="lunarlander-continuous",
        marks=env_marks("LunarLanderContinuous-v3"),
    ),
]


def _rollout(env, steps: int) -> list[np.ndarray]:
    """The reset observation and up to ``steps`` more, from actions the env's seeded space draws."""
    observations = [env.reset()[0]]
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        observations.append(obs)
        if terminated or truncated:
            break
    return observations


@pytest.mark.parametrize(("from_config", "env_id", "velocity"), WRAPPER_CASES)
def test_mask_velocity_zeroes_exactly_the_velocity_entries(from_config, env_id, velocity):
    plain = from_config({"name": env_id, "seed": 3})
    masked = from_config({"name": env_id, "seed": 3, "mask_velocity": True})
    try:
        kept = [i for i in range(plain.observation_space.shape[0]) if i not in velocity]
        for raw, obs in zip(_rollout(plain, 5), _rollout(masked, 5), strict=True):
            np.testing.assert_array_equal(
                obs[list(velocity)], 0.0, err_msg=f"{env_id}: velocity entries should be zero"
            )
            np.testing.assert_array_equal(
                obs[kept], raw[kept], err_msg=f"{env_id}: masking changed a non-velocity entry"
            )
            assert masked.observation_space.contains(obs), "a masked observation left its space"
        assert masked.observation_space.shape == plain.observation_space.shape, (
            "masking should keep the observation's length"
        )
    finally:
        plain.close()
        masked.close()


@pytest.mark.parametrize("make_env", challenge_env_factories())
def test_frame_stack_stacks_the_observations_of_every_challenge(make_env):
    plain = make_env(seed=4)
    stacked = make_env(seed=4, frame_stack=3)
    env_id = plain.env_name
    try:
        d = plain.observation_space.shape[0]
        assert gym.spaces.flatdim(stacked.observation_space) == 3 * d, (
            f"{env_id}: a stack of 3 should triple the observation's length, which sizes the agent"
        )
        raw = _rollout(plain, 4)
        for t, obs in enumerate(_rollout(stacked, 4)):
            # Frame j of step t is raw step t - 2 + j, with the reset observation repeated before it.
            expected = np.concatenate([raw[max(t - 2 + j, 0)] for j in range(3)])
            np.testing.assert_array_equal(obs, expected, err_msg=f"{env_id}: wrong stack at {t}")
            assert stacked.observation_space.contains(obs), "a stacked observation left its space"
    finally:
        plain.close()
        stacked.close()


def test_a_frame_keeps_the_noise_it_was_observed_with():
    env = cartpole_from_config({"seed": 5, "obs_noise_std": 0.1, "frame_stack": 2})
    try:
        observations = _rollout(env, 5)
        for before, after in zip(observations[1:], observations[2:], strict=False):
            np.testing.assert_array_equal(
                after[:4],
                before[4:],
                err_msg="the older frame of a stack must be the newer frame of the step before, "
                "noise included: noise is drawn once per observation, before stacking",
            )
    finally:
        env.close()


def test_masked_entries_stay_zero_with_observation_noise():
    env = cartpole_from_config({"seed": 6, "obs_noise_std": 0.5, "mask_velocity": True})
    try:
        for obs in _rollout(env, 5):
            np.testing.assert_array_equal(
                obs[[1, 3]], 0.0, err_msg="the mask applies after the noise, so it leaves zeros"
            )
    finally:
        env.close()


def test_frame_stack_restarts_at_every_reset():
    env = cartpole_from_config({"seed": 7, "frame_stack": 3})
    try:
        _rollout(env, 10)
        obs, _ = env.reset()
        np.testing.assert_array_equal(
            obs,
            np.tile(obs[:4], 3),
            err_msg="after a reset the stack should hold the new start state three times",
        )
    finally:
        env.close()


class _SameArray(gym.ObservationWrapper):
    """Returns one held array for every observation, as a caching wrapper might."""

    def __init__(self, env):
        super().__init__(env)
        self.held = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype)

    def observation(self, observation):
        self.held[:] = observation
        return self.held


def test_mask_velocity_does_not_write_into_the_environments_array():
    env = cartpole_from_config({"seed": 8, "mask_velocity": True})
    same = _SameArray(env.env)
    env.wrap_env(same)
    try:
        env.reset()
        env.step(0)
        assert np.all(same.held[[1, 3]] != 0.0), (
            "the mask must copy: it zeroed the velocities in the array the environment returned"
        )
    finally:
        env.close()


def test_mask_velocity_refuses_an_environment_with_no_known_velocity_layout():
    with pytest.raises(ValueError, match=r"mask_velocity.*Acrobot-v1"):
        CartPoleEnv(env_name="Acrobot-v1", mask_velocity=True)


@pytest.mark.parametrize(("from_config", "env_id", "velocity"), WRAPPER_CASES)
def test_without_the_keys_the_spaces_are_the_environments_own(from_config, env_id, velocity):
    env = from_config({"name": env_id, "seed": 9})
    raw = gym.make(env_id)
    try:
        assert env.observation_space == raw.observation_space, (
            "with frame_stack and mask_velocity unset, the observation space must not change"
        )
    finally:
        env.close()
        raw.close()


@pytest.mark.parametrize(("from_config", "env_id", "velocity"), WRAPPER_CASES)
def test_frame_deltas_re_express_the_stack_of_every_challenge(from_config, env_id, velocity):
    stacked = from_config({"name": env_id, "seed": 4, "frame_stack": 3, "mask_velocity": True})
    deltas = from_config(
        {"name": env_id, "seed": 4, "frame_stack": 3, "mask_velocity": True, "frame_deltas": True}
    )
    try:
        d = stacked.observation_space.shape[0] // 3
        assert deltas.observation_space.shape == stacked.observation_space.shape
        for t, (stack, obs) in enumerate(
            zip(_rollout(stacked, 4), _rollout(deltas, 4), strict=True)
        ):
            np.testing.assert_array_equal(
                obs, frame_differences(stack, 3), err_msg=f"{env_id}: wrong differences at {t}"
            )
            for block in range(3):
                np.testing.assert_array_equal(
                    obs[[block * d + i for i in velocity]], 0.0, err_msg="a masked entry moved"
                )
            assert deltas.observation_space.contains(obs), f"{env_id}: left its space at {t}"
        np.testing.assert_array_equal(
            _rollout(deltas, 0)[0][d:], 0.0, err_msg="at a reset every difference is zero"
        )
    finally:
        stacked.close()
        deltas.close()


@pytest.mark.parametrize("stack", [None, 1])
def test_frame_deltas_without_a_stack_are_refused_naming_both_keys(stack, monkeypatch):
    closed = []
    original_close = gym.Wrapper.close  # gym.make returns the environment inside wrappers
    monkeypatch.setattr(
        gym.Wrapper, "close", lambda self: closed.append(self) or original_close(self)
    )
    config = (
        {"frame_deltas": True} if stack is None else {"frame_deltas": True, "frame_stack": stack}
    )
    with pytest.raises(ValueError, match="frame_deltas.*frame_stack"):
        cartpole_from_config(config)
    assert closed, "a refused frame_deltas must close the environment it opened"


@pytest.mark.parametrize("value", ["yes", "false", 1])
def test_a_bad_frame_deltas_is_refused_naming_the_key(value):
    with pytest.raises(ValueError, match="frame_deltas"):
        cartpole_from_config({"frame_stack": 2, "frame_deltas": value})


@pytest.mark.parametrize("value", [0, -2, 2.5, True, "4"])
def test_a_bad_frame_stack_is_refused_naming_the_key(value):
    with pytest.raises(ValueError, match="frame_stack"):
        cartpole_from_config({"frame_stack": value})


@pytest.mark.parametrize("value", ["yes", "false", 1, 0])
def test_a_bad_mask_velocity_is_refused_naming_the_key(value):
    with pytest.raises(ValueError, match="mask_velocity"):
        cartpole_from_config({"mask_velocity": value})


# --- Vectorized environments --------------------------------------------------------------------

VEC_CASES = [
    pytest.param(cartpole_from_config, 4, (1, 3), False, id="cartpole-sync"),
    pytest.param(cartpole_from_config, 4, (1, 3), True, id="cartpole-async"),
    pytest.param(pendulum_from_config, 3, (2,), False, id="pendulum-sync"),
]


@pytest.mark.parametrize(("from_config", "d", "velocity", "async_envs"), VEC_CASES)
def test_vector_envs_stack_and_mask_each_sub_environment(from_config, d, velocity, async_envs):
    config = {"seed": 10, "frame_stack": 3, "mask_velocity": True}
    vec_env = vector_env(from_config, config, num_envs=2, async_envs=async_envs)
    try:
        obs, _ = vec_env.reset()
        assert obs.shape == (2, 3 * d), "each sub-environment should return a stack of 3 frames"
        for row in obs:
            np.testing.assert_array_equal(row, np.tile(row[:d], 3), err_msg="reset should pad")
        for _ in range(3):
            obs, *_ = vec_env.step(vec_env.action_space.sample())
            for frame in range(3):
                np.testing.assert_array_equal(
                    obs[:, [frame * d + i for i in velocity]],
                    0.0,
                    err_msg="every frame's velocity entries should be zero",
                )
    finally:
        vec_env.close()


def test_a_vectorized_auto_reset_restarts_the_stack():
    vec_env = vector_env(cartpole_from_config, {"seed": 11, "frame_stack": 3}, num_envs=2)
    try:
        vec_env.reset()
        vec_env.action_space.seed(11)
        for _ in range(500):
            _, _, terminated, truncated, _ = vec_env.step(vec_env.action_space.sample())
            done = np.logical_or(terminated, truncated)
            if done.any():
                break
        assert done.any(), "no sub-environment ended in 500 random steps"
        # Gymnasium's default next-step auto-reset resets the ended sub-environment on this step.
        obs, *_ = vec_env.step(vec_env.action_space.sample())
        for row in obs[done]:
            np.testing.assert_array_equal(
                row,
                np.tile(row[:4], 3),
                err_msg="an auto-reset sub-environment should start a fresh stack, with no frame "
                "from the episode that ended",
            )
    finally:
        vec_env.close()


def test_vector_envs_take_frame_deltas_and_an_auto_reset_zeroes_the_differences():
    config = {"seed": 11, "frame_stack": 3, "mask_velocity": True, "frame_deltas": True}
    vec_env = vector_env(cartpole_from_config, config, num_envs=2)
    try:
        obs, _ = vec_env.reset()
        np.testing.assert_array_equal(obs[:, 4:], 0.0, err_msg="a reset's differences are zero")
        vec_env.action_space.seed(11)
        for _ in range(500):
            _, _, terminated, truncated, _ = vec_env.step(vec_env.action_space.sample())
            done = np.logical_or(terminated, truncated)
            if done.any():
                break
        assert done.any(), "no sub-environment ended in 500 random steps"
        obs, *_ = vec_env.step(vec_env.action_space.sample())  # the auto-reset step
        np.testing.assert_array_equal(
            obs[done][:, 4:], 0.0, err_msg="an auto-reset must start fresh differences"
        )
        np.testing.assert_array_equal(obs[:, [1, 3, 5, 7, 9, 11]], 0.0, err_msg="masked moved")
    finally:
        vec_env.close()


# --- Through the scripts ------------------------------------------------------------------------


@pytest.mark.usefixtures("clear_sdl")
@pytest.mark.parametrize("deltas", [False, True], ids=["stack", "stack-deltas"])
def test_train_and_record_run_with_a_stacked_masked_observation(deltas, tmp_path, monkeypatch):
    """train.py sizes the agent for the stack and checkpoints it; record.py plays that checkpoint
    back with the same environment section."""
    pytest.importorskip("moviepy")
    config_path = minimal_config(
        tmp_path,
        environment={"frame_stack": 4, "mask_velocity": True, "frame_deltas": deltas},
        training={"save_frequency": 2},
        agent={"type": "reinforce", "config": {"seed": 1}},
    )

    run_main(["--config", config_path], monkeypatch)
    checkpoint = tmp_path / "checkpoints" / "cartpole" / "reinforce" / "agent_episode_4.pt"
    agent = ReinforceAgent(observation_dim=16, action_dim=2, config={})
    agent.load(str(checkpoint))
    # REINFORCE loads any size; acting on 16 inputs fails if train.py sized the agent for 4.
    assert agent.select_action(np.zeros(16), training=False) in (0, 1)

    video_dir = tmp_path / "videos"
    run_main(
        ["--config", config_path, "--checkpoint", str(checkpoint), "--episodes", "1"]
        + ["--video-dir", str(video_dir)],
        monkeypatch,
        module="record",
    )
    assert (video_dir / "cartpole-playback-full.mp4").exists(), "record.py wrote no video"


def test_frame_stack_refuses_an_environment_without_a_flat_box_observation(monkeypatch):
    closed = []
    original_close = gym.Wrapper.close  # gym.make returns the environment inside wrappers
    monkeypatch.setattr(
        gym.Wrapper, "close", lambda self: closed.append(self) or original_close(self)
    )
    with pytest.raises(ValueError, match="Box"):
        CartPoleEnv(env_name="FrozenLake-v1", frame_stack=2)
    assert closed, "the environment opened before the refusal should be closed"
