"""The wrappers' schedulable settings: what a curriculum may change between episodes, and how."""

import importlib
import inspect
from contextlib import closing

import numpy as np
import pytest

from rl_acrobot.environments.acrobot_env import AcrobotEnv
from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_core.environments import BaseEnvWrapper, SchedulableEnv
from rl_mountaincar.environments.mountaincar_env import MountainCarEnv
from rl_navigation.environments.navigation_env import NavigationEnv
from rl_pendulum.environments.pendulum_env import PendulumEnv

# The module, not the make_env function its package re-exports under the same name.
factories = importlib.import_module("rl_cartpole.environments.make_env")

CARTPOLE_DR = {"length": [0.3, 0.9], "masspole": [0.05, 0.2]}
PENDULUM_DR = {"l": [0.8, 1.2], "g": [9.0, 10.5]}


def _wrapper_classes():
    """Every challenge's wrapper, the Box2D ones (LunarLander, BipedalWalker) included where Box2D
    is not installed: their classes import without it."""
    classes = [CartPoleEnv, PendulumEnv, AcrobotEnv, MountainCarEnv, NavigationEnv]
    box2d = [
        importlib.import_module("rl_lunarlander.environments.lunarlander_env").LunarLanderEnv,
        importlib.import_module("rl_bipedalwalker.environments.bipedalwalker_env").BipedalWalkerEnv,
    ]
    return classes + box2d


def _cartpole(**settings):
    defaults = {
        "seed": 3,
        "obs_noise_std": 0.05,
        "action_noise_prob": 0.2,
        "domain_randomization": CARTPOLE_DR,
    }
    return CartPoleEnv(**{**defaults, **settings})


def _pendulum(**settings):
    defaults = {
        "seed": 3,
        "obs_noise_std": 0.02,
        "action_noise_std": 0.3,
        "domain_randomization": PENDULUM_DR,
    }
    return PendulumEnv(**{**defaults, **settings})


@pytest.fixture
def cartpole():
    env = _cartpole()
    yield env
    env.close()


@pytest.fixture
def pendulum():
    env = _pendulum()
    yield env
    env.close()


# --- What each wrapper can schedule ---------------------------------------------------------------


def test_each_wrapper_lists_its_schedulable_settings():
    expected = {
        CartPoleEnv: ("obs_noise_std", "action_noise_prob", "domain_randomization"),
        PendulumEnv: ("obs_noise_std", "action_noise_std", "domain_randomization"),
        NavigationEnv: (
            "obs_noise_std",
            "action_noise_std",
            "action_noise_prob",
            "domain_randomization",
        ),
    }
    for cls in _wrapper_classes():
        assert expected.get(cls, ("obs_noise_std",)) == cls.SCHEDULABLE_SETTINGS, cls.__name__


def test_every_schedulable_setting_is_a_constructor_argument():
    for cls in _wrapper_classes():
        parameters = inspect.signature(cls.__init__).parameters
        missing = [name for name in cls.SCHEDULABLE_SETTINGS if name not in parameters]
        assert missing == [], f"{cls.__name__} schedules settings it cannot be built with"


def test_the_wrappers_are_schedulable_envs(cartpole, pendulum):
    assert isinstance(cartpole, SchedulableEnv)
    assert isinstance(pendulum, SchedulableEnv)


def test_scheduled_settings_are_the_settings_on_at_construction(cartpole):
    assert cartpole.scheduled_settings() == {
        "obs_noise_std": 0.05,
        "action_noise_prob": 0.2,
        "domain_randomization": {"length": (0.3, 0.9), "masspole": (0.05, 0.2)},
    }
    with closing(_cartpole(obs_noise_std=0.0, domain_randomization=None)) as env:
        assert env.scheduled_settings() == {"action_noise_prob": 0.2}


def test_scheduled_settings_use_canonical_names_in_construction_order():
    with closing(
        _pendulum(domain_randomization={"Length": [0.8, 1.2], "gravity": [9.0, 10.5]})
    ) as env:
        randomization = env.scheduled_settings()["domain_randomization"]
    assert list(randomization) == ["l", "g"]
    assert randomization == {"l": (0.8, 1.2), "g": (9.0, 10.5)}


def test_scheduled_settings_are_a_copy(cartpole):
    settings = cartpole.scheduled_settings()
    settings["obs_noise_std"] = 1.0
    settings["domain_randomization"]["length"] = (0.1, 0.1)
    assert cartpole.scheduled_settings()["obs_noise_std"] == 0.05
    assert cartpole.scheduled_settings()["domain_randomization"]["length"] == (0.3, 0.9)


# --- Applying settings -----------------------------------------------------------------------------


def test_applied_observation_noise_takes_effect_at_the_next_step(cartpole):
    cartpole.reset()
    cartpole.apply_settings({"obs_noise_std": 0.0})
    obs, *_ = cartpole.step(0)
    np.testing.assert_array_equal(
        obs, np.asarray(cartpole.env.unwrapped.state, dtype=obs.dtype), "noise 0 leaves the state"
    )


def test_applied_action_noise_takes_effect_at_the_next_step(cartpole):
    cartpole.reset()
    cartpole.apply_settings({"action_noise_prob": 1.0, "obs_noise_std": 0.0})
    before = np.array(cartpole.env.unwrapped.state)
    cartpole.step(0)  # flipped to 1: the cart is pushed right, so its velocity rises
    assert cartpole.env.unwrapped.state[1] > before[1], "a certain flip turns push-left into right"


def test_applied_observation_noise_already_noises_the_next_reset(cartpole):
    cartpole.apply_settings({"obs_noise_std": 0.0})
    obs, _ = cartpole.reset()
    np.testing.assert_array_equal(obs, np.asarray(cartpole.env.unwrapped.state, dtype=obs.dtype))


def test_pendulum_applies_action_noise_at_the_next_step(pendulum):
    """With no noise left anywhere, two Pendulums from one seed stay identical; with action noise
    left on in one of them, the torque it applies differs at the first step."""
    with closing(_pendulum()) as noisy:
        for env in (pendulum, noisy):
            env.apply_settings({"obs_noise_std": 0.0, "domain_randomization": PENDULUM_DR})
        pendulum.apply_settings({"action_noise_std": 0.0})
        pendulum.reset()
        noisy.reset()
        quiet_obs = pendulum.step([0.5])[0]
        noisy_obs = noisy.step([0.5])[0]
    assert not np.array_equal(quiet_obs, noisy_obs), "action noise 0.3 should move the pendulum"
    with closing(_pendulum(obs_noise_std=0.0)) as reference:
        reference.apply_settings({"action_noise_std": 0.0})
        reference.reset()
        np.testing.assert_array_equal(reference.step([0.5])[0], quiet_obs)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"action_noise_std": -1.0}, "action_noise_std must be a finite, non-negative number"),
        (
            {"domain_randomization": {"l": [0.0, 1.0], "g": [9.0, 10.0]}},
            "must be strictly positive",
        ),
        ({"domain_randomization": {"l": [0.8, 1.2]}}, "the same parameters as at construction"),
        ({"frame_deltas": True}, "cannot change frame_deltas between episodes"),
    ],
)
def test_pendulum_apply_settings_refuses(pendulum, settings, message):
    before = pendulum.scheduled_settings()
    with pytest.raises(ValueError, match=message):
        pendulum.apply_settings(settings)
    assert pendulum.scheduled_settings() == before


def test_pendulum_action_noise_off_at_construction_cannot_be_switched_on():
    with (
        closing(_pendulum(action_noise_std=0.0)) as env,
        pytest.raises(ValueError, match="action_noise_std was off"),
    ):
        env.apply_settings({"action_noise_std": 0.1})


def test_applied_randomization_is_drawn_at_the_next_reset(cartpole):
    cartpole.reset()
    cartpole.apply_settings(
        {"domain_randomization": {"length": [0.5, 0.5], "masspole": [0.1, 0.1]}}
    )
    assert cartpole.env.unwrapped.length != 0.5, "the episode under way keeps its physics"
    cartpole.reset()
    base = cartpole.env.unwrapped
    assert (base.length, base.masspole) == (0.5, 0.1)
    assert base.polemass_length == pytest.approx(0.05), "derived physics follows the new draw"
    assert base.total_mass == pytest.approx(base.masscart + 0.1)


def test_applied_randomization_keeps_the_construction_order_of_draws():
    """Pendulum draws in its settings' order: a mapping given in another order, or under long
    names, must still draw each parameter from the same number as before."""
    ranges = {"l": [0.9, 1.1], "g": [9.5, 10.0]}
    physics = []
    for given in (ranges, {"gravity": [9.5, 10.0], "length": [0.9, 1.1]}):
        with closing(_pendulum()) as env:
            env.apply_settings({"domain_randomization": given})
            env.reset()
            physics.append((env.env.unwrapped.l, env.env.unwrapped.g))
    assert physics[0] == physics[1]


def test_applying_the_current_settings_changes_nothing(cartpole):
    """Applying settings draws no random numbers, so a run whose settings never move is the run
    without a curriculum."""
    with closing(_cartpole()) as untouched:
        cartpole.apply_settings(cartpole.scheduled_settings())
        for env in (cartpole, untouched):
            env.reset()
        np.testing.assert_array_equal(cartpole.step(1)[0], untouched.step(1)[0])
        assert cartpole.rng_state().keys() == untouched.rng_state().keys()
        for key, value in cartpole.rng_state().items():
            np.testing.assert_array_equal(value, untouched.rng_state()[key], err_msg=key)


# --- Refusals ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"frame_stack": 2}, "cannot change frame_stack between episodes"),
        ({"obs_noise_std": -0.1}, "obs_noise_std must be a finite, non-negative number"),
        ({"action_noise_prob": 1.5}, r"action_noise_prob must be in the interval \[0, 1\]"),
        ({"domain_randomization": {"length": [0.9, 0.3]}}, "has min > max"),
        (
            {"domain_randomization": {"length": [0.3, 0.9]}},
            "the same parameters as at construction",
        ),
        (
            {"domain_randomization": {**CARTPOLE_DR, "gravity": [9.0, 10.0]}},
            "the same parameters as at construction",
        ),
        ({"domain_randomization": None}, "the same parameters as at construction"),
    ],
)
def test_apply_settings_refuses(cartpole, settings, message):
    with pytest.raises(ValueError, match=message):
        cartpole.apply_settings(settings)


def test_a_setting_off_at_construction_cannot_be_switched_on():
    with closing(_cartpole(obs_noise_std=0.0, domain_randomization=None)) as env:
        with pytest.raises(
            ValueError, match="obs_noise_std was off when the environment was built"
        ):
            env.apply_settings({"obs_noise_std": 0.1})
        with pytest.raises(ValueError, match="domain_randomization was off"):
            env.apply_settings({"domain_randomization": {"length": [0.4, 0.6]}})


def test_a_refused_call_changes_no_setting(cartpole):
    before = cartpole.scheduled_settings()
    with pytest.raises(ValueError):
        cartpole.apply_settings({"obs_noise_std": 0.01, "action_noise_prob": 2.0})
    assert cartpole.scheduled_settings() == before


def test_pendulum_refuses_a_parameter_given_twice_under_two_names():
    with pytest.raises(ValueError, match="'g' is given twice"):
        _pendulum(domain_randomization={"g": [9.0, 10.0], "gravity": [9.0, 10.0]})


def test_check_setting_is_the_constructors_rule():
    """The class-level check is what both the constructor and apply_settings apply, so the
    CartPole factory can check a config's ranges before opening any environment."""
    assert CartPoleEnv.check_setting("domain_randomization", {"length": [0.3, 0.9]}) == {
        "length": (0.3, 0.9)
    }
    assert PendulumEnv.check_setting("domain_randomization", {"Gravity": [9, 10]}) == {
        "g": (9.0, 10.0)
    }
    assert BaseEnvWrapper.check_setting("obs_noise_std", 1) == 1.0
    with pytest.raises(ValueError, match="cannot change frame_stack between episodes"):
        CartPoleEnv.check_setting("frame_stack", 2)


def _deprecated_vec_env_from_config(config):
    """The vector factory, deprecated in rl-challenges 0.2.0 (removed with this case in 0.3.0)."""
    with pytest.deprecated_call():
        return factories.make_vec_env_from_config(config)


@pytest.mark.parametrize(
    "factory",
    [factories.make_env_from_config, _deprecated_vec_env_from_config],
    ids=["make_env_from_config", "make_vec_env_from_config"],
)
def test_the_cartpole_factories_check_ranges_before_building_anything(factory, monkeypatch):
    """The factories check a config's randomization with the wrapper's own rule, in this process,
    before any environment (or worker process) is built."""

    def must_not_build(*args, **kwargs):
        raise AssertionError("an environment was built from an invalid config")

    class NotBuilt(CartPoleEnv):
        """CartPoleEnv's own setting checks, without building one."""

        def __init__(self, *args, **kwargs):
            must_not_build()

    monkeypatch.setattr(factories, "CartPoleEnv", NotBuilt)
    monkeypatch.setattr(factories, "SyncVectorEnv", must_not_build)
    monkeypatch.setattr(factories, "AsyncVectorEnv", must_not_build)
    with pytest.raises(ValueError, match="Unsupported domain randomization parameter 'foo'"):
        factory({"domain_randomization": {"foo": [1.0, 2.0]}})


@pytest.mark.parametrize("param", ["gravity", "masscart", "masspole", "length"])
def test_cartpole_refuses_a_physical_range_that_reaches_zero(param):
    """CartPole's dynamics divide by the pole's length and the total mass, so a range reaching 0
    (a curriculum's start, or the target) could draw a zero and give NaN or inf states. Gravity
    follows the same rule, as Pendulum's does."""
    with pytest.raises(ValueError, match=f"Physical parameter '{param}' must be strictly positive"):
        CartPoleEnv(seed=0, domain_randomization={param: [0.0, 1.0]})
