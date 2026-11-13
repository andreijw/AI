"""CartPole's and Pendulum's keyword and vector factories (``make_env``, ``make_vec_env``,
``make_vec_env_from_config``) are deprecated in rl-challenges 0.2.0 and removed in 0.3.0; the
config factory the challenges build with, ``make_env_from_config``, is not. This file goes with
them in 0.3.0."""

import contextlib
import importlib
import warnings

import pytest

import rl_cartpole.environments as cartpole
import rl_pendulum.environments as pendulum
from rl_cartpole.benchmark import CartPoleChallenge
from rl_pendulum.benchmark import PendulumChallenge

PACKAGES = ["rl_cartpole", "rl_pendulum"]
CALLS = {
    "make_env": lambda factory: factory(seed=0),
    "make_vec_env": lambda factory: factory(num_envs=2, seed=0, async_envs=False),
    "make_vec_env_from_config": lambda factory: factory({"seed": 0, "async_envs": False}, 2),
}


FACTORIES = {
    ("rl_cartpole", "make_env"): cartpole.make_env,
    ("rl_cartpole", "make_vec_env"): cartpole.make_vec_env,
    ("rl_cartpole", "make_vec_env_from_config"): cartpole.make_vec_env_from_config,
    ("rl_cartpole", "make_env_from_config"): cartpole.make_env_from_config,
    ("rl_pendulum", "make_env"): pendulum.make_env,
    ("rl_pendulum", "make_vec_env"): pendulum.make_vec_env,
    ("rl_pendulum", "make_vec_env_from_config"): pendulum.make_vec_env_from_config,
    ("rl_pendulum", "make_env_from_config"): pendulum.make_env_from_config,
}


def _factory(package: str, name: str):
    return FACTORIES[package, name]


@pytest.mark.parametrize("name", sorted(CALLS))
@pytest.mark.parametrize("package", PACKAGES)
def test_each_legacy_factory_warns_once_and_still_builds(package, name):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        built = CALLS[name](_factory(package, name))
    with contextlib.closing(built):
        assert built.observation_space is not None
    messages = [str(w.message) for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 1, f"{package}.{name} warned {len(messages)} times: {messages}"
    assert messages[0].startswith(
        f"{package}.{name} is deprecated since rl-challenges 0.2.0 and is removed in 0.3.0: "
    )
    assert record[0].filename == __file__, "attributed to the caller's line"


@pytest.mark.parametrize("package", PACKAGES)
def test_the_package_reexports_are_the_deprecated_factories(package):
    top = importlib.import_module(package)
    assert top.make_env is _factory(package, "make_env")
    assert top.make_vec_env is _factory(package, "make_vec_env")


@pytest.mark.parametrize("package", PACKAGES)
def test_make_env_from_config_does_not_warn(package):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _factory(package, "make_env_from_config")({"seed": 0}).close()


@pytest.mark.parametrize("challenge", [CartPoleChallenge, PendulumChallenge])
def test_the_challenges_build_without_a_warning(challenge):
    """Training and the benchmark build through the challenge: no run warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        challenge().make_env({"seed": 0}, render_mode=None).close()


@pytest.mark.parametrize("package", PACKAGES)
def test_the_replacement_is_named(package):
    with pytest.deprecated_call(match=r"Env\(\.\.\.\)|make_env_from_config"):
        _factory(package, "make_env")(seed=0).close()
    with pytest.deprecated_call(match="training.num_envs"):
        CALLS["make_vec_env"](_factory(package, "make_vec_env")).close()
