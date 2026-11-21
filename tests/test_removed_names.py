"""Names deprecated in 0.2.0 and removed in 0.3.0 stay removed; their replacements still work."""

import importlib
import warnings

import pytest

from rl_cartpole.benchmark import CartPoleChallenge
from rl_pendulum.benchmark import PendulumChallenge

PACKAGES = ["rl_cartpole", "rl_pendulum"]
REMOVED = ["make_env", "make_vec_env", "make_vec_env_from_config"]


@pytest.mark.parametrize("name", REMOVED)
@pytest.mark.parametrize("module", [*PACKAGES, *(f"{p}.environments" for p in PACKAGES)])
def test_the_legacy_factories_are_gone(module, name):
    namespace = vars(importlib.import_module(module))
    # The package re-exports the make_env *module* under that name in ``environments``.
    value = namespace.get(name)
    assert value is None or type(value).__name__ == "module", f"{module}.{name} is still there"
    assert name not in getattr(importlib.import_module(module), "__all__", [])


@pytest.mark.parametrize("package", PACKAGES)
def test_the_modules_define_none_of_them(package):
    module = importlib.import_module(f"{package}.environments.make_env")
    for name in ("make_vec_env", "make_vec_env_from_config"):
        assert not hasattr(module, name)
    assert not hasattr(module, "make_env")


def test_the_robustness_shim_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rl_core.training.robustness")


@pytest.mark.parametrize("package", PACKAGES)
def test_make_env_from_config_stays_and_does_not_warn(package):
    factory = importlib.import_module(f"{package}.environments").make_env_from_config
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factory({"seed": 0}).close()


@pytest.mark.parametrize("challenge", [CartPoleChallenge, PendulumChallenge])
def test_the_challenges_build_without_a_warning(challenge):
    """Training and the benchmark build through the challenge: no run warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        challenge().make_env({"seed": 0}, render_mode=None).close()


@pytest.mark.parametrize("package", PACKAGES)
def test_async_envs_is_no_longer_an_environment_setting(package):
    """Set to a value it is rejected like any unknown setting; left off (false, null) it is accepted."""
    factory = importlib.import_module(f"{package}.environments").make_env_from_config
    with pytest.raises(ValueError, match="does not support"):
        factory({"seed": 0, "async_envs": True})
    factory({"seed": 0, "async_envs": None}).close()
    factory({"seed": 0, "async_envs": False}).close()
