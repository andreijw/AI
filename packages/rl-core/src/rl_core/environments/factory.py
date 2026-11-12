"""Building a challenge's environment wrapper from a config's ``environment`` section."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from rl_core.utils.config import read_env_settings


def env_from_config[EnvT](
    wrapper: Callable[..., EnvT], config: Mapping[str, Any], defaults: Mapping[str, Any]
) -> EnvT:
    """
    The wrapper for ``config``, built with the settings ``defaults`` lists (each with its default),
    the config's ``render_mode`` and ``seed``, and the environment id as ``env_name``.

    ``defaults`` must hold ``name`` (the environment id); every other key is passed to ``wrapper``
    under its config name, so it must match a keyword of the wrapper.

    Raises:
        ValueError: If the config sets a setting ``defaults`` does not list.
    """
    settings = read_env_settings(config, defaults, defaults["name"])
    return wrapper(
        env_name=settings.pop("name"),
        render_mode=config.get("render_mode"),
        seed=config.get("seed"),
        **settings,
    )
