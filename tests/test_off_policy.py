"""OffPolicyAgent's own rules, on a minimal subclass with only critics: the class constants a
subclass must set, and the critic step for one critic and for twin critics."""

import copy
from typing import Any, ClassVar

import numpy as np
import pytest

from rl_core.agents.mlp import init_mlp
from rl_core.agents.off_policy import OffPolicyAgent
from rl_core.agents.optim import Adam
from tests.agent_internals import critic_loss_and_grads, critic_step, off_policy_networks

OBS_DIM, ACT_DIM, B = 3, 2, 6
DEFAULTS = {
    "hidden_sizes": [5, 4],
    "learning_rate": 0.01,
    "gamma": 0.9,
    "tau": 0.1,
    "batch_size": 4,
    "buffer_capacity": 50,
    "learning_starts": 0,
    "updates_per_step": 1.0,
}


class _Critics(OffPolicyAgent):
    """Critics only, each with its own Adam; records each gradient and each step it takes."""

    CRITICS: ClassVar[tuple[str, ...]] = ("critic",)

    def _build_networks(self, weights_rng: np.random.Generator) -> None:
        sizes = [self.observation_dim + self.action_dim, *self.hidden_sizes, 1]
        self._nets = {name: init_mlp(sizes, weights_rng) for name in self.CRITICS}
        self._adams = {
            name: Adam(params, self.max_grad_norm) for name, params in self._nets.items()
        }
        self.calls: list[tuple[str, str]] = []

    def _name_of(self, params: Any) -> str:
        """The critic ``params`` is; None means the default, the first."""
        if params is None:
            return self.CRITICS[0]
        return next(name for name, net in self._nets.items() if net is params)

    def _critic_loss_and_grads(self, obs, actions, targets, params=None):
        self.calls.append(("gradients", self._name_of(params)))
        return super()._critic_loss_and_grads(obs, actions, targets, params)

    def _apply(self, params, optimizer, grads):
        self.calls.append(("step", self._name_of(params)))
        super()._apply(params, optimizer, grads)

    def _networks(self):
        return self._nets

    def _optimizers_by_net(self):
        return self._adams

    def _critic_names(self):
        return self.CRITICS

    def _target_names(self):
        return []

    def _greedy_action(self, obs):
        return self.bias.copy()

    def _training_action(self, obs):
        return self.bias.copy()

    def _train_steps(self, num_updates):
        return {}


class OneCritic(_Critics):
    SHORT_NAME = "One"
    DEFAULTS = DEFAULTS


class TwinCritics(OneCritic):
    CRITICS = ("critic", "critic_2")


def _data():
    rng = np.random.default_rng(4)
    obs = rng.standard_normal((B, OBS_DIM))
    actions = rng.uniform(-1.0, 1.0, (B, ACT_DIM))
    return obs, actions, rng.standard_normal(B)


def test_a_subclass_without_its_class_constants_says_which():
    class NoName(_Critics):
        DEFAULTS = DEFAULTS

    class NoDefaults(_Critics):
        SHORT_NAME = "Bare"

    class ShortDefaults(_Critics):
        SHORT_NAME = "Short"
        DEFAULTS = {k: v for k, v in DEFAULTS.items() if k != "tau"}

    with pytest.raises(TypeError, match="NoName must set SHORT_NAME"):
        NoName(OBS_DIM, ACT_DIM, {})
    with pytest.raises(TypeError, match="NoDefaults must set DEFAULTS"):
        NoDefaults(OBS_DIM, ACT_DIM, {})
    with pytest.raises(TypeError, match="ShortDefaults.DEFAULTS lacks tau"):
        ShortDefaults(OBS_DIM, ACT_DIM, {})


def test_one_critic_steps_once_and_returns_its_own_metrics():
    agent = OneCritic(OBS_DIM, ACT_DIM, {"seed": 1})
    obs, actions, targets = _data()
    expected, _ = critic_loss_and_grads(copy.deepcopy(agent), obs, actions, targets)
    before = copy.deepcopy(off_policy_networks(agent))
    agent.calls.clear()

    assert critic_step(agent, obs, actions, targets) == expected
    assert agent.calls == [("gradients", "critic"), ("step", "critic")]
    assert not np.array_equal(off_policy_networks(agent)["critic"]["W0"], before["critic"]["W0"])


def test_twin_critics_take_every_gradient_before_any_step_and_combine_their_metrics():
    """Stepping the first critic before the second's gradient would be harmless only because the
    two are independent; the order is the rule TD3 (and SAC's twin critics) rely on."""
    agent = TwinCritics(OBS_DIM, ACT_DIM, {"seed": 1})
    obs, actions, targets = _data()
    reference = copy.deepcopy(agent)
    first, _ = critic_loss_and_grads(
        reference, obs, actions, targets, off_policy_networks(reference)["critic"]
    )
    second, _ = critic_loss_and_grads(
        reference, obs, actions, targets, off_policy_networks(reference)["critic_2"]
    )
    agent.calls.clear()

    metrics = critic_step(agent, obs, actions, targets)

    assert agent.calls == [
        ("gradients", "critic"),
        ("gradients", "critic_2"),
        ("step", "critic"),
        ("step", "critic_2"),
    ]
    assert metrics == {
        "critic_loss": (first["critic_loss"] + second["critic_loss"]) / 2.0,
        "mean_q": first["mean_q"],
        "critic_grad_norm": float(np.hypot(first["critic_grad_norm"], second["critic_grad_norm"])),
    }
    assert first["critic_loss"] != second["critic_loss"], "the critics should start apart"
    for name in ("critic", "critic_2"):
        assert not np.array_equal(
            off_policy_networks(agent)[name]["W0"], off_policy_networks(reference)[name]["W0"]
        ), f"{name} should have taken a step"
