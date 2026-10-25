"""DDPG's gradient step written in PyTorch, as a reference for the NumPy agent's.

The same step as one iteration of DDPGAgent.update: Bellman targets from the target networks, a
critic Adam step on mean 1/2 (Q - y)^2, an actor Adam step on -mean Q(s, mu(s)) against the updated
critic, then Polyak averaging of both targets. ReLU hidden layers, tanh output scaled to the action
bounds, Adam (0.9, 0.999, 1e-8), no gradient clipping. time_torch.py times it;
tests/test_ddpg_torch_reference.py checks it against the NumPy agent. Results:
docs/pytorch_jetson_timing.md.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

NETWORKS = ("actor", "critic", "target_actor", "target_critic")


@dataclass
class TorchDDPG:
    """The four networks and two optimizers of one DDPG agent, with its fixed settings."""

    actor: nn.Sequential
    critic: nn.Sequential
    target_actor: nn.Sequential
    target_critic: nn.Sequential
    actor_opt: torch.optim.Adam
    critic_opt: torch.optim.Adam
    gamma: float
    tau: float
    scale: float
    bias: float


def _mlp(sizes: list[int], device: str, dtype: torch.dtype) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers).to(device, dtype)


def build(
    obs_dim: int,
    action_dim: int,
    hidden: list[int],
    device: str,
    dtype: torch.dtype,
    *,
    learning_rate: float = 1e-3,
    gamma: float = 0.98,
    tau: float = 0.005,
    action_low: float = -2.0,
    action_high: float = 2.0,
    capturable: bool = False,
) -> TorchDDPG:
    """Fresh networks; ``capturable`` Adam is needed to replay the step as a CUDA graph."""
    actor = _mlp([obs_dim, *hidden, action_dim], device, dtype)
    critic = _mlp([obs_dim + action_dim, *hidden, 1], device, dtype)
    target_actor, target_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    for param in [*target_actor.parameters(), *target_critic.parameters()]:
        param.requires_grad_(False)
    return TorchDDPG(
        actor=actor,
        critic=critic,
        target_actor=target_actor,
        target_critic=target_critic,
        actor_opt=torch.optim.Adam(
            actor.parameters(), lr=learning_rate, eps=1e-8, capturable=capturable
        ),
        critic_opt=torch.optim.Adam(
            critic.parameters(), lr=learning_rate, eps=1e-8, capturable=capturable
        ),
        gamma=gamma,
        tau=tau,
        scale=(action_high - action_low) / 2,
        bias=(action_high + action_low) / 2,
    )


def _linears(net: nn.Sequential) -> list[nn.Linear]:
    return [layer for layer in net if isinstance(layer, nn.Linear)]


def load_weights(agent: TorchDDPG, weights: dict[str, np.ndarray]) -> None:
    """Copy NumPy weights named like ``actor_W0`` and ``target_critic_b1`` into the networks.

    NumPy stores ``W`` as inputs x outputs and torch as outputs x inputs, hence the transpose.
    """
    with torch.no_grad():
        for net_name in NETWORKS:
            for i, linear in enumerate(_linears(getattr(agent, net_name))):
                linear.weight.copy_(torch.from_numpy(weights[f"{net_name}_W{i}"].T))
                linear.bias.copy_(torch.from_numpy(weights[f"{net_name}_b{i}"]))


def numpy_weights(agent: TorchDDPG) -> dict[str, np.ndarray]:
    """Every weight as a NumPy array, named and laid out as :func:`load_weights` expects."""
    weights = {}
    for net_name in NETWORKS:
        for i, linear in enumerate(_linears(getattr(agent, net_name))):
            weights[f"{net_name}_W{i}"] = linear.weight.detach().cpu().numpy().T.copy()
            weights[f"{net_name}_b{i}"] = linear.bias.detach().cpu().numpy().copy()
    return weights


def policy(agent: TorchDDPG, net: nn.Sequential, obs: torch.Tensor) -> torch.Tensor:
    return agent.bias + agent.scale * torch.tanh(net(obs))


def q_value(net: nn.Sequential, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return net(torch.cat([obs, actions], dim=1))[:, 0]


def ddpg_step(agent: TorchDDPG, batch: dict[str, torch.Tensor]) -> None:
    """One gradient step on ``batch`` (keys ``obs``, ``act``, ``rew``, ``next_obs``, ``done``)."""
    with torch.no_grad():
        next_actions = policy(agent, agent.target_actor, batch["next_obs"])
        next_q = q_value(agent.target_critic, batch["next_obs"], next_actions)
        targets = batch["rew"] + agent.gamma * (1 - batch["done"]) * next_q
    critic_loss = 0.5 * ((q_value(agent.critic, batch["obs"], batch["act"]) - targets) ** 2).mean()
    agent.critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    agent.critic_opt.step()
    actor_loss = -q_value(
        agent.critic, batch["obs"], policy(agent, agent.actor, batch["obs"])
    ).mean()
    agent.actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    agent.actor_opt.step()
    with torch.no_grad():
        for online, target in (
            (agent.actor, agent.target_actor),
            (agent.critic, agent.target_critic),
        ):
            torch._foreach_lerp_(list(target.parameters()), list(online.parameters()), agent.tau)
