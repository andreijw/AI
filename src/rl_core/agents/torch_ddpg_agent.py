"""DDPG with its gradient steps in PyTorch, on the GPU when there is one.

Imported only when a config sets ``backend: torch`` (``BaseAgent.class_for_config``), so nothing
else in the framework needs PyTorch. The measurements behind it are in
docs/pytorch_jetson_timing.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import torch

from .ddpg_agent import DDPGAgent
from .mlp import Params

DEVICES = ("auto", "cpu", "cuda")
DTYPES = {"float32": torch.float32, "float64": torch.float64}
# The metrics DDPGAgent.update returns, in the order the device accumulates them.
METRICS = (
    "critic_loss",
    "mean_q",
    "critic_grad_norm",
    "actor_loss",
    "saturation",
    "actor_grad_norm",
)
# Batches sampled and moved to the device at a time.
STAGE_CHUNK = 64
# Eager steps run on a side stream before capturing the CUDA graph, then undone.
CAPTURE_WARMUP = 3
NETWORKS = ("actor", "critic", "target_actor", "target_critic")

TensorParams = dict[str, torch.Tensor]


def _forward(params: TensorParams, x: torch.Tensor) -> torch.Tensor:
    """``rl_core.agents.mlp.mlp_forward`` in PyTorch: ReLU hidden layers, linear output."""
    n = len(params) // 2
    h = x
    for i in range(n):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < n - 1:
            h = torch.relu(h)
    return h


def _global_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    """The L2 norm of all gradients together, as DDPGAgent and its Adam compute it."""
    return torch.sqrt(sum(torch.sum(g * g) for g in grads))  # type: ignore[arg-type]


class TorchDDPGAgent(DDPGAgent):
    """
    DDPGAgent with each gradient step computed in PyTorch: the same algorithm, defaults, config
    and checkpoints. Everything else is DDPGAgent's: exploration, the replay buffer and its
    sampling stream, and acting, which stays on the CPU with NumPy (acting on the GPU costs
    about 1 ms per environment step).

    * The device holds the four networks and both Adam states, laid out as NumPy's (``W`` is
      inputs x outputs), starting from the weights DDPGAgent drew for the seed. After every
      :meth:`update` that trained, and after :meth:`load`, all of it is mirrored back into
      DDPGAgent's NumPy arrays in float64, so acting, checkpoints and test helpers read the
      current values. Writing those NumPy arrays does not reach the device.
    * On CUDA the step is captured once as a CUDA graph and replayed (``cuda_graph``, on by
      default there): without it the GPU is no faster than NumPy for networks this small.
      Capture runs a few warm-up steps on a side stream and then restores every tensor, so it
      trains nothing. The graph holds the address of every tensor it touches, so loading
      writes into them in place. The learning rate is part of the graph; a schedule would have
      to become a tensor.
    * Batches are sampled from the NumPy buffer in the same order DDPGAgent samples them,
      ``STAGE_CHUNK`` at a time, and moved to the device in one copy per chunk.
    * Per-step metrics are summed on the device and read once per :meth:`update`.

    float64 matches DDPGAgent to rounding, on the CPU and on CUDA (tests/test_torch_ddpg_agent.py);
    float32 on the GPU agrees to about 1e-3 relative over a few updates and is not bit-exact
    across runs. Adam's step count is held in the agent's dtype, so in float32 it stops counting
    exactly after 2**24 (16.7 million) gradient steps. The agent changes no global PyTorch state: no seeding, default dtype or matmul
    precision.

    Config keys beyond DDPGAgent's (``backend: torch`` selects this class):

    * ``device``: ``auto`` (default: cuda if available, else cpu), ``cpu`` or ``cuda``.
    * ``dtype``: ``float32`` (default) or ``float64``.
    * ``cuda_graph``: null (default: on with cuda), true or false.
    """

    BACKEND: ClassVar[str] = "torch"

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Raises:
            ValueError: For an invalid DDPG hyperparameter or torch setting, or ``device: cuda``
                without CUDA.
        """
        super().__init__(observation_dim, action_dim, config)
        self.device, self.dtype, self.cuda_graph = _torch_settings(config)
        self._nets: dict[str, TensorParams] = {
            net: {
                name: torch.tensor(value, dtype=self.dtype, device=self.device)
                for name, value in params.items()
            }
            for net, params in self._networks().items()
        }
        for net in ("actor", "critic"):
            for tensor in self._nets[net].values():
                tensor.requires_grad_(True)
                tensor.grad = torch.zeros_like(tensor)
        self._optimizers = {
            net: torch.optim.Adam(
                list(self._nets[net].values()),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                capturable=self.device.type == "cuda",  # one Adam code path, graph or not
            )
            for net in ("actor", "critic")
        }
        for optimizer in self._optimizers.values():
            for tensor in optimizer.param_groups[0]["params"]:
                # Filled now, not on the first step: a captured graph needs these tensors to exist.
                optimizer.state[tensor] = {
                    # In the agent's dtype: capturable Adam computes its bias corrections from this
                    # tensor, and a float32 count makes a float64 run drift from the NumPy agent.
                    "step": torch.zeros(
                        (),
                        dtype=self.dtype,
                        device=self.device if self.device.type == "cuda" else "cpu",
                    ),
                    "exp_avg": torch.zeros_like(tensor, memory_format=torch.preserve_format),
                    "exp_avg_sq": torch.zeros_like(tensor, memory_format=torch.preserve_format),
                }
        self._scale_t = torch.tensor(self.scale, dtype=self.dtype, device=self.device)
        self._bias_t = torch.tensor(self.bias, dtype=self.dtype, device=self.device)

        b, o, a = self.batch_size, observation_dim, action_dim
        self._inputs = {
            "observations": torch.zeros((b, o), dtype=self.dtype, device=self.device),
            "actions": torch.zeros((b, a), dtype=self.dtype, device=self.device),
            "rewards": torch.zeros((b,), dtype=self.dtype, device=self.device),
            "next_observations": torch.zeros((b, o), dtype=self.dtype, device=self.device),
            "dones": torch.zeros((b,), dtype=self.dtype, device=self.device),
        }
        pin = self.device.type == "cuda"
        self._host_chunk = {
            key: torch.zeros((STAGE_CHUNK, *tensor.shape), dtype=self.dtype, pin_memory=pin)
            for key, tensor in self._inputs.items()
        }
        self._device_chunk = {
            key: torch.zeros((STAGE_CHUNK, *tensor.shape), dtype=self.dtype, device=self.device)
            for key, tensor in self._inputs.items()
        }
        self._chunk_copied: torch.cuda.Event | None = None
        self._metric_sums = torch.zeros(len(METRICS), dtype=self.dtype, device=self.device)
        self._graph: torch.cuda.CUDAGraph | None = None

    # ------------------------------------------------------------------
    # The gradient step
    # ------------------------------------------------------------------

    def _policy(self, params: TensorParams, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.tanh(_forward(params, obs))
        return self._bias_t + self._scale_t * u, u

    def _q(self, params: TensorParams, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return _forward(params, torch.cat([obs, actions], dim=1))[:, 0]

    def _clip_and_set(self, params: TensorParams, grads: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Clip the gradients by their global norm as DDPGAgent's Adam does, write them into each
        parameter's ``.grad`` in place, and return the norm before clipping."""
        norm = _global_norm(list(grads))
        coef = None
        if self.max_grad_norm is not None:
            coef = torch.clamp(self.max_grad_norm / (norm + 1e-6), max=1.0)
        for tensor, grad in zip(params.values(), grads, strict=True):
            assert tensor.grad is not None
            tensor.grad.copy_(grad if coef is None else grad * coef)
        return norm

    def _step(self) -> None:
        """One gradient step on the batch in ``self._inputs``: critic, then actor, then Polyak.

        The same step as DDPGAgent's; it allocates nothing outside the graph's pool and reads
        nothing back to the host, so it can be captured.
        """
        x = self._inputs
        nets = self._nets
        with torch.no_grad():
            next_actions, _ = self._policy(nets["target_actor"], x["next_observations"])
            next_q = self._q(nets["target_critic"], x["next_observations"], next_actions)
            targets = x["rewards"] + self.gamma * (1.0 - x["dones"]) * next_q

        q = self._q(nets["critic"], x["observations"], x["actions"])
        critic_loss = torch.mean(0.5 * (q - targets) ** 2)
        critic_params = list(nets["critic"].values())
        critic_norm = self._clip_and_set(
            nets["critic"], torch.autograd.grad(critic_loss, critic_params)
        )
        self._optimizers["critic"].step()

        mu, u = self._policy(nets["actor"], x["observations"])
        actor_loss = -torch.mean(self._q(nets["critic"], x["observations"], mu))
        # Gradients for the actor only: the critic's stay those of its own loss.
        actor_norm = self._clip_and_set(
            nets["actor"], torch.autograd.grad(actor_loss, list(nets["actor"].values()))
        )
        self._optimizers["actor"].step()

        with torch.no_grad():
            for online, target in (("actor", "target_actor"), ("critic", "target_critic")):
                torch._foreach_lerp_(
                    list(nets[target].values()), list(nets[online].values()), self.tau
                )
            saturation = torch.mean((torch.abs(u) > 0.99).to(self.dtype))
            self._metric_sums += torch.stack(
                [critic_loss, q.mean(), critic_norm, actor_loss, saturation, actor_norm]
            ).detach()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_steps(self, num_updates: int) -> dict[str, float]:
        """Run ``num_updates`` steps on the device, then mirror the result into NumPy."""
        if self.cuda_graph and self._graph is None:
            self._capture()
        self._metric_sums.zero_()
        done = 0
        while done < num_updates:
            count = min(STAGE_CHUNK, num_updates - done)
            self._stage(count)
            for i in range(count):
                for key, tensor in self._inputs.items():
                    tensor.copy_(self._device_chunk[key][i])
                if self._graph is not None:
                    self._graph.replay()
                else:
                    self._step()
            done += count
        self._gradient_steps += num_updates
        means = (self._metric_sums / num_updates).cpu().numpy()
        self._mirror_to_numpy()
        return {name: float(value) for name, value in zip(METRICS, means, strict=True)}

    def _stage(self, count: int) -> None:
        """Sample ``count`` batches from the replay buffer and copy them to the device at once."""
        if self._chunk_copied is not None:
            self._chunk_copied.synchronize()  # the last chunk has left the host buffer
        for i in range(count):
            sample = self.replay_buffer.sample(self.batch_size)
            for key, host in self._host_chunk.items():
                host[i].copy_(torch.from_numpy(np.asarray(sample[key], dtype=np.float64)))
        for key, host in self._host_chunk.items():
            self._device_chunk[key][:count].copy_(host[:count], non_blocking=True)
        if self.device.type == "cuda":
            self._chunk_copied = torch.cuda.Event()
            self._chunk_copied.record()

    def _capture(self) -> None:
        """Capture :meth:`_step` as a CUDA graph without training: warm-up steps run on a side
        stream (as capture requires), then every parameter, Adam state and gradient is put back."""
        saved = [t.detach().clone() for t in self._state_tensors()]
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(CAPTURE_WARMUP):
                self._step()
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step()
        with torch.no_grad():
            for tensor, value in zip(self._state_tensors(), saved, strict=True):
                tensor.copy_(value)
        self._graph = graph

    def _state_tensors(self) -> list[torch.Tensor]:
        """Every tensor a step changes: weights, gradients, Adam state and the metric sums."""
        tensors = [t for net in NETWORKS for t in self._nets[net].values()]
        for net in ("actor", "critic"):
            tensors += [t.grad for t in self._nets[net].values() if t.grad is not None]
            for state in self._optimizers[net].state.values():
                tensors += [state["step"], state["exp_avg"], state["exp_avg_sq"]]
        return [*tensors, self._metric_sums]

    # ------------------------------------------------------------------
    # The NumPy mirror and checkpoints
    # ------------------------------------------------------------------

    def _adam_arrays(self, net: str) -> dict[str, np.ndarray]:
        """An Adam state of the device, keyed as DDPGAgent's NumPy Adam stores it."""
        optimizer = self._optimizers[net]
        states = [optimizer.state[tensor] for tensor in self._nets[net].values()]
        steps = torch.stack([state["step"] for state in states]).cpu()
        if not torch.all(steps == steps[0]):
            raise RuntimeError(
                f"{net} Adam step counts differ between parameters: {steps.tolist()}"
            )
        arrays: dict[str, np.ndarray] = {"t": np.array(int(round(float(steps[0]))))}
        for name, state in zip(self._nets[net], states, strict=True):
            arrays[f"m_{name}"] = state["exp_avg"].detach().cpu().numpy().astype(np.float64)
            arrays[f"v_{name}"] = state["exp_avg_sq"].detach().cpu().numpy().astype(np.float64)
        return arrays

    def _mirror_to_numpy(self) -> None:
        """Copy all four networks and both Adam states from the device into DDPGAgent's arrays."""
        for net, params in self._networks().items():
            for name, tensor in self._nets[net].items():
                params[name] = tensor.detach().cpu().numpy().astype(np.float64)
        for net, numpy_adam in (
            ("actor", self._actor_optimizer),
            ("critic", self._critic_optimizer),
        ):
            numpy_adam.load_state(self._adam_arrays(net), prefix="")

    def _push_to_device(self) -> None:
        """Copy DDPGAgent's NumPy networks and Adam states into the device tensors, in place."""
        with torch.no_grad():
            for net, params in self._networks().items():
                for name, tensor in self._nets[net].items():
                    tensor.copy_(torch.from_numpy(params[name]))
            for net, numpy_adam in (
                ("actor", self._actor_optimizer),
                ("critic", self._critic_optimizer),
            ):
                stored = numpy_adam.state(prefix="")
                for name, tensor in self._nets[net].items():
                    state = self._optimizers[net].state[tensor]
                    state["step"].fill_(float(stored["t"]))
                    state["exp_avg"].copy_(torch.from_numpy(stored[f"m_{name}"]))
                    state["exp_avg_sq"].copy_(torch.from_numpy(stored[f"v_{name}"]))

    def _load_checkpoint_arrays(self, data: Mapping[str, Any]) -> None:
        super()._load_checkpoint_arrays(data)
        self._push_to_device()

    # DDPGAgent's NumPy training maths would train the mirror, which the device then overwrites.

    def _bellman_targets(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("TorchDDPGAgent trains on the device")

    def _critic_loss_and_grads(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("TorchDDPGAgent trains on the device")

    def _actor_loss_and_grads(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("TorchDDPGAgent trains on the device")

    def _apply(self, params: Params, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("TorchDDPGAgent trains on the device")

    def _polyak(self) -> None:
        raise NotImplementedError("TorchDDPGAgent trains on the device")


def _torch_settings(config: Mapping[str, Any]) -> tuple[torch.device, torch.dtype, bool]:
    """``device``, ``dtype`` and ``cuda_graph`` from the config, checked."""
    device_name = config.get("device", "auto")
    if device_name not in DEVICES:
        raise ValueError(f"device must be one of {list(DEVICES)}, got {device_name!r}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("The config asks for device cuda, but CUDA is not available")
    dtype_name = config.get("dtype", "float32")
    if dtype_name not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}, got {dtype_name!r}")
    graph = config.get("cuda_graph")
    if graph not in (None, True, False):
        raise ValueError(f"cuda_graph must be true, false or null, got {graph!r}")
    if graph and device_name != "cuda":
        raise ValueError("cuda_graph needs device cuda")
    return (
        torch.device(device_name),
        DTYPES[dtype_name],
        device_name == "cuda" and graph is not False,
    )
