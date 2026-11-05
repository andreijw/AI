"""What the PyTorch backends of the replay-buffer agents share: DDPG, TD3 and SAC.

Imported only through a PyTorch agent, which a config selects with ``backend: torch``
(``BaseAgent.class_for_config``), so nothing else in the framework needs PyTorch. The measurements
behind it are in docs/pytorch_jetson_timing.md.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from collections.abc import Mapping
from functools import reduce
from typing import Any, ClassVar

import numpy as np
import torch

from .off_policy import OffPolicyAgent

DEVICES = ("auto", "cpu", "cuda")
DTYPES = {"float32": torch.float32, "float64": torch.float64}
# Batches sampled and moved to the device at a time.
STAGE_CHUNK = 64
# Eager steps run on a side stream before capturing the CUDA graph, then undone.
CAPTURE_WARMUP = 3

TensorParams = dict[str, torch.Tensor]


def forward(params: TensorParams, x: torch.Tensor) -> torch.Tensor:
    """``rl_core.agents.mlp.mlp_forward`` in PyTorch: ReLU hidden layers, linear output."""
    n = len(params) // 2
    h = x
    for i in range(n):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < n - 1:
            h = torch.relu(h)
    return h


def _global_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    """The L2 norm of all gradients together, as the NumPy agents and their Adam compute it."""
    return torch.sqrt(sum(torch.sum(g * g) for g in grads))  # type: ignore[arg-type]


class TorchOffPolicyAgent(OffPolicyAgent):
    """
    An OffPolicyAgent with each gradient step computed in PyTorch: the same algorithm, defaults,
    config and checkpoints as the NumPy agent it is combined with
    (``class TorchX(TorchOffPolicyAgent, X)``). Everything but the gradient step is the NumPy
    agent's: exploration, the replay buffer and its sampling stream, and acting, which stays on
    the CPU with NumPy (acting on the GPU costs about 1 ms per environment step).

    * The device holds every parameter group and Adam state the NumPy agent declares
      (``_networks`` and ``_optimizers_by_net``; a group need not be an MLP), laid out as NumPy's
      (``W`` is inputs x outputs), starting from the values the NumPy agent drew for the seed.
      After every :meth:`update` that trained the actor is mirrored back into the NumPy arrays in
      float64, so acting reads the current policy; the rest follows before a checkpoint is
      written or a test helper reads it (:meth:`_sync_numpy`), and after :meth:`load`. Writing
      those NumPy arrays does not reach the device.
    * Each group is clipped as its NumPy Adam clips it (its ``max_grad_norm``), so a group the
      NumPy agent never clips (SAC's temperature) is not clipped here either.
    * On CUDA the step is captured once as a CUDA graph and replayed (``cuda_graph``, on by
      default there): without it the GPU is no faster than NumPy for networks this small.
      Capture runs a few warm-up steps on a side stream and then restores every tensor, so it
      trains nothing. The graph holds the address of every tensor it touches, so loading
      writes into them in place. The learning rate is part of the graph; a schedule would have
      to become a tensor.
    * Each step's inputs (the batch, and any noise the step draws) are sampled on the host in
      the order the NumPy agent draws them, ``STAGE_CHUNK`` steps at a time, and moved to the
      device in one copy per chunk. Noise drawn on the host from the NumPy agent's generators is
      what lets float64 training match the NumPy agent.
    * Per-step metrics are summed on the device and read once per :meth:`update`.

    float64 matches the NumPy agent to rounding, on the CPU and on CUDA
    (tests/test_torch_ddpg_agent.py); float32 on the GPU agrees to about 1e-3 relative over a few
    updates and is not bit-exact across runs. Adam's step count is held in the agent's dtype, so
    in float32 it stops counting exactly after 2**24 (16.7 million) gradient steps. The agent
    changes no global PyTorch state: no seeding, default dtype or matmul precision.

    A subclass supplies its device step: :meth:`_step` for each kind :meth:`_step_kind` names
    (each kind is captured as its own graph), the inputs a step reads (:meth:`_input_shapes`,
    :meth:`_sample_inputs`), ``STEP_METRICS`` and :meth:`_reduce_metrics`; and stubs the NumPy
    agent's own training maths, which would train the mirror the device then overwrites.

    Config keys beyond the NumPy agent's (``backend: torch`` selects the subclass):

    * ``device``: ``auto`` (default: cuda if available, else cpu), ``cpu`` or ``cuda``.
    * ``dtype``: ``float32`` (default) or ``float64``.
    * ``cuda_graph``: null (default: on with cuda), true or false.
    """

    BACKEND: ClassVar[str] = "torch"
    # The metrics one step adds to the device's sums, in order.
    STEP_METRICS: ClassVar[tuple[str, ...]]

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Raises:
            ValueError: For an invalid hyperparameter or torch setting, or ``device: cuda``
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
        trained = self._optimizers_by_net()
        for net in trained:
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
            for net in trained
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

        self._inputs = {
            key: torch.zeros(shape, dtype=self.dtype, device=self.device)
            for key, shape in self._input_shapes().items()
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
        # True when an update has left the NumPy critics, targets and Adam states behind the
        # device: only the actor, which select_action reads, is copied after each update.
        self._numpy_stale = False
        self._metric_sums = torch.zeros(
            len(self.STEP_METRICS), dtype=self.dtype, device=self.device
        )
        self._graphs: dict[str, torch.cuda.CUDAGraph] = {}

    # ------------------------------------------------------------------
    # The gradient step: what a subclass supplies
    # ------------------------------------------------------------------

    @abstractmethod
    def _step(self, kind: str) -> None:
        """One gradient step of ``kind`` on the inputs in ``self._inputs``, adding its
        ``STEP_METRICS`` to ``self._metric_sums``. ``_gradient_steps`` is not advanced until
        :meth:`_train_steps` returns: a step that needs its number gets it through ``kind``.

        It must allocate nothing outside the graph's pool and read nothing back to the host, so
        it can be captured."""

    def _input_shapes(self) -> dict[str, tuple[int, ...]]:
        """The shape of each tensor one step reads, by name; :meth:`_sample_inputs` fills them."""
        b, o, a = self.batch_size, self.observation_dim, self.action_dim
        return {
            "observations": (b, o),
            "actions": (b, a),
            "rewards": (b,),
            "next_observations": (b, o),
            "dones": (b,),
        }

    def _sample_inputs(self) -> Mapping[str, np.ndarray]:
        """One step's inputs, drawn on the host in the order the NumPy agent draws them. Called
        for a whole chunk before its steps run; ``_gradient_steps`` counts only the steps of
        earlier calls until :meth:`_train_steps` returns."""
        return self.replay_buffer.sample(self.batch_size)

    def _step_kind(self, gradient_step: int) -> str:
        """The kind of step ``gradient_step`` (counted from 1 over the agent's life) is: by
        default all steps are alike."""
        return "full"

    def _reduce_metrics(self, sums: torch.Tensor, kinds: Counter[str]) -> dict[str, float]:
        """What :meth:`update` returns, from the device's sums of ``STEP_METRICS`` and the number
        of steps of each kind: by default the mean of each over all the steps."""
        means = (sums / sum(kinds.values())).cpu().numpy()
        return {name: float(value) for name, value in zip(self.STEP_METRICS, means, strict=True)}

    # ------------------------------------------------------------------
    # Pieces of a step
    # ------------------------------------------------------------------

    def _q(self, params: TensorParams, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``Q(s, a)``, (B,), of the critic ``params``."""
        return forward(params, torch.cat([obs, actions], dim=1))[:, 0]

    def _next_q(self, next_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``OffPolicyAgent._min_target_q`` on the device: ``Q'``, or ``min(Q1', Q2')``."""
        target_of = dict(self._target_names())
        values = [
            self._q(self._nets[target_of[name]], next_obs, actions) for name in self._critic_names()
        ]
        return reduce(torch.minimum, values)

    def _bootstrap(self, next_values: torch.Tensor) -> torch.Tensor:
        """``r + gamma (1 - terminated) next_values`` for the batch in ``self._inputs``."""
        x = self._inputs
        return x["rewards"] + self.gamma * (1.0 - x["dones"]) * next_values

    def _clip_and_set(self, net: str, grads: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Clip group ``net``'s gradients by their global norm as its NumPy Adam does (not at
        all without a ``max_grad_norm``), write them into each parameter's ``.grad`` in place,
        and return the norm before clipping."""
        norm = _global_norm(list(grads))
        limit = self._optimizers_by_net()[net].max_grad_norm
        coef = None if limit is None else torch.clamp(limit / (norm + 1e-6), max=1.0)
        for tensor, grad in zip(self._nets[net].values(), grads, strict=True):
            assert tensor.grad is not None
            tensor.grad.copy_(grad if coef is None else grad * coef)
        return norm

    def _adam_step(self, net: str, loss: torch.Tensor) -> torch.Tensor:
        """One Adam step of group ``net`` on ``loss``, the gradients clipped as
        :meth:`_clip_and_set` says; every other group's gradients are left alone. Returns the
        gradient norm before clipping."""
        params = list(self._nets[net].values())
        norm = self._clip_and_set(net, torch.autograd.grad(loss, params))
        self._optimizers[net].step()
        return norm

    def _critic_update(
        self, net: str, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One Adam step of critic ``net`` on ``mean ½ (Q(s, a) - targets)²``. Returns the loss,
        Q (B,) and the gradient norm before clipping."""
        x = self._inputs
        q = self._q(self._nets[net], x["observations"], x["actions"])
        loss = torch.mean(0.5 * (q - targets) ** 2)
        return loss, q, self._adam_step(net, loss)

    def _critics_update(self, targets: torch.Tensor) -> list[torch.Tensor]:
        """``OffPolicyAgent._critic_step`` on the device: a step for each critic towards
        ``targets``. Returns ``critic_loss``, ``mean_q`` and ``critic_grad_norm`` as it does:
        for twins the mean loss, the first critic's Q and the norm of both together."""
        results = [self._critic_update(net, targets) for net in self._critic_names()]
        if len(results) == 1:
            loss, q, norm = results[0]
            return [loss, q.mean(), norm]
        (loss_1, q, norm_1), (loss_2, _, norm_2) = results
        return [(loss_1 + loss_2) / 2.0, q.mean(), torch.hypot(norm_1, norm_2)]

    def _move_targets(self) -> None:
        """Move every target network a fraction ``tau`` towards its online network."""
        with torch.no_grad():
            for online, target in self._target_names():
                torch._foreach_lerp_(
                    list(self._nets[target].values()), list(self._nets[online].values()), self.tau
                )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_steps(self, num_updates: int) -> dict[str, float]:
        """Run ``num_updates`` steps on the device, then mirror the actor into NumPy."""
        self._metric_sums.zero_()
        kinds: Counter[str] = Counter()
        done = 0
        while done < num_updates:
            count = min(STAGE_CHUNK, num_updates - done)
            self._stage(count)
            for i in range(count):
                kind = self._step_kind(self._gradient_steps + done + i + 1)
                if self.cuda_graph and kind not in self._graphs:
                    self._capture(kind)
                for key, tensor in self._inputs.items():
                    tensor.copy_(self._device_chunk[key][i])
                graph = self._graphs.get(kind)
                if graph is not None:
                    graph.replay()
                else:
                    self._step(kind)
                kinds[kind] += 1
            done += count
        self._gradient_steps += num_updates
        metrics = self._reduce_metrics(self._metric_sums, kinds)
        self._mirror_actor()
        return metrics

    def _stage(self, count: int) -> None:
        """Sample ``count`` steps' inputs and copy them to the device at once."""
        if self._chunk_copied is not None:
            self._chunk_copied.synchronize()  # the last chunk has left the host buffer
        for i in range(count):
            sample = self._sample_inputs()
            for key, host in self._host_chunk.items():
                host[i].copy_(torch.from_numpy(np.asarray(sample[key], dtype=np.float64)))
        for key, host in self._host_chunk.items():
            self._device_chunk[key][:count].copy_(host[:count], non_blocking=True)
        if self.device.type == "cuda":
            self._chunk_copied = torch.cuda.Event()
            self._chunk_copied.record()

    def _capture(self, kind: str) -> None:
        """Capture :meth:`_step` of ``kind`` as a CUDA graph without training: warm-up steps run on
        a side stream (as capture requires), then every parameter, Adam state and gradient is put
        back."""
        saved = [t.detach().clone() for t in self._state_tensors()]
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(CAPTURE_WARMUP):
                self._step(kind)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step(kind)
        with torch.no_grad():
            for tensor, value in zip(self._state_tensors(), saved, strict=True):
                tensor.copy_(value)
        self._graphs[kind] = graph

    def _state_tensors(self) -> list[torch.Tensor]:
        """Every tensor a step changes: weights, gradients, Adam state and the metric sums."""
        tensors = [t for params in self._nets.values() for t in params.values()]
        for net, optimizer in self._optimizers.items():
            tensors += [t.grad for t in self._nets[net].values() if t.grad is not None]
            for state in optimizer.state.values():
                tensors += [state["step"], state["exp_avg"], state["exp_avg_sq"]]
        return [*tensors, self._metric_sums]

    # ------------------------------------------------------------------
    # The NumPy mirror and checkpoints
    # ------------------------------------------------------------------

    def _adam_arrays(self, net: str) -> dict[str, np.ndarray]:
        """An Adam state of the device, keyed as the NumPy Adam stores it."""
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

    def _mirror_actor(self) -> None:
        """Copy the actor, the one network :meth:`select_action` reads, from the device; the rest
        waits for :meth:`_sync_numpy`. With ``update_every: 1`` a full copy after every step made
        training several times slower."""
        self._copy_from_device("actor")
        self._numpy_stale = True

    def _sync_numpy(self) -> None:
        """Bring every NumPy network and Adam state up to date with the device, if an update left
        them behind."""
        if self._numpy_stale:
            self._mirror_to_numpy()
            self._numpy_stale = False

    def _checkpoint_arrays(self) -> dict[str, Any]:
        self._sync_numpy()
        return super()._checkpoint_arrays()

    def _mirror_to_numpy(self) -> None:
        """Copy every network and Adam state from the device into the NumPy agent's arrays."""
        for net in self._networks():
            self._copy_from_device(net)
        for net, numpy_adam in self._optimizers_by_net().items():
            numpy_adam.load_state(self._adam_arrays(net), prefix="")

    def _copy_from_device(self, net: str) -> None:
        """Copy group ``net`` from the device into the NumPy agent's arrays, as float64."""
        params = self._networks()[net]
        for name, tensor in self._nets[net].items():
            params[name] = tensor.detach().cpu().numpy().astype(np.float64)

    def _push_to_device(self) -> None:
        """Copy the NumPy agent's networks and Adam states into the device tensors, in place."""
        with torch.no_grad():
            for net, params in self._networks().items():
                for name, tensor in self._nets[net].items():
                    tensor.copy_(torch.from_numpy(params[name]))
            for net, numpy_adam in self._optimizers_by_net().items():
                stored = numpy_adam.state(prefix="")
                for name, tensor in self._nets[net].items():
                    state = self._optimizers[net].state[tensor]
                    state["step"].fill_(float(stored["t"]))
                    state["exp_avg"].copy_(torch.from_numpy(stored[f"m_{name}"]))
                    state["exp_avg_sq"].copy_(torch.from_numpy(stored[f"v_{name}"]))

    def _load_checkpoint_arrays(self, data: Mapping[str, Any]) -> None:
        super()._load_checkpoint_arrays(data)
        self._push_to_device()
        self._numpy_stale = False

    # OffPolicyAgent's NumPy training maths would train the mirror, which the device then
    # overwrites; a subclass stubs its NumPy agent's own maths the same way.

    def _on_device(self) -> NotImplementedError:
        """The error a NumPy training method raises on this backend."""
        return NotImplementedError(f"{type(self).__name__} trains on the device")

    def _critic_loss_and_grads(self, *args: Any, **kwargs: Any) -> Any:
        raise self._on_device()

    def _critic_step(self, *args: Any, **kwargs: Any) -> Any:
        raise self._on_device()

    def _apply(self, *args: Any, **kwargs: Any) -> None:
        raise self._on_device()

    def _polyak(self) -> None:
        raise self._on_device()


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
