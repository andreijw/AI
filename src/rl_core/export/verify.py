"""
Compare an exported policy with the NumPy reference
(:class:`~rl_core.agents.inference.InferencePolicy`), and check a C header by compiling it into a
small harness and running it on observations. The ONNX model is run in ``onnx_model`` and compared
here too.

The observations are rounded to float32 (as the exported code takes them) and the reference is
computed in float64 from those same values; the C harness reads and prints hexadecimal floats, so
nothing is lost on the way. Outputs agree when every one is within ``ATOL + RTOL · |reference|``;
the report gives the worst error as a share of that. A discrete action may differ only where the
reference's two largest outputs are that close (a near-tie, counted apart); a continuous action
within the same tolerance.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess  # nosec B404 - runs the C compiler and the harness it built, no shell
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from rl_core.agents.inference import InferencePolicy

ATOL = 1e-5
RTOL = 1e-5
TIMEOUT_SECONDS = 120
HEADER_NAME = "policy_weights.h"
_COMPILERS = ("cc", "gcc", "clang")


class CompilerNotFoundError(RuntimeError):
    """No C compiler: ``CC`` is unset and none of cc, gcc or clang is on the path."""


def find_c_compiler() -> list[str] | None:
    """The compiler command: ``CC`` split as a shell would, else the first of cc, gcc and clang
    on the path; None if there is none (MSVC's ``cl`` is not supported)."""
    from_env = os.environ.get("CC", "").strip()
    if from_env:
        return shlex.split(from_env)
    if sys.platform == "win32":
        return None
    for name in _COMPILERS:
        path = shutil.which(name)
        if path:
            return [path]
    return None


def near_ties(outputs: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """For each row of ``outputs`` (B, O), whether its two largest values are within the tolerance:
    where float32 and float64 may legitimately take different greedy actions."""
    values = np.atleast_2d(np.asarray(outputs, dtype=np.float64))
    if values.shape[1] < 2:
        return np.zeros(len(values), dtype=bool)
    top_two = np.sort(values, axis=1)[:, -2:]
    return np.asarray((top_two[:, 1] - top_two[:, 0]) <= ATOL + RTOL * np.abs(top_two[:, 1]))


@dataclass(frozen=True)
class VerificationReport:
    """How an exported policy's outputs and actions compare with the reference's."""

    target: str
    observations: int
    max_abs_error: float
    max_rel_error: float
    worst_tolerance_share: float
    outputs_within: bool
    action_mismatches: int
    near_ties: int
    outputs: npt.NDArray[np.float64]
    actions: npt.NDArray[np.float64]

    @property
    def passed(self) -> bool:
        return self.outputs_within and self.action_mismatches == 0

    def summary(self) -> str:
        verdict = "passed" if self.passed else "FAILED"
        return (
            f"{self.target} verification {verdict} on {self.observations} observations: worst output error "
            f"{self.worst_tolerance_share:.2g} of the tolerance ({ATOL:g} + {RTOL:g} x |output|; "
            f"max abs error {self.max_abs_error:.3g}, max rel error {self.max_rel_error:.3g}), "
            f"{self.action_mismatches} action mismatch(es), {self.near_ties} near-tie(s)"
        )


def compare(
    reference_outputs: npt.ArrayLike,
    reference_actions: npt.ArrayLike,
    outputs: npt.ArrayLike,
    actions: npt.ArrayLike,
    discrete: bool,
    *,
    target: str,
) -> VerificationReport:
    """Compare an exported policy's outputs (B, O) and actions with the reference's (see the module
    docstring); ``target`` names the format checked ("C", "ONNX") in the report."""
    ref = np.asarray(reference_outputs, dtype=np.float64)
    got = np.asarray(outputs, dtype=np.float64)
    ref_actions = np.asarray(reference_actions)
    got_actions = np.asarray(actions)
    error = np.abs(got - ref)
    tolerance = ATOL + RTOL * np.abs(ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(ref != 0.0, error / np.abs(ref), np.where(error == 0.0, 0.0, np.inf))
    ties = 0
    mismatches = 0
    if discrete:
        close = near_ties(ref)
        differ = got_actions != ref_actions
        ties = int(np.sum(differ & close))
        mismatches = int(np.sum(differ & ~close))
    else:
        action_error = np.abs(got_actions.astype(np.float64) - ref_actions)
        bad = action_error > ATOL + RTOL * np.abs(ref_actions.astype(np.float64))
        mismatches = int(np.sum(bad.reshape(len(bad), -1).any(axis=1)))
    return VerificationReport(
        target=target,
        observations=len(ref),
        max_abs_error=float(error.max(initial=0.0)),
        max_rel_error=float(relative.max(initial=0.0)),
        worst_tolerance_share=float((error / tolerance).max(initial=0.0)),
        outputs_within=bool(np.all(error <= tolerance)),
        action_mismatches=mismatches,
        near_ties=ties,
        outputs=got,
        actions=got_actions,
    )


def _harness(policy: InferencePolicy) -> str:
    """A C program: read POLICY_INPUT_DIM numbers at a time, print outputs then the action."""
    if policy.discrete:
        act = ['        printf("%d", policy_act(obs));']
    else:
        act = [
            "        float action[POLICY_ACTION_DIM];",
            "        policy_act(obs, action);",
            '        for (int j = 0; j < POLICY_ACTION_DIM; ++j) printf(" %a", (double)action[j]);',
        ]
    return "\n".join(
        [
            "#include <stdio.h>",
            f'#include "{HEADER_NAME}"',
            "",
            "int main(void) {",
            "    float obs[POLICY_INPUT_DIM], out[POLICY_OUTPUT_DIM];",
            "    double v;",
            "    for (;;) {",
            "        for (int i = 0; i < POLICY_INPUT_DIM; ++i) {",
            '            if (scanf("%lf", &v) != 1) return 0;',
            "            obs[i] = (float)v;",
            "        }",
            "        policy_outputs(obs, out);",
            '        for (int j = 0; j < POLICY_OUTPUT_DIM; ++j) printf("%a ", (double)out[j]);',
            '        printf("| ");',
            *act,
            '        printf("\\n");',
            "    }",
            "}",
            "",
        ]
    )


def _run(argv: list[str], what: str, stdin: str | None = None) -> str:
    result = subprocess.run(  # nosec B603 - argv is a list; no shell
        argv, input=stdin, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {result.returncode}):\n{result.stderr.strip()}")
    return result.stdout


def run_c_header(
    header: str,
    policy: InferencePolicy,
    observations: npt.ArrayLike,
    workdir: str | Path,
    compiler: list[str] | None = None,
    werror: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compile ``header`` with a harness in ``workdir`` and run it on ``observations`` (B, input),
    rounded to float32. Returns C's outputs (B, O) and actions ((B,) or (B, A)).

    Raises:
        CompilerNotFoundError: If ``compiler`` is None and :func:`find_c_compiler` finds none.
        RuntimeError: If compiling or running fails, with the compiler's message.
    """
    command = compiler or find_c_compiler()
    if command is None:
        raise CompilerNotFoundError("no C compiler found: set CC, or install gcc or clang")
    directory = Path(workdir)
    (directory / HEADER_NAME).write_text(header, encoding="utf-8")
    (directory / "harness.c").write_text(_harness(policy), encoding="utf-8")
    binary = directory / ("harness.exe" if sys.platform == "win32" else "harness")
    flags = ["-std=c99", "-O2", "-Wall", "-Wextra", *(["-Werror"] if werror else [])]
    _run([*command, *flags, "-o", str(binary), str(directory / "harness.c"), "-lm"], "compiling")

    obs = np.asarray(observations, dtype=np.float32).reshape(-1, policy.input_dim)
    text = "\n".join(" ".join(float(v).hex() for v in row) for row in obs) + "\n"
    rows = _run([str(binary)], "the C harness", stdin=text).splitlines()
    if len(rows) != len(obs):
        raise RuntimeError(f"the C harness answered {len(rows)} of {len(obs)} observations")
    outputs, actions = [], []
    for row in rows:
        out_text, act_text = row.split("|")
        outputs.append([float.fromhex(v) for v in out_text.split()])
        actions.append(
            int(act_text) if policy.discrete else [float.fromhex(v) for v in act_text.split()]
        )
    return np.array(outputs, dtype=np.float64), np.array(actions)


def verify_c_header(
    header: str,
    policy: InferencePolicy,
    observations: npt.ArrayLike,
    workdir: str | Path | None = None,
    compiler: list[str] | None = None,
    werror: bool = False,
) -> VerificationReport:
    """
    Run ``header`` on ``observations`` (see :func:`run_c_header`) and compare it with ``policy``
    computed in float64 on the same float32-rounded observations. ``workdir`` defaults to a
    temporary directory, removed afterwards; ``werror`` turns compiler warnings into errors.
    """
    obs = np.asarray(observations, dtype=np.float32).reshape(-1, policy.input_dim)
    if workdir is None:
        with tempfile.TemporaryDirectory() as temporary:
            c_outputs, c_actions = run_c_header(header, policy, obs, temporary, compiler, werror)
    else:
        c_outputs, c_actions = run_c_header(header, policy, obs, workdir, compiler, werror)
    return compare_with_policy(policy, obs, c_outputs, c_actions, target="C")


def compare_with_policy(
    policy: InferencePolicy,
    observations: npt.NDArray[np.float32],
    outputs: npt.ArrayLike,
    actions: npt.ArrayLike,
    target: str,
) -> VerificationReport:
    """Compare what an export gave on ``observations`` (float32, as it took them) with ``policy``
    computed in float64 from those same values."""
    exact = np.asarray(observations, dtype=np.float32).astype(np.float64)
    return compare(
        policy.outputs(exact), policy.act(exact), outputs, actions, policy.discrete, target=target
    )
