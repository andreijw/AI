"""Export a trained agent's greedy policy as a self-contained C header or an ONNX model, and check
it (compiled, or in onnxruntime) against NumPy before writing it."""

import argparse
import hashlib
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_core.export import c_header, onnx_model, verify
from rl_core.export.rollout import greedy_rollout, policy_disagreements
from rl_core.seeding import eval_seed_for
from train import add_challenge_arguments, load_run_config, select_run_challenge

WHAT = {"c": "C header", "onnx": "ONNX model"}


def _source(checkpoint):
    with open(checkpoint, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return f"{os.path.basename(checkpoint)} (sha256 {digest})"


def _render(export_format, policy, source):
    """The exported file's bytes (a header in UTF-8 with LF line endings on every platform)."""
    if export_format == "onnx":
        return onnx_model.render_onnx_model(policy, source=source)
    return c_header.render_c_header(policy, source=source).encode("utf-8")


def _write(path, exported, policy):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(exported)
    print(f"Wrote {policy.agent_name}'s greedy policy to '{path}'")


def _fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def _check_onnx_packages(verify_episodes):
    """Stop before loading anything if the onnx extra is missing: an ONNX model is written only
    once checked, so onnxruntime is needed too unless the check is skipped."""
    missing = onnx_model.missing_packages(verify=verify_episodes > 0)
    if not missing:
        return
    skip = (
        " or pass --verify-episodes 0 to write the model unchecked"
        if missing == ["onnxruntime"]
        else ""
    )
    _fail(
        f"--format onnx needs the onnx extra ({', '.join(missing)} not installed): "
        f"{onnx_model.INSTALL_COMMAND}{skip}."
    )


def main():
    """Load a checkpoint as record.py does, export it, and verify it on greedy episodes."""
    parser = argparse.ArgumentParser(
        description="Export a trained RL agent as a C header or an ONNX model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cartpole_default.yaml",
        help="Configuration the checkpoint was trained with",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Agent checkpoint to export (e.g. checkpoints/ppo/agent_episode_1400.pt)",
    )
    parser.add_argument(
        "--format",
        choices=sorted(WHAT),
        default="c",
        help="c: a C99 header (default); onnx: an ONNX model (needs the onnx extra)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="File to write (e.g. policy_weights.h, or policy.onnx with --format onnx)",
    )
    parser.add_argument(
        "--verify-episodes",
        type=int,
        default=3,
        help="Greedy episodes whose observations check the export against NumPy, compiled or in "
        "onnxruntime (default: 3; 0 skips the check)",
    )
    add_challenge_arguments(parser)
    args = parser.parse_args()
    if args.verify_episodes < 0:
        parser.error("--verify-episodes must be 0 or more.")
    if args.format == "onnx":
        _check_onnx_packages(args.verify_episodes)

    config, agent_type = load_run_config(args.config, args.agent_type)
    run_challenge = select_run_challenge(config, agent_type, args.challenge)
    # A PyTorch-backend checkpoint loads into the NumPy agent: exporting needs no PyTorch.
    agent_config = {**config["agent"]["config"], "backend": "numpy"}
    env = run_challenge.make_env(config["environment"], render_mode=None)
    try:
        agent = run_challenge.make_agent(agent_type, agent_config, env)
        agent.inference_policy()  # refuse an agent with no network before reading the checkpoint
        agent.load(args.checkpoint)
        policy = agent.inference_policy()
        exported = _render(args.format, policy, _source(args.checkpoint))
        if args.verify_episodes == 0:
            _write(args.output, exported, policy)
            return
        rollout = greedy_rollout(
            env, agent, args.verify_episodes, seed=eval_seed_for(config["environment"].get("seed"))
        )
    finally:
        env.close()

    # The export is written only once it is checked. The one exception is a C header without a
    # compiler (exit status 1): the header is still useful, on Windows for example. So a failed
    # export never leaves a file behind.
    agreement = policy_disagreements(policy, rollout)
    if agreement.mismatches:
        _fail(
            f"The exported policy acts unlike the agent on {agreement.mismatches} step(s) in NumPy"
        )
    if agreement.near_ties:
        print(f"NumPy: {agreement.near_ties} near-tie(s) with the agent's actions")
    if args.format == "onnx":
        report = onnx_model.verify_onnx_model(exported, policy, rollout.observations)
    else:
        if verify.find_c_compiler() is None:
            _write(args.output, exported, policy)
            _fail(
                "No C compiler found to verify the header: set CC or install gcc or clang, "
                "or pass --verify-episodes 0 to skip the check."
            )
        report = verify.verify_c_header(exported.decode("utf-8"), policy, rollout.observations)
    print(report.summary())
    if not report.passed:
        _fail(f"Not writing '{args.output}': the {WHAT[args.format]} disagrees with NumPy")
    _write(args.output, exported, policy)


if __name__ == "__main__":
    main()
