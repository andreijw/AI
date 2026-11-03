"""Export a trained agent's greedy policy as one self-contained C header, and check it in C."""

import argparse
import hashlib
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_core.export import c_header, verify
from rl_core.export.rollout import greedy_rollout, policy_disagreements
from rl_core.seeding import eval_seed_for
from train import add_challenge_arguments, load_run_config, select_run_challenge


def _source(checkpoint):
    with open(checkpoint, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return f"{os.path.basename(checkpoint)} (sha256 {digest})"


def _write(path, header, policy):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
    print(f"Wrote {policy.agent_name}'s greedy policy to '{path}'")


def _fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def main():
    """Load a checkpoint as record.py does, write its C header, and verify it on greedy episodes."""
    parser = argparse.ArgumentParser(description="Export a trained RL agent as a C header")
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
        "--output", type=str, required=True, help="Header to write (e.g. policy_weights.h)"
    )
    parser.add_argument(
        "--verify-episodes",
        type=int,
        default=3,
        help="Greedy episodes whose observations check the compiled header against NumPy "
        "(default: 3; 0 skips the check)",
    )
    add_challenge_arguments(parser)
    args = parser.parse_args()
    if args.verify_episodes < 0:
        parser.error("--verify-episodes must be 0 or more.")

    config, agent_type = load_run_config(args.config, args.agent_type)
    run_challenge = select_run_challenge(config, agent_type, args.challenge)
    # A PyTorch-backend checkpoint loads into the NumPy agent: exporting needs no PyTorch.
    agent_config = {**config["agent"]["config"], "backend": "numpy"}
    env = run_challenge.make_env(config["environment"], render_mode=None)
    try:
        agent = run_challenge.make_agent(agent_type, agent_config, env)
        agent.load(args.checkpoint)
        policy = agent.inference_policy()
        header = c_header.render_c_header(policy, source=_source(args.checkpoint))
        if args.verify_episodes == 0:
            _write(args.output, header, policy)
            return
        rollout = greedy_rollout(
            env, agent, args.verify_episodes, seed=eval_seed_for(config["environment"].get("seed"))
        )
    finally:
        env.close()

    # The header is written only once it is checked, or unchecked for want of a compiler (with
    # exit status 1), so a failed export never leaves one behind.
    agreement = policy_disagreements(policy, rollout)
    if agreement.mismatches:
        _fail(
            f"The exported policy acts unlike the agent on {agreement.mismatches} step(s) in NumPy"
        )
    if agreement.near_ties:
        print(f"NumPy: {agreement.near_ties} near-tie(s) with the agent's actions")
    if verify.find_c_compiler() is None:
        _write(args.output, header, policy)
        _fail(
            "No C compiler found to verify the header: set CC or install gcc or clang, "
            "or pass --verify-episodes 0 to skip the check."
        )
    report = verify.verify_c_header(header, policy, rollout.observations)
    print(report.summary())
    if not report.passed:
        _fail(f"Not writing '{args.output}': the compiled header disagrees with NumPy")
    _write(args.output, header, policy)


if __name__ == "__main__":
    main()
