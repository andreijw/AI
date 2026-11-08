"""Play a trained agent greedily with one environment setting perturbed at a time, inside and
beyond the range it trained on, and report how it holds up."""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_core.seeding import eval_seed_for
from rl_core.training.checkpoints import file_sha256
from rl_core.training.curriculum import curriculum_from_config
from rl_core.training.robustness import (
    Perturbation,
    RobustnessRow,
    difficulty_reached,
    parse_sweep,
    perturbed_section,
    robustness_rows,
)
from train import (
    add_challenge_arguments,
    add_checkpoint_arguments,
    load_run_config,
    numpy_agent_config,
    select_run_challenge,
)


def _sweeps(parser, texts):
    """Each --sweep's perturbations, in order, refusing a malformed one."""
    sweeps = []
    for text in texts:
        try:
            sweeps.append(parse_sweep(text))
        except ValueError as error:
            parser.error(str(error))
    return sweeps


def _check_sweeps(parser, section, env, sweeps):
    """Refuse a setting the wrapper cannot sweep, a value it refuses, or a setting (under any of
    its names) in two sweeps."""
    swept = set()
    for sweep in sweeps:
        try:
            settings = {perturbed_section(section, env, p)[0] for p in sweep}
        except ValueError as error:
            parser.error(str(error))
        (setting,) = settings
        if setting in swept:
            parser.error(f"{setting} is swept twice: give its values in one --sweep")
        swept.add(setting)


def _range_note(curriculum, difficulty):
    if curriculum is None:
        return "the config's `environment` settings (no curriculum)"
    if difficulty is None:
        return (
            f"the configured curriculum ({curriculum.mode}), start to target: no trainer file "
            "beside the checkpoint, so the difficulty reached is unknown"
        )
    return (
        f"the curriculum ({curriculum.mode}) from its start to difficulty {difficulty:.2f}, "
        "replayed from the trainer file beside the checkpoint"
    )


def _row(row: RobustnessRow, threshold, with_success):
    if row.setting is None:
        setting, value, status, trained = "nominal (the target)", "–", "–", "–"
    else:
        setting, value = row.setting, f"{row.value:g}"
        status = "in range" if row.in_range else "beyond"
        low, high = row.trained_range
        trained = f"{low:g}" if low == high else f"{low:g} to {high:g}"
    solved = row.solved_fraction(threshold)
    solved_text = "N/A" if solved is None else f"{solved * 100:.0f}%"
    line = (
        f"| {setting} | {value} | {status} | {trained} | {row.mean_reward:.1f} | "
        f"±{row.std_reward:.1f} | {row.mean_length:.1f} | {solved_text} |"
    )
    if with_success:
        rate = row.success_rate
        line += " N/A |" if rate is None else f" {rate * 100:.0f}% |"
    return line


def render_report(rows, *, title, header, threshold):
    """The Markdown report: what was played, then one table row per perturbation. The threshold
    column is the share of episodes whose reward reached it; a Success column, the share that
    succeeded, is added when the environment reports success."""
    threshold_text = "N/A" if threshold is None else f"{threshold:g}"
    with_success = any(row.success_rate is not None for row in rows)
    return "\n".join(
        [
            f"# {title}",
            "",
            *(f"- **{name}**: {value}" for name, value in header),
            "",
            "Each row plays the same start states: episode i of every row resets with the same "
            "seed. Noise draws are not paired once a pinned parameter adds a draw at each reset.",
            "",
            f"| Setting | Value | Trained On | Training Range | Mean Reward | Reward Std "
            f"| Mean Length | Reward ≥ {threshold_text} |" + (" Success |" if with_success else ""),
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
            + (" :---: |" if with_success else ""),
            *(_row(row, threshold, with_success) for row in rows),
            "",
        ]
    )


def main():
    """Load a checkpoint as export.py does and play it on each perturbation of its config."""
    parser = argparse.ArgumentParser(description="Measure a trained RL agent's robustness")
    add_checkpoint_arguments(parser, "evaluate")
    parser.add_argument(
        "--sweep",
        action="append",
        required=True,
        metavar="SETTING=V1,V2,...",
        help="A noise setting or physics parameter and the values to play it at; repeatable, "
        "one --sweep per setting (e.g. length=0.25,0.5,1.0,2.0)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Greedy episodes per perturbation (default: 20)",
    )
    parser.add_argument(
        "--report", type=str, help="Markdown file to write the report to (default: print only)"
    )
    add_challenge_arguments(parser)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer.")
    sweeps = _sweeps(parser, args.sweep)
    perturbations: list[Perturbation] = [p for sweep in sweeps for p in sweep]

    config, agent_type = load_run_config(args.config, args.agent_type)
    run_challenge = select_run_challenge(config, agent_type, args.challenge)
    section = config["environment"]
    agent_config = numpy_agent_config(config)
    env = run_challenge.make_env(section, render_mode=None)
    try:
        _check_sweeps(parser, section, env, sweeps)
        agent = run_challenge.make_agent(agent_type, agent_config, env)
        agent.load(args.checkpoint)
        curriculum = curriculum_from_config(section, env)
    finally:
        env.close()

    try:
        difficulty = difficulty_reached(args.checkpoint, curriculum)
    except ValueError as error:
        parser.error(str(error))
    eval_seed = eval_seed_for(section.get("seed"))
    rows = robustness_rows(
        run_challenge,
        config,
        agent,
        perturbations,
        episodes=args.episodes,
        eval_seed=eval_seed,
        start=None if curriculum is None else curriculum.start,
        difficulty=1.0 if difficulty is None else difficulty,
    )
    report = render_report(
        rows,
        title=f"Robustness: {agent_type} on {run_challenge.name}",
        header=[
            ("Config", f"`{args.config}`"),
            (
                "Checkpoint",
                f"`{os.path.basename(args.checkpoint)}` (sha256 {file_sha256(args.checkpoint)})",
            ),
            ("Episodes per row", f"{args.episodes} greedy"),
            ("Eval seed", f"{eval_seed} (episode i resets with {eval_seed} + i)"),
            ("Training range", _range_note(curriculum, difficulty)),
        ],
        threshold=run_challenge.solve_threshold,
    )
    print(report)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Robustness report saved to '{args.report}'")


if __name__ == "__main__":
    main()
