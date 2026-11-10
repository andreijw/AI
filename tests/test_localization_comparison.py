"""The EKF against MCL on the same seeded trials: the comparison, its records, report and CLI."""

import dataclasses
import inspect
import json
import math

import numpy as np
import pytest

from rl_navigation.estimation import EkfLocalizer, MclSettings
from rl_navigation.estimation import comparison as comparison_module
from rl_navigation.estimation.comparison import (
    ComparisonRow,
    EkfReset,
    TrialRecord,
    check_records,
    compare_localizers,
    confidently_wrong,
    format_markdown,
    load_records,
    main,
    merge_records,
    nees_per_step,
    run_records,
)
from rl_navigation.estimation.evaluation import LOST_ERROR, Trial, average_nees
from rl_navigation.sim import lidar

# Short runs: 2 seeds of 120 steps, kidnapped at 60, localized within 20 steps and held for 10,
# with few particles. "lidar_scaled" is the same lidar under a second name, so its trials are
# the "lidar" ones and only the MCL covariance scale given for it differs.
SEEDS = (1, 2)
STEPS = 120
KIDNAP_AT = 60
DEADLINE = 20
HOLD = 10
FEW_PARTICLES = MclSettings(num_particles=150)
SCALE = (2.0, 2.0, 2.0)
SETS = ("ekf_", "mcl_", "mcl_kidnap_", "ekf_kidnap_")
TRACKING_KEYS = (
    "position_rmse",
    "heading_rmse",
    "mean_nees",
    "nees_below_upper_95",
    "axis_ratio_x",
    "axis_ratio_y",
    "axis_ratio_theta",
    "lost",
    "trials",
    "ms_per_step",
    "dead_reckoning_position_rmse",
    "dead_reckoning_heading_rmse",
    "dead_reckoning_lost",
)
KIDNAP_KEYS = (
    "global_share",
    "median_steps_to_localize",
    "seconds_to_localize",
    "recovered_share",
    "median_steps_to_recover",
    "seconds_to_recover",
    "confidently_wrong",
    "confidently_wrong_localized",
    "trials",
    "ms_per_step",
)


@pytest.fixture(scope="module")
def comparison():
    """One short comparison of the lidar under two names, the second with an MCL scale."""
    return compare_localizers(
        SEEDS,
        {"lidar": lidar, "lidar_scaled": lidar},
        (None,),
        steps=STEPS,
        kidnap_at=KIDNAP_AT,
        deadline=DEADLINE,
        hold=HOLD,
        mcl_settings=FEW_PARTICLES,
        mcl_scales={"lidar_scaled": SCALE},
        jobs_note="pytest",
    )


def _without_timings(metrics):
    return {key: value for key, value in metrics.items() if not key.endswith("ms_per_step")}


def test_compare_localizers_reports_every_column_for_every_row(comparison):
    """One row per sensor and dropout, with the tracking columns of both filters, the kidnap
    columns of both, the EKF's reset maximum, and four trial sets of every seed recorded."""
    assert [(row.sensor, row.dropout) for row in comparison.rows] == [
        ("lidar", None),
        ("lidar_scaled", None),
    ], "rows should come one per sensor and dropout, sorted by sensor"
    for row in comparison.rows:
        expected = {f"{prefix}{key}" for prefix in ("ekf_", "mcl_") for key in TRACKING_KEYS}
        expected |= {
            f"{prefix}{key}" for prefix in ("mcl_kidnap_", "ekf_kidnap_") for key in KIDNAP_KEYS
        }
        expected |= {"ekf_max_recent_resets", "ekf_kidnap_max_recent_resets"}
        missing = expected - set(row.metrics)
        assert not missing, f"row {row.sensor} lacks the columns {sorted(missing)}"
        assert row.metrics["ekf_trials"] == len(SEEDS) == row.metrics["mcl_kidnap_trials"]
        for prefix in ("mcl_kidnap_", "ekf_kidnap_"):
            share = row.metrics[f"{prefix}confidently_wrong"]
            assert 0.0 <= share <= 1.0, f"{prefix}confidently_wrong {share} is not a share"
            assert row.metrics[f"{prefix}ms_per_step"] > 0.0, "every set should be timed"
    assert len(comparison.records) == 2 * len(SETS) * len(SEEDS), "one record per trial"
    ekf_resets = []
    for record in comparison.records:
        resets = record.meta["resets"]
        if record.meta["mode"].startswith("ekf"):
            assert resets["max_recent"] <= len(resets["steps"]), f"inconsistent resets {resets}"
            ekf_resets += resets["steps"]
        else:
            assert resets is None, "MCL has no resets to record"
    # Carried a metre or more, the EKF gates its lidar beams: some kidnapped run resets.
    assert ekf_resets, "no EKF record noted a reset, though the kidnap set moved every EKF"


def test_both_filters_see_the_same_dead_reckoning(comparison):
    """Both tracking sets run the same seeds, so their encoders and dead reckoning are one."""
    for row in comparison.rows:
        for key in ("position_rmse", "heading_rmse", "lost"):
            assert (
                row.metrics[f"ekf_dead_reckoning_{key}"] == row.metrics[f"mcl_dead_reckoning_{key}"]
            ), f"dead reckoning {key} differs between the EKF and MCL trials of {row.sensor}"


def test_the_mcl_scale_reaches_only_its_sensors_mcl(comparison):
    """A scale of 2 on every axis halves MCL's NEES and axis ratios on that sensor's row and
    changes no pose and nothing of the EKF."""
    plain, scaled = (row.metrics for row in comparison.rows)
    for key in ("mean_nees", "axis_ratio_x", "axis_ratio_y", "axis_ratio_theta"):
        np.testing.assert_allclose(
            scaled[f"mcl_{key}"], plain[f"mcl_{key}"] / 2.0, rtol=1e-9, err_msg=key
        )
        assert scaled[f"ekf_{key}"] == plain[f"ekf_{key}"], f"the scale changed the EKF's {key}"
    assert scaled["mcl_position_rmse"] == plain["mcl_position_rmse"], "the scale moved MCL"


def test_compare_localizers_rejects_a_scale_for_an_unknown_sensor():
    with pytest.raises(ValueError, match="sonar"):
        compare_localizers(
            SEEDS, {"lidar": lidar}, (None,), mcl_scales={"sonar": SCALE}, steps=STEPS
        )


def _trial(errors, variances, kidnap_at):
    """A hand-built trial at the origin with x errors ``errors`` and isotropic ``variances``."""
    count = len(errors)
    estimates = np.zeros((count, 3))
    estimates[:, 0] = errors
    return Trial(
        true_poses=np.zeros((count, 3)),
        estimates=estimates,
        covariances=np.array([variance * np.eye(3) for variance in variances]),
        dead_reckoning=np.zeros((count, 3)),
        beams_used=np.full(count, 4),
        collided=False,
        kidnap_at=kidnap_at,
    )


def test_confidently_wrong_needs_a_large_error_an_uncovered_one_and_no_recent_reset():
    """From the kidnapping on: error over LOST_ERROR, NEES over 9.35 and, for a filter with
    resets, none within the window. Step 5 resets; with a window of 3 it covers steps 5 to 7."""
    small, large = 1e-4, 10.0
    errors = [1.0] * 10 + [LOST_ERROR - 0.1, 1.0]
    variances = [small] * 9 + [large, small, small]
    trial = _trial(errors, variances, kidnap_at=4)
    # Post-kidnap steps 4..11: 4 counts; 5-7 had a reset in the window; 8 counts; 9's covariance
    # covers the error (NEES 0.1); 10's error is under LOST_ERROR; 11 counts.
    np.testing.assert_array_equal(
        confidently_wrong(trial, reset_steps=[5], window=3),
        [True, False, False, False, True, False, False, True],
        err_msg="an EKF step is confidently wrong only when off, uncovered and not resetting",
    )
    np.testing.assert_array_equal(
        confidently_wrong(trial, reset_steps=None, window=3),
        [True, True, True, True, True, False, False, True],
        err_msg="without resets (MCL) only the error and the NEES count",
    )


def test_confidently_wrong_needs_a_kidnapped_trial():
    with pytest.raises(ValueError, match="kidnap"):
        confidently_wrong(_trial([1.0] * 3, [1e-4] * 3, kidnap_at=None), None, 3)


def _metrics(**changes):
    """A row's metrics with fixed values, the dead reckoning on the EKF's keys."""
    metrics = {
        "ekf_position_rmse": 0.00183,
        "ekf_heading_rmse": 0.00171,
        "ekf_mean_nees": 3.181,
        "ekf_axis_ratio_x": 1.02,
        "ekf_axis_ratio_y": 0.98,
        "ekf_axis_ratio_theta": 1.104,
        "ekf_nees_below_upper_95": 0.9704,
        "ekf_lost": 0.0,
        "ekf_trials": 2.0,
        "ekf_ms_per_step": 1.234,
        "ekf_max_recent_resets": 1.0,
        "ekf_dead_reckoning_position_rmse": 0.93312,
        "ekf_dead_reckoning_heading_rmse": 0.32549,
        "ekf_dead_reckoning_lost": 2.0,
        "mcl_position_rmse": 0.007,
        "mcl_heading_rmse": 0.009,
        "mcl_mean_nees": 0.5712,
        "mcl_axis_ratio_x": 0.15,
        "mcl_axis_ratio_y": 0.08,
        "mcl_axis_ratio_theta": 0.32,
        "mcl_nees_below_upper_95": 1.0,
        "mcl_lost": 0.0,
        "mcl_trials": 2.0,
        "mcl_ms_per_step": 45.678,
        "mcl_kidnap_global_share": 1.0,
        "mcl_kidnap_median_steps_to_localize": 31.0,
        "mcl_kidnap_seconds_to_localize": 3.1,
        "mcl_kidnap_recovered_share": 0.5,
        "mcl_kidnap_median_steps_to_recover": 40.0,
        "mcl_kidnap_seconds_to_recover": 4.0,
        "mcl_kidnap_confidently_wrong": 0.0124,
        "mcl_kidnap_confidently_wrong_localized": 0.0062,
        "mcl_kidnap_trials": 2.0,
        "mcl_kidnap_ms_per_step": 44.0,
        "ekf_kidnap_global_share": 1.0,
        "ekf_kidnap_median_steps_to_localize": 0.0,
        "ekf_kidnap_seconds_to_localize": 0.0,
        "ekf_kidnap_recovered_share": 0.0,
        "ekf_kidnap_median_steps_to_recover": math.nan,
        "ekf_kidnap_seconds_to_recover": math.nan,
        "ekf_kidnap_confidently_wrong": 0.9,
        "ekf_kidnap_confidently_wrong_localized": math.nan,
        "ekf_kidnap_trials": 2.0,
        "ekf_kidnap_ms_per_step": 1.1,
        "ekf_kidnap_max_recent_resets": 4.0,
    }
    return {**metrics, **changes}


REPORT_META = {
    "seeds": [1, 2, 3],
    "steps": 600,
    "kidnap_at": 300,
    "deadline": 250,
    "hold": 50,
    "mcl_settings": {"num_particles": 2000, "max_beams": 16, "model_std": 0.15},
    "mcl_scales": {"hc_sr04": [1.0, 1.0, 1.0], "lidar": [0.5, 0.25, 2.0]},
    "ekf_reset": {
        "reset_after": 3,
        "min_candidates": 2,
        "reset_std": [0.2, 0.2, 0.1],
        "reset_window": 20,
        "nis_quantile": 0.999,
    },
    "machines": ["Linux-aarch64"],
    "blas_threads": ["OPENBLAS_NUM_THREADS=1"],
    "jobs_notes": ["2 jobs"],
}

EXPECTED_REPORT = """\
# Localization comparison: EKF and MCL

Seeds 1-3; 600 steps a run; the kidnap sets carry the robot off at step 300.
Localized: within 0.2 m and 0.2 rad at each of the 50 steps that begin 250 steps after the \
start or the kidnapping; a
step is 0.1 s.

MCL: 2000 particles, 16 beams, model_std 0.15; covariance scale hc_sr04 (1, 1, 1), lidar \
(0.5, 0.25, 2).
EKF reset: after 3 counting scans of at least 2 candidates, variance floor (0.2, 0.2, 0.1) \
squared, window 20 scans.
EKF hold: an update that does not converge or has an NIS over the chi-square 0.999 quantile \
keeps P.

## Tracking from the noisy start

| Sensor | Dropout | Filter | Position RMSE (m) | Heading RMSE (rad) | Mean NEES | Mean \
e_i^2/P_ii (x, y, theta) | NEES <= 9.35 | Lost | Max recent resets | ms/step |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| lidar | own | EKF | 0.0018 | 0.0017 | 3.18 | 1.02, 0.98, 1.10 | 0.970 | 0/2 | 1 | 1.23 |
| lidar | own | MCL | 0.0070 | 0.0090 | 0.57 | 0.15, 0.08, 0.32 | 1.000 | 0/2 | - | 45.68 |
| lidar | own | Dead reckoning | 0.9331 | 0.3255 | - | - | - | 2/2 | - | - |
| hc_sr04 | 0.3 | EKF | 0.0018 | 0.0017 | 3.18 | 1.02, 0.98, 1.10 | 0.970 | 0/2 | 1 | 1.23 |
| hc_sr04 | 0.3 | MCL | 0.0070 | 0.0090 | 0.57 | 0.15, 0.08, 0.32 | 1.000 | 0/2 | - | 45.68 |
| hc_sr04 | 0.3 | Dead reckoning | 0.9331 | 0.3255 | - | - | - | 2/2 | - | - |

## Global start and kidnapping

Confidently wrong: the share of steps after the kidnapping, over all kidnapped runs, with \
position error over 0.5 m,
NEES over 9.35 and, for the EKF, no reset in the last 20 scans.
Localized runs: the same share over only the runs that had localized before the kidnapping \
(n/a when none had).

| Sensor | Dropout | Filter | Start | Localized | Median to localize | Recovered | Median to \
recover | Confidently wrong (all runs) | Confidently wrong (localized runs) | Max recent \
resets | ms/step |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| lidar | own | MCL | anywhere | 1.00 | 31 steps (3.1 s) | 0.50 | 40 steps (4.0 s) | 0.012 | \
0.006 | - | 44.00 |
| lidar | own | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.00 | n/a | 0.900 | n/a | 4 | \
1.10 |
| hc_sr04 | 0.3 | MCL | anywhere | 0.00 | n/a | 0.50 | 40 steps (4.0 s) | 0.012 | 0.006 | - | \
44.00 |
| hc_sr04 | 0.3 | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.00 | n/a | 0.900 | n/a | 4 | \
1.10 |

Timings: Linux-aarch64; OPENBLAS_NUM_THREADS=1; concurrent jobs: 2 jobs. ms/step: the \
filter's predict, update, pose and
covariance per step.
"""


def test_format_markdown_gives_the_report_for_fixed_rows():
    rows = [
        ComparisonRow("lidar", None, _metrics()),
        ComparisonRow(
            "hc_sr04",
            0.3,
            _metrics(
                mcl_kidnap_global_share=0.0,
                mcl_kidnap_median_steps_to_localize=math.nan,
                mcl_kidnap_seconds_to_localize=math.nan,
            ),
        ),
    ]
    assert format_markdown(rows, REPORT_META) == EXPECTED_REPORT


def test_format_markdown_keeps_prose_within_the_markdown_line_length():
    """CI's markdownlint allows 120 characters outside tables; the report's prose must fit."""
    meta = {
        **REPORT_META,
        "machines": ["Linux-6.8.12-1021-tegra-aarch64-with-glibc2.39"],
        "blas_threads": ["OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1"],
        "jobs_notes": ["2 jobs (one per sensor), 1 BLAS thread each; review agents alongside"],
        "mcl_scales": {"hc_sr04": [0.174, 0.18, 0.525], "lidar": [0.13, 0.124, 0.308]},
    }
    prose = [
        line
        for line in format_markdown([ComparisonRow("lidar", None, _metrics())], meta).splitlines()
        if not line.startswith("|")
    ]
    assert max(len(line) for line in prose) <= 120


def test_format_markdown_says_when_the_ekf_reset_is_off():
    meta = {**REPORT_META, "ekf_reset": {**REPORT_META["ekf_reset"], "reset_after": None}}
    assert "EKF reset: off." in format_markdown([], meta)


@pytest.mark.parametrize("nis_quantile", [None, "missing"])
def test_format_markdown_says_when_the_ekf_hold_is_off(nis_quantile):
    """``nis_quantile=None`` turns the hold off, and records written before the hold have no key."""
    reset = {**REPORT_META["ekf_reset"], "nis_quantile": nis_quantile}
    if nis_quantile == "missing":
        del reset["nis_quantile"]
    assert "EKF hold: off." in format_markdown([], {**REPORT_META, "ekf_reset": reset})


SETTINGS = {
    "steps": 12,
    "kidnap_at": 4,
    "ekf_reset": {
        "reset_after": 3,
        "min_candidates": 2,
        "reset_std": [0.2, 0.2, 0.1],
        "reset_window": 20,
        "nis_quantile": 0.999,
    },
    "mcl_settings": {"num_particles": 150, "covariance_scale": [1.0, 1.0, 1.0]},
}


def _record(seed=1, sensor="lidar", mode="ekf_kidnap", timings=0.5, **settings):
    """A record of a hand-built kidnapped trial with the given seed, labels and settings."""
    meta = {
        "seed": seed,
        "sensor": sensor,
        "dropout": None,
        "mode": mode,
        "settings": {**SETTINGS, **settings},
        "resets": {"steps": [5], "max_recent": 1},
        "timings": {
            "seconds": {"predict": timings, "update": 0.25, "pose": 0.0, "covariance": 1e-6},
            "machine": "test",
            "blas_threads": "OPENBLAS_NUM_THREADS=1",
            "jobs_note": "none",
        },
    }
    trial = _trial([1.0] * 12, [1e-4] * 12, kidnap_at=4)
    return TrialRecord(trial, meta)


@pytest.mark.parametrize("kidnap_at, collided", [(4, False), (None, True)])
def test_trial_record_round_trips_through_an_npz_without_pickle(tmp_path, kidnap_at, collided):
    record = _record()
    trial = dataclasses.replace(record.trial, kidnap_at=kidnap_at, collided=collided)
    record = TrialRecord(trial, {**record.meta, "dropout": 0.3})
    path = tmp_path / record.file_name
    record.save(path)
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data["meta"])) == record.meta, "meta should be one JSON string"
    loaded = TrialRecord.load(path)
    pairs = {
        "true_poses": (trial.true_poses, loaded.trial.true_poses),
        "estimates": (trial.estimates, loaded.trial.estimates),
        "covariances": (trial.covariances, loaded.trial.covariances),
        "dead_reckoning": (trial.dead_reckoning, loaded.trial.dead_reckoning),
        "beams_used": (trial.beams_used, loaded.trial.beams_used),
    }
    for name, (expected, actual) in pairs.items():
        np.testing.assert_array_equal(actual, expected, err_msg=f"{name} changed on the way")
        assert actual.dtype == expected.dtype, f"{name} came back as {actual.dtype}"
    assert loaded.trial.collided is collided and loaded.trial.kidnap_at == kidnap_at
    assert loaded.meta == record.meta


def test_record_file_names_differ_by_sensor_dropout_mode_and_seed():
    names = {
        _record(seed=seed, sensor=sensor, mode=mode).file_name
        for seed in (1, 2)
        for sensor in ("lidar", "hc_sr04")
        for mode in ("ekf_kidnap", "mcl_kidnap")
    }
    assert len(names) == 8, f"record file names collide: {sorted(names)}"


def test_check_records_ignores_timings():
    check_records([_record(seed=1, timings=0.5), _record(seed=2, timings=9.0)])


def test_check_records_rejects_a_duplicate_trial_even_with_other_timings():
    with pytest.raises(ValueError, match="duplicate"):
        check_records([_record(seed=1, timings=0.5), _record(seed=1, timings=9.0)])


@pytest.mark.parametrize(
    "change",
    [
        {"steps": 13},
        {"kidnap_at": 5},
        {"ekf_reset": {**SETTINGS["ekf_reset"], "reset_after": 4}},
        {"mcl_settings": {"num_particles": 300, "covariance_scale": [1.0, 1.0, 1.0]}},
        {"mcl_settings": {"num_particles": 150, "covariance_scale": [2.0, 1.0, 1.0]}},
    ],
)
def test_check_records_rejects_mismatched_settings(change):
    with pytest.raises(ValueError, match="settings"):
        check_records([_record(seed=1), _record(seed=2, **change)])


# The EKF's settings as records written before the hold (2026-11-09) carry them.
RESET_BEFORE_THE_HOLD = {k: v for k, v in SETTINGS["ekf_reset"].items() if k != "nis_quantile"}


def test_check_records_ignores_the_ekf_settings_of_mcl_records():
    """MCL does not read ``ekf_reset``, so MCL records from before the hold merge with new EKF
    records; EST-3 reruns only the EKF that way."""
    check_records(
        [_record(seed=1), _record(seed=1, mode="mcl_kidnap", ekf_reset=RESET_BEFORE_THE_HOLD)]
    )


def test_a_record_without_nis_quantile_ran_without_the_hold():
    """A missing ``nis_quantile`` reads as None (the hold off), which is what such a run did."""
    off = {**SETTINGS["ekf_reset"], "nis_quantile": None}
    check_records(
        [_record(seed=1, ekf_reset=RESET_BEFORE_THE_HOLD), _record(seed=2, ekf_reset=off)]
    )
    with pytest.raises(ValueError, match="settings"):
        check_records([_record(seed=1, ekf_reset=RESET_BEFORE_THE_HOLD), _record(seed=2)])


def test_check_records_allows_each_sensor_its_own_mcl_scale():
    scaled = {"num_particles": 150, "covariance_scale": [2.0, 1.0, 1.0]}
    check_records([_record(sensor="lidar"), _record(sensor="hc_sr04", mcl_settings=scaled)])
    with pytest.raises(ValueError, match="settings"):
        check_records(
            [
                _record(
                    sensor="hc_sr04",
                    mcl_settings={"num_particles": 300, "covariance_scale": [2.0, 1.0, 1.0]},
                ),
                _record(sensor="lidar"),
            ]
        )


def test_merge_records_needs_all_four_sets_of_a_row():
    with pytest.raises(ValueError, match="mcl_tracking"):
        merge_records([_record(mode="ekf_kidnap")], deadline=1, hold=1)


CLI_RUN = ["--sensors", "lidar", "--dropouts", "own", "--steps", str(STEPS)]
CLI_RUN += ["--kidnap-at", str(KIDNAP_AT), "--deadline", str(DEADLINE), "--hold", str(HOLD)]
CLI_RUN += ["--particles", str(FEW_PARTICLES.num_particles), "--mcl-scale", "lidar", "2", "2", "2"]


@pytest.fixture(scope="module")
def cli_records(tmp_path_factory):
    """Seeds 1 and 2 run apart through the CLI, each into its own record directory."""
    root = tmp_path_factory.mktemp("cli")
    directories = [root / f"seed{seed}" for seed in SEEDS]
    for seed, directory in zip(SEEDS, directories, strict=True):
        main(["--seeds", str(seed), "--records", str(directory), *CLI_RUN])
    return directories


def _row(comparison, sensor):
    (row,) = [row for row in comparison.rows if row.sensor == sensor]
    return row


def test_records_run_apart_merge_to_the_one_run_comparison(comparison, cli_records):
    """Seeds 1 and 2 run apart with lidar scaled by 2 merge to the one-run lidar_scaled row,
    whatever order the records come in."""
    expected = _without_timings(_row(comparison, "lidar_scaled").metrics)
    for directories in (cli_records, cli_records[::-1]):
        merged = merge_records(load_records(directories), deadline=DEADLINE, hold=HOLD)
        # assert_equal, because a set that never recovered has NaN medians
        np.testing.assert_equal(
            _without_timings(_row(merged, "lidar").metrics),
            expected,
            err_msg="records of seeds 1 and 2 should merge to the one-run comparison",
        )


def _merge_args(cli_records):
    args = ["--from-records", *map(str, cli_records)]
    return [*args, "--deadline", str(DEADLINE), "--hold", str(HOLD)]


def test_cli_report_goes_to_the_file_and_stdout(cli_records, tmp_path, capsys):
    report = tmp_path / "out" / "report.md"
    main([*_merge_args(cli_records), "--report", str(report)])
    printed = capsys.readouterr().out
    assert report.read_text(encoding="utf-8") == printed, "--report should hold what is printed"
    assert printed.startswith("# Localization comparison"), printed[:200]


def test_cli_without_report_writes_nothing(cli_records, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    watched = [tmp_path, *cli_records]
    before = [sorted(directory.rglob("*")) for directory in watched]
    main(_merge_args(cli_records))
    assert capsys.readouterr().out.startswith("# Localization comparison")
    after = [sorted(directory.rglob("*")) for directory in watched]
    assert after == before, "without --report nothing should be written"


def test_cli_reuses_the_records_already_written(cli_records, monkeypatch, capsys):
    """A rerun into the same directory loads its records and runs no trial."""
    times = sorted((path.name, path.stat().st_mtime_ns) for path in cli_records[0].glob("*"))

    def no_trial(*args, **kwargs):
        raise AssertionError("a trial ran though its record was there")

    monkeypatch.setattr(comparison_module, "run_trial", no_trial)
    main(["--seeds", "1", "--records", str(cli_records[0]), *CLI_RUN])
    assert capsys.readouterr().out.startswith("# Localization comparison")
    assert sorted((p.name, p.stat().st_mtime_ns) for p in cli_records[0].glob("*")) == times


def test_run_records_saves_each_trial_as_it_ends(tmp_path):
    """A run that fails keeps the records of the trials it finished."""
    calls = []

    def flaky_lidar():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("the second trial fails")
        return lidar()

    with pytest.raises(RuntimeError, match="second"):
        run_records(
            (1,),
            {"lidar": flaky_lidar},
            (None,),
            steps=20,
            kidnap_at=10,
            mcl_settings=FEW_PARTICLES,
            records_dir=tmp_path,
        )
    (saved,) = tmp_path.iterdir()
    assert saved.name == "lidar_own_ekf_tracking_seed1.npz"
    assert len(TrialRecord.load(saved).trial.true_poses) == 20


@pytest.mark.parametrize(
    "argv, message",
    [
        (
            ["--seeds", "1", "--sensors", "lidar", "--mcl-scale", "hc_sr04", "1", "1", "1"],
            "hc_sr04",
        ),
        (["--seeds", "5-3"], "backwards"),
        (["--seeds", "1-3", "2"], "repeats"),
        (["--seeds", "x"], "not a seed"),
        (["--seeds", "1", "--kidnap-at", "60", "--deadline", "50", "--hold", "20"], "at most"),
        (["--seeds", "1", "--mcl-scale", "lidar", "1", "-1", "1"], "mcl_scales"),
        (["--from-records", "somewhere", "--steps", "120"], "--steps"),
        (["--from-records", "somewhere", "--sensors", "lidar", "--jobs-note", "x"], "--jobs-note"),
        (["--from-records", "somewhere", "--seeds", "1"], "--seeds"),
        (["--steps", "10"], "--seeds is required"),
    ],
)
def test_cli_refuses_bad_arguments_before_running(argv, message, monkeypatch, capsys):
    monkeypatch.setattr(comparison_module, "run_trial", None)  # nothing may run
    with pytest.raises(SystemExit):
        main(argv)
    assert message in capsys.readouterr().err


def test_ekf_reset_defaults_are_the_filters():
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(EkfLocalizer).parameters.items()
        if name in {field.name for field in dataclasses.fields(EkfReset)}
    }
    assert dataclasses.asdict(EkfReset()) == defaults, "EkfReset should default as EkfLocalizer"


def test_nees_per_step_agrees_with_the_evaluations_nees():
    trial = _trial([0.3, 0.1, 0.7], [0.01, 0.2, 0.05], kidnap_at=1)
    np.testing.assert_allclose(nees_per_step(trial), average_nees([trial]), rtol=1e-12)


def _known_record(seed, mode):
    """A record with figures worked out by hand (``test_merge_records_computes_...``)."""
    if mode.endswith("tracking"):
        # Constant error (0.1, 0.2, 0.3) against variances 0.01: e_i^2/P_ii = 1, 4, 9.
        estimates = np.tile([0.1, 0.2, 0.3], (10, 1))
        covariances = np.tile(0.01 * np.eye(3), (10, 1, 1))
        kidnap_at = None
    else:
        # Kidnapped at row 4. Seed 1 is off by 1 m at rows 4-5 only: localized (rows 2-3 on),
        # recovered after 2 steps. Seed 2 is also off at rows 2-3 (not localized) and 6-7 (not
        # recovered).
        off = [4, 5] if seed == 1 else [2, 3, 4, 5, 6, 7]
        estimates = np.zeros((10, 3))
        estimates[off, 0] = 1.0
        covariances = np.tile(1e-4 * np.eye(3), (10, 1, 1))
        kidnap_at = 4
    trial = Trial(
        true_poses=np.zeros((10, 3)),
        estimates=estimates,
        covariances=covariances,
        dead_reckoning=np.zeros((10, 3)),
        beams_used=np.full(10, 4),
        collided=False,
        kidnap_at=kidnap_at,
    )
    settings = {**SETTINGS, "steps": 10, "kidnap_at": 4}
    settings["ekf_reset"] = {**SETTINGS["ekf_reset"], "reset_window": 2}
    seconds = {"predict": 0.01, "update": 0.02, "pose": 0.0, "covariance": 0.0}
    if seed == 2:
        seconds["update"] = 0.04
    meta = {
        "seed": seed,
        "sensor": "lidar",
        "dropout": None,
        "mode": mode,
        "settings": settings,
        # The EKF resets at row 4; with a window of 2 that covers rows 4 and 5.
        "resets": {"steps": [4], "max_recent": 2 * seed + 1} if mode.startswith("ekf") else None,
        "timings": {"seconds": seconds, "machine": "m", "blas_threads": "b", "jobs_note": "n"},
    }
    return TrialRecord(trial, meta)


KNOWN_MODES = ("ekf_tracking", "mcl_tracking", "mcl_kidnap", "ekf_kidnap")


def test_merge_records_computes_each_sets_figures_by_hand():
    records = [_known_record(seed, mode) for seed in (2, 1) for mode in KNOWN_MODES]
    (row,) = merge_records(records, deadline=2, hold=2).rows
    metrics = row.metrics
    expected = {
        # 0.03 s and 0.05 s of filter calls over 2 x 10 steps.
        "ekf_ms_per_step": 4.0,
        "mcl_kidnap_ms_per_step": 4.0,
        "ekf_axis_ratio_x": 1.0,
        "ekf_axis_ratio_y": 4.0,
        "ekf_axis_ratio_theta": 9.0,
        "mcl_axis_ratio_theta": 9.0,
        "ekf_mean_nees": 14.0,
        # max_recent 3 and 5: the larger.
        "ekf_max_recent_resets": 5.0,
        "ekf_kidnap_max_recent_resets": 5.0,
        "mcl_kidnap_global_share": 0.5,
        "mcl_kidnap_median_steps_to_localize": 0.0,
        "mcl_kidnap_recovered_share": 0.5,
        "mcl_kidnap_median_steps_to_recover": 2.0,
        "mcl_kidnap_seconds_to_recover": 0.2,
        # MCL: rows 4-5 of seed 1 and 4-7 of seed 2 are off and uncovered: 6 of 12 steps;
        # among the localized runs (seed 1) 2 of 6.
        "mcl_kidnap_confidently_wrong": 0.5,
        "mcl_kidnap_confidently_wrong_localized": 1.0 / 3.0,
        # EKF: the reset covers rows 4-5, leaving rows 6-7 of seed 2: 2 of 12; seed 1 none.
        "ekf_kidnap_confidently_wrong": 1.0 / 6.0,
        "ekf_kidnap_confidently_wrong_localized": 0.0,
        "ekf_kidnap_seconds_to_localize": 0.0,
    }
    for key, value in expected.items():
        np.testing.assert_allclose(metrics[key], value, rtol=1e-12, atol=1e-15, err_msg=key)
    assert "max_recent_resets" not in {key.removeprefix("mcl_") for key in metrics}, (
        "MCL has no resets, so no reset column"
    )


def test_merge_records_takes_the_ekf_settings_from_an_ekf_record():
    """The report's EKF settings and the reset window come from the EKF records even when an MCL
    record from before the hold comes first."""
    ekf_records = [_known_record(seed, m) for seed in (1, 2) for m in KNOWN_MODES if "ekf" in m]
    mcl_records = []
    for seed in (1, 2):
        for mode in ("mcl_tracking", "mcl_kidnap"):
            record = _known_record(seed, mode)
            settings = {**record.meta["settings"], "ekf_reset": RESET_BEFORE_THE_HOLD}
            mcl_records.append(TrialRecord(record.trial, {**record.meta, "settings": settings}))
    merged = merge_records(mcl_records + ekf_records, deadline=2, hold=2)
    expected = merge_records(ekf_records + mcl_records, deadline=2, hold=2)
    assert merged.meta["ekf_reset"] == ekf_records[0].meta["settings"]["ekf_reset"]
    assert merged.rows == expected.rows, "The record order should not change any figure."


def test_merge_records_refuses_sets_of_different_seeds():
    records = [_known_record(seed, mode) for seed in (1, 2) for mode in KNOWN_MODES]
    with pytest.raises(ValueError, match="sets of different seeds"):
        merge_records(records[:-1], deadline=2, hold=2)


def test_merge_records_refuses_rows_of_different_seeds():
    """The report prints one seed list, so every row must have run the same seeds."""
    records = [_known_record(seed, mode) for seed in (1, 2) for mode in KNOWN_MODES]
    other = [
        TrialRecord(record.trial, {**record.meta, "sensor": "lidar_b"})
        for record in records
        if record.meta["seed"] == 1
    ]
    with pytest.raises(ValueError, match="same seeds"):
        merge_records(records + other, deadline=2, hold=2)


def test_cli_turns_records_that_do_not_merge_into_an_argument_error(tmp_path, capsys):
    _record(mode="ekf_kidnap").save(tmp_path / "one.npz")
    (tmp_path / "empty").mkdir()
    for directories in ([str(tmp_path)], [str(tmp_path / "empty")]):
        with pytest.raises(SystemExit):
            main(["--from-records", *directories, "--deadline", "1", "--hold", "1"])
    errors = capsys.readouterr().err
    assert "mcl_tracking" in errors and "No trial records" in errors


def test_load_records_refuses_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No trial records"):
        load_records([tmp_path])


def test_compare_localizers_refuses_a_window_past_the_kidnapping_before_running(monkeypatch):
    monkeypatch.setattr(comparison_module, "run_trial", None)  # nothing may run
    with pytest.raises(ValueError, match="deadline"):
        compare_localizers(SEEDS, {"lidar": lidar}, (None,), kidnap_at=60, deadline=50, hold=20)
