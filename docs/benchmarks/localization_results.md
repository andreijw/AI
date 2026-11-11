# Localization comparison: EKF and MCL

Seeds 101-150; 600 steps a run; the kidnap sets carry the robot off at step 300.
Localized: within 0.2 m and 0.2 rad at each of the 50 steps that begin 250 steps after the start or the kidnapping; a
step is 0.1 s.

MCL: 2000 particles, 16 beams, model_std 0.15; covariance scale hc_sr04 (0.174, 0.18, 0.525), lidar (0.13, 0.124,
0.308).
EKF reset: after 3 counting scans of at least 2 candidates, variance floor (0.2, 0.2, 0.1) squared, window 20 scans.
EKF hold: an update that does not converge or has an NIS over the chi-square 0.999 quantile keeps P.
EKF off-map count: a scan with no candidate beam while the estimate is not in free space counts toward the reset.

## Tracking from the noisy start

| Sensor | Dropout | Filter | Position RMSE (m) | Heading RMSE (rad) | Mean NEES | Mean e_i^2/P_ii (x, y, theta) | NEES <= 9.35 | Lost | Max recent resets | ms/step |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| hc_sr04 | own | EKF | 0.0062 | 0.0120 | 3.14 | 1.01, 1.04, 1.07 | 0.970 | 0/50 | 3 | 3.17 |
| hc_sr04 | own | MCL | 0.0118 | 0.0229 | 2.94 | 1.00, 1.01, 0.95 | 0.963 | 0/50 | - | 51.13 |
| hc_sr04 | own | Dead reckoning | 0.7988 | 0.2832 | - | - | - | 50/50 | - | - |
| hc_sr04 | 0.3 | EKF | 0.0068 | 0.0147 | 3.38 | 1.03, 1.04, 1.20 | 0.967 | 0/50 | 1 | 3.04 |
| hc_sr04 | 0.3 | MCL | 0.0132 | 0.0264 | 3.28 | 1.13, 1.15, 1.02 | 0.949 | 0/50 | - | 38.29 |
| hc_sr04 | 0.3 | Dead reckoning | 0.7988 | 0.2832 | - | - | - | 50/50 | - | - |
| lidar | own | EKF | 0.0018 | 0.0018 | 3.09 | 1.03, 1.07, 1.02 | 0.972 | 0/50 | 0 | 4.40 |
| lidar | own | MCL | 0.0070 | 0.0087 | 3.10 | 1.00, 1.10, 0.97 | 0.957 | 0/50 | - | 37.04 |
| lidar | own | Dead reckoning | 0.7988 | 0.2832 | - | - | - | 50/50 | - | - |
| lidar | 0.3 | EKF | 0.0021 | 0.0021 | 3.09 | 1.04, 1.06, 1.02 | 0.970 | 0/50 | 0 | 4.30 |
| lidar | 0.3 | MCL | 0.0070 | 0.0088 | 3.16 | 1.08, 1.04, 1.00 | 0.955 | 0/50 | - | 36.71 |
| lidar | 0.3 | Dead reckoning | 0.7988 | 0.2832 | - | - | - | 50/50 | - | - |

## Global start and kidnapping

Confidently wrong: the share of steps after the kidnapping, over all kidnapped runs, with position error over 0.5 m,
NEES over 9.35 and, for the EKF, no reset in the last 20 scans.
Localized runs: the same share over only the runs that had localized before the kidnapping (n/a when none had).
Wrong with σ < 5 cm: the confidently wrong steps whose position standard deviation (larger axis) is under 0.05 m.

| Sensor | Dropout | Filter | Start | Localized | Median to localize | Recovered | Median to recover | Confidently wrong (all runs) | Confidently wrong (localized runs) | Wrong with σ < 5 cm (all runs) | Max recent resets | ms/step |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| hc_sr04 | own | MCL | anywhere | 0.80 | 2.5 steps (0.2 s) | 0.94 | 55 steps (5.5 s) | 0.281 | 0.245 | 0.280 | - | 50.84 |
| hc_sr04 | own | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.20 | 78 steps (7.8 s) | 0.163 | 0.163 | 0.095 | 7 | 2.69 |
| hc_sr04 | 0.3 | MCL | anywhere | 0.88 | 5.5 steps (0.6 s) | 0.86 | 39 steps (3.9 s) | 0.284 | 0.256 | 0.282 | - | 37.55 |
| hc_sr04 | 0.3 | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.12 | 90 steps (9.0 s) | 0.189 | 0.189 | 0.126 | 7 | 2.53 |
| lidar | own | MCL | anywhere | 0.96 | 0 steps (0.0 s) | 0.96 | 21.5 steps (2.1 s) | 0.127 | 0.129 | 0.127 | - | 37.23 |
| lidar | own | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.28 | 111.5 steps (11.2 s) | 0.124 | 0.124 | 0.052 | 7 | 4.02 |
| lidar | 0.3 | MCL | anywhere | 0.96 | 0 steps (0.0 s) | 0.98 | 21 steps (2.1 s) | 0.118 | 0.102 | 0.118 | - | 37.21 |
| lidar | 0.3 | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.24 | 90 steps (9.0 s) | 0.165 | 0.165 | 0.094 | 7 | 3.98 |

Timings: Linux-6.8.12-1021-tegra-aarch64-with-glibc2.39; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1;
concurrent jobs: 2 J3 jobs (one per sensor), 1 BLAS thread each; review agents may run tests alongside / EKF-only rerun
with the off-map count: MCL records reused from 2026-11-09; queue jobs alongside, 1 BLAS thread each. ms/step: the
filter's predict, update, pose and covariance per step.
