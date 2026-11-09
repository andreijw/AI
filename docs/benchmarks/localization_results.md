# Localization comparison: EKF and MCL

Seeds 101-150; 600 steps a run; the kidnap sets carry the robot off at step 300.
Localized: within 0.2 m and 0.2 rad at each of the 50 steps that begin 250 steps after the start or the kidnapping; a
step is 0.1 s.

MCL: 2000 particles, 16 beams, model_std 0.15; covariance scale hc_sr04 (0.174, 0.18, 0.525), lidar (0.13, 0.124,
0.308).
EKF reset: after 3 counting scans of at least 2 candidates, variance floor (0.2, 0.2, 0.1) squared, window 20 scans.

## Tracking from the noisy start

| Sensor | Dropout | Filter | Position RMSE (m) | Heading RMSE (rad) | Mean NEES | Mean e_i^2/P_ii (x, y, theta) | NEES <= 9.35 | Lost | Max recent resets | ms/step |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| hc_sr04 | own | EKF | 0.0068 | 0.0121 | 3.52 | 1.06, 1.14, 1.19 | 0.964 | 0/50 | 3 | 2.22 |
| hc_sr04 | own | MCL | 0.0118 | 0.0229 | 2.94 | 1.00, 1.01, 0.95 | 0.963 | 0/50 | - | 51.13 |
| hc_sr04 | own | Dead reckoning | 0.7988 | 0.2832 | - | - | - | 50/50 | - | - |
| hc_sr04 | 0.3 | EKF | 0.3268 | 0.1347 | 279.22 | 76.99, 4.01, 19.62 | 0.955 | 1/50 | 4 | 2.18 |
| hc_sr04 | 0.3 | MCL | 0.0132 | 0.0264 | 3.28 | 1.13, 1.15, 1.02 | 0.949 | 0/50 | - | 38.29 |
| hc_sr04 | 0.3 | Dead reckoning | 0.7988 | 0.2832 | - | - | - | 50/50 | - | - |
| lidar | own | EKF | 0.0018 | 0.0018 | 3.39 | 1.06, 1.09, 1.19 | 0.970 | 0/50 | 0 | 3.24 |
| lidar | own | MCL | 0.0070 | 0.0087 | 3.10 | 1.00, 1.10, 0.97 | 0.957 | 0/50 | - | 37.04 |
| lidar | own | Dead reckoning | 0.7988 | 0.2832 | - | - | - | 50/50 | - | - |
| lidar | 0.3 | EKF | 0.0021 | 0.0022 | 3.48 | 1.07, 1.10, 1.24 | 0.969 | 0/50 | 0 | 3.11 |
| lidar | 0.3 | MCL | 0.0070 | 0.0088 | 3.16 | 1.08, 1.04, 1.00 | 0.955 | 0/50 | - | 36.71 |
| lidar | 0.3 | Dead reckoning | 0.7988 | 0.2832 | - | - | - | 50/50 | - | - |

## Global start and kidnapping

Confidently wrong: the share of steps after the kidnapping, over all kidnapped runs, with position error over 0.5 m,
NEES over 9.35 and, for the EKF, no reset in the last 20 scans.
Localized runs: the same share over only the runs that had localized before the kidnapping (n/a when none had).

| Sensor | Dropout | Filter | Start | Localized | Median to localize | Recovered | Median to recover | Confidently wrong (all runs) | Confidently wrong (localized runs) | Max recent resets | ms/step |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| hc_sr04 | own | MCL | anywhere | 0.80 | 2.5 steps (0.2 s) | 0.94 | 55 steps (5.5 s) | 0.281 | 0.245 | - | 50.84 |
| hc_sr04 | own | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.16 | 152 steps (15.2 s) | 0.321 | 0.321 | 7 | 1.95 |
| hc_sr04 | 0.3 | MCL | anywhere | 0.88 | 5.5 steps (0.6 s) | 0.86 | 39 steps (3.9 s) | 0.284 | 0.256 | - | 37.55 |
| hc_sr04 | 0.3 | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.18 | 136 steps (13.6 s) | 0.308 | 0.308 | 7 | 1.88 |
| lidar | own | MCL | anywhere | 0.96 | 0 steps (0.0 s) | 0.96 | 21.5 steps (2.1 s) | 0.127 | 0.129 | - | 37.23 |
| lidar | own | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.16 | 138.5 steps (13.9 s) | 0.239 | 0.239 | 7 | 3.02 |
| lidar | 0.3 | MCL | anywhere | 0.96 | 0 steps (0.0 s) | 0.98 | 21 steps (2.1 s) | 0.118 | 0.102 | - | 37.21 |
| lidar | 0.3 | EKF | noisy start | 1.00 | 0 steps (0.0 s) | 0.10 | 41 steps (4.1 s) | 0.286 | 0.286 | 7 | 2.90 |

Timings: Linux-6.8.12-1021-tegra-aarch64-with-glibc2.39; OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1;
concurrent jobs: 2 J3 jobs (one per sensor), 1 BLAS thread each; review agents may run tests alongside. ms/step: the
filter's predict, update, pose and covariance per step.
