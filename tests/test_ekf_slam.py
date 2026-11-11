"""rl_navigation.slam.ekf_slam: landmark EKF-SLAM with first-estimate Jacobians, against finite
differences, the unobservable directions, hand-worked values and the guards."""

import math

import numpy as np
import pytest

import rl_navigation.slam.ekf_slam as ekf_slam_module
from rl_navigation.estimation import EkfLocalizer, SensorMap, kalman_update, motion_jacobians
from rl_navigation.estimation.ekf import chi2_quantile
from rl_navigation.sim import Arena, WheelOdometry, lidar, unicycle_step, wrap_angle
from rl_navigation.slam.ekf_slam import (
    AMBIGUOUS,
    NEW_LANDMARK,
    EkfSlam,
    SlamSettings,
    associate_ml,
    fej_motion_jacobian,
    initialize_landmark,
    landmark_measurement,
)
from rl_navigation.slam.features import CircleFeature
from tests.navigation_cases import (
    TWO_LANDMARK_POSE,
    TWO_LANDMARKS,
    fd_jacobian,
    slam_null_space,
)

SENSOR = lidar(num_beams=360)
ODOMETRY = WheelOdometry()
NO_FLOOR = SlamSettings(floor_factor=0.0)
QUARTER_TURN = np.array([[0.0, -1.0], [1.0, 0.0]])
FEATURE_Q = 1e-4 * np.eye(3)


def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def feature(z, covariance=None, beams=8, first_beam=0):
    covariance = FEATURE_Q if covariance is None else covariance
    return CircleFeature(
        np.asarray(z, dtype=float),
        np.array(covariance, dtype=float),
        np.arange(first_beam, first_beam + beams),
        1.0,
    )


def body_frame(pose, landmark):
    """A landmark ``(m_x, m_y, rho)`` as the robot at ``pose`` sees it."""
    pose = np.asarray(pose, dtype=float)
    landmark = np.asarray(landmark, dtype=float)
    return np.array([*(rotation(pose[2]).T @ (landmark[:2] - pose[:2])), landmark[2]])


def oracle(labels):
    return lambda features: list(labels)[: len(features)]


def random_pose(rng):
    return np.array([*rng.uniform(0.0, 5.0, 2), rng.uniform(-math.pi, math.pi)])


def filter_from(state, covariance, settings=NO_FLOOR, labels=None, first=None, **kwargs):
    return EkfSlam.from_state(
        SENSOR,
        ODOMETRY,
        state,
        covariance,
        settings,
        labels=labels,
        first_estimates=first,
        **kwargs,
    )


def two_landmark_filter(settings=NO_FLOOR, pose_variance=(1e-4, 1e-4, 0.0), landmark_std=0.05):
    state = np.concatenate([TWO_LANDMARK_POSE, TWO_LANDMARKS.ravel()])
    covariance = np.diag([*pose_variance, *[landmark_std**2] * 6])
    return filter_from(state, covariance, settings, labels=[0, 1])


def stacked_rows(pose_block, landmark_block, j, num_landmarks):
    """The ``(3, 3 + 3L)`` rows of the stacked measurement Jacobian for landmark ``j``."""
    rows = np.zeros((3, 3 + 3 * num_landmarks))
    rows[:, :3] = pose_block
    rows[:, 3 + 3 * j : 6 + 3 * j] = landmark_block
    return rows


# Jacobians against central differences ----------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_the_fej_motion_jacobian_at_the_estimate_is_the_motion_jacobian(seed):
    rng = np.random.default_rng(seed)
    pose = random_pose(rng)
    distance, turn = rng.uniform(-0.05, 0.05), rng.uniform(-0.3, 0.3)
    prediction = unicycle_step(pose, distance, turn, 1.0)
    F, _ = motion_jacobians(pose, distance, turn)
    np.testing.assert_allclose(fej_motion_jacobian(pose, prediction), F, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        fd_jacobian(lambda p: unicycle_step(p, distance, turn, 1.0), pose),
        F,
        rtol=1e-6,
        atol=1e-8,
    )


def test_the_fej_motion_jacobian_uses_the_previous_prediction():
    """F_k = [[1, 0, -(y_{k+1|k} - y_{k|k-1})], [0, 1, x_{k+1|k} - x_{k|k-1}], [0, 0, 1]]."""
    previous = np.array([1.0, 2.0, 0.3])
    prediction = np.array([1.5, 1.25, 0.7])
    expected = np.array([[1.0, 0.0, 0.75], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(fej_motion_jacobian(previous, prediction), expected)


@pytest.mark.parametrize("seed", range(5))
def test_the_measurement_jacobians_match_central_differences(seed):
    rng = np.random.default_rng(seed)
    pose = random_pose(rng)
    landmark = np.array([*rng.uniform(0.0, 5.0, 2), rng.uniform(0.2, 0.5)])
    h, H_pose, H_landmark = landmark_measurement(pose, landmark)
    np.testing.assert_allclose(h, body_frame(pose, landmark), atol=1e-12)
    numeric = fd_jacobian(
        lambda v: landmark_measurement(v[:3], v[3:])[0],
        np.concatenate([pose, landmark]),
        angle_index=None,
    )
    np.testing.assert_allclose(numeric[:, :3], H_pose, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(numeric[:, 3:], H_landmark, rtol=1e-6, atol=1e-8)
    _, fej_pose, fej_landmark = landmark_measurement(pose, landmark, lever=landmark[:2])
    np.testing.assert_array_equal(fej_pose, H_pose)
    np.testing.assert_array_equal(fej_landmark, H_landmark)


def test_the_fej_measurement_jacobian_takes_the_heading_lever_from_the_first_estimate():
    pose = np.array([1.0, 2.0, 0.0])
    _, H_pose, _ = landmark_measurement(pose, [3.0, 2.0, 0.3], lever=[1.0, 5.0])
    # d h / d theta = R'(0)^T (first - p) = [[0, 1], [-1, 0]] (0, 3) = (3, 0).
    np.testing.assert_allclose(H_pose[:2, 2], [3.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(H_pose[:2, :2], -np.eye(2), atol=1e-15)


def test_the_measurement_is_vectorised_over_landmarks():
    pose = np.array([1.0, 2.0, 0.4])
    h, H_pose, H_landmark = landmark_measurement(pose, TWO_LANDMARKS)
    assert h.shape == (2, 3) and H_pose.shape == (2, 3, 3) and H_landmark.shape == (2, 3, 3)
    for j, landmark in enumerate(TWO_LANDMARKS):
        single = landmark_measurement(pose, landmark)
        np.testing.assert_array_equal(h[j], single[0])
        np.testing.assert_array_equal(H_pose[j], single[1])
        np.testing.assert_array_equal(H_landmark[j], single[2])


@pytest.mark.parametrize("seed", range(5))
def test_the_initialization_jacobians_match_central_differences(seed):
    rng = np.random.default_rng(seed)
    pose = random_pose(rng)
    z = np.array([*rng.uniform(-2.0, 2.0, 2), rng.uniform(0.2, 0.5)])
    m, G_pose, G_z = initialize_landmark(pose, z)
    np.testing.assert_allclose(landmark_measurement(pose, m)[0], z, atol=1e-12)
    np.testing.assert_allclose(
        fd_jacobian(lambda p: initialize_landmark(p, z)[0], pose, angle_index=None),
        G_pose,
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fd_jacobian(lambda v: initialize_landmark(pose, v)[0], z, angle_index=None),
        G_z,
        rtol=1e-6,
        atol=1e-8,
    )


# The unobservable directions (deterministic, 1e-12) ---------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_the_fej_measurement_jacobian_annihilates_the_null_space(seed):
    """H_fej N = 0 with N built at the current pose and the first estimates: -R^T J p +
    R'^T (first - p) + R^T J first = 0, and the radius row sees neither move."""
    rng = np.random.default_rng(seed)
    pose = random_pose(rng)
    landmarks = np.column_stack([rng.uniform(0, 5, (3, 2)), rng.uniform(0.2, 0.5, 3)])
    first = landmarks[:, :2] + rng.normal(0.0, 0.3, (3, 2))
    _, H_pose, H_landmark = landmark_measurement(pose, landmarks, lever=first)
    null = slam_null_space(pose, first)
    for j in range(3):
        rows = stacked_rows(H_pose[j], H_landmark[j], j, 3)
        np.testing.assert_allclose(rows @ null, 0.0, atol=1e-12)
    # The standard Jacobian, at the current landmarks, annihilates the null space built there.
    _, H_pose, H_landmark = landmark_measurement(pose, landmarks)
    for j in range(3):
        rows = stacked_rows(H_pose[j], H_landmark[j], j, 3)
        np.testing.assert_allclose(rows @ slam_null_space(pose, landmarks[:, :2]), 0.0, atol=1e-12)


def test_the_fej_motion_jacobian_carries_the_null_space_to_the_next_prediction():
    """Phi_k N(x_{k|k-1}) = N(x_{k+1|k}) over 50 random predicts, each after an update moved the
    estimate; the standard F misses by exactly J (p_{k|k} - p_{k|k-1}) in the rotation column."""
    rng = np.random.default_rng(7)
    landmarks = rng.uniform(0.0, 5.0, (2, 2))
    previous = random_pose(rng)
    for _ in range(50):
        updated = previous + np.array([*rng.normal(0.0, 0.05, 2), rng.normal(0.0, 0.05)])
        distance, turn = rng.uniform(0.0, 0.05), rng.uniform(-0.3, 0.3)
        prediction = unicycle_step(updated, distance, turn, 1.0)
        Phi = np.eye(9)
        Phi[:3, :3] = fej_motion_jacobian(previous, prediction)
        np.testing.assert_allclose(
            Phi @ slam_null_space(previous, landmarks),
            slam_null_space(prediction, landmarks),
            atol=1e-12,
        )
        Phi[:3, :3] = motion_jacobians(updated, distance, turn)[0]
        residual = slam_null_space(prediction, landmarks) - Phi @ slam_null_space(
            previous, landmarks
        )
        expected = np.zeros((9, 3))
        expected[:2, 2] = QUARTER_TURN @ (updated[:2] - previous[:2])
        np.testing.assert_allclose(residual, expected, atol=1e-12)
        previous = prediction


@pytest.mark.parametrize("seed", range(10))
def test_initializing_a_landmark_keeps_the_null_space_form(seed):
    """The new rows of N are G_pose N_pose: (I, 0) for the translations and (J m_new, 0) for the
    rotation, the null-space form at first = m_new."""
    rng = np.random.default_rng(seed)
    pose = random_pose(rng)
    z = np.array([*rng.uniform(-2.0, 2.0, 2), rng.uniform(0.2, 0.5)])
    m, G_pose, _ = initialize_landmark(pose, z)
    new_rows = G_pose @ slam_null_space(pose, np.zeros((0, 2)))
    np.testing.assert_allclose(new_rows, slam_null_space(pose, m[:2])[3:], atol=1e-12)


# The two-landmark hand case --------------------------------------------------------------------


def test_the_two_landmark_update_gives_the_hand_worked_values():
    """Pose (1, 2, 0), landmarks (3, 2, 0.3) and (1, 4, 0.3), P_pp = diag(0.01^2, 0.01^2, 0),
    landmark blocks 0.05^2 I, Q = 0.01^2 I, no floor. R(0) = I, so h_1 = (2, 0, 0.3) and
    h_2 = (0, 2, 0.3); d h / d theta = R'(0)^T (m - p) = (0, -2) and (2, 0). The measurements
    (2.02, 0, 0.3) and (0, 2.01, 0.3) give the innovations (0.02, 0, 0) and (0, 0.01, 0).

    With sigma_theta = 0 the model is linear. x enters both features' first rows as -x (prior
    variance a = 1e-4) beside an independent landmark coordinate and the noise (c = 0.05^2 +
    0.01^2 = 0.0026 each), so x' = x - a (0.02 + 0) / (c + 2a) = 1 - 7.142857e-4; likewise
    y' = 2 - a (0 + 0.01) / (c + 2a)."""
    slam = two_landmark_filter()
    x0, P0 = slam.state, slam.state_covariance
    result = slam.update_features([feature([2.02, 0.0, 0.3]), feature([0.0, 2.01, 0.3])])
    assert result.associated == 2 and not result.rejected_update
    H = np.zeros((6, 9))
    H[0:2, 0:2] = H[3:5, 0:2] = -np.eye(2)
    H[0:2, 2] = [0.0, -2.0]
    H[3:5, 2] = [2.0, 0.0]
    H[0:2, 3:5] = H[3:5, 6:8] = np.eye(2)
    H[2, 5] = H[5, 8] = 1.0
    innovation = np.array([0.02, 0.0, 0.0, 0.0, 0.01, 0.0])
    R = 1e-4 * np.eye(6)
    S = H @ P0 @ H.T + R
    K = P0 @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(9) - K @ H
    np.testing.assert_allclose(slam.state, x0 + K @ innovation, atol=1e-12)
    np.testing.assert_allclose(slam.state_covariance, I_KH @ P0 @ I_KH.T + K @ R @ K.T, atol=1e-12)
    a, c = 1e-4, 0.0026
    assert slam.pose[0] == pytest.approx(1.0 - a * 0.02 / (c + 2 * a), abs=1e-12)
    assert slam.pose[1] == pytest.approx(2.0 - a * 0.01 / (c + 2 * a), abs=1e-12)
    assert slam.pose[2] == 0.0


# Prediction ------------------------------------------------------------------------------------


def test_a_zero_travel_predict_changes_nothing():
    slam = two_landmark_filter(pose_variance=(1e-4, 1e-4, 1e-3))
    state, covariance = slam.state, slam.state_covariance
    slam.predict([0.0, 0.0])
    np.testing.assert_array_equal(slam.state, state)
    np.testing.assert_array_equal(slam.state_covariance, covariance)


@pytest.mark.parametrize("jacobians", ["fej", "standard"])
def test_predict_moves_only_the_pose_rows_and_columns(jacobians):
    rng = np.random.default_rng(3)
    slam = two_landmark_filter(SlamSettings(jacobians=jacobians), (1e-4, 2e-4, 1e-3))
    A = rng.normal(size=(9, 9))
    covariance = A @ A.T * 1e-3
    slam = filter_from(slam.state, covariance, SlamSettings(jacobians=jacobians), labels=[0, 1])
    travel = np.array([0.031, 0.036])
    distance, turn = ODOMETRY.increments(travel)
    pose = slam.pose
    F, G = motion_jacobians(pose, distance, turn)
    GA = G @ ODOMETRY.increments_jacobian()
    slam.predict(travel)
    F_full = np.eye(9)
    F_full[:3, :3] = F  # FEJ's F equals this at the first predict: previous prediction = x_{0|0}
    Q_full = np.zeros((9, 9))
    Q_full[:3, :3] = GA @ ODOMETRY.wheel_covariance(travel) @ GA.T
    np.testing.assert_allclose(
        slam.state_covariance, F_full @ covariance @ F_full.T + Q_full, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_array_equal(slam.state_covariance[3:, 3:], covariance[3:, 3:])
    np.testing.assert_allclose(slam.pose, unicycle_step(pose, distance, turn, 1.0), atol=1e-15)
    np.testing.assert_array_equal(slam.landmarks, TWO_LANDMARKS)


def test_with_no_landmarks_predict_is_the_localizers_predict():
    """Bit for bit with standard Jacobians; with FEJ the previous prediction is the estimate
    between predicts (no update), so F agrees to rounding."""
    start, covariance = [1.0, 2.0, 0.3], np.diag([0.05, 0.05, 0.05]) ** 2
    rng = np.random.default_rng(11)
    travels = rng.uniform(0.0, 0.04, (40, 2))
    localizer = EkfLocalizer(SensorMap(Arena(5.0, 5.0), SENSOR), ODOMETRY, start, covariance)
    standard = EkfSlam(SENSOR, ODOMETRY, start, covariance, SlamSettings(jacobians="standard"))
    fej = EkfSlam(SENSOR, ODOMETRY, start, covariance)
    for travel in travels:
        localizer.predict(travel)
        standard.predict(travel)
        fej.predict(travel)
        np.testing.assert_array_equal(standard.pose, localizer.pose)
        np.testing.assert_array_equal(standard.covariance, localizer.covariance)
        np.testing.assert_allclose(fej.pose, localizer.pose, rtol=0, atol=0)
        np.testing.assert_allclose(fej.covariance, localizer.covariance, rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(fej.previous_prediction, fej.pose)


# Initialization --------------------------------------------------------------------------------


def test_a_landmark_seen_from_a_known_pose_has_the_rotated_feature_covariance():
    pose = np.array([1.0, 2.0, 0.7])
    Q = np.array([[4e-4, 1e-4, 2e-5], [1e-4, 2e-4, -3e-5], [2e-5, -3e-5, 3e-4]])
    slam = filter_from(pose, np.zeros((3, 3)), SlamSettings(floor_factor=0.0))
    result = slam.update_features([feature([1.5, -0.4, 0.3], Q)], associate=oracle([7]))
    assert result.new == 1 and slam.labels.tolist() == [7]
    G_z = np.eye(3)
    G_z[:2, :2] = rotation(0.7)
    np.testing.assert_allclose(slam.landmark_covariances[0], G_z @ Q @ G_z.T, atol=1e-18)
    np.testing.assert_array_equal(slam.state_covariance[:3, 3:], 0.0)
    np.testing.assert_array_equal(slam.first_estimates, slam.landmarks)


def test_initialization_augments_the_covariance_with_the_dense_formula():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 6))
    P = A @ A.T * 1e-3
    state = np.array([1.0, 2.0, 0.7, 3.0, 3.0, 0.3])
    slam = filter_from(state, P, SlamSettings(floor_factor=0.0), labels=[0])
    z = np.array([-0.5, 1.2, 0.25])
    slam.update_features([feature(z)], associate=oracle([3]))
    m, G_pose, G_z = initialize_landmark(state[:3], z)
    G_x = np.zeros((3, 6))
    G_x[:, :3] = G_pose
    expected = np.block([[P, P @ G_x.T], [G_x @ P, G_x @ P @ G_x.T + G_z @ FEATURE_Q @ G_z.T]])
    np.testing.assert_allclose(slam.state_covariance, expected, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(slam.landmarks[1], m, atol=1e-15)


def test_the_floor_is_added_once_to_the_features_covariance_and_never_to_the_fit():
    floor = 1.5 * SENSOR.noise_std  # 15 mm on the 1 cm lidar
    slam = filter_from([1.0, 2.0, 0.0], np.zeros((3, 3)), SlamSettings(floor_factor=1.5))
    given = feature([1.0, 0.5, 0.3])
    slam.update_features([given], associate=oracle([0]))
    np.testing.assert_allclose(
        slam.landmark_covariances[0], FEATURE_Q + floor**2 * np.eye(3), rtol=1e-12, atol=0
    )
    np.testing.assert_array_equal(given.covariance, FEATURE_Q)  # the fit's own covariance
    # The gate sees Q + floor^2 I too: 7 cm off a perfectly known landmark is NIS 0.07^2 /
    # (1e-4 + 2.25e-4) = 15.1, between the gate and the new-landmark threshold (ambiguous);
    # without the floor it would be 49, a new candidate.
    certain = filter_from(
        [1.0, 2.0, 0.0, 2.0, 2.5, 0.3], np.zeros((6, 6)), SlamSettings(floor_factor=1.5)
    )
    result = certain.update_features([feature([1.07, 0.5, 0.3])])
    assert result.ambiguous == 1 and certain.num_candidates == 0


def test_the_floor_scales_with_the_sensors_range_noise():
    """1.5 sigma by default: 15 mm on a 1 cm lidar, 30 mm on a 2 cm one."""
    noisy = lidar(num_beams=360, noise_std=0.02)
    slam = EkfSlam.from_state(noisy, ODOMETRY, [1.0, 2.0, 0.0], np.zeros((3, 3)))
    slam.update_features([feature([1.0, 0.5, 0.3])], associate=oracle([0]))
    np.testing.assert_allclose(
        slam.landmark_covariances[0], FEATURE_Q + 0.03**2 * np.eye(3), rtol=1e-12, atol=0
    )


# Heading observability -------------------------------------------------------------------------


def heading_std_after_30_scans(jacobians):
    """sigma_theta of a robot that starts with sigma_theta = 0.3 rad and sees two circles for
    30 scans (seeded features with 1 cm noise), started 0.1 rad off."""
    rng = np.random.default_rng(1)
    landmarks = np.array([[3.0, 2.5, 0.3], [2.5, 3.5, 0.3]])
    truth = np.array([1.0, 2.0, 0.0])
    start = truth + np.array([0.03, -0.02, 0.1])
    slam = EkfSlam(
        SENSOR,
        ODOMETRY,
        start,
        np.diag([0.05, 0.05, 0.3]) ** 2,
        SlamSettings(floor_factor=0.0, jacobians=jacobians),
    )
    travel = ODOMETRY.wheel_travel(0.3, 0.5, 0.1)
    for _ in range(30):
        truth = unicycle_step(truth, *ODOMETRY.increments(travel), 1.0)
        slam.predict(travel)
        seen = [
            feature(body_frame(truth, landmark) + 0.01 * rng.standard_normal(3))
            for landmark in landmarks
        ]
        slam.update_features(seen, associate=oracle([0, 1]))
    return math.sqrt(slam.covariance[2, 2])


def test_fej_gains_no_information_about_the_absolute_heading():
    """Nothing observes the global heading. FEJ keeps sigma_theta at its prior; the standard
    Jacobians invent heading information. This fails if FEJ is silently disabled."""
    assert heading_std_after_30_scans("fej") >= 0.29
    assert heading_std_after_30_scans("standard") < 0.15


# Association -----------------------------------------------------------------------------------


def test_associate_ml_gates_creates_and_discards():
    """Gate 11.34, new-landmark threshold 30: inside the gate is associated; beyond 30 from every
    landmark is a new one; between the two is ambiguous."""
    nis = np.array([[1.0, 50.0], [40.0, 45.0], [20.0, 60.0]])
    assert associate_ml(nis, 11.34, 30.0).tolist() == [0, NEW_LANDMARK, AMBIGUOUS]
    assert associate_ml(np.zeros((2, 0)), 11.34, 30.0).tolist() == [NEW_LANDMARK] * 2


def test_associate_ml_is_one_to_one_and_prefers_the_better_feature():
    """Two features near one landmark: the one with the smaller NIS takes it; the other, inside
    the new-landmark threshold of it, is ambiguous rather than a duplicate."""
    assert associate_ml(np.array([[4.0], [1.0]]), 11.34, 30.0).tolist() == [AMBIGUOUS, 0]
    # Pairs go in increasing NIS: (0, 0) at 1.0 first, so feature 1 takes landmark 1 at 9.0.
    nis = np.array([[1.0, 2.0], [1.5, 9.0]])
    assert associate_ml(nis, 11.34, 30.0).tolist() == [0, 1]


def test_associate_ml_treats_an_undefined_nis_as_no_match():
    nis = np.array([[np.inf, np.nan]])
    assert associate_ml(nis, 11.34, 30.0).tolist() == [NEW_LANDMARK]


def test_the_gate_weighs_the_innovation_by_its_covariance():
    """A landmark known to 0.2 m along x and 0.01 m along y: a feature 10 cm off along x passes
    the gate, one 10 cm off along y is far beyond it (a new candidate)."""
    state = np.array([1.0, 2.0, 0.0, 3.0, 2.0, 0.3])
    covariance = np.diag([0.0, 0.0, 0.0, 0.2**2, 0.01**2, 0.01**2])
    along = filter_from(state, covariance).update_features([feature([2.1, 0.0, 0.3])])
    assert along.associated == 1
    across_filter = filter_from(state, covariance)
    across = across_filter.update_features([feature([2.0, 0.1, 0.3])])
    assert across.associated == 0 and across_filter.num_candidates == 1


def test_features_are_associated_through_update_with_the_ml_rule():
    slam = two_landmark_filter()
    result = slam.update_features(
        [feature([0.0, 2.0, 0.3], first_beam=40), feature([2.0, 0.0, 0.3])]
    )
    assert result.associated == 2
    assert sorted(result.landmarks.tolist()) == [0, 1]
    assert result.used.shape == (360,) and result.used.sum() == 16
    assert result.used[40:48].all() and result.used[:8].all()


# The provisional list --------------------------------------------------------------------------


def empty_filter(settings=None):
    return EkfSlam(SENSOR, ODOMETRY, [1.0, 2.0, 0.0], np.diag([1e-4, 1e-4, 1e-4]), settings)


def test_a_candidate_becomes_a_landmark_on_its_third_sighting():
    slam = empty_filter()
    for sighting in range(1, 4):
        result = slam.update_features([feature([1.0, 0.5, 0.3])])
        assert slam.num_landmarks == (1 if sighting == 3 else 0)
        assert result.new == (1 if sighting == 3 else 0)
    assert slam.num_candidates == 0
    np.testing.assert_allclose(slam.landmarks[0], [2.0, 2.5, 0.3], atol=1e-12)


def test_a_candidate_unseen_for_the_timeout_is_dropped():
    slam = empty_filter()
    slam.update_features([feature([1.0, 0.5, 0.3])])
    for _ in range(49):
        slam.update_features([])
    assert slam.num_candidates == 1
    slam.update_features([])
    assert slam.num_candidates == 0


def test_two_fits_of_one_circle_in_a_scan_are_one_sighting():
    """Two new features 2 cm apart in one scan (a circle split in two segments) agree with one
    candidate, but a scan sights a candidate once: three scans are still needed to confirm it."""
    slam = empty_filter()
    slam.update_features([feature([1.0, 0.5, 0.3]), feature([1.02, 0.5, 0.3], first_beam=20)])
    slam.update_features([feature([1.0, 0.5, 0.3])])
    assert slam.num_landmarks == 0
    slam.update_features([feature([1.0, 0.5, 0.3])])
    assert slam.num_landmarks == 1


def test_two_different_spots_seen_once_each_make_no_landmark():
    slam = empty_filter()
    slam.update_features([feature([1.0, 0.5, 0.3])])
    slam.update_features([feature([1.0, -0.5, 0.3])])
    assert slam.num_landmarks == 0 and slam.num_candidates == 2


# The labelled (oracle) path ---------------------------------------------------------------------


def certain_landmark(label=5):
    """A landmark at (2, 2.5, 0.3) known exactly, seen from a known pose (1, 2, 0): S = Q =
    1e-4 I, so a feature d metres off along x has NIS d^2 / 1e-4."""
    state = np.array([1.0, 2.0, 0.0, 2.0, 2.5, 0.3])
    return filter_from(state, np.zeros((6, 6)), labels=[label])


def test_labels_bypass_the_ml_gate_but_not_the_oracle_gate():
    """NIS 14 lies between the ML gate (chi2_3 at 0.99) and the oracle's (at 0.999): the label
    associates it where the ML rule calls it ambiguous. NIS 20 is beyond both: rejected."""
    assert chi2_quantile(0.99, 3) < 14.0 < chi2_quantile(0.999, 3) < 20.0
    off = math.sqrt(14.0 * 1e-4)
    labelled = certain_landmark().update_features(
        [feature([1.0 + off, 0.5, 0.3])], associate=oracle([5])
    )
    assert labelled.associated == 1 and labelled.rejected == 0
    unlabelled = certain_landmark().update_features([feature([1.0 + off, 0.5, 0.3])])
    assert unlabelled.associated == 0 and unlabelled.ambiguous == 1
    far = certain_landmark()
    state = far.state
    rejected = far.update_features(
        [feature([1.0 + math.sqrt(20.0 * 1e-4), 0.5, 0.3])], associate=oracle([5])
    )
    assert rejected.rejected == 1 and rejected.associated == 0
    np.testing.assert_array_equal(far.state, state)


def test_an_unknown_label_initializes_a_landmark_at_once_and_a_negative_one_is_ignored():
    slam = certain_landmark()
    result = slam.update_features(
        [feature([1.0, -1.0, 0.25]), feature([2.0, 0.0, 0.4])], associate=oracle([9, -1])
    )
    assert result.new == 1 and slam.labels.tolist() == [5, 9]
    np.testing.assert_allclose(slam.landmarks[1], [2.0, 1.0, 0.25], atol=1e-12)
    assert slam.num_candidates == 0


def test_a_label_given_twice_in_a_scan_is_used_once():
    result = certain_landmark().update_features(
        [feature([1.0, 0.5, 0.3]), feature([1.001, 0.5, 0.3])], associate=oracle([5, 5])
    )
    assert result.associated == 1 and result.ambiguous == 1


def test_an_association_function_must_label_every_feature():
    with pytest.raises(ValueError, match="one label per feature"):
        certain_landmark().update_features([feature([1.0, 0.5, 0.3])], associate=lambda f: [])


# Guards: the filter returns, counts and keeps its prior ------------------------------------------


@pytest.mark.parametrize(
    "covariance",
    [np.full((3, 3), np.nan), np.diag([-1.0, 1e-4, 1e-4])],
    ids=["nan", "not-positive-definite"],
)
@pytest.mark.parametrize("labelled", [False, True])
def test_a_feature_with_an_unsound_covariance_is_skipped(covariance, labelled):
    slam = certain_landmark()
    state, P = slam.state, slam.state_covariance
    result = slam.update_features(
        [feature([1.0, 0.5, 0.3], covariance)], associate=oracle([5]) if labelled else None
    )
    assert result.skipped == 1 and result.associated == 0 and not result.used.any()
    np.testing.assert_array_equal(slam.state, state)
    np.testing.assert_array_equal(slam.state_covariance, P)


def test_a_landmark_whose_innovation_covariance_is_not_positive_definite_is_skipped():
    state = np.array([1.0, 2.0, 0.0, 2.0, 2.5, 0.3])
    corrupted = np.diag([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])  # a corrupted landmark block
    slam = filter_from(state, corrupted, labels=[5])
    result = slam.update_features([feature([1.0, 0.5, 0.3])], associate=oracle([5]))
    assert result.skipped == 1 and result.associated == 0
    np.testing.assert_array_equal(slam.state, state)


def test_an_indefinite_innovation_covariance_is_skipped_not_gated():
    """A landmark block indefinite only in the radius: S's x entry is tiny, so a 10 cm innovation
    would read as NIS ~ 99 and be counted rejected. An S that is not positive definite has no NIS:
    the feature is skipped."""
    state = np.array([1.0, 2.0, 0.0, 2.0, 2.5, 0.3])
    corrupted = np.diag([0.0, 0.0, 0.0, 1e-6, 1e-6, -1.0])
    slam = filter_from(state, corrupted, labels=[5])
    result = slam.update_features([feature([1.1, 0.5, 0.3])], associate=oracle([5]))
    assert (result.skipped, result.rejected, result.associated) == (1, 0, 0)


def test_an_indefinite_stacked_innovation_covariance_falls_back_feature_by_feature():
    """Each landmark's own S is positive definite but the stacked S is not (a cross covariance
    no valid P has): the update goes feature by feature, the first applied, and nothing raises."""
    state = np.concatenate([TWO_LANDMARK_POSE, TWO_LANDMARKS.ravel()])
    P = np.zeros((9, 9))
    P[3:6, 3:6] = P[6:9, 6:9] = np.eye(3) * 1e-2
    P[3:6, 6:9] = P[6:9, 3:6] = np.eye(3) * 5e-2  # |correlation| 5: not a covariance
    slam = filter_from(state, P, labels=[0, 1])
    result = slam.update_features(
        [feature([2.0, 0.0, 0.3]), feature([0.0, 2.0, 0.3])], associate=oracle([0, 1])
    )
    assert result.associated + result.skipped + int(result.rejected_update) >= 1
    assert np.isfinite(slam.state).all() and np.isfinite(slam.state_covariance).all()


def test_an_update_with_a_non_finite_result_is_rejected_and_the_prior_kept(monkeypatch):
    slam = certain_landmark()
    state, P = slam.state, slam.state_covariance
    monkeypatch.setattr(
        ekf_slam_module,
        "kalman_update",
        lambda x, P, *args: (np.full_like(x, np.nan), np.asarray(P)),
    )
    result = slam.update_features([feature([1.0, 0.5, 0.3])], associate=oracle([5]))
    assert result.rejected_update and result.associated == 0 and not result.used.any()
    np.testing.assert_array_equal(slam.state, state)
    np.testing.assert_array_equal(slam.state_covariance, P)


def test_an_update_that_carries_the_pose_out_of_bounds_is_rejected():
    """A pose known to 0.1 m and a certain landmark: a 5 cm innovation carries the pose about
    5 cm, past bounds 3 cm ahead."""
    state = np.array([1.0, 2.0, 0.0, 2.0, 2.5, 0.3])
    P = np.diag([0.1**2, 0.1**2, 0.0, 0.0, 0.0, 0.0])
    loose = filter_from(state, P, labels=[5], bounds=(0.0, 0.0, 5.0, 5.0))
    loose.update_features([feature([0.95, 0.5, 0.3])], associate=oracle([5]))
    assert loose.pose[0] > 1.04
    tight = filter_from(state, P, labels=[5], bounds=(0.95, 1.95, 1.03, 2.05))
    result = tight.update_features([feature([0.95, 0.5, 0.3])], associate=oracle([5]))
    assert result.rejected_update
    np.testing.assert_array_equal(tight.state, state)


def test_an_update_leaving_the_covariance_indefinite_is_rejected(monkeypatch):
    slam = certain_landmark()
    state = slam.state
    monkeypatch.setattr(
        ekf_slam_module,
        "kalman_update",
        lambda x, P, *args: (np.asarray(x), -np.eye(len(x))),
    )
    result = slam.update_features([feature([1.0, 0.5, 0.3])], associate=oracle([5]))
    assert result.rejected_update
    np.testing.assert_array_equal(slam.state, state)


# The heading wrap ------------------------------------------------------------------------------


def test_an_update_either_side_of_pi_gives_the_same_posterior():
    """The same landmark straight ahead of a robot facing pi - 1e-9 and -pi + 1e-9."""
    posteriors = []
    for heading in (math.pi - 1e-9, -math.pi + 1e-9):
        state = np.array([3.0, 2.0, heading, 1.0, 2.0, 0.3])
        covariance = np.diag([1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-4])
        slam = filter_from(state, covariance, labels=[0])
        slam.update_features([feature([2.02, 0.01, 0.31])], associate=oracle([0]))
        posteriors.append((slam.state, slam.state_covariance))
    (a, Pa), (b, Pb) = posteriors
    assert -math.pi <= a[2] < math.pi and -math.pi <= b[2] < math.pi
    difference = a - b
    difference[2] = wrap_angle(difference[2])
    np.testing.assert_allclose(difference, 0.0, atol=1e-7)
    np.testing.assert_allclose(Pa, Pb, atol=1e-10)


def test_a_predict_across_pi_wraps_the_heading_and_the_previous_prediction():
    slam = EkfSlam(SENSOR, ODOMETRY, [2.0, 2.0, math.pi - 0.01], np.diag([1e-4, 1e-4, 1e-4]))
    travel = np.array([-0.002, 0.002])  # a turn of +0.02 rad in place on a 0.2 m wheel base
    slam.predict(travel)
    assert slam.pose[2] == pytest.approx(-math.pi + 0.01, abs=1e-12)
    np.testing.assert_array_equal(slam.previous_prediction, slam.pose)
    pose = np.array([2.0, 2.0, math.pi - 1e-9])
    F, _ = motion_jacobians(pose, 0.03, 0.02)
    np.testing.assert_allclose(
        fd_jacobian(lambda p: unicycle_step(p, 0.03, 0.02, 1.0), pose), F, rtol=1e-6, atol=1e-8
    )


# Batch against sequential, and a long run --------------------------------------------------------


def test_one_batch_update_equals_three_sequential_updates_on_a_linear_model():
    """sigma_theta = 0 and no floor: h is linear in (p, m), so the batch update over three
    features equals three sequential Kalman updates."""
    landmarks = np.array([[3.0, 2.0, 0.3], [1.0, 4.0, 0.4], [-0.5, 1.0, 0.25]])
    pose = np.array([1.0, 2.0, 0.4])
    state = np.concatenate([pose, landmarks.ravel()])
    rng = np.random.default_rng(2)
    A = rng.normal(size=(12, 12)) * 0.03
    P = A @ A.T
    P[2, :] = P[:, 2] = 0.0
    zs = [body_frame(pose, m) + rng.normal(0.0, 0.01, 3) for m in landmarks]
    slam = filter_from(state, P, labels=[0, 1, 2])
    slam.update_features([feature(z) for z in zs], associate=oracle([0, 1, 2]))
    x, cov = state.copy(), P.copy()
    for j, z in enumerate(zs):
        h, H_pose, H_landmark = landmark_measurement(x[:3], x[3 + 3 * j : 6 + 3 * j])
        H = stacked_rows(H_pose, H_landmark, j, 3)
        x, cov = kalman_update(x, cov, z - h, H, FEATURE_Q)
    np.testing.assert_allclose(slam.state, x, atol=1e-12)
    np.testing.assert_allclose(slam.state_covariance, cov, atol=1e-12)


def test_the_feature_by_feature_fallback_equals_the_batch_on_a_linear_model(monkeypatch):
    """The fallback applies each feature with its innovation moved to the current estimate,
    nu_i - H_i (x - x_0), so on a linear model it lands where the batch does. The stacked S is
    made to look indefinite (a check that refuses anything larger than one feature's 3 x 3)."""
    landmarks = np.array([[3.0, 2.0, 0.3], [1.0, 4.0, 0.4], [-0.5, 1.0, 0.25]])
    pose = np.array([1.0, 2.0, 0.4])
    state = np.concatenate([pose, landmarks.ravel()])
    rng = np.random.default_rng(6)
    A = rng.normal(size=(12, 12)) * 0.03
    P = A @ A.T
    P[2, :] = P[:, 2] = 0.0
    seen = [feature(body_frame(pose, m) + rng.normal(0.0, 0.01, 3)) for m in landmarks]
    batch = filter_from(state, P, labels=[0, 1, 2])
    batch.update_features(seen, associate=oracle([0, 1, 2]))
    checked = ekf_slam_module.positive_definite
    monkeypatch.setattr(
        ekf_slam_module,
        "positive_definite",
        lambda m: checked(m) if m.shape[-1] == 3 else np.zeros(m.shape[:-2], dtype=bool),
    )
    fallback = filter_from(state, P, labels=[0, 1, 2])
    result = fallback.update_features(seen, associate=oracle([0, 1, 2]))
    assert result.associated == 3 and result.skipped == 0
    np.testing.assert_allclose(fallback.state, batch.state, atol=1e-12)
    np.testing.assert_allclose(fallback.state_covariance, batch.state_covariance, atol=1e-12)


def test_the_covariance_stays_symmetric_and_positive_definite_over_2000_cycles():
    rng = np.random.default_rng(4)
    landmarks = np.array([[3.0, 2.5, 0.3], [1.5, 4.0, 0.4], [4.0, 4.0, 0.25]])
    truth = np.array([2.0, 2.0, 0.0])
    slam = EkfSlam(SENSOR, ODOMETRY, truth, np.diag([0.05, 0.05, 0.05]) ** 2)
    for _ in range(2000):
        travel = rng.uniform(0.0, 0.04, 2)
        truth = unicycle_step(truth, *ODOMETRY.increments(travel), 1.0)
        slam.predict(travel)
        seen = [i for i in range(3) if rng.random() < 0.7]
        slam.update_features(
            [feature(body_frame(truth, landmarks[i]) + rng.normal(0, 0.01, 3)) for i in seen],
            associate=oracle(seen),
        )
    P = slam.state_covariance
    assert slam.num_landmarks == 3
    np.testing.assert_allclose(P, P.T, atol=1e-12, rtol=0)
    assert np.linalg.eigvalsh(P).min() > 0.0


# Zero features, input checks, determinism, the scan path -----------------------------------------


def test_an_update_without_features_changes_nothing_but_the_candidates_clock():
    slam = two_landmark_filter()
    slam.predict([0.01, 0.012])
    before = (slam.state, slam.state_covariance, slam.first_estimates, slam.previous_prediction)
    result = slam.update_features([])
    assert (result.associated, result.new, result.skipped, result.rejected_update) == (
        0,
        0,
        0,
        False,
    )
    after = (slam.state, slam.state_covariance, slam.first_estimates, slam.previous_prediction)
    for old, new in zip(before, after, strict=True):
        np.testing.assert_array_equal(old, new)
    empty = slam.update(np.full(360, np.inf))
    assert not empty.used.any()
    np.testing.assert_array_equal(slam.state, before[0])


def test_wrong_inputs_are_refused():
    slam = empty_filter()
    with pytest.raises(ValueError, match="wheel_travel"):
        slam.predict([0.1])
    with pytest.raises(ValueError, match="readings"):
        slam.update(np.ones(64))
    with pytest.raises(ValueError, match="pose"):
        EkfSlam(SENSOR, ODOMETRY, [1.0, 2.0], np.eye(3))
    with pytest.raises(ValueError, match="state"):
        EkfSlam.from_state(SENSOR, ODOMETRY, np.zeros(5), np.eye(5))
    with pytest.raises(ValueError, match="symmetric"):
        EkfSlam.from_state(SENSOR, ODOMETRY, np.zeros(3), np.triu(np.ones((3, 3))))
    with pytest.raises(ValueError, match="noise"):
        EkfSlam(lidar(noise_std=0.0), ODOMETRY, [1.0, 2.0, 0.0], np.eye(3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"floor_factor": -0.5},
        {"gate_quantile": 1.0},
        {"oracle_quantile": 0.0},
        {"new_landmark_nis": 5.0},
        {"confirm": 0},
        {"candidate_radius": 0.0},
        {"candidate_timeout": 0},
        {"jacobians": "iterated"},
    ],
)
def test_slam_settings_refuse_values_out_of_range(kwargs):
    with pytest.raises(ValueError, match="SlamSettings"):
        SlamSettings(**kwargs)


def test_the_filter_is_deterministic():
    rng = np.random.default_rng(9)
    arena = Arena(5.0, 5.0, circle_centers=[[3.0, 2.0], [1.5, 3.8]], circle_radii=[0.3, 0.4])
    pose = np.array([1.0, 2.0, 0.0])
    scans = [SENSOR.read(arena, pose, rng)[0] for _ in range(5)]
    runs = []
    for _ in range(2):
        slam = EkfSlam(SENSOR, ODOMETRY, pose, np.diag([1e-4, 1e-4, 1e-4]))
        for scan in scans:
            slam.predict([0.0, 0.0])
            slam.update(scan)
        runs.append((slam.state, slam.state_covariance))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


def test_three_scans_of_two_circles_map_both():
    rng = np.random.default_rng(9)
    arena = Arena(5.0, 5.0, circle_centers=[[3.0, 2.0], [1.5, 3.8]], circle_radii=[0.3, 0.4])
    pose = np.array([1.0, 2.0, 0.0])
    slam = EkfSlam(SENSOR, ODOMETRY, pose, np.diag([1e-6, 1e-6, 1e-6]))
    for scan in range(3):
        result = slam.update(SENSOR.read(arena, pose, rng)[0])
        assert slam.num_landmarks == (2 if scan == 2 else 0)
    assert result.new == 2
    order = np.argsort(slam.landmarks[:, 0])[::-1]
    np.testing.assert_allclose(slam.landmarks[order, :2], arena.circle_centers, atol=0.03)
    np.testing.assert_allclose(slam.landmarks[order, 2], arena.circle_radii, atol=0.03)
