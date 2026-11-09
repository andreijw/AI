"""Monte Carlo localization: resampling, the recovery rates, the beam likelihood and the cluster
estimate against hand-computed values; the sampled motion against the wheel model; dropouts,
collisions, symmetric arenas, a still robot and a bump."""

import math
import warnings

import numpy as np
import pytest

from rl_navigation.estimation import (
    MclLocalizer,
    MclSettings,
    SensorMap,
    effective_sample_size,
    heaviest_cluster,
    low_variance_resample,
    measurement_log_likelihood,
    pose_difference,
    recovery_update,
    sample_free_poses,
)
from rl_navigation.sim import Arena, RangeSensor, WheelOdometry, lidar, unicycle_step, wrap_angle

ODOMETRY = WheelOdometry(wheel_base=0.2, noise_per_metre=2.5e-4)
STILL_ODOMETRY = WheelOdometry(wheel_base=0.2, noise_per_metre=0.0)
ROOM = Arena(
    5.0,
    5.0,
    circle_centers=[[3.5, 3.5]],
    circle_radii=[0.4],
    boxes=[[1.0, 3.2, 1.8, 4.0]],
)
TRUTH = np.array([2.5, 2.0, 0.3])
TIGHT = np.diag([1e-8, 1e-8, 1e-8])
ROUGHENING = np.array(MclSettings().roughening)


def _mcl(range_map, pose=None, covariance=None, seed=0, odometry=ODOMETRY, **settings):
    """A filter with 300 particles unless the test asks for more."""
    return MclLocalizer(
        range_map,
        odometry,
        pose,
        covariance,
        rng=np.random.default_rng(seed),
        settings=MclSettings(**{"num_particles": 300, **settings}),
    )


def _scan(sensor, pose, seed=0, arena=ROOM):
    return sensor.read(arena, pose, np.random.default_rng(seed))[0]


def _weight_near(mcl, pose, radius=0.3, heading=0.3):
    """Share of the weight within ``radius`` metres and ``heading`` radians of ``pose``."""
    difference = pose_difference(mcl.particles, pose)
    near = (np.hypot(difference[:, 0], difference[:, 1]) < radius) & (
        np.abs(difference[:, 2]) < heading
    )
    return float(mcl.weights[near].sum())


class _OneBeamMap:
    """A one-beam particle map in a 10 x 10 m square whose range is ``offset + slope . pose``,
    and ``inf`` from poses at or beyond ``finite_below_x``: the seam lets a test choose the
    measurement exactly."""

    min_range, max_range, noise_std, num_readings = 0.02, 4.0, 0.01, 1
    bounds = (0.0, 0.0, 10.0, 10.0)

    def __init__(self, offset, slope, finite_below_x=np.inf):
        self.offset, self.slope = offset, np.asarray(slope, float)
        self.finite_below_x = finite_below_x

    def expected_ranges(self, poses):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        ranges = self.offset + poses @ self.slope
        return np.where(poses[:, 0] < self.finite_below_x, ranges, np.inf)[:, None]

    def expected_beam_ranges(self, poses, beams):
        return self.expected_ranges(poses)[:, np.asarray(beams)]

    def free(self, points, margin):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return np.all((points > margin) & (points < 10.0 - margin), axis=1)


class _NoFreeSpaceMap(_OneBeamMap):
    def free(self, points, margin):
        return np.zeros(len(np.asarray(points).reshape(-1, 2)), dtype=bool)


class _TwoBeamMap(_OneBeamMap):
    """Beam 0 reads 2 m from everywhere; beam 1 reads the pose's x, so it tells a particle at
    x = 2 from one drawn over the 10 x 10 m square."""

    num_readings = 2

    def __init__(self):
        super().__init__(0.0, [1.0, 0.0, 0.0])

    def expected_ranges(self, poses):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        return np.column_stack([np.full(len(poses), 2.0), poses[:, 0]])


# Pure functions ---------------------------------------------------------------------------------


def test_low_variance_resampling_by_hand():
    """Positions (k + 0.5) / 4 = 0.125, 0.375, 0.625, 0.875 against the cumulative weights
    0.5, 0.75, 1, 1: the first two fall in particle 0, then 1, then 2; the empty one never."""
    np.testing.assert_array_equal(low_variance_resample([0.5, 0.25, 0.25, 0.0], 0.5), [0, 0, 1, 2])
    np.testing.assert_array_equal(low_variance_resample([2.0, 1.0, 1.0, 0.0], 0.5), [0, 0, 1, 2])


@pytest.mark.parametrize("u0", [0.0, 0.5, float(np.nextafter(1.0, 0.0))])
def test_resampling_never_draws_a_trailing_zero_weight(u0):
    """Ten weights of 0.1 sum to 1 - 1.1e-16 in floating point: a cumulative sum forced to end at
    1 would leave a sliver for the empty last particle."""
    weights = [0.1] * 10 + [0.0]
    indices = low_variance_resample(weights, u0)
    assert indices.max() <= 9, f"A zero-weight particle was drawn at u0={u0}: {indices}"
    assert len(indices) == 11


def test_effective_sample_size():
    assert effective_sample_size(np.full(4, 0.25)) == pytest.approx(4.0)
    assert effective_sample_size([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert effective_sample_size([2.0, 2.0]) == pytest.approx(2.0), "Weights are normalised first."


def test_recovery_rates_by_hand():
    """From w_slow = w_fast = 1 a scan averaging 0.1: w_slow = 1 + 0.001 (0.1 - 1) = 0.9991,
    w_fast = 1 + 0.1 (0.1 - 1) = 0.91, and p_inject = 1 - 0.91 / 0.9991."""
    w_slow, w_fast, p_inject = recovery_update(1.0, 1.0, 0.1, 0.001, 0.1)
    np.testing.assert_allclose([w_slow, w_fast, p_inject], [0.9991, 0.91, 1 - 0.91 / 0.9991])
    assert recovery_update(1.0, 1.0, 2.0, 0.001, 0.1)[2] == 0.0, (
        "A rising likelihood should inject nothing."
    )


def test_beam_likelihood_by_hand():
    """sigma = sqrt(0.01^2 + 0.15^2); p = 0.9 N(z; z_hat, sigma^2) + 0.1 / (4 - 0.02) per beam,
    an expected range of inf counting as max_range: a reading of 3.95 m is then 0.05 m off."""
    rmap = _OneBeamMap(0.0, [1.0, 0.0, 0.0])
    sigma = math.hypot(0.01, 0.15)
    peak, floor = 0.9 / (math.sqrt(2.0 * math.pi) * sigma), 0.1 / 3.98
    expected = np.array([[3.95], [3.85], [np.inf]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        log_likelihood = measurement_log_likelihood(np.array([3.95]), expected, rmap, 0.15, 0.1)
    np.testing.assert_allclose(
        log_likelihood,
        [
            math.log(peak + floor),
            math.log(peak * math.exp(-0.5 * (0.1 / sigma) ** 2) + floor),
            math.log(peak * math.exp(-0.5 * (0.05 / sigma) ** 2) + floor),
        ],
        rtol=1e-12,
    )


def test_beam_likelihoods_add_over_beams():
    rmap = _OneBeamMap(0.0, [1.0, 0.0, 0.0])
    one = measurement_log_likelihood(np.array([1.0]), np.array([[1.2]]), rmap, 0.15, 0.1)
    two = measurement_log_likelihood(np.array([1.0, 1.0]), np.array([[1.2, 1.2]]), rmap, 0.15, 0.1)
    np.testing.assert_allclose(two, 2 * one, rtol=1e-12)


def test_the_heavier_cluster_is_reported():
    """600 equal particles at (1, 1, 0) and 400 at (3, 1, pi): the estimate is the first cluster
    exactly, its share 0.6, and its covariance the roughening kernel alone."""
    particles = np.array([[1.0, 1.0, 0.0]] * 600 + [[3.0, 1.0, math.pi - 1e-12]] * 400)
    pose, covariance, share = heaviest_cluster(particles, np.ones(1000), ROUGHENING)
    np.testing.assert_allclose(pose, [1.0, 1.0, 0.0], atol=1e-12)
    assert share == pytest.approx(0.6)
    np.testing.assert_allclose(covariance, np.diag(ROUGHENING**2), rtol=1e-12, atol=1e-20)


def test_the_cluster_is_chosen_by_weight_not_count():
    particles = np.array([[1.0, 1.0, 0.0]] * 900 + [[3.0, 1.0, 1.0]] * 100)
    weights = np.array([1.0] * 900 + [20.0] * 100)  # 900 against 2000
    pose, _, share = heaviest_cluster(particles, weights, ROUGHENING)
    np.testing.assert_allclose(pose, [3.0, 1.0, 1.0], atol=1e-12)
    assert share == pytest.approx(2000 / 2900)


def test_a_cluster_across_pi_reports_pi_not_zero():
    """Headings pi - 0.01 and -pi + 0.01 average to pi: an arithmetic mean would give 0."""
    particles = np.array([[1.0, 1.0, math.pi - 0.01]] * 50 + [[1.0, 1.0, -math.pi + 0.01]] * 50)
    pose, covariance, _ = heaviest_cluster(particles, np.ones(100), ROUGHENING)
    assert abs(wrap_angle(pose[2] - math.pi)) < 1e-9, f"Heading {pose[2]} should be +-pi."
    assert covariance[2, 2] == pytest.approx(0.01**2 + ROUGHENING[2] ** 2), (
        "The heading spread is 0.01 rad either side of pi, not 2 pi - 0.01."
    )


def test_a_cluster_spread_gives_its_weighted_covariance():
    """Two particles 0.2 m apart in x with weights 3 and 1: mean x = 0.05 from the first, variance
    0.75 * 0.05^2 + 0.25 * 0.15^2 = 0.0075, plus the kernel."""
    particles = np.array([[1.0, 1.0, 0.0], [1.2, 1.0, 0.0]])
    pose, covariance, _ = heaviest_cluster(particles, np.array([3.0, 1.0]), ROUGHENING)
    np.testing.assert_allclose(pose, [1.05, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(covariance[0, 0], 0.0075 + ROUGHENING[0] ** 2, rtol=1e-12)


def test_free_poses_are_free_and_in_range():
    rmap = SensorMap(ROOM, lidar())
    poses = sample_free_poses(rmap, 500, 0.1, np.random.default_rng(3))
    assert poses.shape == (500, 3)
    assert rmap.free(poses[:, :2], 0.1).all(), "Every sampled pose should be clear of obstacles."
    assert np.all((poses[:, 2] >= -math.pi) & (poses[:, 2] < math.pi))


def test_a_map_without_free_space_is_named():
    with pytest.raises(ValueError, match="NoFreeSpaceMap"):
        sample_free_poses(_NoFreeSpaceMap(0.0, [1, 0, 0]), 10, 0.1, np.random.default_rng(0))


# Predict ----------------------------------------------------------------------------------------


def test_noiseless_prediction_follows_the_arc():
    """With noiseless encoders every particle moves on the exact arc of (left, right) travel
    (0.02, 0.03): distance 0.025 m, turn 0.05 rad."""
    mcl = _mcl(
        SensorMap(ROOM, lidar()), TRUTH, np.diag([0.01, 0.01, 0.01]), odometry=STILL_ODOMETRY
    )
    before = mcl.particles
    mcl.predict([0.02, 0.03])
    np.testing.assert_allclose(
        mcl.particles, unicycle_step(before, 0.025, 0.05, 1.0), rtol=0.0, atol=1e-14
    )


def test_prediction_noise_follows_the_wheel_model():
    """Each wheel rolls 0.1 m with variance k 0.1: the distance (their mean) has variance
    k 0.1 / 2 = 1.25e-5 m^2 and the turn (their difference over b) 2 k 0.1 / b^2 = 1.25e-3."""
    arena = Arena(20.0, 20.0)
    mcl = _mcl(SensorMap(arena, lidar()), [10.0, 10.0, 0.0], 1e-12 * np.eye(3), num_particles=20000)
    mcl.predict([0.1, 0.1])
    particles = mcl.particles
    np.testing.assert_allclose(np.var(particles[:, 0] - 10.0), 1.25e-5, rtol=0.05)
    np.testing.assert_allclose(np.var(particles[:, 2]), 1.25e-3, rtol=0.05)


def test_a_particle_that_would_hit_a_box_stays_put():
    """0.25 m below the box at y = 3.2: driving 0.2 m up would leave 0.05 m, inside the robot's
    0.1 m radius, so every particle stays; driving down moves them all."""
    rmap = SensorMap(ROOM, lidar())
    towards = _mcl(rmap, [1.4, 2.95, math.pi / 2], TIGHT, odometry=STILL_ODOMETRY)
    before = towards.particles
    towards.predict([0.2, 0.2])
    np.testing.assert_array_equal(towards.particles, before)
    away = _mcl(rmap, [1.4, 2.95, -math.pi / 2], TIGHT, odometry=STILL_ODOMETRY)
    away.predict([0.2, 0.2])
    np.testing.assert_allclose(away.particles[:, 1], 2.75, atol=1e-3)


def test_predict_refuses_a_bad_wheel_travel():
    mcl = _mcl(SensorMap(ROOM, lidar()), TRUTH, TIGHT)
    for travel in ([0.1], [0.1, np.nan], [[0.1, 0.1]]):
        with pytest.raises(ValueError, match="wheel_travel"):
            mcl.predict(travel)


# Update -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("lost", ["max_range", "nan", "inf", "min_range", "mixed"])
def test_a_scan_with_nothing_usable_changes_nothing(lost):
    """Nothing moves and the generator does not advance."""
    sensor = lidar()
    rng = np.random.default_rng(5)
    mcl = MclLocalizer(
        SensorMap(ROOM, sensor),
        ODOMETRY,
        TRUTH,
        np.diag([0.01, 0.01, 0.004]),
        rng=rng,
        settings=MclSettings(num_particles=300),
    )
    mcl.update(_scan(sensor, TRUTH))  # uneven weights
    fills = {"max_range": sensor.max_range, "nan": np.nan, "inf": np.inf}
    if lost == "mixed":
        scan = np.resize([sensor.max_range, np.nan, sensor.min_range, np.inf], sensor.num_readings)
    else:
        scan = np.full(sensor.num_readings, fills.get(lost, sensor.min_range))
    particles, weights, pose, state = mcl.particles, mcl.weights, mcl.pose, rng.bit_generator.state
    result = mcl.update(scan)
    assert not result.used.any(), f"A scan of {lost} readings should use no beam."
    np.testing.assert_array_equal(mcl.particles, particles)
    np.testing.assert_array_equal(mcl.weights, weights)
    np.testing.assert_array_equal(mcl.pose, pose)
    assert rng.bit_generator.state == state, "An unusable scan should draw nothing."


def test_a_lost_reading_equals_a_sensor_without_that_beam():
    sensor = lidar(16)
    lost = 5
    fewer = RangeSensor(
        mount_angles=np.delete(sensor.mount_angles, lost),
        min_range=sensor.min_range,
        max_range=sensor.max_range,
        noise_std=sensor.noise_std,
    )
    readings = _scan(sensor, TRUTH)
    start, spread = TRUTH + [0.05, -0.03, 0.02], np.diag([0.01, 0.01, 0.004])
    with_nan = readings.copy()
    with_nan[lost] = np.nan
    full = _mcl(SensorMap(ROOM, sensor), start, spread, seed=2)
    result = full.update(with_nan)
    reduced = _mcl(SensorMap(ROOM, fewer), start, spread, seed=2)
    reduced.update(np.delete(readings, lost))
    assert result.used.sum() == 15 and not result.used[lost]
    np.testing.assert_allclose(full.particles, reduced.particles, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(full.weights, reduced.weights, rtol=1e-9)


def test_beams_beyond_the_limit_are_spread_over_the_scan():
    sensor = lidar()  # 64 beams
    mcl = _mcl(SensorMap(ROOM, sensor), TRUTH, TIGHT, max_beams=16)
    used = np.flatnonzero(mcl.update(_scan(sensor, TRUTH)).used)
    np.testing.assert_array_equal(used, np.round(np.linspace(0, 63, 16)).astype(int))


def test_casts_of_inf_give_finite_weights_without_warnings():
    """Half the particles stand where the beam sees nothing (x >= 5): their likelihood is the
    max-range one, small but finite."""
    rmap = _OneBeamMap(1.0, [0.0, 0.0, 0.0], finite_below_x=5.0)
    mcl = _mcl(rmap, [5.0, 5.0, 0.0], np.diag([0.04, 0.04, 0.01]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mcl.update([1.0])
    weights = mcl.weights
    assert np.isfinite(weights).all() and weights.sum() == pytest.approx(1.0)
    particles = mcl.particles
    assert weights[particles[:, 0] < 5.0].sum() > 0.9, "Particles that see the wall should win."


def test_a_scan_of_the_wrong_length_is_refused():
    mcl = _mcl(SensorMap(ROOM, lidar()), TRUTH, TIGHT)
    with pytest.raises(ValueError, match="readings"):
        mcl.update(np.ones(63))


def test_a_still_robot_localizes_globally():
    sensor = lidar()
    mcl = _mcl(SensorMap(ROOM, sensor), seed=1, num_particles=2000)
    for step in range(30):
        mcl.predict([0.0, 0.0])
        mcl.update(_scan(sensor, TRUTH, seed=step))
    error = mcl.pose - TRUTH
    error[2] = wrap_angle(error[2])
    assert np.hypot(error[0], error[1]) < 0.05 and abs(error[2]) < 0.05, (
        f"A still robot should localize from anywhere to within 5 cm and 0.05 rad, ended {error} off."
    )
    assert mcl.cluster_weight > 0.9


SYMMETRIC_ROOM = Arena(4.0, 3.0)
SYMMETRIC_TRUTH = np.array([1.2, 1.0, 0.4])
SYMMETRIC_MIRROR = np.array([2.8, 2.0, 0.4 - math.pi])


def test_a_pose_and_its_mirror_explain_a_scan_equally():
    """An empty 4 x 3 m room looks the same from (x, y, theta) and (4 - x, 3 - y, theta + pi):
    every beam's expected range is the same, so no scan can tell the two apart."""
    rmap = SensorMap(SYMMETRIC_ROOM, lidar())
    beams = np.arange(rmap.num_readings)
    expected = rmap.expected_beam_ranges(np.array([SYMMETRIC_TRUTH, SYMMETRIC_MIRROR]), beams)
    scan = _scan(lidar(), SYMMETRIC_TRUTH, arena=SYMMETRIC_ROOM)
    truth, mirror = measurement_log_likelihood(scan, expected, rmap, 0.15, 0.1)
    assert truth == pytest.approx(mirror, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_symmetric_arena_is_localized_up_to_its_symmetry(seed):
    """The filter settles on the truth or its mirror, whichever its particles happened to fit
    best. Resampling collapses the other hypothesis within one to three scans (measured
    2026-11-06), so cluster_weight ends near 1 either way: it cannot flag this ambiguity."""
    sensor = lidar()
    mcl = _mcl(SensorMap(SYMMETRIC_ROOM, sensor), seed=seed, num_particles=2000)
    for step in range(30):
        mcl.predict([0.0, 0.0])
        mcl.update(_scan(sensor, SYMMETRIC_TRUTH, seed=step, arena=SYMMETRIC_ROOM))
    near_truth = _weight_near(mcl, SYMMETRIC_TRUTH)
    near_mirror = _weight_near(mcl, SYMMETRIC_MIRROR)
    assert near_truth + near_mirror > 0.99, (
        f"Seed {seed}: the weight should sit at the truth or its mirror, not "
        f"{near_truth:.2f} and {near_mirror:.2f}."
    )


def _flat_filter(reading_offsets):
    """300 particles that all expect 2 m on one beam (equal weights, so the effective sample size
    never asks for resampling), fed readings ``2 + offset``; returns the updates."""
    mcl = _mcl(_OneBeamMap(2.0, [0.0, 0.0, 0.0]), [5.0, 5.0, 0.0], np.diag([0.01, 0.01, 0.01]))
    return [mcl.update([2.0 + offset]) for offset in reading_offsets]


def test_injection_needs_at_least_one_particle():
    """A reading 2 cm off after an exact one lowers the per-beam likelihood by 0.9 %:
    p_inject = 1 - w_fast / w_slow is about 0.099 x 0.0087 = 8.6e-4, under 1 / 300, so nothing is
    injected or resampled. A reading 1 m off gives p_inject near 0.1: about 30 particles."""
    small = _flat_filter([0.0, 0.02])[-1]
    assert not small.resampled and small.injected == 0, (
        f"A p_inject below 1 / P should change nothing, got {small}."
    )
    large = _flat_filter([0.0, 1.0])[-1]
    assert large.resampled and 10 <= large.injected <= 60, (
        f"A drop to the floor should inject about 30 of 300 particles, got {large}."
    )


def test_a_cloud_of_only_injected_particles_keeps_injecting():
    """Four particles, readings 1.5 m off after one good scan, and alpha_fast = 1: nearly every
    particle is injected on every scan, so on some scan all of them were just injected. Left out
    of w_avg, they would leave it 0 / 0 = NaN, and injection would stop for good; the fallback
    counts them all."""
    mcl = _mcl(
        _OneBeamMap(2.0, [0.0, 0.0, 0.0]),
        [5.0, 5.0, 0.0],
        np.diag([0.01, 0.01, 0.01]),
        num_particles=4,
        alpha_fast=1.0,
    )
    mcl.update([2.0])
    updates = [mcl.update([3.5]) for _ in range(12)]
    assert sum(update.injected for update in updates[2:]) > 0, (
        f"Injection should go on while the readings stay off: {updates}."
    )
    assert all(np.isfinite(update.effective_size) for update in updates)


def test_injected_particles_count_again_on_the_scan_after_next():
    """Only the scan right after an injection leaves the injected particles out of w_avg. Scan 1
    (both beams 1.5 m off) injects most of 300 particles; scan 2 reads beam 0 alone, which every
    particle explains equally, so nothing resamples; scan 3 reads both beams at the settled
    particles' x = 2. The injected ones, spread over the square, explain beam 1 poorly: counted
    again, they pull w_avg under w_slow and more are injected."""
    mcl = _mcl(_TwoBeamMap(), [2.0, 5.0, 0.0], TIGHT, alpha_fast=1.0, resample_threshold=1e-6)
    mcl.update([2.0, 2.0])
    burst = mcl.update([3.5, 3.5])
    quiet = mcl.update([2.0, np.nan])
    again = mcl.update([2.0, 2.0])
    assert burst.injected > 100 and not quiet.resampled, (burst, quiet)
    assert again.injected > 0, (
        f"The particles injected two scans ago should count in w_avg again: {again}."
    )


def test_headings_stay_wrapped_after_resampling():
    """Particles at a heading of pi - 1e-4, jittered by 5 mrad, cross pi half the time."""
    sensor = lidar()
    pose = TRUTH.copy()
    pose[2] = math.pi - 1e-4
    mcl = _mcl(SensorMap(ROOM, sensor), pose, TIGHT, resample_threshold=1.0)
    result = mcl.update(_scan(sensor, pose))
    headings = mcl.particles[:, 2]
    assert result.resampled
    assert np.all((headings >= -math.pi) & (headings < math.pi)), "Headings should wrap at pi."
    assert np.any(headings < 0.0), "Some jittered headings should have crossed pi."


def _bumped_still_robot(scans_after):
    """300 particles on a still robot at TRUTH for 10 scans, then pushed 0.1 m sideways for
    ``scans_after`` scans; returns the filter, the bumped pose and every update."""
    sensor = lidar()
    mcl = _mcl(SensorMap(ROOM, sensor), TRUTH, np.diag([1e-4, 1e-4, 1e-4]), seed=3)
    bumped = TRUTH + [0.0, 0.1, 0.0]
    updates = []
    for step in range(10 + scans_after):
        mcl.predict([0.0, 0.0])
        updates.append(mcl.update(_scan(sensor, TRUTH if step < 10 else bumped, seed=step)))
    return mcl, bumped, updates


def test_a_bumped_still_robot_recovers():
    """Pushed 0.1 m sideways while still, which an EKF without its reset never recovers from.
    Measured 2026-11-06: within 3 cm for good after 18 scans at 300 particles."""
    mcl, bumped, _ = _bumped_still_robot(50)
    error = pose_difference(mcl.pose, bumped)
    assert np.hypot(error[0], error[1]) < 0.03 and abs(error[2]) < 0.03, (
        f"The filter should follow the bump to within 3 cm, ended {error} off."
    )


def test_a_recovered_robot_injects_little():
    """The particles injected to find the robot again would drag the likelihood's average down,
    keep the fast average under the slow one and so keep injecting, for the slow average's
    ~1000-scan memory. Leaving the particles injected on the previous scan out of that average
    breaks the loop: in the last 100 of 200 scans after the bump, 282 of 300 x 100 particle-scans
    were injected (measured 2026-11-06) against 1892 with them counted."""
    mcl, bumped, updates = _bumped_still_robot(200)
    late = sum(update.injected for update in updates[-100:])
    assert late <= 700, f"A recovered still robot injected {late} particles in its last 100 scans."
    error = pose_difference(mcl.pose, bumped)
    assert np.hypot(error[0], error[1]) < 0.03, f"The filter should have found the bump: {error}."


def test_a_localized_still_robot_rarely_resamples():
    """With nothing to correct, the weights stay even: 200 scans of a still robot on the real map
    resample only a few times (the reviewer measured 5 of 200 at 2000 particles)."""
    sensor = lidar()
    mcl = _mcl(SensorMap(ROOM, sensor), TRUTH, np.diag([1e-4, 1e-4, 1e-4]), seed=3)
    resampled = 0
    for step in range(200):
        mcl.predict([0.0, 0.0])
        resampled += mcl.update(_scan(sensor, TRUTH, seed=step)).resampled
    assert resampled <= 20, f"A localized still robot resampled on {resampled} of 200 scans."


def test_mean_shift_stops_when_no_particle_is_near_the_mean():
    """Two equal particles 0.74 m apart along the diagonal: their mean is 0.52 m from each, beyond
    the 0.5 m radius, so the estimate stays their mean with both as members: variance and
    covariance 0.37^2 in x and y, plus the kernel."""
    particles = np.array([[0.0, 0.0, 0.0], [0.74, 0.74, 0.0]])
    pose, covariance, share = heaviest_cluster(particles, np.ones(2), ROUGHENING)
    np.testing.assert_allclose(pose, [0.37, 0.37, 0.0], atol=1e-12)
    assert share == pytest.approx(1.0)
    expected = np.full((2, 2), 0.37**2) + np.diag(ROUGHENING[:2] ** 2)
    np.testing.assert_allclose(covariance[:2, :2], expected, rtol=1e-12)


def test_the_heading_seed_takes_its_neighbouring_bins():
    """Weight in heading bins 3 and 5 only (16 bins of pi/8): bin 4 with its neighbours holds all
    of it, so both particles belong to the cluster and its heading is their mean, bin 4's middle."""
    width = 2.0 * math.pi / 16
    headings = [-math.pi + 3.0 * width + 1e-9, -math.pi + 6.0 * width - 1e-9]
    particles = np.array([[1.0, 1.0, heading] for heading in headings])
    pose, _, share = heaviest_cluster(particles, np.ones(2), ROUGHENING)
    assert share == pytest.approx(1.0)
    assert pose[2] == pytest.approx(-math.pi + 4.5 * width, abs=1e-9)


def test_a_seeded_filter_is_reproducible():
    sensor = lidar()
    runs = []
    for seed in (7, 7, 8):
        mcl = _mcl(SensorMap(ROOM, sensor), seed=seed)
        for step in range(3):
            mcl.predict([0.01, 0.012])
            mcl.update(_scan(sensor, TRUTH, seed=step))
        runs.append(mcl.particles)
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2]), "Seeds should differ."


def _scaled_run(scale=None):
    """A seeded filter after three steps, with ``covariance_scale`` left at its default when
    ``scale`` is None."""
    sensor = lidar()
    settings = {} if scale is None else {"covariance_scale": scale}
    mcl = _mcl(SensorMap(ROOM, sensor), TRUTH, np.diag([0.01, 0.01, 0.004]), seed=3, **settings)
    for step in range(3):
        mcl.predict([0.01, 0.012])
        mcl.update(_scan(sensor, TRUTH, seed=step))
    return mcl


def test_the_covariance_scale_rescales_only_the_reported_covariance():
    """With D = diag(scale), the covariance is D^1/2 C D^1/2 for the cluster's C: each variance
    times its scale, each covariance times the root of both; particles and pose unchanged."""
    scale = np.array([0.25, 4.0, 2.0])
    default, scaled = _scaled_run(), _scaled_run(tuple(scale))
    np.testing.assert_array_equal(scaled.particles, default.particles)
    np.testing.assert_array_equal(scaled.pose, default.pose)
    root = np.sqrt(scale)
    np.testing.assert_array_equal(
        scaled.covariance,
        default.covariance * np.outer(root, root),
        err_msg="The covariance should be exactly D^1/2 C D^1/2.",
    )
    _, cluster_covariance, _ = heaviest_cluster(scaled.particles, scaled.weights, ROUGHENING)
    np.testing.assert_array_equal(
        default.covariance, cluster_covariance, err_msg="heaviest_cluster itself is unscaled."
    )


def test_the_default_covariance_scale_changes_nothing():
    """(1, 1, 1) reports the cluster's covariance bit for bit, as before the scale existed."""
    explicit, default = _scaled_run((1.0, 1.0, 1.0)), _scaled_run()
    _, cluster_covariance, _ = heaviest_cluster(default.particles, default.weights, ROUGHENING)
    np.testing.assert_array_equal(default.covariance, cluster_covariance)
    np.testing.assert_array_equal(explicit.covariance, cluster_covariance)
    np.testing.assert_array_equal(explicit.particles, default.particles)


# Construction -----------------------------------------------------------------------------------


def test_an_initial_heading_out_of_range_is_wrapped():
    mcl = _mcl(SensorMap(ROOM, lidar()), [2.5, 2.0, 3.0 * math.pi], TIGHT)
    headings = mcl.particles[:, 2]
    assert np.all((headings >= -math.pi) & (headings < math.pi))
    assert abs(wrap_angle(mcl.pose[2] - math.pi)) < 1e-3


def test_a_start_inside_an_obstacle_is_refused():
    with pytest.raises(ValueError, match="free"):
        _mcl(SensorMap(ROOM, lidar()), [1.4, 3.6, 0.0], TIGHT)


@pytest.mark.parametrize(
    ("pose", "covariance", "match"),
    [
        (None, np.eye(3), "covariance without a pose"),
        ([1.0, 1.0], np.eye(3), "pose"),
        (TRUTH, None, "covariance"),
        (TRUTH, -np.eye(3), "positive-definite"),
        (TRUTH, [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]], "symmetric"),
    ],
)
def test_an_invalid_start_is_refused(pose, covariance, match):
    with pytest.raises(ValueError, match=match):
        _mcl(SensorMap(ROOM, lidar()), pose, covariance)


@pytest.mark.parametrize(
    "settings",
    [
        {"num_particles": 0},
        {"num_particles": 2.5},
        {"num_particles": True},
        {"max_beams": 0},
        {"model_std": 0.0},
        {"z_rand": 0.0},
        {"z_rand": 1.0},
        {"robot_radius": -0.1},
        {"resample_threshold": 0.0},
        {"resample_threshold": 1.5},
        {"alpha_slow": 0.1, "alpha_fast": 0.1},
        {"alpha_fast": 1.5},
        {"roughening": (0.0, 0.005, 0.005)},
        {"roughening": (0.005, 0.005)},
        {"covariance_scale": (0.0, 1.0, 1.0)},
        {"covariance_scale": (1.0, -0.5, 1.0)},
        {"covariance_scale": (1.0, 1.0, np.inf)},
        {"covariance_scale": (1.0, 1.0)},
    ],
)
def test_invalid_settings_are_refused(settings):
    with pytest.raises(ValueError, match="MclSettings"):
        MclSettings(**settings)
