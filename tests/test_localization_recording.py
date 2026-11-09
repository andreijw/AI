"""rl_navigation.estimation.recording: snapshots of the EKF and MCL on one seeded trial, the
covariance ellipse, and the frames painted from them."""

import dataclasses

import numpy as np
import pytest

from rl_navigation.estimation import evaluation
from rl_navigation.estimation import recording as recording_module
from rl_navigation.estimation.evaluation import Trial, ekf, mcl, run_trial
from rl_navigation.estimation.mcl import MclSettings
from rl_navigation.estimation.recording import (
    Recording,
    SnapshotFactory,
    SnapshotLocalizer,
    Snapshots,
    covariance_ellipse,
    record_trials,
    render_frames,
)
from rl_navigation.sim import Arena, Motion, World, lidar

SEED = 7
STEPS = 30
KIDNAP_AT = 15
SETTINGS = MclSettings(num_particles=200)
MAX_PARTICLES = 50


@pytest.fixture(scope="module")
def recording():
    return record_trials(
        SEED,
        lidar(),
        steps=STEPS,
        kidnap_at=KIDNAP_AT,
        mcl_settings=SETTINGS,
        max_particles=MAX_PARTICLES,
    )


# ---------------------------------------------------------------------------
# covariance_ellipse
# ---------------------------------------------------------------------------


def test_covariance_ellipse_of_an_axis_aligned_covariance():
    """Standard deviations 0.2 and 0.1: the 2-sigma ellipse has semi-axes 0.4 and 0.2, and the
    closed outline starts where it ends."""
    covariance = np.diag([0.04, 0.01, 0.1])
    outline = covariance_ellipse([1.0, 2.0, 0.3], covariance)
    assert outline.shape == (49, 2)
    np.testing.assert_allclose(outline[0], outline[-1])
    d = outline - [1.0, 2.0]
    np.testing.assert_allclose((d[:, 0] / 0.4) ** 2 + (d[:, 1] / 0.2) ** 2, 1.0)
    np.testing.assert_allclose(d.max(axis=0), [0.4, 0.2])
    np.testing.assert_allclose(d.min(axis=0), [-0.4, -0.2])


def test_covariance_ellipse_of_a_rotated_covariance_lies_on_two_sigma():
    """Every point d from the mean satisfies d^T S^-1 d = n_sigma^2 = 4."""
    angle = 0.5
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    position = rotation @ np.diag([0.09, 0.01]) @ rotation.T
    covariance = np.zeros((3, 3))
    covariance[:2, :2] = position
    covariance[2, 2] = 0.2
    outline = covariance_ellipse([1.0, -1.0, 0.0], covariance, points=36)
    d = outline - [1.0, -1.0]
    np.testing.assert_allclose(np.einsum("pi,ij,pj->p", d, np.linalg.inv(position), d), 4.0)
    assert np.hypot(*d.T).max() == pytest.approx(0.6)  # 2 x 0.3 along the major axis


def test_covariance_ellipse_with_a_zero_variance_axis_is_a_line():
    """No spread in y (a small negative eigenvalue from round-off is clipped at 0): the outline
    is the segment from x - 0.4 to x + 0.4 at the mean's y."""
    outline = covariance_ellipse([1.0, 2.0, 0.0], np.diag([0.04, -1e-9, 0.1]))
    assert np.isfinite(outline).all()
    np.testing.assert_allclose(outline[:, 1], 2.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose([outline[:, 0].min(), outline[:, 0].max()], [0.6, 1.4])


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_covariance_ellipse_of_a_non_finite_covariance_is_nan(bad):
    covariance = np.eye(3)
    covariance[0, 1] = covariance[1, 0] = bad
    outline = covariance_ellipse([1.0, 2.0, 0.0], covariance)
    assert outline.shape == (49, 2) and np.isnan(outline).all()


# ---------------------------------------------------------------------------
# SnapshotLocalizer
# ---------------------------------------------------------------------------


class _Used:
    used = np.ones(3, dtype=bool)


class _StubFilter:
    """A filter with 2000 particles numbered by row, whose pose counts its updates."""

    def __init__(self, num_particles=2000):
        self.updates = 0
        self.num_particles = num_particles

    @property
    def pose(self):
        return np.array([float(self.updates), 0.0, 0.0])

    @property
    def covariance(self):
        return np.eye(3) * (1.0 + self.updates)

    @property
    def particles(self):
        rows = np.arange(self.num_particles, dtype=np.float64)
        return np.stack([rows + self.updates, rows, rows], axis=1)

    def predict(self, wheel_travel):
        pass

    def update(self, readings):
        self.updates += 1
        return _Used()


def test_snapshot_localizer_records_at_construction_and_after_each_update_only():
    wrapped = SnapshotLocalizer(_StubFilter())
    assert len(wrapped.poses) == 1
    wrapped.predict([0.0, 0.0])
    for _ in range(3):  # reads between updates record nothing
        _ = wrapped.pose, wrapped.covariance
    result = wrapped.update([1.0, 2.0, 3.0])
    assert result.used.all()
    wrapped.update([4.0, 5.0, 6.0])
    np.testing.assert_array_equal(np.array(wrapped.poses)[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(np.array(wrapped.covariances)[:, 0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(wrapped.scans, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(wrapped.pose, [2.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("num_particles", "max_particles", "stride"), [(2000, 600, 3), (200, 50, 4), (100, 600, 1)]
)
def test_snapshot_localizer_keeps_a_fixed_stride_subsample_of_the_particles(
    num_particles, max_particles, stride
):
    """At most max_particles, every stride-th from the first: 2000 to 600 keeps rows 0, 3, ...,
    1797; fewer particles than the cap keeps them all."""
    wrapped = SnapshotLocalizer(_StubFilter(num_particles), max_particles=max_particles)
    wrapped.update([1.0])
    expected_rows = np.arange(0, num_particles, stride)[:max_particles]
    for index, kept in enumerate(wrapped.particles):
        assert kept.shape == (min(num_particles, max_particles), 3)
        np.testing.assert_array_equal(kept[:, 1], expected_rows)
        np.testing.assert_array_equal(kept[:, 0], expected_rows + index)


def test_snapshot_localizer_of_a_filter_without_particles_keeps_none():
    class _NoParticles(_StubFilter):
        particles = None

    wrapped = SnapshotLocalizer(_NoParticles())
    wrapped.update([1.0])
    assert wrapped.particles is None


def test_snapshot_factory_keeps_what_run_trial_built_and_the_map():
    factory = SnapshotFactory(ekf())
    trial = run_trial(SEED, lidar(), steps=5, make_localizer=factory)
    assert factory.built is not None and len(factory.built.poses) == 6
    np.testing.assert_array_equal(factory.built.poses[1:], trial.estimates)
    assert factory.arena is not None and factory.sensor is not None
    assert factory.robot_radius == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# record_trials
# ---------------------------------------------------------------------------


def test_the_two_trials_share_one_trajectory_and_one_scan_per_step(recording):
    """The trajectory does not depend on the filter: the EKF's and MCL's trials see the same true
    poses and the same scans."""
    np.testing.assert_array_equal(recording.ekf_trial.true_poses, recording.mcl_trial.true_poses)
    np.testing.assert_array_equal(recording.ekf.scans, recording.mcl.scans)
    np.testing.assert_array_equal(recording.true_poses, recording.ekf_trial.true_poses)
    np.testing.assert_array_equal(recording.scans, recording.ekf.scans)
    assert recording.scans.shape == (STEPS, lidar().num_readings)
    assert recording.kidnap_at == KIDNAP_AT and not recording.collided


def test_recording_does_not_change_the_trials(recording):
    """Snapshots read the filters but never change them: the same trials run without them."""
    plain_ekf = run_trial(SEED, lidar(), steps=STEPS, kidnap_at=KIDNAP_AT, make_localizer=ekf())
    plain_mcl = run_trial(
        SEED,
        lidar(),
        steps=STEPS,
        kidnap_at=KIDNAP_AT,
        make_localizer=mcl(SETTINGS, global_start=True),
    )
    np.testing.assert_array_equal(recording.ekf_trial.estimates, plain_ekf.estimates)
    np.testing.assert_array_equal(recording.mcl_trial.estimates, plain_mcl.estimates)


@pytest.mark.parametrize("name", ["ekf", "mcl"])
def test_snapshot_k_is_the_filter_after_update_k(recording, name):
    """steps + 1 snapshots: the filter as built, then after each update, which is what run_trial
    recorded as its estimate and covariance of step k - 1."""
    snapshots, trial = {
        "ekf": (recording.ekf, recording.ekf_trial),
        "mcl": (recording.mcl, recording.mcl_trial),
    }[name]
    assert len(snapshots.poses) == len(snapshots.covariances) == STEPS + 1
    np.testing.assert_array_equal(snapshots.poses[1:], trial.estimates)
    np.testing.assert_array_equal(snapshots.covariances[1:], trial.covariances)


def test_the_ekf_starts_near_the_robot_and_mcl_from_anywhere(recording):
    """Frame 0: the EKF at the noisy start (5 cm noise, one 0.1 s step from the first true pose);
    MCL's global start spreads its cloud over most of the 5 x 5 m arena."""
    assert recording.ekf.particles is None
    assert np.hypot(*(recording.ekf.poses[0, :2] - recording.true_poses[0, :2])) < 0.3
    cloud = recording.mcl.particles[0]
    assert recording.mcl.particles.shape == (STEPS + 1, MAX_PARTICLES, 3)
    spread = cloud[:, :2].max(axis=0) - cloud[:, :2].min(axis=0)
    assert (spread > 0.6 * np.array([recording.arena.width, recording.arena.height])).all()


def test_record_trials_refuses_trials_that_part_ways(monkeypatch):
    """Two trials on one seed that do not share a trajectory mean the trial depends on the filter:
    the frames would show one robot and the other filter's estimate."""
    calls = []

    def diverging(*args, **kwargs):
        trial = run_trial(*args, **kwargs)
        calls.append(trial)
        if len(calls) == 2:
            trial = dataclasses.replace(trial, true_poses=trial.true_poses + 0.01)
        return trial

    monkeypatch.setattr(recording_module, "run_trial", diverging)
    with pytest.raises(ValueError, match="true poses"):
        record_trials(SEED, lidar(), steps=5, kidnap_at=None, mcl_settings=SETTINGS)


def test_a_collision_shortens_both_trials_alike(monkeypatch):
    """A world whose 12th move collides ends both trials after 11 steps: 12 snapshots each, and
    frames for those only."""

    class CollidingWorld(World):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.moves = 0

        def step(self, *args, **kwargs):
            self.moves += 1
            motion = super().step(*args, **kwargs)
            return Motion(motion.v, motion.omega, True) if self.moves == 12 else motion

    monkeypatch.setattr(evaluation, "World", CollidingWorld)
    shortened = record_trials(SEED, lidar(), steps=20, kidnap_at=None, mcl_settings=SETTINGS)
    assert shortened.collided
    assert len(shortened.true_poses) == 11
    assert len(shortened.ekf.poses) == len(shortened.mcl.poses) == 12
    assert len(list(render_frames(shortened, pixels_per_metre=10))) == 12


def test_record_trials_refuses_a_kidnap_outside_the_run():
    with pytest.raises(ValueError, match="kidnap_at"):
        record_trials(SEED, lidar(), steps=5, kidnap_at=5, mcl_settings=SETTINGS)


# ---------------------------------------------------------------------------
# render_frames
# ---------------------------------------------------------------------------


def _count(frame, colour):
    return int((frame == colour).all(axis=2).sum())


@pytest.fixture(scope="module")
def frames(recording):
    return list(render_frames(recording, pixels_per_metre=20))


def test_render_frames_paints_one_frame_per_snapshot_at_full_resolution(recording, frames):
    """A 5 x 5 m arena at 20 px/m is 100 x 100, plus the progress strip below it."""
    rows = int(round(recording.arena.height * 20)) + recording_module.STRIP_ROWS
    cols = int(round(recording.arena.width * 20))
    assert len(frames) == STEPS + 1
    assert all(f.shape == (rows, cols, 3) and f.dtype == np.uint8 for f in frames)


def test_frame_zero_shows_the_cloud_and_the_robot_but_no_mcl_estimate(frames):
    first, later = frames[0], frames[1]
    assert _count(first, recording_module.PARTICLE_COLOUR) > 10
    assert _count(first, recording_module.EKF_COLOUR) > 0
    assert _count(first, recording_module.ROBOT_COLOUR) > 0
    assert _count(first, recording_module.MCL_COLOUR) == 0
    assert _count(later, recording_module.MCL_COLOUR) > 0


def test_the_progress_strip_fills_and_marks_the_kidnapping(recording, frames):
    strip_first = frames[0][-recording_module.STRIP_ROWS :]
    strip_last = frames[-1][-recording_module.STRIP_ROWS :]
    assert _count(strip_first, recording_module.KIDNAP_COLOUR) > 0
    assert _count(strip_last, recording_module.PROGRESS_COLOUR) > _count(
        strip_first, recording_module.PROGRESS_COLOUR
    )


# A hand-built recording, so every frame's pieces sit apart and each can be found by its colour:
# 4 steps in an empty 5 x 5 m arena at 40 px/m, kidnapped at step 2 from (1, 2) to (4, 4). The EKF
# sits at x = 2.5 and MCL at x = 3.2, one row higher each frame; their 2-sigma ellipses grow by
# 0.15 m a frame (MCL's 0.1 m larger); MCL keeps one particle a frame, 0.6 m further right each time.
PPM = 40
TRUE_POSES = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [4.0, 4.0, 0.0], [4.0, 3.0, 0.0]])
SYNTHETIC_KIDNAP = 2
FRAMES = len(TRUE_POSES) + 1
ROWS = np.arange(FRAMES)
EKF_POSES = np.stack([np.full(FRAMES, 2.5), 0.5 + 0.9 * ROWS, np.zeros(FRAMES)], axis=1)
MCL_POSES = np.stack([np.full(FRAMES, 3.2), 0.5 + 0.9 * ROWS, np.zeros(FRAMES)], axis=1)
EKF_TWO_SIGMA = 0.15 * (ROWS + 1)
MCL_TWO_SIGMA = 0.1 + 0.15 * (ROWS + 1)
PARTICLES = np.stack([0.3 + 0.6 * ROWS, np.full(FRAMES, 0.3), np.zeros(FRAMES)], axis=1)[:, None]


def _covariances(two_sigma):
    return np.array([np.diag([(r / 2) ** 2, (r / 2) ** 2, 0.01]) for r in two_sigma])


def _trial(poses, covariances):
    return Trial(
        true_poses=TRUE_POSES,
        estimates=poses[1:],
        covariances=covariances[1:],
        dead_reckoning=TRUE_POSES,
        beams_used=np.zeros(len(TRUE_POSES), dtype=np.int_),
        collided=False,
        kidnap_at=SYNTHETIC_KIDNAP,
    )


@pytest.fixture(scope="module")
def synthetic_frames():
    scans = np.zeros((len(TRUE_POSES), 1))  # zero-length beams, under the robot
    ekf_cov, mcl_cov = _covariances(EKF_TWO_SIGMA), _covariances(MCL_TWO_SIGMA)
    hand_built = Recording(
        arena=Arena(width=5.0, height=5.0),
        sensor=lidar(num_beams=1),
        robot_radius=0.1,
        ekf=Snapshots(EKF_POSES, ekf_cov, scans, None),
        mcl=Snapshots(MCL_POSES, mcl_cov, scans, PARTICLES),
        ekf_trial=_trial(EKF_POSES, ekf_cov),
        mcl_trial=_trial(MCL_POSES, mcl_cov),
    )
    return list(render_frames(hand_built, pixels_per_metre=PPM))


def _arena(frame):
    return frame[: -recording_module.STRIP_ROWS]


def _centres(frame, colour):
    """``(K, 2)`` arena coordinates of the pixel centres painted ``colour``."""
    arena = _arena(frame)
    rows, cols = np.nonzero((arena == colour).all(axis=2))
    return np.stack([(cols + 0.5) / PPM, (arena.shape[0] - 1 - rows + 0.5) / PPM], axis=1)


def _colour_at(frame, point):
    arena = _arena(frame)
    return tuple(arena[arena.shape[0] - 1 - int(point[1] * PPM), int(point[0] * PPM)])


def test_synthetic_frames_one_per_snapshot(synthetic_frames):
    assert len(synthetic_frames) == FRAMES


@pytest.mark.parametrize("frame", range(FRAMES))
def test_frame_k_draws_the_robot_after_step_k(synthetic_frames, frame):
    """Frame k's robot is at true_poses[k - 1]; frame 0's at true_poses[0] (after step 1)."""
    centre = _centres(synthetic_frames[frame], recording_module.ROBOT_COLOUR).mean(axis=0)
    np.testing.assert_allclose(centre, TRUE_POSES[max(frame - 1, 0), :2], atol=0.05)


@pytest.mark.parametrize("frame", range(FRAMES))
def test_frame_k_draws_each_filter_after_update_k(synthetic_frames, frame):
    """The EKF's disc and ellipse are centred on ekf.poses[k], MCL's on mcl.poses[k] (none on
    frame 0), and each ellipse reaches 2 sigma of covariances[k] from its centre."""
    image = synthetic_frames[frame]
    ekf_pixels = _centres(image, recording_module.EKF_COLOUR)
    np.testing.assert_allclose(ekf_pixels.mean(axis=0), EKF_POSES[frame, :2], atol=0.05)
    reach = np.hypot(*(ekf_pixels - EKF_POSES[frame, :2]).T).max()
    assert reach == pytest.approx(EKF_TWO_SIGMA[frame], abs=1.5 / PPM)
    mcl_pixels = _centres(image, recording_module.MCL_COLOUR)
    if frame == 0:
        assert len(mcl_pixels) == 0
        return
    np.testing.assert_allclose(mcl_pixels.mean(axis=0), MCL_POSES[frame, :2], atol=0.05)
    reach = np.hypot(*(mcl_pixels - MCL_POSES[frame, :2]).T).max()
    assert reach == pytest.approx(MCL_TWO_SIGMA[frame], abs=1.5 / PPM)


@pytest.mark.parametrize("frame", range(FRAMES))
def test_frame_k_draws_the_particles_of_snapshot_k(synthetic_frames, frame):
    centre = _centres(synthetic_frames[frame], recording_module.PARTICLE_COLOUR).mean(axis=0)
    np.testing.assert_allclose(centre, PARTICLES[frame, 0, :2], atol=1.5 / PPM)


def test_the_kidnap_mark_sits_at_the_first_frame_after_the_kidnapping(synthetic_frames):
    """The robot is first seen at its new pose on frame kidnap_at + 1 = 3 of frames 0 to 4, so
    the mark is centred 3/4 of the way along the strip."""
    strip = synthetic_frames[0][-recording_module.STRIP_ROWS :]
    _, cols = np.nonzero((strip == recording_module.KIDNAP_COLOUR).all(axis=2))
    expected = (strip.shape[1] - 1) * (SYNTHETIC_KIDNAP + 1) / (FRAMES - 1)
    assert cols.mean() == pytest.approx(expected, abs=1.0)


def test_the_path_grows_one_segment_a_frame(synthetic_frames):
    """Frame 2 shows the segment of steps 1 to 2 and frame 1 does not; the last frame shows the
    last segment, after the kidnapping."""
    first_segment = (TRUE_POSES[0, :2] + TRUE_POSES[1, :2]) / 2
    last_segment = (TRUE_POSES[2, :2] + TRUE_POSES[3, :2]) / 2
    colour = recording_module.TRUE_PATH_COLOUR
    assert _colour_at(synthetic_frames[1], first_segment) != colour
    assert _colour_at(synthetic_frames[2], first_segment) == colour
    assert _colour_at(synthetic_frames[3], last_segment) != colour
    assert _colour_at(synthetic_frames[4], last_segment) == colour


@pytest.mark.parametrize("frame", [3, 4])
def test_the_true_path_breaks_at_the_kidnapping(synthetic_frames, frame):
    """No path pixel anywhere along the jump from (1, 2) to (4, 4), away from its two ends."""
    before, after = TRUE_POSES[SYNTHETIC_KIDNAP - 1, :2], TRUE_POSES[SYNTHETIC_KIDNAP, :2]
    for t in np.linspace(0.1, 0.9, 200):
        point = before + t * (after - before)
        assert _colour_at(synthetic_frames[frame], point) != recording_module.TRUE_PATH_COLOUR, (
            f"path pixel at {point} on the kidnapping's jump"
        )
