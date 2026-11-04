"""rl_navigation.sim.kinematics and rl_navigation.sim.world: the unicycle model and the robot's world."""

import math

import numpy as np
import pytest

from rl_navigation.sim.geometry import Arena
from rl_navigation.sim.kinematics import unicycle_step, wrap_angle
from rl_navigation.sim.sensors import hc_sr04
from rl_navigation.sim.world import World

EMPTY = Arena(width=5.0, height=5.0)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.5, 0.5),
        (math.pi, -math.pi),  # the half-open interval [-pi, pi) excludes +pi
        (-math.pi, -math.pi),
        (3 * math.pi, -math.pi),
        (-3 * math.pi, -math.pi),
        (2 * math.pi, 0.0),
        (-2 * math.pi, 0.0),
        (4 * math.pi + 0.25, 0.25),
        (-math.pi / 2 - 2 * math.pi, -math.pi / 2),
    ],
)
def test_wrap_angle(angle, expected):
    assert wrap_angle(angle) == pytest.approx(expected, abs=1e-12)


def test_wrap_angle_stays_below_pi_just_under_minus_pi():
    # x + pi is a tiny negative number here, whose remainder modulo 2 pi rounds up to 2 pi.
    wrapped = wrap_angle(np.nextafter(-math.pi, -math.inf))
    assert -math.pi <= wrapped < math.pi


def test_wrap_angle_is_elementwise_on_arrays():
    angles = np.array([0.0, math.pi, 7.0, -7.0])
    np.testing.assert_allclose(
        wrap_angle(angles), [0.0, -math.pi, 7 - 2 * math.pi, 2 * math.pi - 7]
    )


def test_unicycle_step_straight():
    # 0.5 m/s for 2 s at 45 deg: 1 m along the diagonal, (sqrt(2) / 2, sqrt(2) / 2).
    pose = unicycle_step(np.array([1.0, 2.0, math.pi / 4]), 0.5, 0.0, 2.0)
    np.testing.assert_allclose(
        pose, [1 + math.sqrt(0.5), 2 + math.sqrt(0.5), math.pi / 4], rtol=1e-12
    )


def test_unicycle_step_on_an_arc_is_exact():
    # 1 m/s and pi/2 rad/s for 1 s: a quarter circle of radius R = v / w = 2 / pi, from (0, 0)
    # heading +x to (R, R) heading +y.
    pose = unicycle_step(np.array([0.0, 0.0, 0.0]), 1.0, math.pi / 2, 1.0)
    np.testing.assert_allclose(pose, [2 / math.pi, 2 / math.pi, math.pi / 2], atol=1e-12)


def test_unicycle_step_turning_right_from_a_rotated_heading():
    # Heading +y at (1, 1), turning at -pi/2 rad/s for 1 s with R = 2 / pi: a quarter circle to the
    # right ends at (1 + R, 1 + R) heading +x.
    pose = unicycle_step(np.array([1.0, 1.0, math.pi / 2]), 1.0, -math.pi / 2, 1.0)
    r = 2 / math.pi
    np.testing.assert_allclose(pose, [1 + r, 1 + r, 0.0], atol=1e-12)


def test_unicycle_step_full_circle_returns_to_the_start():
    pose = unicycle_step(np.array([2.0, 3.0, 0.3]), 1.0, 2 * math.pi, 1.0)
    np.testing.assert_allclose(pose, [2.0, 3.0, 0.3], atol=1e-12)


def test_unicycle_step_turning_on_the_spot_wraps_the_heading():
    pose = unicycle_step(np.array([2.0, 3.0, 3.0]), 0.0, 2.0, 0.1)
    np.testing.assert_allclose(pose, [2.0, 3.0, 3.2 - 2 * math.pi], atol=1e-12)


def test_unicycle_step_is_continuous_as_the_turn_rate_vanishes():
    start = np.array([1.0, 1.0, 0.7])
    straight = unicycle_step(start, 0.5, 0.0, 0.1)
    np.testing.assert_allclose(unicycle_step(start, 0.5, 1e-7, 0.1), straight, atol=1e-9)
    np.testing.assert_allclose(unicycle_step(start, 0.5, 1e-12, 0.1), straight, atol=1e-12)


def test_unicycle_step_does_not_modify_its_input():
    start = np.array([1.0, 1.0, 0.7])
    unicycle_step(start, 0.5, 1.0, 0.1)
    np.testing.assert_array_equal(start, [1.0, 1.0, 0.7])


# --- World ------------------------------------------------------------------------------------------


def make_world(pose=(1.0, 2.5, 0.0), arena=EMPTY, **kwargs) -> World:
    return World(arena, hc_sr04(), np.array(pose), np.array([4.0, 2.5]), **kwargs)


def test_world_step_without_slip_executes_the_command():
    world = make_world(dt=0.1)
    motion = world.step(0.5, 0.0, np.random.default_rng(0))
    assert (motion.v, motion.omega, motion.collided) == (0.5, 0.0, False)
    np.testing.assert_allclose(world.pose, [1.05, 2.5, 0.0], rtol=1e-12)


def test_world_step_executes_the_command_reduced_by_wheel_slip():
    # Each of v and w is scaled by (1 - slip * u), u ~ U(0, 1), drawn in that order.
    world = make_world(dt=0.1, wheel_slip=0.4)
    start = world.pose
    motion = world.step(0.5, 1.0, np.random.default_rng(3))
    u = np.random.default_rng(3).random(2)
    assert motion.v == pytest.approx(0.5 * (1 - 0.4 * u[0]), rel=1e-15)
    assert motion.omega == pytest.approx(1.0 * (1 - 0.4 * u[1]), rel=1e-15)
    assert motion.v < 0.5 and motion.omega < 1.0
    np.testing.assert_allclose(world.pose, unicycle_step(start, motion.v, motion.omega, 0.1))


def test_world_step_draws_the_slip_even_without_slip():
    # The generator advances by the same amount whatever the slip, so two runs that differ only in
    # slip see the same later random numbers.
    rng = np.random.default_rng(4)
    make_world(wheel_slip=0.0).step(0.5, 0.0, rng)
    expected = np.random.default_rng(4)
    expected.random(2)
    assert rng.random() == expected.random()


def test_world_step_into_a_wall_collides_and_stays_put():
    world = make_world(pose=(4.85, 2.5, 0.0), dt=0.1)
    motion = world.step(1.0, 0.0, np.random.default_rng(0))  # to x = 4.95, 0.05 m from x = 5
    assert motion.collided, "Moving to within the robot's radius of the x = 5 wall should collide."
    np.testing.assert_array_equal(world.pose, [4.85, 2.5, 0.0])


def test_world_step_through_a_thin_wall_collides():
    # One 1 s step at 0.5 m/s from x = 1.3 to x = 1.8 crosses a 1 cm wall at x = 1.5; both ends of
    # the move are more than the robot's 0.1 m radius from it.
    arena = Arena(5.0, 5.0, boxes=[[1.5, 0.5, 1.51, 4.5]])
    world = make_world(pose=(1.3, 2.5, 0.0), arena=arena, dt=1.0)
    assert world.step(0.5, 0.0, np.random.default_rng(0)).collided
    np.testing.assert_array_equal(world.pose, [1.3, 2.5, 0.0])


def test_world_goal_distance_and_arrival():
    world = make_world(pose=(3.7, 2.5, 0.0), goal_radius=0.2)
    assert world.goal_distance == pytest.approx(0.3, abs=1e-12)
    assert not world.at_goal
    world.step(1.0, 0.0, np.random.default_rng(0))  # dt 0.1: x 3.7 -> 3.8, then 0.2 from the goal
    world.step(1.0, 0.0, np.random.default_rng(0))  # x 3.9, 0.1 from the goal
    assert world.at_goal


def test_world_read_sensors_reads_from_the_current_pose():
    world = make_world(pose=(1.0, 2.5, 0.0))
    readings, true_ranges = world.read_sensors(
        np.random.default_rng(9), noise_std=0.0, dropout_prob=0.0
    )
    expected, _ = hc_sr04().read(EMPTY, world.pose, np.random.default_rng(9), 0.0, 0.0)
    np.testing.assert_array_equal(readings, expected)
    # The forward sonar sits 0.1 m ahead of the centre (hc_sr04's mount_radius), 3.9 m from x = 5.
    assert true_ranges[2] == pytest.approx(3.9, abs=1e-12), "The forward sonar should read 3.9 m."


def test_world_pose_is_a_copy():
    world = make_world()
    world.pose[0] = 3.0
    assert world.pose[0] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pose": (4.95, 2.5, 0.0)}, "collision"),
        ({"wheel_slip": 1.5}, "wheel_slip"),
        ({"dt": 0.0}, "dt"),
        ({"robot_radius": 0.0}, "robot_radius"),
        ({"goal_radius": 0.0}, "goal_radius"),
    ],
)
def test_world_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_world(**kwargs)


# --- review fixes -----------------------------------------------------------------------------------


def test_unicycle_step_is_batched_over_poses_and_commands():
    rng = np.random.default_rng(0)
    poses = np.column_stack([rng.uniform(0, 5, (6, 2)), rng.uniform(-3, 3, 6)])
    v = rng.uniform(0, 1, 6)
    omega = np.array([0.0, 1e-12, -2.0, 2.0, 0.5, -0.5])
    batched = unicycle_step(poses, v, omega, 0.1)
    assert batched.shape == (6, 3)
    for i in range(6):
        np.testing.assert_allclose(
            batched[i], unicycle_step(poses[i], float(v[i]), float(omega[i]), 0.1), atol=1e-15
        )
    # A scalar command broadcasts over a batch of poses.
    assert unicycle_step(poses, 0.5, 1.0, 0.1).shape == (6, 3)


@pytest.mark.parametrize("omega", [1e-6, 1e-8, 2e-9, 5e-10])
def test_unicycle_step_with_a_tiny_turn_rate_matches_the_straight_line_limit(omega):
    # As omega dt -> 0 the arc tends to a straight move along the mid-step heading theta + omega
    # dt / 2; the error is of order v dt (omega dt)^2 / 24, below 1e-15 here. A v / omega form
    # loses about 1e-16 v / omega to cancellation (5e-9 m at omega = 1e-8).
    pose, v, dt = np.array([1.0, 2.0, 0.7]), 0.5, 0.1
    mid = 0.7 + omega * dt / 2
    expected = [1 + v * dt * math.cos(mid), 2 + v * dt * math.sin(mid), 0.7 + omega * dt]
    np.testing.assert_allclose(unicycle_step(pose, v, omega, dt), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "pose",
    [(2.5, 2.5, 0.0), (6.0, 2.5, 0.0)],  # inside the box, outside the arena
    ids=["inside-a-box", "outside-the-arena"],
)
def test_world_rejects_a_start_inside_a_box_or_outside_the_arena(pose):
    arena = Arena(5.0, 5.0, boxes=[[1.0, 1.0, 4.0, 4.0]])
    with pytest.raises(ValueError, match="collision"):
        World(arena, hc_sr04(), np.array(pose), np.array([4.5, 4.5]))


def test_world_step_refuses_a_turn_the_chord_check_cannot_follow():
    # A half turn in one step (pi rad): from (1, 2.5) heading +x at 1 m/s and pi rad/s for 1 s the
    # arc bulges 1 / pi m to the right of its chord and would pass the circle at (1.32, 2.18),
    # which the chord never comes near. World.step refuses any turn over MAX_TURN_PER_STEP.
    arena = Arena(5.0, 5.0, circle_centers=[[1.32, 2.18]], circle_radii=[0.05])
    world = World(arena, hc_sr04(), np.array([1.0, 2.5, 0.0]), np.array([4.0, 4.0]), dt=1.0)
    with pytest.raises(ValueError, match="MAX_TURN_PER_STEP"):
        world.step(1.0, -math.pi, np.random.default_rng(0))
    np.testing.assert_array_equal(world.pose, [1.0, 2.5, 0.0])


def test_world_step_allows_the_largest_supported_turn():
    from rl_navigation.sim.world import MAX_TURN_PER_STEP

    world = make_world(dt=0.1)
    assert not world.step(0.1, MAX_TURN_PER_STEP / 0.1, np.random.default_rng(0)).collided
