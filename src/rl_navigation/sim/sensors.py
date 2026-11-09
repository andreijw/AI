"""Range sensors: ultrasonic sonars and lidar beams, with Gaussian noise and dropouts.

A sensor is a set of mounts at fixed angles to the robot's heading. Each mount reads the nearest
surface within its cone, cast as ``rays_per_cone`` rays spread evenly across the cone's full width
(one ray along the mount for a zero-width cone such as a lidar beam).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.sim.geometry import Arena, FloatArray, cast_rays
from rl_navigation.sim.kinematics import wrap_angle


def _check_noise(noise_std: float, dropout_prob: float) -> None:
    if not noise_std >= 0.0:
        raise ValueError(f"noise_std must be at least 0, got {noise_std}.")
    if not 0.0 <= dropout_prob <= 1.0:
        raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}.")


@dataclass(frozen=True, eq=False)
class RangeSensor:
    """A ring of range sensors fixed to the robot.

    Args:
        mount_angles: ``(N,)`` mount directions in radians relative to the heading.
        cone_width: Full width of each mount's cone in radians (0 for a single ray).
        rays_per_cone: Rays cast across each cone (at least 1).
        min_range: Nearest distance the sensor reports, in metres.
        max_range: Farthest distance the sensor reports, in metres; a dropout also reads it.
        noise_std: Default standard deviation of the Gaussian range noise, in metres.
        dropout_prob: Default probability that a reading is lost.
        mount_radius: Distance in metres from the robot's centre to each mount, along its mount
            angle: rays start there and readings are measured from there (0 for a sensor at the
            centre).

    ``noise_std`` and ``dropout_prob`` are defaults: :meth:`read` takes either as an argument, so an
    environment that randomizes them passes its own values and keeps a single source of truth.

    **Reading contract.** A scan is an ``(N,)`` float array in ``mount_angles`` order, in metres
    from each mount, within ``[min_range, max_range]``, reading exactly ``max_range`` when no echo
    returns. A hardware driver that replaces :meth:`read` on the robot must return the same.

    Raises:
        ValueError: ``min_range >= max_range``, a negative cone width, noise or mount radius, a
            ray count that is not a positive integer, or a dropout probability outside ``[0, 1]``.
    """

    mount_angles: FloatArray
    cone_width: float = 0.0
    rays_per_cone: int = 1
    min_range: float = 0.02
    max_range: float = 4.0
    noise_std: float = 0.0
    dropout_prob: float = 0.0
    mount_radius: float = 0.0
    ray_offsets: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mounts = np.array(self.mount_angles, dtype=np.float64).reshape(-1)
        if not 0.0 <= self.min_range < self.max_range:
            raise ValueError(
                f"RangeSensor needs 0 <= min_range < max_range, got {self.min_range} and {self.max_range}."
            )
        if not self.cone_width >= 0.0:
            raise ValueError(f"RangeSensor cone_width must be at least 0, got {self.cone_width}.")
        if not self.mount_radius >= 0.0:
            raise ValueError(
                f"RangeSensor mount_radius must be at least 0, got {self.mount_radius}."
            )
        if isinstance(self.rays_per_cone, bool) or not (
            isinstance(self.rays_per_cone, int | np.integer) and self.rays_per_cone >= 1
        ):
            raise ValueError(
                f"RangeSensor rays_per_cone must be at least 1, got {self.rays_per_cone}."
            )
        _check_noise(self.noise_std, self.dropout_prob)
        spread = (
            np.linspace(-0.5, 0.5, self.rays_per_cone) if self.rays_per_cone > 1 else np.zeros(1)
        )
        offsets = mounts[:, None] + self.cone_width * spread  # (N, K)
        mounts.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "mount_angles", mounts)
        object.__setattr__(self, "ray_offsets", offsets)

    @property
    def num_readings(self) -> int:
        """Number of readings per scan, one per mount."""
        return len(self.mount_angles)

    def true_ranges(self, arena: Arena, poses: ArrayLike) -> FloatArray:
        """Noiseless, unclipped ranges from many poses at once (for a particle filter's likelihood).

        Args:
            arena: The arena.
            poses: ``(P, 3)`` poses ``(x, y, theta)``.

        Returns:
            ``(P, N)``: for each pose and mount, the minimum distance over the cone's rays,
            measured from the mount.
        """
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        if self.mount_radius == 0.0:  # every ray starts at the centre: one origin per pose
            ranges = cast_rays(arena, poses[:, :2], poses[:, 2, None] + self.ray_offsets.ravel())
        else:  # one origin per mount, on the robot's perimeter
            mounts = poses[:, 2, None] + self.mount_angles  # (P, N)
            origins = poses[:, None, :2] + self.mount_radius * np.stack(
                [np.cos(mounts), np.sin(mounts)], axis=-1
            )  # (P, N, 2)
            angles = (poses[:, 2, None, None] + self.ray_offsets).reshape(-1, self.rays_per_cone)
            ranges = cast_rays(arena, origins.reshape(-1, 2), angles)
        return ranges.reshape(len(poses), self.num_readings, -1).min(axis=-1)

    def beams(self, pose: ArrayLike, readings: ArrayLike) -> FloatArray:
        """Each mount's beam as a segment from the mount to the point it read, for drawing.

        Args:
            pose: ``(3,)`` robot pose ``(x, y, theta)``.
            readings: ``(N,)`` a scan from :meth:`read`; a dropout reads ``max_range`` and is
                drawn at full length.

        Returns:
            ``(N, 2, 2)``: for each mount, its position and the point ``readings`` metres along
            its direction.
        """
        x, y, theta = np.asarray(pose, dtype=np.float64)
        angles = theta + self.mount_angles
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        mounts = np.array([x, y]) + self.mount_radius * directions
        ranges = np.asarray(readings, dtype=np.float64).reshape(-1, 1)
        return np.stack([mounts, mounts + ranges * directions], axis=1)

    def read(
        self,
        arena: Arena,
        pose: ArrayLike,
        rng: np.random.Generator,
        noise_std: float | None = None,
        dropout_prob: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """One scan from ``pose``: true range plus Gaussian noise, clipped, with dropouts.

        Each reading is ``clip(true + noise_std * z, min_range, max_range)`` with ``z ~ N(0, 1)``;
        with probability ``dropout_prob`` it is lost and reads exactly ``max_range``, as an
        HC-SR04 does when no echo returns. The noise and the dropout are drawn for every reading
        whatever their settings, so the generator advances the same way for any noise level.

        Args:
            arena: The arena.
            pose: ``(3,)`` pose ``(x, y, theta)``.
            rng: Generator for the noise and the dropouts.
            noise_std: Noise standard deviation in metres; None uses the sensor's own.
            dropout_prob: Dropout probability; None uses the sensor's own.

        Returns:
            ``(readings, true_ranges)``, each ``(N,)``; the true ranges are not clipped.

        Raises:
            ValueError: A negative ``noise_std`` or a ``dropout_prob`` outside ``[0, 1]``.
        """
        noise_std = self.noise_std if noise_std is None else noise_std
        dropout_prob = self.dropout_prob if dropout_prob is None else dropout_prob
        _check_noise(noise_std, dropout_prob)
        true = self.true_ranges(arena, pose)[0]
        noise = rng.standard_normal(len(true))
        lost = rng.random(len(true)) < dropout_prob
        readings = np.clip(true + noise_std * noise, self.min_range, self.max_range)
        return np.where(lost, self.max_range, readings), true


def hc_sr04() -> RangeSensor:
    """Five HC-SR04 ultrasonic sonars facing 0, +-30 and +-60 degrees.

    From the ElecFreaks HC-SR04 datasheet
    (https://cdn.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf): range 2 cm to 4 m, and a
    measuring angle of 15 degrees, modelled here as the full cone (+-7.5 degrees; the datasheet
    does not say whether 15 degrees is the full or the half angle), cast as 5 rays. The sonars sit
    on the perimeter of the default 0.1 m robot (``mount_radius``), so readings are measured from
    the robot's edge, as the real sensors measure.

    The noise (0.01 m) and dropout probability (0.02) are not in the datasheet: they are
    placeholders until the sensors are calibrated on the robot (Phase 8, step 5).
    """
    return RangeSensor(
        mount_angles=np.radians([-60.0, -30.0, 0.0, 30.0, 60.0]),
        cone_width=math.radians(15.0),
        rays_per_cone=5,
        min_range=0.02,
        max_range=4.0,
        noise_std=0.01,  # placeholder until hardware calibration
        dropout_prob=0.02,  # placeholder until hardware calibration
        mount_radius=0.1,  # the default robot radius (World, LayoutParams)
    )


def lidar(
    num_beams: int = 64,
    field_of_view: float = 2.0 * math.pi,
    min_range: float = 0.05,
    max_range: float = 4.0,
    noise_std: float = 0.01,
    dropout_prob: float = 0.0,
) -> RangeSensor:
    """A scanning lidar of ``num_beams`` single-ray beams across ``field_of_view``.

    A full circle spaces the beams ``2 pi / num_beams`` apart starting straight ahead; a narrower
    field of view spans it end to end, centred on the heading (a single beam points straight
    ahead). The lidar sits at the robot's centre. The figures are generic placeholders, not those
    of a particular lidar.

    Args:
        num_beams: Number of beams.
        field_of_view: Angular span in radians, at most ``2 pi``.
        min_range: Nearest reported distance in metres.
        max_range: Farthest reported distance in metres.
        noise_std: Default range noise standard deviation in metres.
        dropout_prob: Default dropout probability.

    Raises:
        ValueError: ``num_beams`` below 1.
    """
    if num_beams < 1:
        raise ValueError(f"lidar needs num_beams of at least 1, got {num_beams}.")
    if num_beams == 1:
        mounts = np.zeros(1)
    elif field_of_view >= 2.0 * math.pi:
        mounts = wrap_angle(2.0 * math.pi * np.arange(num_beams) / num_beams)
    else:
        mounts = np.linspace(-field_of_view / 2.0, field_of_view / 2.0, num_beams)
    return RangeSensor(
        mount_angles=mounts,
        min_range=min_range,
        max_range=max_range,
        noise_std=noise_std,
        dropout_prob=dropout_prob,
    )
