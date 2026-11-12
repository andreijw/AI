"""Unicycle kinematics: a robot that drives forward at speed v and turns at rate omega."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def wrap_angle(angle: ArrayLike) -> NDArray[np.float64]:
    """Wrap angles in radians to the half-open interval ``[-pi, pi)``, elementwise.

    Args:
        angle: An angle or an array of angles.

    Returns:
        The wrapped angles, as a float64 array (0-d for a scalar input).
    """
    wrapped = np.mod(np.asarray(angle, dtype=np.float64) + math.pi, 2.0 * math.pi) - math.pi
    # A remainder that rounds up to 2 pi would give +pi.
    return np.where(wrapped >= math.pi, wrapped - 2.0 * math.pi, wrapped)


def unicycle_step(
    pose: ArrayLike, v: ArrayLike, omega: ArrayLike, dt: float
) -> NDArray[np.float64]:
    """Integrate the unicycle model exactly over one step of constant ``v`` and ``omega``.

    The robot moves on a circular arc whose chord has length ``v dt sinc(omega dt / 2)`` and points
    along the mid-step heading ``theta + omega dt / 2``:
    ``x' = x + v dt s cos(theta + omega dt / 2)``, ``y' = y + v dt s sin(theta + omega dt / 2)``,
    ``theta' = theta + omega dt``, with ``s = sin(omega dt / 2) / (omega dt / 2)``. This equals the
    ``v / omega`` form of the arc and, unlike it, stays exact as ``omega`` tends to 0 (a straight
    move), with no threshold.

    Args:
        pose: ``(..., 3)`` poses ``(x, y, theta)``; ``(3,)`` for one robot, ``(P, 3)`` for particles.
        v: Forward speeds in m/s, broadcastable to ``pose[..., 0]``.
        omega: Turn rates in rad/s, counter-clockwise positive, broadcastable likewise.
        dt: Step length in seconds.

    Returns:
        New poses of the same shape, headings wrapped to ``[-pi, pi)``.
    """
    pose = np.asarray(pose, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    half_turn = omega * dt / 2.0
    chord = v * dt * np.sinc(half_turn / math.pi)  # np.sinc(x) = sin(pi x) / (pi x)
    mid = pose[..., 2] + half_turn
    moved = np.empty(np.broadcast_shapes(pose.shape[:-1], chord.shape) + (3,))
    moved[..., 0] = pose[..., 0] + chord * np.cos(mid)
    moved[..., 1] = pose[..., 1] + chord * np.sin(mid)
    moved[..., 2] = wrap_angle(pose[..., 2] + omega * dt)
    return moved
