"""Poses shared by the filters: their difference with the heading wrapped, and the checks on a
filter's initial Gaussian and on an encoder reading."""

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.sim.geometry import FloatArray
from rl_navigation.sim.kinematics import wrap_angle


def pose_difference(poses: ArrayLike, reference: ArrayLike) -> FloatArray:
    """``poses - reference`` for ``(..., 3)`` poses, the heading difference wrapped to
    ``[-pi, pi)``: the error between two poses either side of +-pi is small, as it should be."""
    difference = np.asarray(poses, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    difference[..., 2] = wrap_angle(difference[..., 2])
    return difference


def checked_start(
    pose: ArrayLike, covariance: ArrayLike, owner: str
) -> tuple[FloatArray, FloatArray]:
    """A filter's initial Gaussian, validated: new float64 arrays, the heading wrapped to
    ``[-pi, pi)`` (an in-range heading is kept exactly as given).

    Raises:
        ValueError: A pose that is not 3 finite values, or a covariance that is not a finite,
            symmetric, positive-definite 3 x 3 matrix; the message starts with ``owner``.
    """
    pose = np.array(pose, dtype=np.float64)
    covariance = np.array(covariance, dtype=np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError(f"{owner} pose must be 3 finite values, got {pose.tolist()}.")
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError(
            f"{owner} covariance must be a finite 3 x 3 matrix, got shape {covariance.shape}."
        )
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0.0):
        raise ValueError(f"{owner} covariance must be symmetric.")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{owner} covariance must be positive-definite.") from error
    if not -np.pi <= pose[2] < np.pi:
        pose[2] = wrap_angle(pose[2])
    return pose, covariance


def checked_wheel_travel(wheel_travel: ArrayLike, owner: str) -> FloatArray:
    """``(2,)`` float64 ``(left, right)`` travel, per the encoders' reading contract.

    Raises:
        ValueError: Not two finite numbers; the message starts with ``owner``.
    """
    travel = np.asarray(wheel_travel, dtype=np.float64)
    if travel.shape != (2,) or not np.isfinite(travel).all():
        raise ValueError(
            f"{owner} needs wheel_travel of two finite values (left, right) in metres, got "
            f"{travel.tolist()}."
        )
    return travel


def checked_axis_values(values: ArrayLike, owner: str, name: str) -> FloatArray:
    """``(3,)`` float64 values, one per pose axis ``(x, y, theta)``, each finite and positive.

    Raises:
        ValueError: Not 3 finite positive values; the message starts with ``owner`` and names
            ``name``.
    """
    checked = np.array(values, dtype=np.float64)
    if checked.shape != (3,) or not (np.isfinite(checked).all() and (checked > 0).all()):
        raise ValueError(f"{owner} {name} must be 3 positive values (x, y, theta), got {values!r}.")
    return checked
