"""A 2D simulator for a disc robot with range sensors, in NumPy and the standard library only.

It imports neither Gymnasium nor ``rl_core``: the Gymnasium environment is a layer over it, and
the robot's own code (state estimation, mapping, planning) can use it without either.
"""

from rl_navigation.sim.geometry import Arena, cast_rays, clearance, occupancy_grid, segment_hits
from rl_navigation.sim.kinematics import unicycle_step, wrap_angle
from rl_navigation.sim.layouts import Layout, LayoutParams, generate_layout, path_exists
from rl_navigation.sim.sensors import RangeSensor, hc_sr04, lidar
from rl_navigation.sim.world import Motion, World

__all__ = [
    "Arena",
    "Layout",
    "LayoutParams",
    "Motion",
    "RangeSensor",
    "World",
    "cast_rays",
    "clearance",
    "generate_layout",
    "hc_sr04",
    "lidar",
    "occupancy_grid",
    "path_exists",
    "segment_hits",
    "unicycle_step",
    "wrap_angle",
]
