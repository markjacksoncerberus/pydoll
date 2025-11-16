"""
Mouse movement trajectory generation using bezier curves.

This module implements bezier curve-based mouse movement algorithms to simulate
natural human-like cursor motion. The core algorithm is adapted from HumanCursor
(MIT License, Copyright (c) 2023 Flori Batusha) and modified to use Python stdlib
only, removing numpy and pytweening dependencies.

Reference: https://github.com/riflosnake/HumanCursor
"""

from __future__ import annotations

import math
import random
from typing import Callable


def _binomial(n: int, k: int) -> float:
    """
    Calculate binomial coefficient 'n choose k'.

    Args:
        n: Total number of items.
        k: Number of items to choose.

    Returns:
        Binomial coefficient value.
    """
    return math.factorial(n) / float(math.factorial(k) * math.factorial(n - k))


def _bernstein_polynomial_point(t: float, i: int, n: int) -> float:
    """
    Calculate the i-th component of a Bernstein polynomial of degree n.

    Args:
        t: Parameter value in range [0, 1].
        i: Index of the control point.
        n: Degree of the polynomial (number of control points - 1).

    Returns:
        Bernstein basis polynomial value.
    """
    return _binomial(n, i) * (t**i) * ((1 - t) ** (n - i))


def _bernstein_polynomial(points: list[tuple[float, float]]) -> Callable[[float], tuple[float, float]]:
    """
    Create a function that evaluates a bezier curve at parameter t.

    Given a list of control points, returns a function which, given a parameter
    in [0, 1], returns the corresponding point on the bezier curve.

    Args:
        points: List of control points as (x, y) tuples.

    Returns:
        Function that takes t in [0, 1] and returns (x, y) on the curve.
    """

    def bernstein(t: float) -> tuple[float, float]:
        n = len(points) - 1
        x = y = 0.0
        for i, point in enumerate(points):
            bern = _bernstein_polynomial_point(t, i, n)
            x += point[0] * bern
            y += point[1] * bern
        return x, y

    return bernstein


def _easing_out_quad(t: float) -> float:
    """
    Easing function with quadratic deceleration.

    Starts fast and slows down towards the end, mimicking natural
    human movement deceleration.

    Args:
        t: Progress value in range [0, 1].

    Returns:
        Eased value in range [0, 1].
    """
    return 1 - (1 - t) * (1 - t)


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        x1: X coordinate of first point.
        y1: Y coordinate of first point.
        x2: X coordinate of second point.
        y2: Y coordinate of second point.

    Returns:
        Distance between the two points.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def generate_internal_knots(
    from_point: tuple[float, float],
    to_point: tuple[float, float],
    knots_count: int,
    offset_boundary_x: float,
    offset_boundary_y: float,
) -> list[tuple[float, float]]:
    """
    Generate random internal control points (knots) for bezier curve.

    Control points are randomly placed within boundaries to create
    natural curved paths between start and end points.

    Args:
        from_point: Starting point (x, y).
        to_point: Ending point (x, y).
        knots_count: Number of internal control points to generate.
        offset_boundary_x: Horizontal boundary offset from path endpoints.
        offset_boundary_y: Vertical boundary offset from path endpoints.

    Returns:
        List of randomly generated control points.
    """
    left_boundary = int(min(from_point[0], to_point[0]) - offset_boundary_x)
    right_boundary = int(max(from_point[0], to_point[0]) + offset_boundary_x)
    down_boundary = int(min(from_point[1], to_point[1]) - offset_boundary_y)
    up_boundary = int(max(from_point[1], to_point[1]) + offset_boundary_y)

    if left_boundary >= right_boundary:
        right_boundary = left_boundary + 1
    if down_boundary >= up_boundary:
        up_boundary = down_boundary + 1

    knots: list[tuple[float, float]] = []
    for _ in range(knots_count):
        x = random.randint(left_boundary, right_boundary)
        y = random.randint(down_boundary, up_boundary)
        knots.append((float(x), float(y)))

    return knots


def generate_bezier_curve_points(
    from_point: tuple[float, float],
    to_point: tuple[float, float],
    internal_knots: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Generate points along a bezier curve defined by control points.

    Args:
        from_point: Starting point (x, y).
        to_point: Ending point (x, y).
        internal_knots: Internal control points for curve shape.

    Returns:
        List of points along the bezier curve.
    """
    num_points = max(
        abs(int(from_point[0] - to_point[0])),
        abs(int(from_point[1] - to_point[1])),
        2,
    )

    control_points = [from_point] + internal_knots + [to_point]
    bernstein = _bernstein_polynomial(control_points)

    curve_points: list[tuple[float, float]] = []
    for i in range(num_points):
        t = i / (num_points - 1)
        curve_points.append(bernstein(t))

    return curve_points


def apply_distortion(
    points: list[tuple[float, float]],
    distortion_mean: float,
    distortion_stdev: float,
    distortion_frequency: float,
) -> list[tuple[float, float]]:
    """
    Add random distortion to curve points to simulate natural hand tremor.

    Applies random y-axis deviations to intermediate points based on
    normal distribution and frequency parameters.

    Args:
        points: Original curve points.
        distortion_mean: Mean of normal distribution for distortion.
        distortion_stdev: Standard deviation of normal distribution.
        distortion_frequency: Probability (0-1) of applying distortion to each point.

    Returns:
        Distorted curve points.
    """
    if not (0 <= distortion_frequency <= 1):
        distortion_frequency = 0.5

    distorted: list[tuple[float, float]] = [points[0]]

    for i in range(1, len(points) - 1):
        x, y = points[i]
        if random.random() < distortion_frequency:
            delta = random.gauss(distortion_mean, distortion_stdev)
            y += delta
        distorted.append((x, y))

    distorted.append(points[-1])
    return distorted


def apply_easing(
    points: list[tuple[float, float]],
    target_points: int,
    easing_func: Callable[[float], float] = _easing_out_quad,
) -> list[tuple[float, float]]:
    """
    Apply easing function to redistribute points along curve.

    Modifies point distribution to create acceleration/deceleration
    effects mimicking natural human movement.

    Args:
        points: Original curve points.
        target_points: Desired number of output points.
        easing_func: Easing function that maps [0, 1] to [0, 1].

    Returns:
        Redistributed points with easing applied.
    """
    if target_points < 2:
        target_points = 2

    result: list[tuple[float, float]] = []
    for i in range(target_points):
        progress = i / (target_points - 1)
        eased_progress = easing_func(progress)
        index = int(eased_progress * (len(points) - 1))
        result.append(points[index])

    return result


def generate_human_mouse_trajectory(
    from_point: tuple[float, float],
    to_point: tuple[float, float],
    knots_count: int = 2,
    distortion_mean: float = 1.0,
    distortion_stdev: float = 1.0,
    distortion_frequency: float = 0.5,
    target_points: int = 100,
    offset_boundary_x: float = 80.0,
    offset_boundary_y: float = 80.0,
) -> list[tuple[float, float]]:
    """
    Generate a complete human-like mouse movement trajectory.

    Creates a bezier curve path from start to end point with randomized
    control points, distortion, and easing to simulate natural cursor movement.

    Args:
        from_point: Starting coordinates (x, y).
        to_point: Ending coordinates (x, y).
        knots_count: Number of internal control points for curve complexity.
        distortion_mean: Mean offset for random distortion.
        distortion_stdev: Standard deviation for random distortion.
        distortion_frequency: Probability of applying distortion (0-1).
        target_points: Number of points in final trajectory.
        offset_boundary_x: Horizontal boundary for control point generation.
        offset_boundary_y: Vertical boundary for control point generation.

    Returns:
        List of (x, y) coordinates forming the complete trajectory.
    """
    internal_knots = generate_internal_knots(
        from_point, to_point, knots_count, offset_boundary_x, offset_boundary_y
    )

    points = generate_bezier_curve_points(from_point, to_point, internal_knots)

    points = apply_distortion(points, distortion_mean, distortion_stdev, distortion_frequency)

    points = apply_easing(points, target_points)

    return points
