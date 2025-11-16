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

from pydoll.constants import MouseMovement

# Module-level constant (aliased from constants.py for internal use)
_EASING_MIDPOINT = MouseMovement.EASING_MIDPOINT


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


def _bernstein_polynomial(
    points: list[tuple[float, float]],
) -> Callable[[float], tuple[float, float]]:
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


def _ease_out_quad(t: float) -> float:
    """Quadratic easing out - starts fast and decelerates."""
    return -t * (t - 2)


def _ease_out_cubic(t: float) -> float:
    """Cubic easing out - stronger deceleration."""
    t -= 1
    return t * t * t + 1


def _ease_out_quart(t: float) -> float:
    """Quartic easing out - very strong deceleration."""
    t -= 1
    return -(t * t * t * t - 1)


def _ease_out_quint(t: float) -> float:
    """Quintic easing out - extremely strong deceleration."""
    t -= 1
    return t * t * t * t * t + 1


def _ease_out_sine(t: float) -> float:
    """Sinusoidal easing out - smooth deceleration."""
    return math.sin(t * math.pi / 2)


def _ease_out_expo(t: float) -> float:
    """Exponential easing out - rapid initial movement, slow end."""
    return 1 - math.pow(2, -10 * t) if t != 1 else 1


def _ease_out_circ(t: float) -> float:
    """Circular easing out - decelerating circular curve."""
    t -= 1
    return math.sqrt(1 - t * t)


def _ease_in_out_cubic(t: float) -> float:
    """Cubic easing in/out - stronger acceleration/deceleration."""
    if t < _EASING_MIDPOINT:
        return 4 * t * t * t
    t = 2 * t - 2
    return (t * t * t + 2) / 2


def _ease_in_out_quart(t: float) -> float:
    """Quartic easing in/out - very strong acceleration/deceleration."""
    if t < _EASING_MIDPOINT:
        return 8 * t * t * t * t
    t -= 1
    return -(8 * t * t * t * t - 1)


def _ease_in_out_quint(t: float) -> float:
    """Quintic easing in/out - extremely strong acceleration/deceleration."""
    if t < _EASING_MIDPOINT:
        return 16 * t * t * t * t * t
    t = 2 * t - 2
    return (t * t * t * t * t + 2) / 2


def _ease_in_out_sine(t: float) -> float:
    """Sinusoidal easing in/out - smooth acceleration/deceleration."""
    return -(math.cos(math.pi * t) - 1) / 2


def _ease_in_out_expo(t: float) -> float:
    """Exponential easing in/out - exponential acceleration/deceleration."""
    if t == 0:
        return 0
    if t == 1:
        return 1
    if t < _EASING_MIDPOINT:
        return math.pow(2, 20 * t - 10) / 2
    return (2 - math.pow(2, -20 * t + 10)) / 2


def _ease_in_out_circ(t: float) -> float:
    """Circular easing in/out - circular acceleration/deceleration."""
    if t < _EASING_MIDPOINT:
        return (1 - math.sqrt(1 - 4 * t * t)) / 2
    t = -2 * t + 2
    return (math.sqrt(1 - t * t) + 1) / 2


def _linear(t: float) -> float:
    """Linear easing - constant speed."""
    return t


def _get_random_easing_function() -> Callable[[float], float]:
    """Select random easing function with weighted probabilities."""
    # Favor ease-out and linear for smoother starts, avoid aggressive ease-in-out
    easing_functions = [
        _linear,
        _linear,  # More weight to linear
        _ease_out_sine,
        _ease_out_quad,
        _ease_out_cubic,
        _ease_in_out_sine,  # Gentlest ease-in-out
        _ease_in_out_cubic,  # Moderate ease-in-out
    ]
    return random.choice(easing_functions)


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
    easing_func: Callable[[float], float] | None = None,
) -> list[tuple[float, float]]:
    """
    Apply easing function to redistribute points along curve.

    Modifies point distribution to create acceleration/deceleration
    effects mimicking natural human movement.

    Args:
        points: Original curve points.
        target_points: Desired number of output points.
        easing_func: Easing function that maps [0, 1] to [0, 1].
                    If None, randomly selects from available functions.

    Returns:
        Redistributed points with easing applied.
    """
    target_points = max(target_points, 2)

    if easing_func is None:
        easing_func = _get_random_easing_function()

    result: list[tuple[float, float]] = []
    for i in range(target_points):
        progress = i / (target_points - 1)
        eased_progress = easing_func(progress)
        index = int(eased_progress * (len(points) - 1))
        result.append(points[index])

    return result


def _get_random_knots_count() -> int:
    """Get random knots count with weighted probability distribution."""
    knots_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights = [0.15, 0.36, 0.17, 0.12, 0.08, 0.04, 0.03, 0.02, 0.015, 0.005]
    return random.choices(knots_options, weights=weights)[0]


def _get_random_target_points() -> int:
    """Get random target points count with weighted probability distribution."""
    ranges = [range(35, 45), range(45, 60), range(60, 80)]
    weights = [0.53, 0.32, 0.15]
    selected_range = random.choices(ranges, weights=weights)[0]
    return random.choice(selected_range)


def _get_random_offset_boundary() -> tuple[float, float]:
    """Get random offset boundaries with weighted probability distribution."""
    ranges = [range(5, 15), range(15, 25), range(25, 35)]
    weights = [0.2, 0.65, 0.15]
    x_range = random.choices(ranges, weights=weights)[0]
    y_range = random.choices(ranges, weights=weights)[0]
    return float(random.choice(x_range)), float(random.choice(y_range))


def _get_random_distortion_params() -> tuple[float, float, float]:
    """Get random distortion parameters."""
    mean = random.randint(0, 20) / 100.0  # 0-0.2 pixels (very subtle)
    stdev = random.randint(20, 40) / 100.0  # 0.2-0.4 pixels variation
    frequency = random.randint(10, 25) / 100.0  # Only 10-25% of points
    return mean, stdev, frequency


def generate_human_mouse_trajectory(
    from_point: tuple[float, float],
    to_point: tuple[float, float],
    knots_count: int | None = None,
    distortion_mean: float | None = None,
    distortion_stdev: float | None = None,
    distortion_frequency: float | None = None,
    target_points: int | None = None,
    offset_boundary_x: float | None = None,
    offset_boundary_y: float | None = None,
) -> list[tuple[float, float]]:
    """
    Generate a complete human-like mouse movement trajectory.

    Creates a bezier curve path from start to end point with randomized
    control points, distortion, and easing to simulate natural cursor movement.
    All parameters are optional - if not provided, realistic random values
    are generated following HumanCursor best practices.

    Args:
        from_point: Starting coordinates (x, y).
        to_point: Ending coordinates (x, y).
        knots_count: Number of internal control points. If None, randomly selected (1-10, weighted).
        distortion_mean: Mean offset for random distortion. If None, random (0-0.2).
        distortion_stdev: Standard deviation for distortion. If None, random (0.2-0.4).
        distortion_frequency: Probability of distortion (0-1). If None, random (0.1-0.25).
        target_points: Number of points in final trajectory. If None, random (35-80, weighted).
        offset_boundary_x: Horizontal boundary for control points.
                          If None, random (10-55, weighted).
        offset_boundary_y: Vertical boundary for control points.
                          If None, random (10-55, weighted).

    Returns:
        List of (x, y) coordinates forming the complete trajectory.
    """
    # Use random parameters if not specified (HumanCursor approach)
    if knots_count is None:
        knots_count = _get_random_knots_count()
    if target_points is None:
        target_points = _get_random_target_points()
    if offset_boundary_x is None or offset_boundary_y is None:
        offset_boundary_x, offset_boundary_y = _get_random_offset_boundary()
    if distortion_mean is None or distortion_stdev is None or distortion_frequency is None:
        distortion_mean, distortion_stdev, distortion_frequency = _get_random_distortion_params()

    internal_knots = generate_internal_knots(
        from_point, to_point, knots_count, offset_boundary_x, offset_boundary_y
    )

    points = generate_bezier_curve_points(from_point, to_point, internal_knots)

    points = apply_distortion(points, distortion_mean, distortion_stdev, distortion_frequency)

    points = apply_easing(points, target_points, easing_func=None)  # Random easing

    return points
