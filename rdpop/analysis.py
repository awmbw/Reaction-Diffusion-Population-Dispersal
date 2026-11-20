"""Measurements taken from simulation output."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LineFit:
    """Least-squares line y = slope t + intercept, with the slope's standard error."""
    slope: float
    intercept: float
    slope_error: float
    r_squared: float


def front_position(x, u, level, centre=None):
    """Outermost position right of centre where u falls through level.

    The crossing is located by linear interpolation between grid nodes. Returns NaN
    when the density does not reach level.
    """
    centre = x[len(x) // 2] if centre is None else centre
    start = int(np.searchsorted(x, centre))
    above = np.nonzero(u[start:] >= level)[0]
    if above.size == 0:
        return np.nan
    i = start + above[-1]
    if i == len(x) - 1:
        return x[-1]
    fraction = (u[i] - level) / (u[i] - u[i + 1])
    return x[i] + fraction * (x[i + 1] - x[i])


def front_track(solution, level, centre=None):
    """Front position at every saved time."""
    return np.array([front_position(solution.x, frame, level, centre)
                     for frame in solution.u])


def left_front_track(solution, level, centre):
    """Distance of the left-moving front from centre at every saved time."""
    x = solution.x
    mirrored_x = (centre - x)[::-1]
    return np.array([front_position(mirrored_x, frame[::-1], level, 0.0)
                     for frame in solution.u])


def fit_line(t, y, t_min=None, t_max=None):
    """Fit a straight line to the valid samples inside [t_min, t_max]."""
    t, y = np.asarray(t), np.asarray(y)
    keep = np.isfinite(y)
    if t_min is not None:
        keep &= t >= t_min
    if t_max is not None:
        keep &= t <= t_max
    t, y = t[keep], y[keep]
    if t.size < 3:
        raise ValueError("at least three samples are needed for a line fit")

    design = np.column_stack([t, np.ones_like(t)])
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - slope * t - intercept
    variance = residual @ residual / (t.size - 2)
    slope_error = np.sqrt(variance / np.sum((t - t.mean())**2))
    total = np.sum((y - y.mean())**2)
    r_squared = 1.0 - (residual @ residual) / total if total > 0 else 1.0
    return LineFit(float(slope), float(intercept), float(slope_error), float(r_squared))


def fit_log_delay(t, y, t_min, t_max=None):
    """Fit y = v t - c ln t + b and return (v, c, b)."""
    t, y = np.asarray(t), np.asarray(y)
    keep = np.isfinite(y) & (t >= t_min)
    if t_max is not None:
        keep &= t <= t_max
    t, y = t[keep], y[keep]
    design = np.column_stack([t, -np.log(t), np.ones_like(t)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return tuple(float(value) for value in coefficients)


def total_mass(x, u):
    """Trapezoidal integral of the density over the domain."""
    return np.trapezoid(u, x, axis=-1)


def error_norms(numeric, exact, dx):
    """Discrete L2 and maximum norms of the error."""
    error = numeric - exact
    return float(np.sqrt(dx * np.sum(error**2))), float(np.max(np.abs(error)))


def observed_order(steps, errors):
    """Slope of log(error) against log(step)."""
    slope, _ = np.polyfit(np.log(steps), np.log(errors), 1)
    return float(slope)
