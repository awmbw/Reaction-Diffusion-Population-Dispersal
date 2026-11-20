"""Explicit finite difference solver for the 1D reaction-diffusion equation.

    du/dt = D d2u/dx2 + r u (1 - u / K(x))

Two engines implement the same FTCS update:

* solve_loops  - the reference solver. The spatial second-derivative loop is
  nested inside the temporal Euler loop, written with plain Python for-loops.
* solve_vector - the same update written with NumPy slices, used for long runs
  and parameter sweeps where the pure Python loops would be too slow.

Boundary conditions are either no-flux ("neumann") or prescribed values
("dirichlet"), where each end value may be a constant or a function of time.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    """Model parameters. K is a number or an array with one value per grid node."""
    D: float = 1.0
    r: float = 1.0
    K: object = 1.0


@dataclass(frozen=True)
class Solution:
    """Saved snapshots: u[j, i] is the density at time t[j] and position x[i]."""
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray


def make_grid(length, nx):
    """Return nx uniformly spaced nodes on [0, length] and the spacing."""
    x = np.linspace(0.0, length, nx)
    return x, x[1] - x[0]


def stable_dt(D, r, dx, K_min=1.0, K_max=1.0):
    """Largest time step that keeps the scheme monotone (see report, Section 4.4).

    For uniform K this is 1 / (2 D / dx^2 + r). A spatially varying K tightens the
    reaction term by the factor (2 K_max / K_min - 1).
    """
    return 1.0 / (2.0 * D / dx**2 + r * (2.0 * K_max / K_min - 1.0))


def _prepare(params, x, u0, dt, t_end, save_every, allow_unstable):
    """Validate the inputs and return the values shared by both engines."""
    x = np.asarray(x, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a one-dimensional grid with at least 3 nodes")
    if u0.shape != x.shape:
        raise ValueError("u0 must have the same shape as x")
    if (not np.isfinite(dt) or not np.isfinite(t_end) or dt <= 0.0 or t_end <= 0.0
            or not isinstance(save_every, (int, np.integer)) or save_every < 1):
        raise ValueError("dt and t_end must be finite and positive; save_every a positive integer")
    if (not np.isfinite(params.D) or not np.isfinite(params.r)
            or params.D < 0.0 or params.r < 0.0):
        raise ValueError("D and r must be finite and non-negative")

    dx = x[1] - x[0]
    if (not np.all(np.isfinite(x)) or dx <= 0.0
            or not np.allclose(np.diff(x), dx, rtol=1e-9, atol=1e-12 * dx)):
        raise ValueError("x must be finite, increasing and uniformly spaced")
    K = np.broadcast_to(np.asarray(params.K, dtype=float), x.shape).copy()
    if not np.all(np.isfinite(K)) or np.any(K <= 0.0):
        raise ValueError("carrying capacity K must be finite and positive")
    if not np.all(np.isfinite(u0)) or np.any(u0 < 0.0):
        raise ValueError("initial density must be finite and non-negative")

    limit = (stable_dt(params.D, params.r, dx, K.min(), K.max())
             if params.D > 0.0 or params.r > 0.0 else np.inf)
    if dt > limit * (1.0 + 1e-12) and not allow_unstable:
        raise ValueError(f"dt={dt:g} exceeds the stability limit {limit:g}")

    # nextafter prevents an extra step when division rounds an integer upward.
    n_steps = max(1, int(np.ceil(np.nextafter(t_end / dt, -np.inf))))
    return x, u0, dx, K, n_steps


def _boundary_value(value, t):
    """Evaluate a Dirichlet value that is either constant or a function of time."""
    return value(t) if callable(value) else value


def solve_loops(params, x, u0, dt, t_end, save_every=1, bc="neumann",
                left=0.0, right=0.0, allow_unstable=False):
    """Reference FTCS solver: spatial loop nested inside the temporal Euler loop."""
    x, u0, dx, K, n_steps = _prepare(params, x, u0, dt, t_end, save_every, allow_unstable)
    if bc not in ("neumann", "dirichlet"):
        raise ValueError(f"unknown boundary condition: {bc}")
    D, r = params.D, params.r
    K = K.tolist()
    nx = x.size
    dx2 = dx * dx

    u = u0.tolist()
    if bc == "dirichlet":
        u[0], u[-1] = _boundary_value(left, 0.0), _boundary_value(right, 0.0)
    new = [0.0] * nx
    times, frames = [0.0], [np.array(u)]

    for n in range(n_steps):                      # temporal Euler loop
        step = dt if n < n_steps - 1 else t_end - n * dt
        for i in range(1, nx - 1):                # spatial second-derivative loop
            u_xx = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / dx2
            growth = r * u[i] * (1.0 - u[i] / K[i])
            new[i] = u[i] + step * (D * u_xx + growth)

        t_next = min((n + 1) * dt, t_end)
        if bc == "neumann":                       # mirror ghost nodes: u[-1] = u[1]
            for i, inner in ((0, 1), (nx - 1, nx - 2)):
                u_xx = 2.0 * (u[inner] - u[i]) / dx2
                growth = r * u[i] * (1.0 - u[i] / K[i])
                new[i] = u[i] + step * (D * u_xx + growth)
        elif bc == "dirichlet":
            new[0] = _boundary_value(left, t_next)
            new[-1] = _boundary_value(right, t_next)
        else:
            raise ValueError(f"unknown boundary condition: {bc}")

        u, new = new, u
        if (n + 1) % save_every == 0 or n == n_steps - 1:
            times.append(t_next)
            frames.append(np.array(u))

    return Solution(np.array(times), x, np.array(frames))


def solve_vector(params, x, u0, dt, t_end, save_every=1, bc="neumann",
                 left=0.0, right=0.0, allow_unstable=False):
    """Array version of solve_loops; produces the same values to round-off."""
    x, u0, dx, K, n_steps = _prepare(params, x, u0, dt, t_end, save_every, allow_unstable)
    if bc not in ("neumann", "dirichlet"):
        raise ValueError(f"unknown boundary condition: {bc}")
    D, r = params.D, params.r
    dx2 = dx * dx

    u = u0.copy()
    if bc == "dirichlet":
        u[0], u[-1] = _boundary_value(left, 0.0), _boundary_value(right, 0.0)
    u_xx = np.zeros_like(u)
    times, frames = [0.0], [u.copy()]

    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(n_steps):
            step = dt if n < n_steps - 1 else t_end - n * dt
            u_xx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx2
            t_next = min((n + 1) * dt, t_end)
            if bc == "neumann":
                u_xx[0] = 2.0 * (u[1] - u[0]) / dx2
                u_xx[-1] = 2.0 * (u[-2] - u[-1]) / dx2
            elif bc != "dirichlet":
                raise ValueError(f"unknown boundary condition: {bc}")

            u = u + step * (D * u_xx + r * u * (1.0 - u / K))
            if bc == "dirichlet":
                u[0] = _boundary_value(left, t_next)
                u[-1] = _boundary_value(right, t_next)

            if (n + 1) % save_every == 0 or n == n_steps - 1:
                times.append(t_next)
                frames.append(u.copy())

    return Solution(np.array(times), x, np.array(frames))
