"""Numerical experiments. Each function returns a plain dictionary of results."""
import time

import numpy as np
from scipy.integrate import solve_bvp

from . import analysis, solver, theory
from .solver import Params

# Baseline parameters. All quantities are in scaled units: length in units of
# sqrt(D/r), time in units of 1/r and density in units of K, so D = r = K = 1.
BASE = Params(D=1.0, r=1.0, K=1.0)
DX = 0.2
DT = 0.0025
DROP_AMPLITUDE = 0.5
DROP_WIDTH = 1.0
FRONT_LEVEL = 0.5


def drop(x, centre, amplitude=DROP_AMPLITUDE, width=DROP_WIDTH):
    """Localized Gaussian drop of population."""
    return amplitude * np.exp(-(x - centre)**2 / (2.0 * width**2))


def _grid(length, dx):
    return solver.make_grid(length, int(round(length / dx)) + 1)


def baseline():
    """The main experiment, run with the nested-loop reference solver."""
    x, dx = _grid(200.0, DX)
    centre, t_end = 100.0, 40.0
    u0 = drop(x, centre)
    save = int(round(0.5 / DT))

    start = time.perf_counter()
    sol = solver.solve_loops(BASE, x, u0, DT, t_end, save_every=save)
    loop_seconds = time.perf_counter() - start

    start = time.perf_counter()
    check = solver.solve_vector(BASE, x, u0, DT, t_end, save_every=save)
    vector_seconds = time.perf_counter() - start

    right = analysis.front_track(sol, FRONT_LEVEL, centre) - centre
    left = analysis.left_front_track(sol, FRONT_LEVEL, centre)
    window = (20.0, 40.0)
    fit_right = analysis.fit_line(sol.t, right, *window)
    fit_left = analysis.fit_line(sol.t, left, *window)
    levels = {level: analysis.fit_line(
        sol.t, analysis.front_track(sol, level, centre) - centre, *window)
        for level in (0.1, 0.5, 0.9)}

    theory_speed = theory.fisher_speed(BASE.r, BASE.D)
    grid_speed = theory.discrete_speed(BASE.r, BASE.D, dx, DT)
    delay = theory_speed - theory.bramson_mean_speed(*window, BASE.r, BASE.D)
    mass = analysis.total_mass(x, sol.u)
    mass_fit = analysis.fit_line(sol.t, mass, *window)
    return {
        "sol": sol, "centre": centre, "right": right, "left": left, "window": window,
        "fit_right": fit_right, "fit_left": fit_left, "levels": levels,
        "theory_speed": theory_speed, "grid_speed": grid_speed, "delay": delay,
        "predicted": grid_speed - delay, "mass": mass, "mass_fit": mass_fit,
        "engine_difference": float(np.max(np.abs(sol.u - check.u))),
        "loop_seconds": loop_seconds, "vector_seconds": vector_seconds,
        "alpha": BASE.D * DT / dx**2, "dt_limit": solver.stable_dt(BASE.D, BASE.r, dx),
        "updates": (len(x) - 1) * int(round(t_end / DT)),
    }


def long_run():
    """A long run that separates the true speed from the logarithmic delay."""
    t_end = 160.0
    length = 2.0 * (2.0 * t_end + 80.0)  # keeps the leading edge away from the ends
    x, dx = _grid(length, DX)
    centre = length / 2.0
    sol = solver.solve_vector(BASE, x, drop(x, centre), DT, t_end,
                              save_every=int(round(0.5 / DT)))
    front = analysis.front_track(sol, FRONT_LEVEL, centre) - centre
    speed, delay_coefficient, offset = analysis.fit_log_delay(sol.t, front, 20.0)

    windows = [(10.0, 20.0), (20.0, 40.0), (40.0, 80.0), (80.0, 160.0)]
    grid_speed = theory.discrete_speed(BASE.r, BASE.D, dx, DT)
    table = []
    for t1, t2 in windows:
        delay = theory.fisher_speed(BASE.r, BASE.D) - theory.bramson_mean_speed(
            t1, t2, BASE.r, BASE.D)
        measured = analysis.fit_line(sol.t, front, t1, t2).slope
        table.append({"t1": t1, "t2": t2, "measured": measured, "delay": delay,
                      "predicted": grid_speed - delay})

    # Late-time profiles written in the frame moving with the front.
    times = [80.0, 120.0, 160.0]
    profiles = []
    for t_target in times:
        j = int(np.argmin(np.abs(sol.t - t_target)))
        profiles.append((sol.t[j], x - centre - front[j], sol.u[j]))

    # A front with speed exactly v* has the leading edge (A z + B) exp(-z sqrt(r/D)).
    z, u_final = profiles[-1][1], profiles[-1][2]
    edge = (z > 8.0) & (z < 30.0)
    decay = -np.polyfit(z[edge], np.log(u_final[edge] / z[edge]), 1)[0]

    orbit_z, orbit_u, _ = theory.wave_orbit(theory.fisher_speed(BASE.r, BASE.D), BASE.r, BASE.D)
    half = int(np.argmin(np.abs(orbit_u - FRONT_LEVEL)))
    orbit_z = orbit_z - orbit_z[half]
    near = (z > -10.0) & (z < 8.0)
    profile_gap = float(np.max(np.abs(u_final[near] - np.interp(z[near], orbit_z, orbit_u))))
    return {
        "t": sol.t, "front": front, "speed": speed, "delay_coefficient": delay_coefficient,
        "offset": offset, "table": table, "grid_speed": grid_speed, "profiles": profiles,
        "edge_decay": float(decay), "orbit_z": orbit_z, "orbit_u": orbit_u,
        "profile_gap": profile_gap, "t_end": t_end,
    }


def parameter_sweep():
    """Measure the front speed over a grid of (r, D) values on one fixed mesh."""
    rows = []
    dx = 0.15
    for r in (0.25, 0.5, 1.0, 2.0):
        for D in (0.25, 0.5, 1.0, 2.0):
            speed = theory.fisher_speed(r, D)
            t_end = 40.0 / r
            dt = min(0.1 * dx**2 / D, 0.02 / r)
            length = 2.0 * (speed * t_end + 30.0)
            x, _ = _grid(length, dx)
            centre = length / 2.0
            params = Params(D=D, r=r, K=1.0)
            sol = solver.solve_vector(params, x, drop(x, centre), dt, t_end,
                                      save_every=max(1, int(round(0.01 * t_end / dt))))
            front = analysis.front_track(sol, FRONT_LEVEL, centre) - centre
            t1, t2 = t_end / 2.0, t_end
            measured = analysis.fit_line(sol.t, front, t1, t2).slope
            delay = speed - theory.bramson_mean_speed(t1, t2, r, D)
            rows.append({"r": r, "D": D, "theory": speed, "measured": measured,
                         "corrected": measured + delay,
                         "discrete": theory.discrete_speed(r, D, x[1] - x[0], dt)})
    theories = np.array([row["theory"] for row in rows])
    corrected = np.array([row["corrected"] for row in rows])
    # Power-law check v = C (r D)^p, where the theory gives C = 2 and p = 1/2.
    products = np.array([row["r"] * row["D"] for row in rows])
    exponent, log_c = np.polyfit(np.log(products), np.log(corrected), 1)
    return {"rows": rows, "exponent": float(exponent), "prefactor": float(np.exp(log_c)),
            "max_error": float(np.max(np.abs(corrected / theories - 1.0)))}


def discretization_study():
    """Compare measured speeds with the exact speed of the discrete scheme."""
    t_end = 60.0
    window = (30.0, 60.0)
    length = 2.0 * (2.0 * t_end + 30.0)
    delay = 2.0 - theory.bramson_mean_speed(*window, 1.0, 1.0)

    def run(dx, dt):
        x, dx = _grid(length, dx)
        centre = length / 2.0
        sol = solver.solve_vector(BASE, x, drop(x, centre), dt, t_end,
                                  save_every=max(1, int(round(0.25 / dt))))
        front = analysis.front_track(sol, FRONT_LEVEL, centre) - centre
        measured = analysis.fit_line(sol.t, front, *window).slope
        return {"dx": float(dx), "dt": dt, "corrected": measured + delay,
                "discrete": theory.discrete_speed(1.0, 1.0, dx, dt)}

    by_dx = [run(dx, 0.25 * dx**2) for dx in (0.8, 0.4, 0.2, 0.1)]
    by_dt = [run(0.2, dt) for dt in (0.0015, 0.0025, 0.005, 0.01, 0.015, 0.018)]
    return {"by_dx": by_dx, "by_dt": by_dt, "delay": delay}


def convergence():
    """Measure the order of accuracy against the exact Ablowitz-Zeppetella wave."""
    t_end = 5.0
    domain = (-20.0, 50.0)

    def run(dx, dt):
        nx = int(round((domain[1] - domain[0]) / dx)) + 1
        x = np.linspace(*domain, nx)
        return x, solver.solve_vector(
            BASE, x, theory.exact_wave(x, 0.0, 1.0, 1.0), dt, t_end,
            save_every=int(round(t_end / dt)), bc="dirichlet",
            left=lambda t: theory.exact_wave(x[0], t, 1.0, 1.0),
            right=lambda t: theory.exact_wave(x[-1], t, 1.0, 1.0)).u[-1]

    space = []
    for dx in (0.5, 0.25, 0.125, 0.0625):
        dt = 0.25 * dx**2
        dt = t_end / round(t_end / dt)
        x, numeric = run(dx, dt)
        l2, max_norm = analysis.error_norms(numeric, theory.exact_wave(x, t_end, 1.0, 1.0),
                                            x[1] - x[0])
        space.append({"dx": dx, "dt": dt, "l2": l2, "max": max_norm})

    # Time accuracy: compare with a much smaller step on the same mesh, so that the
    # spatial error is common to both runs and cancels.
    dx = 0.1
    base_dt = t_end / 2000.0
    x, reference = run(dx, base_dt / 256.0)
    times = []
    for factor in (8, 4, 2, 1):
        dt = base_dt / factor
        _, numeric = run(dx, dt)
        l2, max_norm = analysis.error_norms(numeric, reference, dx)
        times.append({"dt": dt, "l2": l2, "max": max_norm})
    times.sort(key=lambda row: row["dt"])
    return {
        "space": space, "time": times, "t_end": t_end,
        "space_order": analysis.observed_order([row["dx"] for row in space],
                                               [row["l2"] for row in space]),
        "time_order": analysis.observed_order([row["dt"] for row in times],
                                              [row["l2"] for row in times]),
    }


def stability():
    """Run the scheme on either side of the stability limits."""
    x, dx = _grid(100.0, 0.5)
    t_end = 40.0
    cases = []
    for dt, name in ((0.100, "monotone"), (0.1165, "stable"), (0.125, "unstable")):
        n_steps = int(round(t_end / dt))
        sol = solver.solve_vector(BASE, x, drop(x, 50.0), dt, n_steps * dt, save_every=1,
                                  allow_unstable=True)
        sawtooth = np.max(np.abs(np.diff(sol.u, n=2, axis=1)), axis=1) / 4.0
        cases.append({"dt": dt, "name": name, "alpha": BASE.D * dt / dx**2, "t": sol.t,
                      "sawtooth": sawtooth, "sol": sol})
    return {
        "cases": cases, "dx": dx,
        "monotone_limit": solver.stable_dt(1.0, 1.0, dx),
        "von_neumann_limit": 2.0 / (4.0 / dx**2 + 1.0),
        "diffusion_limit": 0.5 * dx**2,
    }


def pure_diffusion():
    """With r = 0 the model must conserve mass and match the Gaussian solution."""
    length, centre, sigma0, t_end = 200.0, 100.0, 2.0, 40.0
    x, dx = _grid(length, DX)
    u0 = theory.gaussian_diffusion(x, 0.0, BASE.D, 1.0, centre, sigma0)
    params = Params(D=BASE.D, r=0.0, K=1.0)
    sol = solver.solve_vector(params, x, u0, DT, t_end, save_every=int(round(0.5 / DT)))
    mass = analysis.total_mass(x, sol.u)
    variance = analysis.total_mass(x, sol.u * (x - centre)**2) / mass
    exact = theory.gaussian_diffusion(x, t_end, BASE.D, 1.0, centre, sigma0)
    return {
        "sol": sol, "mass": mass, "variance": variance,
        "exact_variance": sigma0**2 + 2.0 * BASE.D * sol.t, "exact_final": exact,
        "centre": centre, "sigma0": sigma0,
        "mass_drift": float(np.max(np.abs(mass / mass[0] - 1.0))),
        "profile_error": float(np.max(np.abs(sol.u[-1] - exact))),
        "variance_slope": analysis.fit_line(sol.t, variance).slope,
    }


def speed_selection():
    """Slowly decaying initial data set the speed by v(lam) = D lam + r / lam."""
    length, centre, t_end = 800.0, 400.0, 40.0
    window = (25.0, 40.0)
    x, _ = _grid(length, DX)
    rows = []
    for lam in (0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0):
        u0 = np.minimum(1.0, np.exp(-lam * (np.abs(x - centre) - 2.0)))
        sol = solver.solve_vector(BASE, x, u0, DT, t_end, save_every=int(round(0.5 / DT)))
        front = analysis.front_track(sol, FRONT_LEVEL, centre) - centre
        rows.append({"lam": lam, "measured": analysis.fit_line(sol.t, front, *window).slope,
                     "theory": float(theory.selected_speed(lam, 1.0, 1.0))})
    return {"rows": rows}


def capacity_profile(x, centre, width, K_centre=1.0, K_edge=0.05):
    """Carrying capacity that is high at the centre and low at the edges."""
    return K_edge + (K_centre - K_edge) * np.exp(-(x - centre)**2 / (2.0 * width**2))


def steady_state(x, K_values, D=1.0, r=1.0):
    """Solve D u'' + r u (1 - u/K(x)) = 0 with no-flux ends as a boundary value problem."""
    def capacity(z):
        return np.interp(z, x, K_values)

    def rhs(z, y):
        return np.vstack([y[1], -r * y[0] * (1.0 - y[0] / capacity(z)) / D])

    def ends(a, b):
        return np.array([a[1], b[1]])

    guess = np.vstack([K_values, np.gradient(K_values, x)])
    result = solve_bvp(rhs, ends, x, guess, tol=1e-6, max_nodes=200000)
    if not result.success:
        raise RuntimeError(result.message)
    return result.sol(x)[0]


def capacity_gradient():
    """Optional extension: a favourable centre with hostile edges."""
    length, centre, width = 100.0, 50.0, 15.0
    x, dx = _grid(length, DX)
    K_values = capacity_profile(x, centre, width)
    params = Params(D=1.0, r=1.0, K=K_values)
    t_end = 150.0
    save = int(round(0.5 / DT))
    sol = solver.solve_vector(params, x, drop(x, centre), DT, t_end, save_every=save)
    uniform = solver.solve_vector(BASE, x, drop(x, centre), DT, 24.0, save_every=save)

    steady = steady_state(x, K_values)
    final = sol.u[-1]
    excess = final > K_values

    level_high, level_low = 0.5, 0.01
    high = analysis.front_track(sol, level_high, centre) - centre
    low = analysis.front_track(sol, level_low, centre) - centre
    uniform_low = analysis.front_track(uniform, level_low, centre) - centre
    stall_theory = width * np.sqrt(2.0 * np.log(0.95 / (level_high - 0.05)))
    reach = sol.t <= 20.0
    speed_low = analysis.fit_line(sol.t[reach], low[reach], 10.0, 20.0).slope
    speed_uniform = analysis.fit_line(uniform.t, uniform_low, 10.0, 20.0).slope

    sweep = []
    for w in (5.0, 10.0, 15.0, 25.0):
        K_w = capacity_profile(x, centre, w)
        u_w = steady_state(x, K_w)
        sweep.append({"width": w, "K": K_w, "u": u_w,
                      "mass": float(analysis.total_mass(x, u_w)),
                      "capacity_mass": float(analysis.total_mass(x, K_w))})
    return {
        "sol": sol, "K": K_values, "centre": centre, "width": width, "steady": steady,
        "pde_bvp_gap": float(np.max(np.abs(final - steady))),
        "high": high, "low": low, "level_high": level_high, "level_low": level_low,
        "stall_measured": float(high[-1]), "stall_theory": float(stall_theory),
        "speed_low": speed_low, "speed_uniform": speed_uniform,
        "peak": float(final.max()), "edge_density": float(final[0]),
        "edge_overshoot": float(final[0] / K_values[0]),
        "mass": float(analysis.total_mass(x, final)),
        "capacity_mass": float(analysis.total_mass(x, K_values)),
        "excess_fraction": float(np.mean(excess)), "sweep": sweep,
    }


def critical_patch():
    """Lethal edges (u = 0): a population persists only above a critical length."""
    critical = theory.critical_patch(1.0, 1.0)
    rows = []
    for fraction in (0.6, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0):
        length = fraction * critical
        x, dx = solver.make_grid(length, 41)
        dt = 0.2 * dx**2
        u0 = 1e-6 * np.sin(np.pi * x / length)
        sol = solver.solve_vector(BASE, x, u0, dt, 8.0, save_every=max(1, int(round(0.2 / dt))),
                                  bc="dirichlet")
        fit = analysis.fit_line(sol.t, np.log(np.max(sol.u, axis=1)), 2.0)

        settle = solver.solve_vector(BASE, x, np.maximum(0.1 * np.sin(np.pi * x / length), 0.0),
                                     dt, 400.0, save_every=int(round(400.0 / dt)),
                                     bc="dirichlet")
        rows.append({"fraction": fraction, "length": length, "growth": fit.slope,
                     "theory": theory.patch_growth_rate(length, 1.0, 1.0),
                     "amplitude": float(settle.u[-1].max())})
    return {"rows": rows, "critical": critical,
            "max_growth_error": float(max(abs(row["growth"] - row["theory"]) for row in rows))}


def run_all(log=print):
    """Run every experiment and return the results keyed by name."""
    steps = [("baseline", baseline), ("long_run", long_run), ("sweep", parameter_sweep),
             ("discretization", discretization_study), ("convergence", convergence),
             ("stability", stability), ("diffusion", pure_diffusion),
             ("selection", speed_selection), ("capacity", capacity_gradient),
             ("patch", critical_patch)]
    results = {}
    for name, function in steps:
        start = time.perf_counter()
        results[name] = function()
        log(f"  {name}: {time.perf_counter() - start:.1f} s")
    return results
