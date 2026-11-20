"""Checks that the solver, theory and analysis code agree with known results."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdpop import analysis, solver, theory  # noqa: E402
from rdpop.solver import Params  # noqa: E402


def drop(x, centre, amplitude=0.5, width=1.0):
    return amplitude * np.exp(-(x - centre)**2 / (2.0 * width**2))


def test_engines_agree_neumann():
    x, dx = solver.make_grid(40.0, 201)
    params = Params(D=0.7, r=1.3, K=2.0)
    u0 = drop(x, 20.0)
    dt = 0.5 * solver.stable_dt(params.D, params.r, dx)
    a = solver.solve_loops(params, x, u0, dt, 3.0, save_every=50)
    b = solver.solve_vector(params, x, u0, dt, 3.0, save_every=50)
    assert np.max(np.abs(a.u - b.u)) < 1e-12


def test_engines_agree_dirichlet_and_varying_capacity():
    x, dx = solver.make_grid(10.0, 101)
    K = 0.2 + np.exp(-(x - 5.0)**2)
    params = Params(D=1.0, r=1.0, K=K)
    dt = 0.5 * solver.stable_dt(1.0, 1.0, dx, K.min(), K.max())
    a = solver.solve_loops(params, x, drop(x, 5.0), dt, 1.0, bc="dirichlet",
                           left=0.0, right=lambda t: 0.1 * t)
    b = solver.solve_vector(params, x, drop(x, 5.0), dt, 1.0, bc="dirichlet",
                            left=0.0, right=lambda t: 0.1 * t)
    assert np.max(np.abs(a.u - b.u)) < 1e-12


def test_unstable_step_is_refused():
    x, dx = solver.make_grid(10.0, 101)
    with pytest.raises(ValueError, match="stability limit"):
        solver.solve_loops(Params(), x, drop(x, 5.0), 0.51 * dx**2, 1.0)


def test_non_positive_capacity_is_refused():
    x, dx = solver.make_grid(10.0, 11)
    with pytest.raises(ValueError, match="positive"):
        solver.solve_vector(Params(K=0.0), x, drop(x, 5.0), 0.1 * dx**2, 1.0)


def test_unknown_boundary_condition_is_refused():
    x, dx = solver.make_grid(10.0, 11)
    with pytest.raises(ValueError, match="unknown boundary"):
        solver.solve_vector(Params(), x, drop(x, 5.0), 0.1 * dx**2, 1.0, bc="periodic")


def test_step_count_reaches_t_end():
    x, dx = solver.make_grid(10.0, 51)
    sol = solver.solve_vector(Params(), x, drop(x, 5.0), 0.01, 1.0, save_every=10)
    assert sol.t[-1] == pytest.approx(1.0)
    assert sol.u.shape == (11, 51)


@pytest.mark.parametrize("engine", [solver.solve_loops, solver.solve_vector])
def test_final_partial_step_and_snapshot_are_not_lost(engine):
    """A non-integral end time must be reached even off the snapshot cadence."""
    x, _ = solver.make_grid(10.0, 11)
    u0 = np.full_like(x, 0.25)
    sol = engine(Params(D=0.0), x, u0, 0.1, 0.25, save_every=10)
    expected = 0.25
    for step in (0.1, 0.1, 0.05):
        expected += step * expected * (1.0 - expected)
    assert sol.t.tolist() == pytest.approx([0.0, 0.25])
    assert sol.u[-1] == pytest.approx(np.full_like(x, expected))


@pytest.mark.parametrize("engine", [solver.solve_loops, solver.solve_vector])
def test_nonuniform_grid_is_refused(engine):
    x = np.array([0.0, 1.0, 2.2, 3.0])
    with pytest.raises(ValueError, match="uniform"):
        engine(Params(), x, np.full_like(x, 0.25), 0.01, 0.1)


def test_diffusion_conserves_mass_with_no_flux():
    x, dx = solver.make_grid(60.0, 301)
    params = Params(D=1.0, r=0.0)
    sol = solver.solve_vector(params, x, drop(x, 30.0, 1.0, 2.0), 0.4 * dx**2, 20.0,
                              save_every=100)
    mass = analysis.total_mass(x, sol.u)
    assert np.max(np.abs(mass / mass[0] - 1.0)) < 1e-11


def test_diffusion_matches_gaussian():
    x, dx = solver.make_grid(80.0, 801)
    D, sigma0, mass = 0.8, 2.0, 3.0
    dt = 0.2 * dx**2 / D
    u0 = theory.gaussian_diffusion(x, 0.0, D, mass, 40.0, sigma0)
    numeric = solver.solve_vector(Params(D=D, r=0.0), x, u0, dt, 10.0, save_every=1000).u[-1]
    exact = theory.gaussian_diffusion(x, 10.0, D, mass, 40.0, sigma0)
    assert np.max(np.abs(numeric - exact)) < 2e-3


def test_density_stays_between_zero_and_capacity():
    x, dx = solver.make_grid(100.0, 501)
    params = Params(D=1.0, r=1.0, K=1.0)
    dt = solver.stable_dt(params.D, params.r, dx)
    u0 = (np.abs(x - 50.0) < 2.0).astype(float)
    sol = solver.solve_vector(params, x, u0, dt, 20.0, save_every=100)
    assert sol.u.min() >= 0.0
    assert sol.u.max() <= 1.0 + 1e-14


def test_exact_wave_satisfies_the_pde():
    x = np.linspace(-20.0, 20.0, 4001)
    r, D, K = 1.5, 0.6, 2.0
    h, dt = x[1] - x[0], 1e-5
    u = theory.exact_wave(x, 1.0, r, D, K)
    u_t = (theory.exact_wave(x, 1.0 + dt, r, D, K)
           - theory.exact_wave(x, 1.0 - dt, r, D, K)) / (2.0 * dt)
    u_xx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / h**2
    residual = u_t[1:-1] - D * u_xx - r * u[1:-1] * (1.0 - u[1:-1] / K)
    assert np.max(np.abs(residual)) < 1e-5


def test_spatial_convergence_is_second_order():
    errors, steps = [], []
    for nx in (121, 241, 481):
        x = np.linspace(-20.0, 40.0, nx)
        dx = x[1] - x[0]
        dt = 0.25 * dx**2
        n_steps = int(round(3.0 / dt))
        sol = solver.solve_vector(Params(), x, theory.exact_wave(x, 0.0, 1.0, 1.0), dt, 3.0,
                                  save_every=n_steps, bc="dirichlet",
                                  left=lambda t: theory.exact_wave(x[0], t, 1.0, 1.0),
                                  right=lambda t: theory.exact_wave(x[-1], t, 1.0, 1.0))
        exact = theory.exact_wave(x, sol.t[-1], 1.0, 1.0)
        errors.append(analysis.error_norms(sol.u[-1], exact, dx)[0])
        steps.append(dx)
    assert analysis.observed_order(steps, errors) == pytest.approx(2.0, abs=0.1)


def test_front_position_interpolates():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert analysis.front_position(x, u, 0.5, centre=0.0) == pytest.approx(2.5)


def test_front_position_without_crossing_is_nan():
    x = np.linspace(0.0, 4.0, 5)
    assert np.isnan(analysis.front_position(x, np.zeros(5), 0.5))


def test_fit_line_recovers_slope():
    t = np.linspace(0.0, 10.0, 50)
    fit = analysis.fit_line(t, 2.5 * t + 1.0, t_min=2.0)
    assert fit.slope == pytest.approx(2.5)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.r_squared == pytest.approx(1.0)


def test_log_delay_fit_recovers_coefficients():
    t = np.linspace(5.0, 100.0, 200)
    v, c, b = analysis.fit_log_delay(t, 2.0 * t - 1.5 * np.log(t) + 3.0, 5.0)
    assert (v, c, b) == pytest.approx((2.0, 1.5, 3.0))


def test_dispersion_minimum_is_the_fisher_speed():
    lam = np.linspace(0.1, 5.0, 5001)
    speeds = theory.dispersion_speed(lam, 2.0, 0.5)
    assert speeds.min() == pytest.approx(theory.fisher_speed(2.0, 0.5), abs=1e-5)
    assert lam[speeds.argmin()] == pytest.approx(theory.critical_decay_rate(2.0, 0.5), abs=1e-2)


def test_discrete_speed_tends_to_continuum_speed():
    assert theory.discrete_speed(1.0, 1.0, 1e-3, 2.5e-7) == pytest.approx(2.0, abs=1e-5)


def test_origin_is_a_spiral_below_the_minimum_speed():
    assert np.iscomplex(theory.origin_eigenvalues(1.5, 1.0, 1.0)).all()
    assert np.isreal(theory.origin_eigenvalues(2.5, 1.0, 1.0)).all()


def test_wave_orbit_stays_positive_only_at_or_above_minimum_speed():
    _, u_fast, _ = theory.wave_orbit(2.0, 1.0, 1.0, z_end=40.0)
    _, u_slow, _ = theory.wave_orbit(1.0, 1.0, 1.0, z_end=40.0)
    assert u_fast.min() > -1e-6
    assert u_slow.min() < -1e-3


def test_simulated_speed_matches_corrected_theory():
    """The measured speed equals the scheme's discrete speed minus the Bramson delay."""
    x, dx = solver.make_grid(200.0, 1001)
    dt = 0.25 * dx**2
    sol = solver.solve_vector(Params(), x, drop(x, 100.0), dt, 40.0, save_every=50)
    fit = analysis.fit_line(sol.t, analysis.front_track(sol, 0.5, 100.0), 20.0, 40.0)
    delay = 2.0 - theory.bramson_mean_speed(20.0, 40.0, 1.0, 1.0)
    predicted = theory.discrete_speed(1.0, 1.0, dx, dt) - delay
    assert fit.slope == pytest.approx(predicted, abs=0.005)


def test_population_dies_below_the_critical_patch():
    length = 0.8 * theory.critical_patch(1.0, 1.0)
    x, dx = solver.make_grid(length, 41)
    u0 = 0.1 * np.sin(np.pi * x / length)
    sol = solver.solve_vector(Params(), x, u0, 0.2 * dx**2, 60.0, save_every=2000,
                              bc="dirichlet")
    assert sol.u[-1].max() < 1e-3


def test_population_survives_above_the_critical_patch():
    length = 1.5 * theory.critical_patch(1.0, 1.0)
    x, dx = solver.make_grid(length, 61)
    u0 = 0.1 * np.sin(np.pi * x / length)
    sol = solver.solve_vector(Params(), x, u0, 0.2 * dx**2, 60.0, save_every=2000,
                              bc="dirichlet")
    assert sol.u[-1].max() > 0.3
