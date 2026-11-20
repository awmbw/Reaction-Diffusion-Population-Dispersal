"""Closed-form results for the Fisher-KPP equation used to check the simulations."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


def fisher_speed(r, D):
    """Minimum travelling wave speed v* = 2 sqrt(r D)."""
    return 2.0 * np.sqrt(r * D)


def critical_decay_rate(r, D):
    """Leading-edge decay rate selected by the minimum speed wave."""
    return np.sqrt(r / D)


def dispersion_speed(lam, r, D):
    """Speed of a leading edge exp(-lam z): v(lam) = D lam + r / lam."""
    return D * lam + r / lam


def selected_speed(lam, r, D):
    """Asymptotic speed from initial data that decays like exp(-lam |x|)."""
    lam = np.asarray(lam, dtype=float)
    lam_star = critical_decay_rate(r, D)
    return np.where(lam < lam_star, dispersion_speed(lam, r, D), fisher_speed(r, D))


def bramson_front(t, r, D, x0=0.0):
    """Front position with the logarithmic delay for compact initial data."""
    return x0 + fisher_speed(r, D) * t - 1.5 * np.sqrt(D / r) * np.log(t)


def bramson_mean_speed(t1, t2, r, D):
    """Average of the Bramson front speed over the window [t1, t2]."""
    delay = 1.5 * np.sqrt(D / r) * np.log(t2 / t1)
    return fisher_speed(r, D) - delay / (t2 - t1)


def discrete_speed(r, D, dx, dt):
    """Minimum front speed of the FTCS scheme itself.

    Substituting exp(-lam (x - v t)) into the scheme linearised about u = 0 gives
    v(lam) = ln(1 + r dt + 2 alpha (cosh(lam dx) - 1)) / (lam dt), with
    alpha = D dt / dx^2. The scheme selects the minimum of this curve.
    """
    alpha = D * dt / dx**2

    def speed(lam):
        return np.log(1.0 + r * dt + 2.0 * alpha * (np.cosh(lam * dx) - 1.0)) / (lam * dt)

    lam_star = critical_decay_rate(r, D)
    best = minimize_scalar(speed, bounds=(0.2 * lam_star, 5.0 * lam_star),
                           method="bounded", options={"xatol": 1e-10})
    return float(best.fun)


def exact_wave(x, t, r, D, K=1.0, C=1.0):
    """Ablowitz-Zeppetella exact travelling wave, speed 5 sqrt(r D / 6)."""
    speed = 5.0 * np.sqrt(r * D / 6.0)
    return K / (1.0 + C * np.exp(np.sqrt(r / (6.0 * D)) * (x - speed * t)))**2


def gaussian_diffusion(x, t, D, mass, centre, sigma0):
    """Pure diffusion of a Gaussian: the variance grows as sigma0^2 + 2 D t."""
    variance = sigma0**2 + 2.0 * D * t
    return mass * np.exp(-(x - centre)**2 / (2.0 * variance)) / np.sqrt(2.0 * np.pi * variance)


def origin_eigenvalues(v, r, D):
    """Eigenvalues of the travelling wave ODE linearised at (U, U') = (0, 0)."""
    return np.roots([D, v, r])


def saddle_eigenvalues(v, r, D):
    """Eigenvalues of the travelling wave ODE linearised at (U, U') = (K, 0)."""
    return np.roots([D, v, -r])


def wave_orbit(v, r, D, K=1.0, z_end=80.0, eps=1e-6, n=4000):
    """Integrate D U'' + v U' + r U (1 - U/K) = 0 from the saddle at U = K.

    Returns z, U and V = U' along the unstable manifold of the saddle, which is the
    travelling wave profile whenever U stays non-negative.
    """
    mu = saddle_eigenvalues(v, r, D).max()

    def rhs(_z, y):
        return [y[1], -(v * y[1] + r * y[0] * (1.0 - y[0] / K)) / D]

    z = np.linspace(0.0, z_end, n)
    sol = solve_ivp(rhs, (0.0, z_end), [K - eps, -eps * mu], t_eval=z,
                    rtol=1e-10, atol=1e-13)
    return sol.t, sol.y[0], sol.y[1]


def critical_patch(r, D):
    """Smallest domain with lethal edges (u = 0) that supports a population."""
    return np.pi * np.sqrt(D / r)


def patch_growth_rate(length, r, D):
    """Growth rate of a small population on [0, length] with u = 0 at both ends."""
    return r - D * (np.pi / length)**2
