"""Figures for the report. Every function takes results from experiments.run_all."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from . import theory  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FULL = (6.3, 3.0)
TALL = (6.3, 3.6)
TIME_RAMP = plt.get_cmap("Blues")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def _ramp(n):
    """Light-to-dark steps of one hue, for curves ordered by time."""
    return [TIME_RAMP(value) for value in np.linspace(0.38, 1.0, n)]


def _save(fig, folder, name):
    fig.savefig(folder / f"{name}.pdf", dpi=220)
    fig.savefig(folder / f"{name}.png", dpi=170)
    plt.close(fig)


def _plain_log_ticks(axis, ticks):
    """Use plain numbers on a short log axis and drop the crowded minor labels."""
    axis.set_ticks(ticks)
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def _frame(sol, time):
    return int(np.argmin(np.abs(sol.t - time)))


def fig_reaction(_results, folder):
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    u = np.linspace(0.0, 1.15, 300)
    ax.plot(u, u * (1.0 - u), color=BLUE)
    ax.axhline(0.0, color=MUTED, lw=0.6)
    ax.plot([0.0], [0.0], "o", ms=7, mfc="white", mec=INK)
    ax.plot([1.0], [0.0], "o", ms=7, color=INK)
    ax.annotate("", (0.75, -0.022), (0.25, -0.022), arrowprops=dict(arrowstyle="->", color=INK))
    ax.annotate("", (1.02, -0.022), (1.14, -0.022), arrowprops=dict(arrowstyle="->", color=INK))
    ax.text(0.03, 0.022, "unstable", color=INK)
    ax.text(0.86, 0.022, "stable", color=INK)
    ax.set(xlabel=r"density $u/K$", ylabel=r"growth rate $f(u)/(rK)$")
    _save(fig, folder, "reaction")


def fig_phase_plane(_results, folder):
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    for ax, v, title in ((axes[0], 1.0, r"$v=1<v^*$: spiral, $U$ turns negative"),
                         (axes[1], 2.0, r"$v=2=v^*$: admissible wave")):
        U, V = np.meshgrid(np.linspace(-0.25, 1.1, 21), np.linspace(-0.75, 0.25, 17))
        dU, dV = V, -(v * V + U * (1.0 - U))
        norm = np.hypot(dU, dV) + 1e-9
        ax.quiver(U, V, dU / norm, dV / norm, color="#b9b8b2", scale=34, width=0.0035)
        _, u, up = theory.wave_orbit(v, 1.0, 1.0, z_end=45.0)
        ax.plot(u, up, color=BLUE, label="unstable manifold of $(K,0)$")
        ax.plot([1.0], [0.0], "o", ms=6, color=INK)
        ax.plot([0.0], [0.0], "o", ms=6, mfc="white", mec=INK)
        ax.set(xlim=(-0.25, 1.1), ylim=(-0.75, 0.25), xlabel=r"$U/K$", title=title)
    axes[0].set_ylabel(r"$U'$")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    _save(fig, folder, "phase_plane")


def fig_dispersion(_results, folder):
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    lam = np.linspace(0.2, 4.0, 400)
    ax.plot(lam, theory.dispersion_speed(lam, 1.0, 1.0), color=BLUE)
    ax.plot([1.0], [2.0], "o", ms=7, color=INK)
    ax.annotate(r"minimum at $\lambda^*=\sqrt{r/D}$, $v^*=2\sqrt{rD}$", (1.0, 2.0), (0.75, 4.9),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set(xlabel=r"decay rate $\lambda\ /\ \sqrt{r/D}$", ylabel=r"speed $v\ /\ \sqrt{rD}$",
           ylim=(1.5, 5.5))
    _save(fig, folder, "dispersion")


def fig_snapshots(results, folder):
    sol = results["baseline"]["sol"]
    times = [0, 5, 10, 20, 30, 40]
    fig, ax = plt.subplots(figsize=TALL)
    for time, colour in zip(times, _ramp(len(times))):
        ax.plot(sol.x, sol.u[_frame(sol, time)], color=colour, label=f"$t={time}$")
    ax.set(xlabel=r"position $x$", ylabel=r"density $u$", xlim=(0, 200), ylim=(0, 1.08))
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.4,
              columnspacing=1.1)
    _save(fig, folder, "snapshots")


def _heatmap(ax, sol, fig, label):
    mesh = ax.pcolormesh(sol.x, sol.t, sol.u, cmap="Blues", shading="auto", rasterized=True)
    fig.colorbar(mesh, ax=ax, label=label, pad=0.02)
    ax.grid(False)
    ax.set(xlabel=r"space $x$", ylabel=r"time $t$")


def fig_heatmap(results, folder):
    base = results["baseline"]
    sol = base["sol"]
    fig, ax = plt.subplots(figsize=TALL)
    _heatmap(ax, sol, fig, r"density $u$")
    t = sol.t[sol.t >= 10.0]
    # Lines of slope v* pinned to the measured front at the end of the run.
    offset = base["right"][-1] - 2.0 * sol.t[-1]
    for sign, label in ((1.0, r"slope $v^*=2$"), (-1.0, None)):
        ax.plot(base["centre"] + sign * (2.0 * t + offset), t, color=ORANGE, ls="--", lw=1.1,
                label=label)
    ax.set(xlim=(0, 200), ylim=(0, 40))
    ax.legend(loc="lower right", labelcolor=INK)
    _save(fig, folder, "heatmap")


def fig_front(results, folder):
    base, long = results["baseline"], results["long_run"]
    sol = base["sol"]
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    keep = sol.t >= 2.0
    ax.plot(sol.t[keep], base["right"][keep], color=BLUE, label="right front")
    ax.plot(sol.t[keep], base["left"][keep], color=ORANGE, ls="--", label="left front")
    fit = base["fit_right"]
    ax.plot(sol.t, fit.slope * sol.t + fit.intercept, color=INK, lw=0.8,
            label=f"fit on $[20,40]$: slope {fit.slope:.3f}")
    ax.set(xlabel=r"time $t$", ylabel=r"distance from centre $X(t)$", xlim=(0, 40), ylim=(0, 85))
    ax.legend(loc="upper left")

    ax = axes[1]
    t, front = long["t"], long["front"]
    start = int(np.searchsorted(t, 6.0))
    stride = 8
    t_s, front_s = t[start::stride], front[start::stride]
    ax.plot(0.5 * (t_s[1:] + t_s[:-1]), np.diff(front_s) / np.diff(t_s), color=BLUE,
            label="measured")
    curve = np.linspace(6.0, long["t_end"], 300)
    ax.plot(curve, long["grid_speed"] - 1.5 / curve, color=ORANGE, ls="--",
            label=r"$v_h-3/(2t)$")
    ax.axhline(2.0, color=INK, lw=0.8, label=r"$v^*=2$")
    ax.set(xlabel=r"time $t$", ylabel=r"front speed $dX/dt$", xscale="log")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, folder, "front")


def fig_profile(results, folder):
    long = results["long_run"]
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    for (time, z, u), colour in zip(long["profiles"], _ramp(3)):
        axes[0].plot(z, u, color=colour, lw=2.4, label=f"PDE, $t={time:.0f}$")
        keep = u > 1e-300
        axes[1].plot(z[keep], u[keep], color=colour, lw=1.6, label=f"PDE, $t={time:.0f}$")
    axes[0].plot(long["orbit_z"], long["orbit_u"], color=ORANGE, ls="--", lw=1.1,
                 label="ODE wave, $v=2$")
    axes[0].set(xlim=(-15, 15), xlabel=r"$z=x-X(t)$", ylabel=r"density $u$")
    axes[0].legend(loc="lower left")
    z = np.linspace(1.0, 40.0, 100)
    axes[1].plot(z, 0.5 * np.exp(1.0 - z), color=INK, lw=0.8, ls=":", label=r"slope of $e^{-z}$")
    axes[1].set(yscale="log", xlim=(0, 40), ylim=(1e-18, 2), xlabel=r"$z=x-X(t)$")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    _save(fig, folder, "profile")


def fig_sweep(results, folder):
    rows = results["sweep"]["rows"]
    theories = np.array([row["theory"] for row in rows])
    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    line = np.array([theories.min() * 0.85, theories.max() * 1.15])
    ax.plot(line, line, color=INK, lw=0.8, label=r"$v=2\sqrt{rD}$")
    ax.plot(theories, [row["measured"] for row in rows], "s", ms=4.5, mfc="white", mec=ORANGE,
            label="measured slope")
    ax.plot(theories, [row["corrected"] for row in rows], "o", ms=5, color=BLUE,
            label="slope + Bramson delay")
    ax.set(xscale="log", yscale="log", xlabel=r"theory $2\sqrt{rD}$", ylabel="simulated speed")
    _plain_log_ticks(ax.xaxis, [0.5, 1, 2, 4])
    _plain_log_ticks(ax.yaxis, [0.5, 1, 2, 4])
    ax.legend(loc="upper left")
    _save(fig, folder, "sweep")


def fig_discrete(results, folder):
    study = results["discretization"]
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True)
    panels = ((axes[0], study["by_dx"], "dx", r"$\Delta x$ (with $\alpha=0.25$)"),
              (axes[1], study["by_dt"], "dt", r"$\Delta t$ (with $\Delta x=0.2$)"))
    for ax, rows, key, label in panels:
        steps = [row[key] for row in rows]
        ax.plot(steps, [row["discrete"] for row in rows], color=ORANGE,
                label=r"discrete theory $v_h$")
        ax.plot(steps, [row["corrected"] for row in rows], "o", ms=5, color=BLUE,
                label="simulation + delay")
        ax.axhline(2.0, color=INK, lw=0.8, label=r"$v^*=2$")
        ax.set(xlabel=label, xscale="log")
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("front speed")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    _save(fig, folder, "discrete")


def fig_convergence(results, folder):
    study = results["convergence"]
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    panels = ((axes[0], study["space"], "dx", 2, r"$\Delta x$"),
              (axes[1], study["time"], "dt", 1, r"$\Delta t$"))
    for ax, rows, key, order, label in panels:
        steps = np.array([row[key] for row in rows])
        l2 = np.array([row["l2"] for row in rows])
        ax.plot(steps, l2, "o-", ms=5, color=BLUE, label=r"$L^2$ error")
        ax.plot(steps, [row["max"] for row in rows], "s-", ms=4.5, color=ORANGE,
                label=r"$L^\infty$ error")
        ax.plot(steps, l2[0] * 1.6 * (steps / steps[0])**order, color=INK, lw=0.8, ls=":",
                label=f"order {order}")
        ax.set(xscale="log", yscale="log", xlabel=label)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(loc="lower right")
    axes[0].set_ylabel("error")
    fig.tight_layout()
    _save(fig, folder, "convergence")


def fig_stability(results, folder):
    study = results["stability"]
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    unstable = study["cases"][-1]
    finite = np.nonzero(np.isfinite(unstable["sawtooth"]) & (unstable["sawtooth"] < 0.6))[0][-1]
    time = unstable["t"][finite]
    for case, colour, style in zip(study["cases"], (BLUE, AQUA, ORANGE), ("-", "--", "-")):
        j = int(np.argmin(np.abs(case["t"] - time)))
        axes[0].plot(case["sol"].x, case["sol"].u[j], color=colour, ls=style, lw=1.1,
                     label=rf"$\Delta t={case['dt']:g}$")
        keep = np.isfinite(case["sawtooth"]) & (case["sawtooth"] < 1e6)
        axes[1].plot(case["t"][keep], case["sawtooth"][keep], color=colour, ls=style, lw=1.3,
                     label=rf"$\Delta t={case['dt']:g}$")
    axes[0].set(xlabel=r"position $x$", ylabel=rf"density at $t={time:.1f}$")
    axes[0].set_ylim(-0.05, 1.95)
    axes[0].legend(loc="upper left", ncol=3, handlelength=1.3, columnspacing=0.8, fontsize=7.5)
    axes[1].set(xlabel=r"time $t$", ylabel="sawtooth amplitude", yscale="log", ylim=(1e-14, 1e6))
    fig.tight_layout()
    _save(fig, folder, "stability")


def fig_diffusion(results, folder):
    study = results["diffusion"]
    sol = study["sol"]
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    times = [0, 10, 40]
    for time, colour in zip(times, _ramp(3)):
        j = _frame(sol, time)
        axes[0].plot(sol.x, sol.u[j], color=colour, lw=2.4, label=f"numerical, $t={time}$")
        exact = theory.gaussian_diffusion(sol.x, sol.t[j], 1.0, 1.0, study["centre"],
                                          study["sigma0"])
        axes[0].plot(sol.x, exact, color=ORANGE, ls="--", lw=0.9,
                     label="exact Gaussian" if time == 40 else None)
    axes[0].set(xlim=(50, 150), xlabel=r"position $x$", ylabel=r"density $u$")
    axes[0].legend(loc="upper right")
    axes[1].plot(sol.t, study["variance"], color=BLUE, lw=2.4, label="numerical")
    axes[1].plot(sol.t, study["exact_variance"], color=ORANGE, ls="--", lw=0.9,
                 label=r"$\sigma_0^2+2Dt$")
    axes[1].set(xlabel=r"time $t$", ylabel="variance of the distribution")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    _save(fig, folder, "diffusion")


def fig_mass(results, folder):
    base = results["baseline"]
    sol = base["sol"]
    fit = base["mass_fit"]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.plot(sol.t, base["mass"], color=BLUE, lw=2.2, label=r"total population $M(t)$")
    ax.plot(sol.t, fit.slope * sol.t + fit.intercept, color=ORANGE, ls="--", lw=1.0,
            label=f"fit: slope {fit.slope:.2f}")
    ax.set(xlabel=r"time $t$", ylabel=r"$M(t)$", ylim=(0, None))
    ax.legend(loc="upper left")
    _save(fig, folder, "mass")


def fig_selection(results, folder):
    rows = results["selection"]["rows"]
    lam = np.linspace(0.2, 2.1, 300)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(lam, theory.selected_speed(lam, 1.0, 1.0), color=ORANGE, label="selection rule")
    ax.plot([row["lam"] for row in rows], [row["measured"] for row in rows], "o", ms=5,
            color=BLUE, label="simulation")
    ax.set(xlabel=r"initial decay rate $\lambda$", ylabel="front speed")
    ax.legend(loc="upper right")
    _save(fig, folder, "selection")


def fig_capacity_snapshots(results, folder):
    study = results["capacity"]
    sol = study["sol"]
    times = [0, 5, 10, 20, 50, 150]
    fig, ax = plt.subplots(figsize=TALL)
    for time, colour in zip(times, _ramp(len(times))):
        ax.plot(sol.x, sol.u[_frame(sol, time)], color=colour, label=f"$t={time}$")
    ax.plot(sol.x, study["K"], color=ORANGE, ls="--", lw=1.1, label=r"$K(x)$")
    ax.set(xlabel=r"position $x$", ylabel=r"density $u$", xlim=(0, 100), ylim=(0, 1.1))
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.3,
              columnspacing=0.9)
    _save(fig, folder, "capacity_snapshots")


def fig_capacity_heatmap(results, folder):
    sol = results["capacity"]["sol"]
    fig, ax = plt.subplots(figsize=TALL)
    _heatmap(ax, sol, fig, r"density $u$")
    ax.set(xlim=(0, 100), ylim=(0, 60))
    _save(fig, folder, "capacity_heatmap")


def fig_capacity_front(results, folder):
    study = results["capacity"]
    sol = study["sol"]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    keep = (sol.t <= 60.0) & (sol.t >= 1.0)
    ax.plot(sol.t[keep], study["low"][keep], color=BLUE,
            label=rf"level $u={study['level_low']}$")
    ax.plot(sol.t[keep], study["high"][keep], color=ORANGE,
            label=rf"level $u={study['level_high']}$")
    ax.axhline(study["stall_theory"], color=INK, lw=0.8, ls=":",
               label=rf"$K(x)={study['level_high']}$")
    ax.set(xlabel=r"time $t$", ylabel="distance from centre", ylim=(0, 53))
    ax.legend(loc="lower right")
    _save(fig, folder, "capacity_front")


def fig_capacity_steady(results, folder):
    study = results["capacity"]
    sol = study["sol"]
    x = sol.x
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    axes[0].plot(x, sol.u[-1], color=BLUE, lw=2.4, label="PDE at $t=150$")
    axes[0].plot(x, study["steady"], color=ORANGE, ls="--", lw=1.0, label="boundary value solve")
    axes[0].plot(x, study["K"], color=INK, lw=0.8, ls=":", label=r"$K(x)$")
    axes[0].set(xlabel=r"position $x$", ylabel=r"density $u$", xlim=(0, 100), ylim=(0, 1.42))
    axes[0].legend(loc="upper left", fontsize=7.5)
    for row, colour in zip(study["sweep"], _ramp(4)):
        axes[1].plot(x, row["u"], color=colour, label=rf"$w={row['width']:.0f}$")
    axes[1].set(xlabel=r"position $x$", xlim=(0, 100), ylim=(0, 1.42))
    axes[1].legend(loc="upper left", ncol=4, handlelength=1.3, columnspacing=0.8)
    fig.tight_layout()
    _save(fig, folder, "capacity_steady")


def fig_patch(results, folder):
    study = results["patch"]
    rows = study["rows"]
    lengths = np.array([row["length"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    curve = np.linspace(lengths.min(), lengths.max(), 300)
    axes[0].plot(curve, theory.patch_growth_rate(curve, 1.0, 1.0), color=ORANGE,
                 label=r"$r-D\pi^2/L^2$")
    axes[0].plot(lengths, [row["growth"] for row in rows], "o", ms=5, color=BLUE,
                 label="simulation")
    axes[0].axhline(0.0, color=MUTED, lw=0.6)
    axes[0].set(xlabel=r"patch length $L$", ylabel=r"growth rate $\sigma$")
    axes[0].legend(loc="lower right")
    axes[1].plot(lengths, [row["amplitude"] for row in rows], "o-", ms=5, color=BLUE,
                 label=r"peak density at $t=400$")
    axes[1].axvline(study["critical"], color=INK, lw=0.8, ls=":", label=r"$L_c=\pi\sqrt{D/r}$")
    axes[1].set(xlabel=r"patch length $L$", ylabel=r"$\max\ u$")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    _save(fig, folder, "patch")


ALL_FIGURES = (fig_reaction, fig_phase_plane, fig_dispersion, fig_snapshots, fig_heatmap,
               fig_front, fig_profile, fig_sweep, fig_discrete, fig_convergence, fig_stability,
               fig_diffusion, fig_mass, fig_selection, fig_capacity_snapshots,
               fig_capacity_heatmap, fig_capacity_front, fig_capacity_steady, fig_patch)


def make_all(results, folder):
    """Write every figure to folder as both PDF and PNG."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for figure in ALL_FIGURES:
        figure(results, folder)
