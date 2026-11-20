"""Animated 1D line plots of the spreading population, written as GIF files."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
import numpy as np  # noqa: E402

from .figures import BLUE, ORANGE  # noqa: E402


def animate(sol, path, last_time, x_limits, capacity=None, y_max=1.1, fps=12):
    """Write one line-plot animation of sol up to last_time."""
    frames = np.nonzero(sol.t <= last_time)[0]
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=90)
    line, = ax.plot(sol.x, sol.u[0], color=BLUE, lw=2.0, label="density $u(x,t)$")
    if capacity is not None:
        ax.plot(sol.x, capacity, color=ORANGE, ls="--", lw=1.1, label="capacity $K(x)$")
    label = ax.text(0.02, 0.9, "", transform=ax.transAxes)
    ax.set(xlim=x_limits, ylim=(0.0, y_max), xlabel="position $x$", ylabel="density $u$")
    ax.legend(loc="upper right")
    fig.tight_layout()

    def update(j):
        line.set_ydata(sol.u[j])
        label.set_text(f"$t = {sol.t[j]:.1f}$")
        return line, label

    movie = FuncAnimation(fig, update, frames=frames, blit=True)
    movie.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_all(results, folder):
    """Write the wave animation and the capacity-gradient animation."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    animate(results["baseline"]["sol"], folder / "wave_spreading.gif", 40.0, (0.0, 200.0))
    capacity = results["capacity"]
    animate(capacity["sol"], folder / "capacity_gradient.gif", 40.0, (0.0, 100.0),
            capacity=capacity["K"])
