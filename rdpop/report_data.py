"""Write the measured values as LaTeX macros and tables, plus machine-readable copies."""
import csv
import json
from pathlib import Path

import numpy as np


def _sci(value):
    """Format a number in scientific notation as LaTeX math."""
    if value == 0.0:
        return "$0$"
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def _macros(results):
    base, long = results["baseline"], results["long_run"]
    sweep, conv = results["sweep"], results["convergence"]
    stab, diff = results["stability"], results["diffusion"]
    cap, patch = results["capacity"], results["patch"]
    fit, left = base["fit_right"], base["fit_left"]
    stable_case = stab["cases"][1]
    unstable = stab["cases"][2]
    blow_up = unstable["t"][np.argmax(~np.isfinite(unstable["sawtooth"]))]
    return {
        "baseSpeed": f"{fit.slope:.4f}", "baseSpeedErr": _sci(fit.slope_error),
        "baseRsq": f"{fit.r_squared:.6f}", "leftSpeed": f"{left.slope:.4f}",
        "gridSpeed": f"{base['grid_speed']:.4f}", "baseDelay": f"{base['delay']:.4f}",
        "basePredicted": f"{base['predicted']:.4f}",
        "baseGap": f"{abs(fit.slope - base['predicted']):.4f}",
        "basePercentLow": f"{100.0 * (1.0 - fit.slope / 2.0):.1f}",
        "speedTen": f"{base['levels'][0.1].slope:.4f}",
        "speedFifty": f"{base['levels'][0.5].slope:.4f}",
        "speedNinety": f"{base['levels'][0.9].slope:.4f}",
        "engineDiff": _sci(base['engine_difference']),
        "loopSeconds": f"{base['loop_seconds']:.1f}",
        "vectorSeconds": f"{base['vector_seconds']:.2f}",
        "speedUp": f"{base['loop_seconds'] / base['vector_seconds']:.0f}",
        "baseAlpha": f"{base['alpha']:.4f}", "baseDtLimit": f"{base['dt_limit']:.4f}",
        "baseUpdates": f"{base['updates'] / 1e6:.0f}",
        "massSlope": f"{base['mass_fit'].slope:.2f}",
        "massPredicted": f"{2.0 * base['predicted']:.2f}",
        "longSpeed": f"{long['speed']:.4f}", "longGrid": f"{long['grid_speed']:.4f}",
        "longDelay": f"{long['delay_coefficient']:.3f}",
        "longGap": f"{100.0 * abs(long['speed'] / long['grid_speed'] - 1.0):.2f}",
        "profileGap": f"{long['profile_gap']:.3f}", "edgeDecay": f"{long['edge_decay']:.3f}",
        "sweepExponent": f"{sweep['exponent']:.4f}", "sweepPrefactor": f"{sweep['prefactor']:.4f}",
        "sweepMaxError": f"{100.0 * sweep['max_error']:.2f}",
        "spaceOrder": f"{conv['space_order']:.2f}", "timeOrder": f"{conv['time_order']:.2f}",
        "monotoneLimit": f"{stab['monotone_limit']:.4f}",
        "vonNeumannLimit": f"{stab['von_neumann_limit']:.4f}",
        "diffusionLimit": f"{stab['diffusion_limit']:.4f}",
        "stableDt": f"{stable_case['dt']:g}", "blowUp": f"{blow_up:.1f}",
        "massDrift": _sci(diff['mass_drift']), "gaussError": _sci(diff['profile_error']),
        "varianceSlope": f"{diff['variance_slope']:.6f}",
        "pdeBvpGap": _sci(cap['pde_bvp_gap']), "stallMeasured": f"{cap['stall_measured']:.2f}",
        "stallTheory": f"{cap['stall_theory']:.2f}", "speedLow": f"{cap['speed_low']:.3f}",
        "speedUniform": f"{cap['speed_uniform']:.3f}", "capPeak": f"{cap['peak']:.3f}",
        "capEdge": f"{cap['edge_density']:.4f}",
        "capEdgeExcess": f"{100.0 * (cap['edge_overshoot'] - 1.0):.1f}",
        "capMass": f"{cap['mass']:.3f}", "capIntK": f"{cap['capacity_mass']:.3f}",
        "capExcess": f"{100.0 * cap['excess_fraction']:.0f}",
        "criticalLength": f"{patch['critical']:.4f}",
        "patchError": f"{patch['max_growth_error']:.4f}",
    }


def _table(path, header, rows):
    lines = [r"\begin{tabular}{" + "c" * len(header) + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_tables(results, folder):
    long, sweep = results["long_run"], results["sweep"]
    conv, sel, patch = results["convergence"], results["selection"], results["patch"]
    cap = results["capacity"]
    _table(folder / "table_windows.tex",
           ["Window", "Measured slope", "Delay", r"Predicted $v_h-{}$delay", "Difference"],
           [[f"$[{row['t1']:.0f},{row['t2']:.0f}]$", f"{row['measured']:.4f}",
             f"{row['delay']:.4f}", f"{row['predicted']:.4f}",
             f"{row['measured'] - row['predicted']:+.4f}"] for row in long["table"]])
    _table(folder / "table_sweep.tex",
           ["$r$", "$D$", r"$2\sqrt{rD}$", "Measured", "Corrected", r"Error (\%)"],
           [[f"{row['r']:g}", f"{row['D']:g}", f"{row['theory']:.4f}", f"{row['measured']:.4f}",
             f"{row['corrected']:.4f}", f"{100.0 * (row['corrected'] / row['theory'] - 1.0):+.2f}"]
            for row in sweep["rows"]])
    _table(folder / "table_space.tex",
           [r"$\Delta x$", r"$\Delta t$", r"$L^2$ error", r"$L^\infty$ error"],
           [[f"{row['dx']:g}", _sci(row["dt"]), _sci(row["l2"]), _sci(row["max"])]
            for row in conv["space"]])
    _table(folder / "table_selection.tex",
           [r"$\lambda$", "Theory", "Measured"],
           [[f"{row['lam']:g}", f"{row['theory']:.4f}", f"{row['measured']:.4f}"]
            for row in sel["rows"]])
    _table(folder / "table_patch.tex",
           [r"$L/L_c$", r"$L$", r"Theory $\sigma$", r"Measured $\sigma$", r"Peak at $t=400$"],
           [[f"{row['fraction']:g}", f"{row['length']:.3f}", f"{row['theory']:+.4f}",
             f"{row['growth']:+.4f}", _sci(row["amplitude"])] for row in patch["rows"]])
    _table(folder / "table_width.tex",
           [r"Width $w$", r"$\int_0^L K\,dx$", r"$\int_0^L u^*\,dx$", "Ratio"],
           [[f"{row['width']:g}", f"{row['capacity_mass']:.3f}", f"{row['mass']:.3f}",
             f"{row['mass'] / row['capacity_mass']:.4f}"] for row in cap["sweep"]])


def write_all(results, report_folder, data_folder):
    """Write macros and tables for the report, and JSON/CSV copies of the data."""
    report_folder, data_folder = Path(report_folder), Path(data_folder)
    report_folder.mkdir(parents=True, exist_ok=True)
    data_folder.mkdir(parents=True, exist_ok=True)

    macros = _macros(results)
    text = "\n".join(rf"\newcommand{{\res{key[0].upper()}{key[1:]}}}{{{value}}}"
                     for key, value in macros.items())
    (report_folder / "results.tex").write_text(text + "\n", encoding="utf8")
    _write_tables(results, report_folder)
    (data_folder / "results.json").write_text(json.dumps(macros, indent=2) + "\n", encoding="utf8")

    base = results["baseline"]
    with open(data_folder / "front_track.csv", "w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "right_front", "left_front", "total_mass"])
        writer.writerows(zip(base["sol"].t, base["right"], base["left"], base["mass"]))
