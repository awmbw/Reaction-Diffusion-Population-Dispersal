"""Reproduce every result: experiments, figures, animations and report data."""
from pathlib import Path

from rdpop import animations, experiments, figures, report_data

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    print("Running experiments")
    results = experiments.run_all()
    print("Writing figures")
    figures.make_all(results, OUTPUTS / "figures")
    print("Writing animations")
    animations.make_all(results, OUTPUTS / "animations")
    print("Writing report data")
    report_data.write_all(results, ROOT / "report", OUTPUTS / "data")
    print(f"Done. Outputs are in {OUTPUTS}")


if __name__ == "__main__":
    main()
