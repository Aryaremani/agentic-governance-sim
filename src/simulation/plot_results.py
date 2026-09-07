"""Generate a chart of resource stock over time, with phases shaded, from a results CSV
produced by run_experiment.py.

Usage:
    python src/simulation/plot_results.py --csv results/mock_demo.csv --out results/mock_demo.png
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

PHASE_COLORS = {
    "no_enforcement": "#fde2e2",
    "enforcement_active": "#e2f0e2",
    "enforcement_withdrawn": "#e2e8fd",
}


def plot(csv_path: str, out_path: str, title: str = "") -> None:
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Shade phases
    for phase in df["phase"].unique():
        rows = df[df["phase"] == phase]
        ax.axvspan(
            rows["round"].min() - 0.5,
            rows["round"].max() + 0.5,
            color=PHASE_COLORS.get(phase, "#eeeeee"),
            label=phase,
        )

    ax.plot(df["round"], df["stock"], color="#1f4e79", linewidth=2, label="Resource stock")
    ax.axhline(y=0, color="gray", linewidth=0.5)

    ax.set_xlabel("Round")
    ax.set_ylabel("Resource stock")
    ax.set_title(title or "Resource stock over time")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    plot(args.csv, args.out, args.title)
