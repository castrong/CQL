import argparse
import pandas as pd
import matplotlib.pyplot as plt

METRICS = [
    ("trainer/QF1 Loss",                        "QF1 Loss"),
    ("trainer/QF2 Loss",                        "QF2 Loss"),
    ("trainer/QF1 in-distribution values Mean", "QF1 In-Distribution Values (Mean)"),
    ("trainer/QF2 in-distribution values Mean", "QF2 In-Distribution Values (Mean)"),
    ("trainer/QF1 random values Mean",          "QF1 Random Values (Mean)"),
    ("trainer/QF2 random values Mean",          "QF2 Random Values (Mean)"),
    ("trainer/QF1 next_actions values Mean",    "QF1 Next-Action Values (Mean)"),
    ("trainer/QF2 next_actions values Mean",    "QF2 Next-Action Values (Mean)"),
    ("trainer/Policy Loss",                     "Policy Loss"),
    ("trainer/Log Pis Mean",        "Log Pis (Mean)"),
    ("trainer/Alpha",               "Alpha"),
    ("trainer/Alpha Loss",          "Alpha Loss"),
    ("evaluation/Average Returns",  "Average Returns"),
    ("evaluation/Returns Std",      "Returns Std"),
]

def plot(csv_path):
    df = pd.read_csv(csv_path)

    n = len(METRICS)
    nrows = 2
    ncols = (n + 1) // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, 8))
    axes = axes.flatten()

    for i, (col, label) in enumerate(METRICS):
        ax = axes[i]
        if col in df.columns:
            ax.plot(df["Epoch"], df[col])
        else:
            ax.text(0.5, 0.5, f"'{col}'\nnot found", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(csv_path, fontsize=9, y=1.01)
    plt.tight_layout()
    out_path = csv_path.replace("progress.csv", "progress_plot.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    args = parser.parse_args()
    plot(args.csv_path)
