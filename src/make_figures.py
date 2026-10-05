"""Figures for the report.

    python src/make_figures.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from harness import CORE, FOLDS, GBM, OUT, ROOT, log_rpm, mi, train

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dcdad4"
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# Figure 1: how the data was split
fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=200)
rows = [(f"Fold {i + 1}", last, test) for i, (last, test) in enumerate(FOLDS)]
rows.append(("Final fit", 10, None))
for y, (name, last, test) in enumerate(reversed(rows)):
    ax.barh(y, last, left=0, color=BLUE, height=0.55, edgecolor="white")
    if test:
        ax.barh(y, 1, left=last, color=ORANGE, height=0.55, edgecolor="white")
        ax.text(last + 1.15, y, f"score {MONTHS[test - 1]}", va="center", fontsize=9, color=MUTED)
    else:
        ax.barh(y, 2, left=last, color=ORANGE, height=0.55, edgecolor="white", alpha=0.55)
        ax.text(last + 2.15, y, "predict Nov-Dec (validation.csv)", va="center", fontsize=9, color=MUTED)
ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
ax.set_xticks(range(0, 13, 2), ["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Jan"][:7])
ax.set_xlim(0, 15.5)
ax.set_xlabel("2025")
ax.text(0.0, 1.08, "Blue: trained on.  Orange: scored on (never seen in that fold).", transform=ax.transAxes,
        fontsize=9, color=MUTED)
clean_axes(ax)
fig.tight_layout()
fig.savefig(FIG / "split.png")
plt.close(fig)

# Figure 2: market_index tracks the daily residual, but each month has its own level
d = train[~train["corrupt"]].reset_index(drop=True)
oof = np.zeros(len(d))
for a, b in KFold(5, shuffle=True, random_state=0).split(d):
    oof[b] = GBM(CORE).fit(d.iloc[a]).predict(d.iloc[b])
d["res"] = log_rpm(d) - oof
day = d.groupby("date").agg(res=("res", "mean")).join(mi.rename("mi"))
day["month"] = day.index.month
corr = day["res"].corr(day["mi"])
(OUT / "market_index_residual_corr.txt").write_text(
    f"out-of-fold daily residual vs market_index: corr {corr:.3f} over {len(day)} days\n")
print(f"daily residual vs market_index corr {corr:.3f}")

fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=200)
others = day[~day["month"].isin([1, 9])]
ax.scatter(others["mi"], others["res"] * 100, s=9, color="#b9b7ae", label="other months", zorder=2)
for m, color, name in ((1, BLUE, "January"), (9, ORANGE, "September")):
    g = day[day["month"] == m]
    ax.scatter(g["mi"], g["res"] * 100, s=14, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    slope, icpt = np.polyfit(g["mi"], g["res"] * 100, 1)
    xs = np.array([g["mi"].min(), g["mi"].max()])
    ax.plot(xs, slope * xs + icpt, color=color, linewidth=2)
    ax.text(xs.mean(), (slope * xs.mean() + icpt) + (1.0 if m == 1 else 1.1), name, color=color, fontsize=10,
            ha="center", fontweight="bold")
ax.set_xlabel("market_index (daily mean)")
ax.set_ylabel("Rate vs static model (%)")
ax.grid(color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Within a month the index tracks the rate closely; the month's level differs",
             loc="left", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "market_index_by_month.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("*.png")))
