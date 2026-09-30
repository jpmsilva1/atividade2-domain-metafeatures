"""Step 5 -- every figure in the report, drawn from the CSVs in results/.

Visual identity shared with the 002 project and Atividade 1: the "Ocean Dusk"
palette, serif Times fonts, no top/right spines, faint grid, PDF output.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import features
from src.analysis import REDUNDANCY_THRESHOLD
from src.data import FIG_DIR, RESULTS_DIR

DARK, TEAL, CORAL, GREY = "#264653", "#2A9D8F", "#E76F51", "#B0BEC5"
APPROACH_STYLE = {  # X-dependent approaches in colour, X-blind references in grey
    "Approach1": dict(color=DARK), "Approach2": dict(color=TEAL), "HARRIS": dict(color=CORAL),
    "AR": dict(color=GREY, ls="--"), "MR": dict(color=GREY, ls=":"),
    "SigWins": dict(color=GREY, ls="-."),
}
ARMS = ["X_base", "X_pruned", "X_ext"]
# Meta-feature groups, coloured as in Meta-IR's Fig. 11.
FEATURE_GROUP = {**{f: "imbalance" for f in ["log_N", "IR", "pct_rare_09", "pct_rare_08"]},
                 **{f: "complexity (Lorena)" for f in ["c2", "s4", "l1"]},
                 "theta_hat": "extremes (EVT)"}
GROUP_COLOR = {"imbalance": DARK, "TSFEL": GREY, "complexity (Lorena)": TEAL, "extremes (EVT)": CORAL}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    # Sized for figures shrunk to page width (~0.55x) in the report.
    "font.size": 14, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 14,
    "legend.fontsize": 12, "legend.frameon": False, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "lines.linewidth": 2.0,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
})


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def cliques(ranks: list, cd: float) -> list:
    """Groups of methods that are not significantly different: maximal runs
    (i, j) of consecutive methods, sorted by rank, whose ranks span <= CD."""
    runs = [(i, max(j for j in range(i, len(ranks)) if ranks[j] - ranks[i] <= cd))
            for i in range(len(ranks))]
    return [(i, j) for i, j in runs
            if j > i and not any(a <= i and j <= b and (a, b) != (i, j) for a, b in runs)]


def cd_panel(ax, avg_rank: dict, cd: float, friedman_p: float, title: str):
    """Demsar-style critical-difference diagram on one axis. Rank 1 (best) on
    the left; a thick bar joins methods whose ranks differ by less than CD,
    i.e. that are not significantly different."""
    items = sorted(avg_rank.items(), key=lambda kv: kv[1])
    k = len(items)
    ax.set_xlim(0.4, k + 0.6), ax.set_ylim(-0.45 - 0.35 * k, 1.0), ax.axis("off")
    ax.plot([1, k], [0, 0], color="black", lw=1.2)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.08], color="black", lw=1.2)
        ax.text(r, 0.15, str(r), ha="center", fontsize=13)
    ax.plot([1, 1 + cd], [0.6, 0.6], color="black", lw=1.5)
    ax.text(1 + cd / 2, 0.68, f"CD = {cd:.2f}", ha="center", fontsize=13)
    for i, (name, rank) in enumerate(items):
        y = -0.55 - 0.35 * i
        ax.plot([rank, rank, k + 0.15], [0, y, y], color="black", lw=1.2)
        ax.text(k + 0.2, y, f"{name} ({rank:.2f})", va="center", fontsize=13)
    ranks = [r for _, r in items]
    for level, (i, j) in enumerate(cliques(ranks, cd)):
        ax.plot([ranks[i] - 0.05, ranks[j] + 0.05], [-0.22 - 0.12 * level] * 2, color="black", lw=4)
    ax.set_title(f"{title}\nFriedman p = {friedman_p:.1g}", fontsize=14)


def fig_cd():
    cd = json.loads((RESULTS_DIR / "cd.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, arm in zip(axes, ARMS):
        c = cd[arm]
        cd_panel(ax, c["avg_rank"], c["cd"], c["friedman_p"], arm)
    save(fig, "fig_cd")


def fig_loss_curves():
    """Three panels side by side, one per arm, shared y-axis."""
    curves = pd.read_csv(RESULTS_DIR / "loss_curves.csv")
    blind = curves[curves["arm"] == "X-blind"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        for name, group in pd.concat([curves[curves["arm"] == arm], blind]).groupby("approach"):
            ax.plot(group["t"], group["loss"], label=name, **APPROACH_STYLE[name])
        # Linear t, as in Atividade 1: the AUC averages the curve over t = 1..52
        # with equal weight, so this is the area the metric actually measures.
        ax.set_title(arm), ax.set_xlabel("t (top-t workflows picked)")
    axes[0].set_ylabel("perda(t)")
    axes[-1].legend(loc="upper right")
    save(fig, "fig_loss_curves")


def bold_new(ticks):
    for tick in ticks:
        if tick.get_text().split(" ")[0] in features.NEW_FEATURES:
            tick.set_fontweight("bold")


def fig_redundancy():
    """Only the question 2.1.2 asks: new features (rows) against the 15 X_base
    columns and against each other, with each new feature's largest |r| in its
    label and every |r| above the redundancy threshold boxed."""
    corr = pd.read_csv(RESULTS_DIR / "redundancy_corr.csv", index_col=0)
    strip = corr.loc[features.NEW_FEATURES, features.X_BASE + features.NEW_FEATURES]
    for f in features.NEW_FEATURES:
        strip.loc[f, f] = np.nan  # self-correlation, not information
    fig, ax = plt.subplots(figsize=(13, 3.2))
    im = ax.imshow(strip, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(strip.shape[1]), strip.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(strip)), [f"{f} (max |r| {strip.loc[f].abs().max():.2f})"
                                      for f in strip.index]), ax.grid(False)
    bold_new(ax.get_yticklabels()), bold_new(ax.get_xticklabels())
    ax.axvline(len(features.X_BASE) - 0.5, color="black", lw=1.5)  # X_base | new
    for i in range(strip.shape[0]):
        for j in range(strip.shape[1]):
            v = strip.iat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.6 else "black")
            if abs(v) > REDUNDANCY_THRESHOLD:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec="black", lw=2.5))
    fig.colorbar(im, ax=ax, label="Spearman r")
    save(fig, "fig_redundancy")


STRATEGY_ORDER = ["none", "under", "over", "smote"]


def importance_bars(ax, imp: pd.Series, title: str):
    """Meta-IR Fig. 11 style: horizontal Gini importance bars coloured by feature group."""
    imp = imp.sort_values()
    ax.barh(imp.index, imp, color=[GROUP_COLOR[FEATURE_GROUP.get(f, "TSFEL")] for f in imp.index])
    ax.set_title(title), ax.set_xlabel("Gini importance")
    bold_new(ax.get_yticklabels())


def group_legend(fig):
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()],
               labels=list(GROUP_COLOR), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))


def fig_importance():
    """Main text: Gini importance of the two Meta-IR Fig. 11 classifiers. (a) best
    learner, the only one that beats chance (predictive_check.csv); (b) best
    resampling family, which does not (the report caption says so)."""
    meta_ir = pd.read_csv(RESULTS_DIR / "importance_meta_ir.csv", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, gridspec_kw=dict(wspace=0.55))
    importance_bars(a, meta_ir["learner"], "(a) best learner")
    importance_bars(b, meta_ir["strategy"], "(b) best resampling family")
    for ax in (a, b):  # shrunk to 0.6 of the page, so larger than the rcParams default
        ax.tick_params(labelsize=16), ax.xaxis.label.set_size(17), ax.title.set_size(18)
    group_legend(fig)
    save(fig, "fig_importance")


def fig_importance_annex():
    """Annex B, Meta-Scaler Fig. 5 style: one row per base learner, importance of
    the classifier predicting that learner's best resampling family; top 2
    labelled. None of them beats chance (predictive_check.csv)."""
    per_learner = pd.read_csv(RESULTS_DIR / "importance_per_learner.csv", index_col=0)
    # Training labels of the (b) classifiers, shown as counts next to each learner.
    labels = pd.read_csv(RESULTS_DIR / "best_strategy_per_learner.csv", index_col=0)
    order = sorted(per_learner.columns, key=lambda f: list(GROUP_COLOR).index(FEATURE_GROUP.get(f, "TSFEL")))
    fig, axes = plt.subplots(len(per_learner), 1, figsize=(7, 5.6), gridspec_kw=dict(hspace=0.25))
    for i, (ax, (learner, imp)) in enumerate(zip(axes, per_learner[order].iterrows())):
        ax.bar(range(len(order)), imp, color=[GROUP_COLOR[FEATURE_GROUP.get(f, "TSFEL")] for f in order])
        for rank, f in enumerate(imp.nlargest(2).index):  # 2nd label higher, so neighbours don't collide
            ax.annotate(f, (order.index(f), imp[f]), textcoords="offset points", xytext=(0, 2 + 11 * rank),
                        ha="center", fontsize=9)
        ax.set_ylim(0, per_learner.to_numpy().max() * 1.55), ax.set_yticks([0, 0.1], ["0", "0.1"], fontsize=9)
        n = labels[learner].value_counts().reindex(STRATEGY_ORDER, fill_value=0)
        ax.text(1.01, 0.5, f"{learner}\n({'/'.join(map(str, n))})", transform=ax.transAxes,
                va="center", fontsize=10)
        if i == len(per_learner) // 2:
            ax.set_ylabel("Gini importance")
        ax.set_xticks(range(len(order)), order if i == len(per_learner) - 1 else [], rotation=90, fontsize=10)
        ax.grid(False)
        if i == 0:
            ax.set_title("best resampling family, per base learner")
    bold_new(ax.get_xticklabels())
    group_legend(fig)
    save(fig, "fig_importance_annex")


PDP_STYLE = {"none": dict(color=GREY, ls="-"), "under": dict(color=DARK, ls="--"),
             "over": dict(color=TEAL, ls="-."), "smote": dict(color=CORAL, ls=":")}


def fig_partial_dependence():
    """FFORMS / MetaFore style: partial dependence of the best-resampling-family
    classifier on each new feature; rug marks the 24 datasets."""
    pdp = pd.read_csv(RESULTS_DIR / "partial_dependence.csv")
    X = pd.read_csv(RESULTS_DIR / "X_all.csv", index_col=0)
    fig, axes = plt.subplots(1, len(features.NEW_FEATURES), figsize=(13, 3.4), sharey=True)
    for ax, f in zip(axes, features.NEW_FEATURES):
        for strategy, style in PDP_STYLE.items():
            group = pdp[(pdp.feature == f) & (pdp.strategy == strategy)]
            ax.plot(group["value"], group["probability"], label=strategy, **style)
        ax.plot(X[f], np.zeros(len(X)) + 0.1, "|", color="black", ms=10)
        ax.set_title(f), ax.set_xlabel("meta-feature value")
    axes[0].set_ylabel("P(best family)"), axes[0].set_ylim(0.08, 0.45)
    axes[-1].legend(loc="upper left", fontsize=10)
    save(fig, "fig_partial_dependence")


def fig_partial_dependence_ice():
    """The curves of fig_partial_dependence one per panel (FFORMS Figs. 7-17 layout),
    each with its 95% band from the ICE curves: rows = new features, columns =
    resampling families. Rug marks the 24 datasets."""
    pdp = pd.read_csv(RESULTS_DIR / "partial_dependence.csv")
    X = pd.read_csv(RESULTS_DIR / "X_all.csv", index_col=0)
    fig, axes = plt.subplots(len(features.NEW_FEATURES), len(PDP_STYLE), figsize=(12, 10),
                             sharey=True, sharex="row")
    for row, f in zip(axes, features.NEW_FEATURES):
        for ax, (strategy, style) in zip(row, PDP_STYLE.items()):
            group = pdp[(pdp.feature == f) & (pdp.strategy == strategy)]
            ax.fill_between(group["value"], group["lower"], group["upper"], color=style["color"],
                            alpha=0.25, lw=0)
            ax.plot(group["value"], group["probability"], color=style["color"])
            ax.axhline(0.25, color="black", lw=0.6, ls=":")  # chance among the 4 families
            ax.plot(X[f], np.zeros(len(X)) + 0.02, "|", color="black", ms=8)
            if f == features.NEW_FEATURES[0]:
                ax.set_title(strategy)
        row[0].set_ylabel(f"{f}\nP(best family)")
    axes[0][0].set_ylim(0, 0.6)
    save(fig, "fig_partial_dependence_ice")


def run():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for draw in [fig_cd, fig_loss_curves, fig_redundancy, fig_importance, fig_importance_annex,
                 fig_partial_dependence, fig_partial_dependence_ice]:
        draw()
