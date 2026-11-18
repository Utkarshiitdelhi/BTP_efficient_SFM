"""Figures and the results table from results/results.csv and the run folders.

    python -m btp.report
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")

# reference palette (light), slots assigned in this fixed order and never cycled
ORDER = ["orbit_single", "orbit_multi", "orbit_high", "oblique_grid", "planner",
         "planner_budget153", "planner_res", "planner_res_budget153"]
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                         "#6250d6", "#e34948"]))
LABEL = {"orbit_single": "Single orbit", "orbit_multi": "Multi-height orbit",
         "orbit_high": "High orbit", "oblique_grid": "Oblique grid",
         "planner": "Planner v1 (24 views)", "planner_budget153": "Planner v1 (153 s)",
         "planner_res": "Planner v2 (24 views)", "planner_res_budget153": "Planner v2 (153 s)"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# Kaggle run_002 v1 (sun only, no sky light): SIFT features per image, from its COLMAP log
KAGGLE_V1_FEATURES = [2423, 2313, 2394, 2378, 2079, 1821, 1902, 1827, 1557, 1220, 781, 359,
                      315, 304, 319, 395, 496, 631, 736, 772, 954, 1317, 1860, 2196]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _fig(w, h, n=1):
    fig, axes = plt.subplots(1, n, figsize=(w, h), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style(ax)
    return fig, axes


def _label_points(ax, df, x, y):
    for _, r in df.iterrows():
        ax.annotate(LABEL.get(r.strategy, r.strategy), (r[x], r[y]), xytext=(7, 5),
                    textcoords="offset points", fontsize=8.5, color=INK)


def quality_vs_time(df):
    fig, axes = _fig(11, 4.2, 2)
    for ax, (m, title) in zip(axes, [("fscore@0.30", "F-score at 0.30 m"),
                                     ("recall@1.00", "Surface covered within 1 m")]):
        for _, r in df.iterrows():
            ax.errorbar(r.flight_time_s, r[m], yerr=r.get(m + "_sd", 0), fmt="o", ms=8,
                        color=COLOR.get(r.strategy, INK2), mec=SURFACE, mew=1.5,
                        elinewidth=2, capsize=0)
        _label_points(ax, df, "flight_time_s", m)
        ax.set_xlabel("Flight time (s)")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylim(0, max(0.05, df[m].max() * 1.25))
        ax.set_xlim(df.flight_time_s.min() * 0.85, df.flight_time_s.max() * 1.25)
    fig.suptitle("Reconstruction quality vs flight time (mean of 3 SfM seeds; bars = 1 sd)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/quality_vs_flight_time.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def predicted_vs_measured(df):
    fig, axes = _fig(5.6, 4.4)
    ax = axes[0]
    for _, r in df.iterrows():
        ax.plot(r.pred_coverage_3views, r["recall@1.00"], "o", ms=8,
                color=COLOR.get(r.strategy, INK2), mec=SURFACE, mew=1.5)
    _label_points(ax, df, "pred_coverage_3views", "recall@1.00")
    if len(df) >= 3:
        rho = df[["pred_coverage_3views", "recall@1.00"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"Geometry-only forecast vs SfM result (Spearman ρ = {rho:.2f})",
                     loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("Predicted: surface seen by ≥3 views (from flight plan only)")
    ax.set_ylabel("Measured: surface reconstructed within 1 m")
    ax.set_xlim(df.pred_coverage_3views.min() - 0.05, df.pred_coverage_3views.max() + 0.12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/predicted_vs_measured.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def layouts(runs, scene):
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1, 1]})
    axes = np.array(axes).reshape(2, n)
    lo, hi = np.array(scene["bbox_lo"]), np.array(scene["bbox_hi"])
    for j, (rid, strat) in enumerate(runs):
        poses = json.load(open(f"{ROOT}/runs/{rid}/poses.json"))["poses"]
        P = np.array([p["position"] for p in poses])
        T = np.array([p["target"] for p in poses])
        for i, (a, b, xl, yl) in enumerate([(0, 1, "x (m)", "y (m)"), (0, 2, "x (m)", "z (m)")]):
            ax = axes[i, j]
            _style(ax)
            ax.add_patch(plt.Rectangle((lo[a], lo[b]), hi[a] - lo[a], hi[b] - lo[b],
                                       color=INK2, alpha=0.25, lw=0))
            loop = np.vstack([P, P[:1]])
            ax.plot(loop[:, a], loop[:, b], color=COLOR.get(strat, INK2), lw=1.2, alpha=0.6)
            ax.quiver(P[:, a], P[:, b], (T - P)[:, a], (T - P)[:, b], angles="xy",
                      scale_units="xy", scale=4, width=0.006, color=COLOR.get(strat, INK2))
            ax.plot(P[:, a], P[:, b], "o", ms=4, color=COLOR.get(strat, INK2))
            ax.set_aspect("equal")
            ax.set_xlim(-70, 75)
            ax.set_ylim(-70, 75) if i == 0 else ax.set_ylim(-5, 80)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(("Top view  " if i == 0 else "Side view  ") + yl, fontsize=8)
            if i == 0:
                ax.set_title(f"{LABEL.get(strat, strat)}\n{len(P)} images", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/flight_layouts.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def coverage_maps(runs):
    """Side elevations of the true surface: reconstructed within 1 m vs missed."""
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 4.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (rid, strat) in zip(axes, runs):
        z = np.load(f"{ROOT}/runs/{rid}/sfm/seed0/surface_eval.npz")
        P, obs, d = z["P"], z["observable"], z["d_g2r"]
        seen = obs                                   # same denominator for every strategy
        got = obs & (d < 1.0)
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        # unroll the four elevations by azimuth round the building centre
        c = P[:, :2].mean(0)
        az = np.degrees(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
        ax.scatter(az[~seen], P[~seen, 2], s=0.3, color="#d9d8d3", rasterized=True)
        ax.scatter(az[seen & ~got], P[seen & ~got, 2], s=0.3, color="#9a9890", rasterized=True)
        ax.scatter(az[got], P[got, 2], s=0.3, color=COLOR.get(strat, INK), rasterized=True)
        ax.set_title(f"{LABEL.get(strat, strat)}\n{got[seen].mean():.0%} of observable surface",
                     fontsize=8.5, color=INK)
    fig.suptitle("Where each flight reconstructed the building (unrolled round the facades; "
                 "colour = reconstructed within 1 m, dark grey = missed, light = not observable by any flight)",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/coverage_maps.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def lighting_figure(run="run_002"):
    path = f"{ROOT}/runs/{run}/sfm/seed0/sfm_stats.json"
    if not os.path.exists(path):
        return
    after = list(json.load(open(path))["features_per_image"].values())
    fig, axes = _fig(7.5, 3.6)
    ax = axes[0]
    x = np.arange(24) * 15
    ax.plot(x, KAGGLE_V1_FEATURES, "-o", lw=2, ms=4, color="#eb6834")
    ax.plot(x, after, "-o", lw=2, ms=4, color="#2a78d6")
    ax.annotate("sun only: no reconstruction", (x[13], KAGGLE_V1_FEATURES[13]),
                xytext=(10, -14), textcoords="offset points", fontsize=8.5, color=INK)
    ax.annotate("sun + sky light: 24/24 registered", (x[13], after[13]),
                xytext=(10, 8), textcoords="offset points", fontsize=8.5, color=INK)
    ax.legend(["Sun only (Kaggle v1)", "Sun + sky light"], frameon=False, fontsize=8.5,
              loc="upper center", ncol=2)
    ax.set_xlabel("Camera azimuth around the orbit (degrees)")
    ax.set_ylabel("SIFT features per image")
    ax.set_title("Illumination decides whether the shadowed facade can be matched",
                 loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, max(max(after), max(KAGGLE_V1_FEATURES)) * 1.2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/lighting_features.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def table(df) -> str:
    cols = [("strategy", "Strategy"), ("n_images", "Images"), ("flight_time_s", "Flight (s)"),
            ("path_length_m", "Path (m)"), ("registered", "Registered"), ("points", "3D points"),
            ("track_length", "Track len."), ("pose_pos_err_m", "Pose err. (m)"),
            ("acc_median_m", "Median err. (m)"), ("precision@0.30", "Precision@0.3"),
            ("recall@0.30", "Recall@0.3"), ("fscore@0.30", "F@0.3"),
            ("recall@1.00", "Recall@1.0"), ("pred_coverage_3views", "Pred. cov.")]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for k, _ in cols:
            v = r[k]
            if k == "strategy":
                cells.append(LABEL.get(v, v))
            elif k in ("n_images", "points", "flight_time_s", "path_length_m"):
                cells.append(f"{v:,.0f}")
            elif k == "registered":
                cells.append(f"{v:.0f}/{r.n_images:.0f}")
            elif k in ("fscore@0.30", "recall@1.00", "recall@0.30") and r.get(k + "_sd", 0) > 0.0005:
                cells.append(f"{v:.3f} ± {r[k + '_sd']:.3f}")
            else:
                cells.append(f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(f"{RES}/results.csv")
    df["order"] = df.strategy.map({s: i for i, s in enumerate(ORDER)}).fillna(99)
    df = df.sort_values("order").reset_index(drop=True)
    scene = json.load(open(f"{ROOT}/data/scene.json"))
    runs = list(zip(df.run, df.strategy))
    quality_vs_time(df)
    predicted_vs_measured(df)
    layouts(runs, scene)
    coverage_maps(runs)
    lighting_figure()
    md = table(df)
    open(f"{RES}/results_table.md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
