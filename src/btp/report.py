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

LABEL = {"orbit_single": "Single orbit", "orbit_multi": "Multi-height orbit",
         "orbit_high": "High orbit", "oblique_grid": "Oblique grid",
         "planner": "Planner v1 (24 views)", "planner_budget153": "Planner v1 (153 s)",
         "planner_res": "Planner v2 (24 views)", "planner_res_budget153": "Planner v2 (153 s)",
         "planner_chain": "Planner v3 (24 views)", "planner_chain_budget153": "Planner v3 (153 s)"}
ALL = list(LABEL)

# Main comparison: the four baselines and the final planner. Reference palette
# (light), slots assigned in this fixed order and never cycled.
MAIN = ["orbit_single", "orbit_multi", "orbit_high", "oblique_grid",
        "planner_chain", "planner_chain_budget153"]
COLOR = dict(zip(MAIN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
# Label offsets (points) where the default up-right label would collide
OFFSET = {"orbit_single": (8, -13), "planner_res_budget153": (8, -12)}
# Runs that are ablations of a strategy rather than a strategy of their own
ABLATION = {"run_012": "Single orbit, sun only (no sky light)"}
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9890"

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


def _save(fig, name):
    fig.savefig(f"{FIG}/{name}", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def quality_vs_time(df):
    d = df[df.strategy.isin(MAIN)]
    fig, axes = _fig(11.5, 4.4, 2)
    for ax, (m, title) in zip(axes, [("fscore@0.30", "F-score at 0.30 m"),
                                     ("recall@1.00", "Observable surface reconstructed within 1 m")]):
        for _, r in d.iterrows():
            ax.errorbar(r.flight_time_s, r[m], yerr=r.get(m + "_sd", 0), fmt="o", ms=9,
                        color=COLOR[r.strategy], mec=SURFACE, mew=1.5, elinewidth=2, capsize=0)
            reg = (f"  ({r.cams_correct:.0f}/{r.n_images:.0f} placed)"
                   if r.cams_correct < r.n_images - 0.01 else "")
            ax.annotate(LABEL[r.strategy] + reg, (r.flight_time_s, r[m]),
                        xytext=OFFSET.get(r.strategy, (8, 4)),
                        textcoords="offset points", fontsize=8.5, color=INK)
        ax.set_xlabel("Flight time (s)  —  3 m/s cruise + 2 s per photo")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylim(0, d[m].max() * 1.25)
        ax.set_xlim(d.flight_time_s.min() * 0.75, d.flight_time_s.max() * 1.35)
    fig.suptitle("Reconstruction quality vs flight time, 24 images per flight unless noted "
                 "(mean of 3 SfM seeds; bars = 1 sd)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, "quality_vs_flight_time.png")


def planner_iterations(df):
    """F-score of each planner version against the single-orbit reference."""
    ref = df[df.strategy == "orbit_single"].iloc[0]
    vers = [("v1\nvisibility only", "planner", "planner_budget153"),
            ("v2\n+ resolution", "planner_res", "planner_res_budget153"),
            ("v3\n+ chained views", "planner_chain", "planner_chain_budget153")]
    fig, axes = _fig(8.5, 4.2)
    ax = axes[0]
    w = 0.36
    for k, (col, lab) in enumerate([(1, "24 views"), (2, "153 s budget")]):
        for i, v in enumerate(vers):
            r = df[df.strategy == v[col]]
            if not len(r):
                continue
            r = r.iloc[0]
            x = i + (k - 0.5) * (w + 0.04)
            ax.bar(x, r["fscore@0.30"], w, color=["#2a78d6", "#eb6834"][k],
                   label=lab if i == 0 else None)
            ax.annotate(f"{r.cams_correct:.0f}/{r.n_images:.0f}\nplaced", (x, r["fscore@0.30"]),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.axhline(ref["fscore@0.30"], color=INK, lw=1.2, ls="--")
    ax.annotate(f"single orbit {ref['fscore@0.30']:.3f}", (2.45, ref["fscore@0.30"]),
                xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.set_xticks(range(3), [v[0] for v in vers], fontsize=9)
    ax.set_ylabel("F-score at 0.30 m")
    ax.set_ylim(0, max(ref["fscore@0.30"], df["fscore@0.30"].max()) * 1.35)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Planner versions vs the single orbit (labels: cameras SfM placed correctly)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, "planner_iterations.png")


def predicted_vs_measured(df):
    from scipy.stats import spearmanr
    fig, axes = _fig(11, 4.6, 2)
    ok = df.cams_correct >= df.n_images - 0.01
    for ax, (col, name) in zip(axes, [("pred_coverage_3views", "Visibility only"),
                                      ("pred_coverage_res", "Resolution-aware (< 0.12 m/px)")]):
        if col not in df:
            continue
        ax.plot(df.loc[ok, col], df.loc[ok, "recall@1.00"], "o", ms=8, color="#2a78d6",
                mec=SURFACE, mew=1.5, label="every camera placed correctly")
        ax.plot(df.loc[~ok, col], df.loc[~ok, "recall@1.00"], "o", ms=8, color="#eb6834",
                mec=SURFACE, mew=1.5, label="SfM chain broke")
        for _, r in df.iterrows():
            short = (LABEL[r.strategy].replace("Planner ", "P").replace(" views", "")
                     .replace("(", "").replace(")", ""))
            dx, dy = OFFSET.get(r.strategy, (6, 3))
            ax.annotate(short, (r[col], r["recall@1.00"]), xytext=(dx - 2, dy + (2 if dy < 0 else 0)),
                        textcoords="offset points", fontsize=7.5, color=INK2)
        rho_all = spearmanr(df[col], df["recall@1.00"]).correlation
        rho_ok = spearmanr(df.loc[ok, col], df.loc[ok, "recall@1.00"]).correlation
        ax.set_title(f"{name}\nSpearman ρ = {rho_all:.2f} (all), {rho_ok:.2f} (unbroken chains)",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Forecast: observable surface seen by ≥3 useful views")
        ax.set_xlim(df[col].min() - 0.05, df[col].max() + 0.15)
    axes[0].set_ylabel("Measured: reconstructed within 1 m")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Can the flight plan alone predict reconstruction quality?",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, "predicted_vs_measured.png")


def layouts(df, scene):
    d = df[df.strategy.isin(MAIN)]
    runs = list(zip(d.run, d.strategy))
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.4), facecolor=SURFACE)
    axes = np.array(axes).reshape(2, n)
    lo, hi = np.array(scene["bbox_lo"]), np.array(scene["bbox_hi"])
    for j, (rid, strat) in enumerate(runs):
        poses = json.load(open(f"{ROOT}/runs/{rid}/poses.json"))["poses"]
        P = np.array([p["position"] for p in poses])
        T = np.array([p["target"] for p in poses])
        for i, (a, b, yl) in enumerate([(0, 1, "y (m)"), (0, 2, "z (m)")]):
            ax = axes[i, j]
            _style(ax)
            ax.add_patch(plt.Rectangle((lo[a], lo[b]), hi[a] - lo[a], hi[b] - lo[b],
                                       color=INK2, alpha=0.25, lw=0))
            loop = np.vstack([P, P[:1]])
            ax.plot(loop[:, a], loop[:, b], color=COLOR[strat], lw=1.2, alpha=0.6)
            ax.quiver(P[:, a], P[:, b], (T - P)[:, a], (T - P)[:, b], angles="xy",
                      scale_units="xy", scale=4, width=0.006, color=COLOR[strat])
            ax.plot(P[:, a], P[:, b], "o", ms=4, color=COLOR[strat])
            ax.set_aspect("equal")
            ax.set_xlim(-70, 75)
            ax.set_ylim(-70, 75) if i == 0 else ax.set_ylim(-5, 80)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(("Top view  " if i == 0 else "Side view  ") + yl, fontsize=8)
            if i == 0:
                ax.set_title(f"{LABEL[strat]}\n{len(P)} images", fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, "flight_layouts.png")


def coverage_maps(df):
    """True surface unrolled round the facades: reconstructed within 1 m vs missed."""
    d = df[df.strategy.isin(MAIN)]
    runs = list(zip(d.run, d.strategy))
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 4.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (rid, strat) in zip(axes, runs):
        z = np.load(f"{ROOT}/runs/{rid}/sfm/seed0/surface_eval.npz")
        P, obs, dist = z["P"], z["observable"], z["d_g2r"]
        got = obs & (dist < 1.0)
        _style(ax)
        ax.grid(False)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlim(-185, 185)
        ax.tick_params(labelsize=7)
        if ax is not axes[0]:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Height (m)", fontsize=8)
        ax.set_xlabel("Azimuth (°)", fontsize=8)
        c = P[:, :2].mean(0)
        az = np.degrees(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
        ax.scatter(az[~obs], P[~obs, 2], s=0.3, color="#e9e8e4", rasterized=True)
        ax.scatter(az[obs & ~got], P[obs & ~got, 2], s=0.3, color=MUTED, rasterized=True)
        ax.scatter(az[got], P[got, 2], s=0.3, color=COLOR[strat], rasterized=True)
        ax.set_title(f"{LABEL[strat]}\n{got[obs].mean():.0%} of observable surface",
                     fontsize=8.5, color=INK)
    fig.suptitle("Where each flight reconstructed the building — facades unrolled by azimuth "
                 "about the centre. Colour: within 1 m · grey: missed · pale: not observable "
                 "by any flight", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    _save(fig, "coverage_maps.png")


def _stats(run, seed=0):
    path = f"{ROOT}/runs/{run}/sfm/seed{seed}/sfm_stats.json"
    return json.load(open(path)) if os.path.exists(path) else None


def lighting_figure(run="run_002", ablation="run_012"):
    on = _stats(run)
    if on is None:
        return
    after = list(on["features_per_image"].values())
    off = _stats(ablation)
    if off is not None:
        before = list(off["features_per_image"].values())
        regs = [s["registered"] for s in (_stats(ablation, k) for k in range(3)) if s]
        lab_b = "Sun only (ablation, run_012)"
        txt_b = f"sun only: {min(regs)}–{max(regs)}/24 registered" if min(regs) != max(regs) \
            else f"sun only: {regs[0]}/24 registered"
    else:
        before, lab_b, txt_b = KAGGLE_V1_FEATURES, "Sun only (Kaggle, first attempt)", \
            "sun only: SfM could not start"
    fig, axes = _fig(7.8, 3.8)
    ax = axes[0]
    x = np.arange(24) * 15
    ax.plot(x, before, "-o", lw=2, ms=4, color="#eb6834")
    ax.plot(x, after, "-o", lw=2, ms=4, color="#2a78d6")
    k = int(np.argmin(before))
    ax.annotate(txt_b, (x[k], before[k]),
                xytext=(10, -16), textcoords="offset points", fontsize=8.5, color=INK)
    ax.annotate(f"sun + sky light: {on['registered']}/24 registered", (x[k], after[k]),
                xytext=(10, 16), textcoords="offset points", fontsize=8.5, color=INK)
    ax.legend([lab_b, "Sun + sky light"], frameon=False,
              fontsize=8.5, loc="upper center", ncol=2)
    ax.set_xlabel("Camera azimuth around the single orbit (degrees)")
    ax.set_ylabel("SIFT features per image")
    ax.set_title("The rear elevation is featureless; without sky light it is unmatchable",
                 loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, max(max(after), max(before)) * 1.25)
    fig.tight_layout()
    _save(fig, "lighting_features.png")


def table(df) -> str:
    cols = [("strategy", "Strategy"), ("n_images", "Images"), ("flight_time_s", "Flight (s)"),
            ("path_length_m", "Path (m)"), ("cams_correct", "Correctly placed"), ("points", "3D points"),
            ("track_length", "Track len."), ("pose_pos_err_m", "Pose err. (m)"),
            ("acc_median_m", "Median err. (m)"), ("precision@0.30", "Precision@0.3"),
            ("recall@0.30", "Recall@0.3"), ("fscore@0.30", "F@0.3"),
            ("recall@1.00", "Recall@1.0"), ("pred_coverage_res", "Forecast")]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |",
             "|" + "|".join(["---"] + ["--:"] * (len(cols) - 1)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for k, _ in cols:
            v = r.get(k, np.nan)
            if k == "strategy":
                cells.append(LABEL.get(v, v))
            elif k in ("n_images", "points", "flight_time_s", "path_length_m"):
                cells.append(f"{v:,.0f}")
            elif k == "cams_correct":
                cells.append(f"{v:.0f}/{r.n_images:.0f}" if r.get("cams_correct_sd", 0) < 0.5
                             else f"{v:.1f}/{r.n_images:.0f}")
            elif k in ("fscore@0.30", "recall@1.00") and r.get(k + "_sd", 0) > 0.0005:
                cells.append(f"{v:.3f} ± {r[k + '_sd']:.3f}")
            else:
                cells.append(f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(f"{RES}/results.csv")
    df["order"] = df.strategy.map({s: i for i, s in enumerate(ALL)}).fillna(99)
    df = df.sort_values("order").reset_index(drop=True)
    abl = df[df.run.isin(ABLATION)]
    df = df[~df.run.isin(ABLATION)].reset_index(drop=True)
    scene = json.load(open(f"{ROOT}/data/scene.json"))
    quality_vs_time(df)
    planner_iterations(df)
    predicted_vs_measured(df)
    layouts(df, scene)
    coverage_maps(df)
    lighting_figure()
    md = table(df)
    if len(abl):
        ref = df[df.strategy == "orbit_single"]
        both = pd.concat([ref, abl]).copy()
        both["strategy"] = ["Single orbit, sun + sky light"] + [ABLATION[r] for r in abl.run]
        md += "\n\n**Illumination ablation** (same flight, lighting changed only)\n\n" + table(both)
    open(f"{RES}/results_table.md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
