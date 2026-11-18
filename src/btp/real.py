"""Real photographs: the same SfM pipeline on phone pictures of a real object.

    python -m btp.real --src "~/Downloads/real/orbit_single" --run-id real_orbit_single

Photos are resized (longest side 2000 px) into runs/<run-id>/images. The phone's
focal length comes from EXIF (35 mm-equivalent) and is refined with one radial
distortion term. There is no ground-truth mesh, so the outputs are SfM statistics,
a sparse cloud, and a scale-free shape check against tape measurements
(see measure()).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageOps

from btp.reconstruct import load_cameras, load_points, reconstruct

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
MAX_SIDE = 2000


def ingest(src: str, out: str, max_side: int = MAX_SIDE) -> dict:
    """Copy photos (sorted by name) into out/images as resized JPEGs; return the camera prior."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    src = os.path.expanduser(src)
    files = sorted(f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in EXTS)
    if not files:
        raise SystemExit(f"no photos in {src}")
    os.makedirs(f"{out}/images", exist_ok=True)
    f35s, sizes = [], set()
    for i, f in enumerate(files):
        im = Image.open(os.path.join(src, f))
        exif = im.getexif()
        f35 = exif.get_ifd(0x8769).get(0xA405) or exif.get(0xA405)   # FocalLengthIn35mmFilm
        if f35:
            f35s.append(float(f35))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        sizes.add(im.size)
        im.save(f"{out}/images/img_{i:03d}.jpg", quality=95)
    if len(sizes) > 1:
        print(f"warning: mixed image sizes {sizes}; keep one orientation and zoom level")
    w, h = max(sizes)
    # 35 mm-equivalent focal is defined on the 36 mm side of a 36x24 frame (diagonal match);
    # use the long image side. Without EXIF, COLMAP's usual 1.2 x long side guess.
    f35 = float(np.median(f35s)) if f35s else None
    f_px = f35 / 36.0 * max(w, h) if f35 else 1.2 * max(w, h)
    prior = {"f_px": f_px, "cx": w / 2, "cy": h / 2, "width": w, "height": h,
             "focal_35mm": f35, "n_photos": len(files),
             "source_files": files}
    json.dump(prior, open(f"{out}/camera_prior.json", "w"), indent=2)
    return prior


def run(src: str, run_id: str, seeds=(0, 1, 2)) -> dict:
    out = os.path.join(ROOT, "runs", run_id)
    prior = ingest(src, out)
    print(f"[{run_id}] {prior['n_photos']} photos, focal prior {prior['f_px']:.0f} px "
          f"({'EXIF' if prior['focal_35mm'] else 'guess'})", flush=True)
    rows = []
    for seed in seeds:
        s = reconstruct(out, prior, focal="prior", seed=seed)
        rows.append(s)
        print(f"[{run_id}] seed {seed}: registered {s['registered']}/{prior['n_photos']}, "
              f"points {s.get('points')}, reproj {s.get('reproj_err_px', float('nan')):.2f} px",
              flush=True)
    keys = ["registered", "points", "track_length", "reproj_err_px", "focal_estimated_px"]
    summary = {"run": run_id, "n_photos": prior["n_photos"],
               **{k: float(np.mean([r.get(k, np.nan) for r in rows])) for k in keys},
               "registered_min": int(min(r["registered"] for r in rows)),
               "n_models_max": int(max(r["n_models"] for r in rows))}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
    return summary


def measure(run_id: str, seed: int = 0) -> dict:
    """Extent of the sparse cloud along its principal axes, in reconstruction units.

    Real SfM has no metric scale, so compare *ratios* (e.g. height / width) with
    tape measurements: a ratio error is a scale-free accuracy check. Outlier points
    (beyond the 2nd-98th percentile) are trimmed.
    """
    pts = load_points(os.path.join(ROOT, "runs", run_id, "sfm", f"seed{seed}", "model"))
    c = np.median(pts, 0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    proj = (pts - c) @ Vt.T
    ext = np.percentile(proj, 98, 0) - np.percentile(proj, 2, 0)
    return {"extent_principal": ext.tolist(), "ratios": (ext / ext[0]).tolist(),
            "n_points": len(pts)}


def plot(run_id: str, seed: int = 0) -> str:
    """Top and side view of the sparse cloud with the recovered camera path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model = os.path.join(ROOT, "runs", run_id, "sfm", f"seed{seed}", "model")
    pts = load_points(model)
    cams = load_cameras(model)
    C = np.array([cams[k][1] for k in sorted(cams)])
    c = np.median(pts, 0)
    _, _, Vt = np.linalg.svd(np.vstack([pts, C]) - c, full_matrices=False)
    # the camera path is roughly planar; its normal (smallest axis) is "up"
    _, _, Vc = np.linalg.svd(C - C.mean(0), full_matrices=False)
    up = Vc[2]
    a = Vt[0] - up * (Vt[0] @ up)
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    P = np.c_[(pts - c) @ a, (pts - c) @ b, (pts - c) @ up]
    Q = np.c_[(C - c) @ a, (C - c) @ b, (C - c) @ up]
    lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
    keep = np.all((P >= lo) & (P <= hi), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor="#fcfcfb")
    for ax, (i, j, t) in zip(axes, [(0, 1, "Top view"), (0, 2, "Side view")]):
        ax.set_facecolor("#fcfcfb")
        ax.scatter(P[keep, i], P[keep, j], s=0.4, color="#52514e", alpha=0.6, rasterized=True)
        ax.plot(Q[:, i], Q[:, j], "-o", ms=3, lw=1, color="#2a78d6")
        ax.set_aspect("equal")
        ax.set_title(t, loc="left", fontsize=10)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{run_id}: sparse reconstruction and recovered camera positions "
                 f"({len(C)} cameras, {len(pts):,} points)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    path = os.path.join(ROOT, "results", "figures", f"{run_id}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of photos")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args()
    s = run(a.src, a.run_id, a.seeds)
    print(json.dumps(s, indent=1))
    if s["registered"] >= 3:
        print(json.dumps(measure(a.run_id), indent=1))
        print("figure:", plot(a.run_id))
