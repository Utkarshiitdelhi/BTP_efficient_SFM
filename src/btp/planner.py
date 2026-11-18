"""Coverage planner: choose viewpoints from the structure's geometry, then route them.

1. Candidates: camera positions on rings at several heights and two standoffs,
   plus rings above the roof looking down. Side views aim horizontally at their
   own height; overhead views aim at the roof centre.
2. Score: ray-cast each candidate against surface samples (visibility.py).
   A view is *useful* to a sample if it sees it at incidence < MAX_INC.
   v2 ("planner_res") also requires the view to RESOLVE the sample: effective
   ground sampling distance  distance / f_px / cos(incidence)  below
   MAX_EFF_GSD. v1 counted grazing, distant views as coverage; the baselines
   showed SfM cannot use them (the high orbit's walls), see README.
3. Select greedily. Each sample's utility saturates at 3 useful views from
   directions >= MIN_SEP apart, so near-duplicate views earn nothing. That
   is a concave function of a coverage count - submodular - so greedy
   selection is within (1 - 1/e) of the best possible set of that size.
   Each new view must also overlap enough with views already chosen, or SfM
   could not register it.
4. Route: shortest closed tour through the chosen views (OR-Tools TSP), with
   legs that would cut through the building replaced by a detour around it.
"""
from __future__ import annotations

import math

import numpy as np

from btp.visibility import load_mesh, sample_surface, visibility_matrix

MAX_INC = 70.0           # degrees; grazing views beyond this are not useful
MIN_SEP = 5.0            # degrees between two views of the same point to add parallax
TARGET_VIEWS = 3         # views per surface point at which utility saturates
GAIN = np.array([0.40, 0.35, 0.25])   # utility of the 1st, 2nd, 3rd useful view
MIN_OVERLAP = 0.25       # fraction of a new view's points already seen by one chosen view
SAFETY_M = 3.0           # clearance around the building's bounding box
MAX_EFF_GSD = 0.12       # v2: metres per pixel on the surface for a view to count
MAX_LINK_DEG = 35.0      # v2: a new view must overlap a chosen view within this angle


# ---------------------------------------------------------------- candidates
def candidates(scene: dict, radius_mults=(2.8, 2.0), n_az=24) -> list[dict]:
    """Candidate poses. Every position gets two aims: level (looks at its own
    height, a facade view) and at the structure's centre (an oblique view that
    also takes in a second facade or the roof)."""
    ctr, lo, hi, H = scene["centre"], scene["bbox_lo"], scene["bbox_hi"], scene["height"]
    r0 = scene["struct_radius"]
    poses = []
    for ri, rm in enumerate(radius_mults):
        R = r0 * rm
        for hf in (0.10, 0.30, 0.50, 0.70, 0.90, 1.25):
            z = lo[2] + hf * H
            for k in range(n_az):
                a = 2 * math.pi * (k + 0.5 * ri) / n_az
                pos = (ctr[0] + R * math.cos(a), ctr[1] + R * math.sin(a), z)
                if hf <= 1.0:
                    poses.append({"position": pos, "target": (ctr[0], ctr[1], z),
                                  "ring": f"r{rm}_h{hf}_level"})
                poses.append({"position": pos, "target": tuple(ctr), "ring": f"r{rm}_h{hf}_centre"})
    # close-in ring above the roof, looking down at the roof
    R = r0 * 1.0
    for k in range(n_az // 2):
        a = 2 * math.pi * k / (n_az // 2)
        pos = (ctr[0] + R * math.cos(a), ctr[1] + R * math.sin(a), hi[2] + 0.35 * H)
        poses.append({"position": pos, "target": (ctr[0], ctr[1], hi[2] - 0.1 * H),
                      "ring": "roof_down"})
    return [p for p in poses if not _inside_box(p["position"], scene, SAFETY_M)]


# ---------------------------------------------------------------- selection
def select(V, I, dirs, n_views: int, min_overlap=MIN_OVERLAP, useful=None, axes=None,
           max_link_deg=None):
    """Greedy submodular selection of n_views (equal-image-count planner).

    V: (C, S) bool visibility, I: (C, S) incidence degrees,
    dirs: (C, S, 3) unit viewing directions candidate -> sample.
    """
    cov = _Coverage(V, I, dirs, useful)
    seen_by = np.zeros((0, V.shape[1]), bool)
    chosen: list[int] = []
    for _ in range(n_views):
        best, best_gain = -1, 0.0
        for c in range(V.shape[0]):
            if c in chosen or not _linked(c, chosen, V, seen_by, axes, min_overlap, max_link_deg):
                continue
            g = cov.gain(c)
            if g > best_gain:
                best, best_gain = c, g
        if best < 0:
            break
        chosen.append(best)
        cov.add(best)
        seen_by = np.vstack([seen_by, V[best][None]])
    return chosen, cov.count

def useful_mask(V, D, I, f_px: float, max_eff_gsd: float | None):
    """Views that see a sample usefully: facing it, and (v2) resolving it finely enough."""
    u = V & (I < MAX_INC)
    if max_eff_gsd is not None:
        eff = D / f_px / np.cos(np.radians(np.minimum(I, 89.0)))
        u &= eff < max_eff_gsd
    return u


def _linked(c, chosen, V, seen_by, axes, min_overlap, max_link_deg):
    """Can SfM register view c? It must share enough surface with one chosen view
    and (v2) not differ from that view's direction by more than max_link_deg."""
    if not chosen:
        return True
    ov = (seen_by & V[c]).sum(1) / max(V[c].sum(), 1)
    ok = ov >= min_overlap
    if max_link_deg is not None:
        ang = np.degrees(np.arccos(np.clip(axes[chosen] @ axes[c], -1, 1)))
        ok &= ang <= max_link_deg
    return bool(ok.any())


class _Coverage:
    """Tracks per-sample useful-view counts and gives each candidate's marginal gain."""

    def __init__(self, V, I, dirs, useful=None):
        self.V, self.dirs = V, dirs
        self.useful = useful if useful is not None else V & (I < MAX_INC)
        S = V.shape[1]
        self.count = np.zeros(S, int)
        self.kept = np.zeros((S, TARGET_VIEWS, 3))
        self.cos_sep = np.cos(np.radians(MIN_SEP))

    def _adds(self, c):
        idx = np.where(self.useful[c] & (self.count < TARGET_VIEWS))[0]
        d = self.dirs[c, idx]
        ok = np.ones(len(idx), bool)
        for j in range(TARGET_VIEWS):
            has = self.count[idx] > j
            ok &= ~has | (np.einsum("ij,ij->i", self.kept[idx, j], d) < self.cos_sep)
        return idx[ok], d[ok]

    def gain(self, c) -> float:
        idx, _ = self._adds(c)
        return float(GAIN[self.count[idx]].sum())

    def add(self, c) -> None:
        idx, d = self._adds(c)
        self.kept[idx, self.count[idx]] = d
        self.count[idx] += 1


def select_budget(V, I, dirs, L, budget_s: float, speed: float, dwell: float,
                  min_overlap=MIN_OVERLAP, max_views: int = 60, mode: str = "ratio",
                  useful=None, axes=None, max_link_deg=None):
    """Cost-aware greedy (orienteering): add the view with the best gain per
    extra second of flight, inserted at its cheapest point in the current tour,
    until the flight-time budget is spent. L is the (C, C) leg-length matrix."""
    C = V.shape[0]
    cov = _Coverage(V, I, dirs, useful)
    tour: list[int] = []                 # route order
    added: list[int] = []                # selection order; rows of seen_by follow this
    tour_len = 0.0
    seen_by = np.zeros((0, V.shape[1]), bool)
    while len(tour) < max_views:
        best, best_ratio, best_ins = -1, 0.0, None
        for c in range(C):
            if c in tour:
                continue
            if tour and not _linked(c, added, V, seen_by, axes, min_overlap, max_link_deg):
                continue
            g = cov.gain(c)
            if g <= 0:
                continue
            if not tour:
                dlen, pos = 0.0, 0
            elif len(tour) == 1:
                dlen, pos = 2 * L[tour[0], c], 1
            else:
                a = np.array(tour)
                b = np.roll(a, -1)
                ins = L[a, c] + L[c, b] - L[a, b]
                pos = int(np.argmin(ins)) + 1
                dlen = float(ins[pos - 1])
            new_time = (tour_len + dlen) / speed + dwell * (len(tour) + 1)
            if new_time > budget_s:
                continue
            ratio = g / (dlen / speed + dwell) if mode == "ratio" else g
            if ratio > best_ratio:
                best, best_ratio, best_ins = c, ratio, (pos, dlen)
        if best < 0:
            break
        tour.insert(best_ins[0], best)
        tour, tour_len = two_opt(tour, L)     # keep the route short so the budget is not misjudged
        added.append(best)
        cov.add(best)
        seen_by = np.vstack([seen_by, V[best][None]])
    return tour, cov.count


def coverage_value(counts: np.ndarray) -> float:
    """Concave utility of per-sample useful-view counts (saturates at TARGET_VIEWS)."""
    cum = np.concatenate([[0.0], np.cumsum(GAIN)])
    return float(cum[np.minimum(counts, TARGET_VIEWS)].sum())


def local_search(tour: list[int], useful, V, L, budget_s, speed, dwell, axes=None,
                 min_overlap=MIN_OVERLAP, max_link_deg=None, max_passes: int = 6):
    """Swap / add / drop moves that raise coverage_value while staying within budget.

    Each accepted move is re-routed with 2-opt. Moves must keep every view
    linked to the rest (overlap and, for v2, direction), so SfM can chain them.
    """
    U = useful.astype(np.int16)
    tour, length = two_opt(tour, L)
    counts = U[tour].sum(0)
    value = coverage_value(counts)
    C = U.shape[0]

    def fits(t, ln):
        return ln / speed + dwell * len(t) <= budget_s + 1e-9

    def linked_set(t):
        if len(t) < 2:
            return True
        for c in t:
            others = [o for o in t if o != c]
            if not _linked(c, others, V, V[others], axes, min_overlap, max_link_deg):
                return False
        return True

    for _ in range(max_passes):
        improved = False
        # add moves (cheapest insertion), then swaps
        for c in np.argsort(-U.sum(1)):
            c = int(c)
            if c in tour:
                continue
            new_val = coverage_value(counts + U[c])
            if new_val <= value + 1e-9:
                continue
            t2, l2 = two_opt(tour + [c], L)
            if fits(t2, l2) and _linked(c, tour, V, V[tour], axes, min_overlap, max_link_deg):
                tour, length, counts, value, improved = t2, l2, counts + U[c], new_val, True
        for out in list(tour):
            if out not in tour:
                continue
            base = counts - U[out]
            rest = [t for t in tour if t != out]
            gains = np.array([coverage_value(base + U[c]) if c not in tour else -1.0
                              for c in range(C)])
            for c in np.argsort(-gains)[:40]:
                c = int(c)
                if gains[c] <= value + 1e-9:
                    break
                t2, l2 = two_opt(rest + [c], L)
                if fits(t2, l2) and linked_set(t2):
                    tour, length, counts, value, improved = t2, l2, base + U[c], gains[c], True
                    break
        if not improved:
            break
    return tour, length, counts


def two_opt(tour: list[int], L: np.ndarray) -> tuple[list[int], float]:
    """2-opt improvement of a closed tour on leg matrix L; returns (tour, length)."""
    t = list(tour)
    n = len(t)
    improved = n > 3
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                if L[a, c] + L[b, d] < L[a, b] + L[c, d] - 1e-9:
                    t[i + 1:j + 1] = reversed(t[i + 1:j + 1])
                    improved = True
    length = float(sum(L[t[k], t[(k + 1) % n]] for k in range(n))) if n > 1 else 0.0
    return t, length


def leg_matrix(positions, scene) -> np.ndarray:
    n = len(positions)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            L[i, j] = L[j, i] = leg_cost(positions[i], positions[j], scene)
    return L


# ---------------------------------------------------------------- routing
def leg_cost(a, b, scene: dict) -> float:
    """Straight-line distance, or a detour round the building if the leg would cut through it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not _segment_hits_box(a, b, scene, SAFETY_M):
        return float(np.linalg.norm(b - a))
    ctr = np.asarray(scene["centre"][:2])
    ra, rb = np.linalg.norm(a[:2] - ctr), np.linalg.norm(b[:2] - ctr)
    da = math.atan2(a[1] - ctr[1], a[0] - ctr[0])
    db = math.atan2(b[1] - ctr[1], b[0] - ctr[0])
    dth = abs((db - da + math.pi) % (2 * math.pi) - math.pi)
    horiz = max(ra, rb) * dth + abs(ra - rb)
    return float(math.hypot(horiz, b[2] - a[2]))


def route(positions, scene: dict) -> tuple[list[int], float]:
    """Shortest closed tour (OR-Tools), returns visiting order and length in metres."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(positions)
    M = [[int(round(100 * leg_cost(positions[i], positions[j], scene))) for j in range(n)]
         for i in range(n)]
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    rt = pywrapcp.RoutingModel(mgr)
    cb = rt.RegisterTransitCallback(lambda i, j: M[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    prm.time_limit.seconds = 10
    sol = rt.SolveWithParameters(prm)
    order, i = [], rt.Start(0)
    while not rt.IsEnd(i):
        order.append(mgr.IndexToNode(i))
        i = sol.Value(rt.NextVar(i))
    return order, tour_length([positions[k] for k in order], scene)


def tour_length(positions, scene: dict) -> float:
    """Closed tour length with building detours (the same rule for every strategy)."""
    p = list(positions)
    return float(sum(leg_cost(p[i], p[(i + 1) % len(p)], scene) for i in range(len(p))))


# ---------------------------------------------------------------- entry point
def _setup(scene, cam, mesh_path, n_samples, seed, max_eff_gsd):
    from btp.geometry import intrinsics
    mesh = load_mesh(mesh_path)
    P, N, _ = sample_surface(mesh, n_samples, seed=seed)
    cand = candidates(scene)
    V, D, I = visibility_matrix(mesh, P, N, cand, cam)
    pos = np.array([c["position"] for c in cand])
    dirs = P[None] - pos[:, None]
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    axes = np.array([np.subtract(c["target"], c["position"]) for c in cand], float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    f_px = intrinsics(cam["lens_mm"], cam["sensor_mm"], cam["res_x"], cam["res_y"])[0]
    useful = useful_mask(V, D, I, f_px, max_eff_gsd)
    return cand, pos, V, I, dirs, axes, useful


def plan(scene: dict, n_views: int, cam: dict, mesh_path: str, n_samples: int = 8000,
         seed: int = 0, min_overlap: float = MIN_OVERLAP, max_eff_gsd: float | None = None,
         max_link_deg: float | None = None, return_details: bool = False):
    """Best n_views views, then the shortest tour through them."""
    cand, pos, V, I, dirs, axes, useful = _setup(scene, cam, mesh_path, n_samples, seed,
                                                 max_eff_gsd)
    chosen, count = select(V, I, dirs, n_views, min_overlap=min_overlap, useful=useful,
                           axes=axes, max_link_deg=max_link_deg)
    picked = [cand[i] for i in chosen]
    order, length = route([p["position"] for p in picked], scene)
    poses = [picked[i] for i in order]
    if return_details:
        return poses, {"n_candidates": len(cand), "chosen": [chosen[i] for i in order],
                       "planned_cov3": float((count >= TARGET_VIEWS).mean()),
                       "tour_length_m": length}
    return poses


def plan_budget(scene: dict, budget_s: float, cam: dict, mesh_path: str,
                speed: float = 3.0, dwell: float = 2.0, n_samples: int = 8000, seed: int = 0,
                max_eff_gsd: float | None = None, max_link_deg: float | None = None,
                return_details: bool = False):
    """Most coverage within a flight-time budget (orienteering)."""
    cand, pos, V, I, dirs, axes, useful = _setup(scene, cam, mesh_path, n_samples, seed,
                                                 max_eff_gsd)
    L = leg_matrix(pos, scene)
    # Budgeted submodular maximisation: run the gain-per-second greedy and the
    # plain-gain greedy, keep the better (the standard remedy for ratio-greedy's
    # weakness on cheap low-value items; Khuller, Moss & Naor 1999).
    best = None
    for mode in ("ratio", "gain"):
        tour, count = select_budget(V, I, dirs, L, budget_s, speed, dwell, mode=mode,
                                    useful=useful, axes=axes, max_link_deg=max_link_deg)
        score = sum(GAIN[:k].sum() for k in np.minimum(count, TARGET_VIEWS))
        if best is None or score > best[0]:
            best = (score, tour, count, mode)
    _, tour, count, mode_used = best

    # Local search from two starts - the greedy tour, and the best whole ring of
    # candidates that fits the budget (a template start) - keeping the better.
    starts = {"greedy": tour}
    rings: dict[str, list[int]] = {}
    for i, c in enumerate(cand):
        rings.setdefault(c.get("ring", ""), []).append(i)
    U16 = useful.astype(np.int16)
    best_ring, best_val = None, -1.0
    for name, idx in rings.items():
        t, ln = two_opt(idx, L)
        if ln / speed + dwell * len(t) <= budget_s:
            v = coverage_value(U16[t].sum(0))
            if v > best_val:
                best_ring, best_val = name, v
    if best_ring is not None:
        starts["ring:" + best_ring] = rings[best_ring]
    results = {}
    for name, t0 in starts.items():
        t, ln, cnt = local_search(list(t0), useful, V, L, budget_s, speed, dwell, axes=axes,
                                  max_link_deg=max_link_deg)
        results[name] = (coverage_value(cnt), t, cnt)
    start_used = max(results, key=lambda k: results[k][0])
    _, tour, count = results[start_used]
    picked = [cand[i] for i in tour]
    order, length = route([p["position"] for p in picked], scene)   # polish the order
    poses = [picked[i] for i in order]
    if return_details:
        return poses, {"n_candidates": len(cand), "n_views": len(poses), "mode": mode_used,
                       "start": start_used,
                       "start_values": {k: round(v[0], 1) for k, v in results.items()},
                       "planned_cov3": float((count >= TARGET_VIEWS).mean()),
                       "tour_length_m": length,
                       "flight_time_s": length / speed + dwell * len(poses)}
    return poses


def plan_from_spec(strategy: str, scene: dict, n: int, cam: dict):
    """Strategy names:
    planner               v1, best n views (visibility-only usefulness)
    planner_budget<T>     v1, flight-time budget of T seconds
    planner_res           v2, resolution-aware usefulness + angular link rule
    planner_res_budget<T> v2 with a flight-time budget
    """
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mesh = os.path.join(root, "data/gt_mesh.ply")
    v2 = strategy.startswith("planner_res")
    kw = {"max_eff_gsd": MAX_EFF_GSD, "max_link_deg": MAX_LINK_DEG} if v2 else {}
    rest = strategy[len("planner_res"):] if v2 else strategy[len("planner"):]
    if rest.startswith("_budget"):
        return plan_budget(scene, float(rest[len("_budget"):]), cam, mesh, **kw)
    return plan(scene, n, cam, mesh, **kw)


# ---------------------------------------------------------------- helpers
def _box(scene, margin):
    lo = np.asarray(scene["bbox_lo"]) - margin
    hi = np.asarray(scene["bbox_hi"]) + margin
    lo[2] = -1e9                                       # ground: no clearance below
    return lo, hi


def _inside_box(p, scene, margin):
    lo, hi = _box(scene, margin)
    p = np.asarray(p)
    return bool(np.all(p > lo) and np.all(p < hi))


def _segment_hits_box(a, b, scene, margin) -> bool:
    """Slab test: does segment a->b pass through the building's bounding box?"""
    lo, hi = _box(scene, margin)
    d = b - a
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if a[k] < lo[k] or a[k] > hi[k]:
                return False
            continue
        ta, tb = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:
            return False
    return True
