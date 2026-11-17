"""Coverage planner: choose viewpoints from the structure's geometry, then route them.

1. Candidates: camera positions on rings at several heights and two standoffs,
   plus rings above the roof looking down. Side views aim horizontally at their
   own height; overhead views aim at the roof centre.
2. Score: ray-cast each candidate against surface samples (visibility.py).
   A view is *useful* to a sample if it sees it at incidence < MAX_INC.
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
                    poses.append({"position": pos, "target": (ctr[0], ctr[1], z)})
                poses.append({"position": pos, "target": tuple(ctr)})
    # close-in ring above the roof, looking down at the roof
    R = r0 * 1.0
    for k in range(n_az // 2):
        a = 2 * math.pi * k / (n_az // 2)
        pos = (ctr[0] + R * math.cos(a), ctr[1] + R * math.sin(a), hi[2] + 0.35 * H)
        poses.append({"position": pos, "target": (ctr[0], ctr[1], hi[2] - 0.1 * H)})
    return [p for p in poses if not _inside_box(p["position"], scene, SAFETY_M)]


# ---------------------------------------------------------------- selection
def select(V, I, dirs, n_views: int, min_overlap=MIN_OVERLAP, start: int | None = None):
    """Greedy submodular selection.

    V: (C, S) bool visibility, I: (C, S) incidence degrees,
    dirs: (C, S, 3) unit viewing directions candidate -> sample.
    """
    C, S = V.shape
    useful = V & (I < MAX_INC)
    cos_sep = np.cos(np.radians(MIN_SEP))
    count = np.zeros(S, int)                         # useful views counted so far
    kept = np.zeros((S, TARGET_VIEWS, 3))            # directions of counted views
    seen_by = np.zeros((0, S), bool)
    chosen: list[int] = []
    for step in range(n_views):
        best, best_gain = -1, -1.0
        # candidate gain = sum over samples of the next increment, if the view adds parallax
        for c in range(C):
            if c in chosen:
                continue
            u = useful[c] & (count < TARGET_VIEWS)
            if not u.any():
                continue
            if chosen:
                ov = (seen_by & V[c]).sum(1).max() / max(V[c].sum(), 1)
                if ov < min_overlap:
                    continue
            idx = np.where(u)[0]
            d = dirs[c, idx]
            sep_ok = np.ones(len(idx), bool)
            for j in range(TARGET_VIEWS):
                has = count[idx] > j
                sep_ok &= ~has | (np.einsum("ij,ij->i", kept[idx, j], d) < cos_sep)
            g = GAIN[count[idx[sep_ok]]].sum()
            if g > best_gain:
                best, best_gain = c, g
        if start is not None and step == 0:
            best = start
        if best < 0:
            break
        chosen.append(best)
        seen_by = np.vstack([seen_by, V[best][None]])
        idx = np.where(useful[best] & (count < TARGET_VIEWS))[0]
        d = dirs[best, idx]
        ok = np.ones(len(idx), bool)
        for j in range(TARGET_VIEWS):
            has = count[idx] > j
            ok &= ~has | (np.einsum("ij,ij->i", kept[idx, j], d) < cos_sep)
        idx = idx[ok]
        kept[idx, count[idx]] = d[ok]
        count[idx] += 1
    return chosen, count


class _Coverage:
    """Tracks per-sample useful-view counts and gives each candidate's marginal gain."""

    def __init__(self, V, I, dirs):
        self.V, self.useful, self.dirs = V, V & (I < MAX_INC), dirs
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
                  min_overlap=MIN_OVERLAP, max_views: int = 60, mode: str = "ratio"):
    """Cost-aware greedy (orienteering): add the view with the best gain per
    extra second of flight, inserted at its cheapest point in the current tour,
    until the flight-time budget is spent. L is the (C, C) leg-length matrix."""
    C = V.shape[0]
    cov = _Coverage(V, I, dirs)
    tour: list[int] = []
    tour_len = 0.0
    seen_by = np.zeros((0, V.shape[1]), bool)
    while len(tour) < max_views:
        best, best_ratio, best_ins = -1, 0.0, None
        for c in range(C):
            if c in tour:
                continue
            if tour:
                ov = (seen_by & V[c]).sum(1).max() / max(V[c].sum(), 1)
                if ov < min_overlap:
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
        tour_len += best_ins[1]
        cov.add(best)
        seen_by = np.vstack([seen_by, V[best][None]])
    return tour, cov.count


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
def plan(scene: dict, n_views: int, cam: dict, mesh_path: str, n_samples: int = 8000,
         seed: int = 0, min_overlap: float = MIN_OVERLAP, return_details: bool = False):
    mesh = load_mesh(mesh_path)
    P, N, _ = sample_surface(mesh, n_samples, seed=seed)
    cand = candidates(scene)
    V, D, I = visibility_matrix(mesh, P, N, cand, cam)
    pos = np.array([c["position"] for c in cand])
    dirs = P[None] - pos[:, None]
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    chosen, count = select(V, I, dirs, n_views, min_overlap=min_overlap)
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
                return_details: bool = False):
    mesh = load_mesh(mesh_path)
    P, N, _ = sample_surface(mesh, n_samples, seed=seed)
    cand = candidates(scene)
    V, D, I = visibility_matrix(mesh, P, N, cand, cam)
    pos = np.array([c["position"] for c in cand])
    dirs = P[None] - pos[:, None]
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    L = leg_matrix(pos, scene)
    # Budgeted submodular maximisation: run the gain-per-second greedy and the
    # plain-gain greedy, keep the better (the standard remedy for ratio-greedy's
    # weakness on cheap low-value items; Khuller, Moss & Naor 1999).
    best = None
    for mode in ("ratio", "gain"):
        tour, count = select_budget(V, I, dirs, L, budget_s, speed, dwell, mode=mode)
        score = sum(GAIN[:k].sum() for k in np.minimum(count, TARGET_VIEWS))
        if best is None or score > best[0]:
            best = (score, tour, count, mode)
    _, tour, count, mode_used = best
    picked = [cand[i] for i in tour]
    order, length = route([p["position"] for p in picked], scene)   # polish the order
    poses = [picked[i] for i in order]
    if return_details:
        return poses, {"n_candidates": len(cand), "n_views": len(poses), "mode": mode_used,
                       "planned_cov3": float((count >= TARGET_VIEWS).mean()),
                       "tour_length_m": length,
                       "flight_time_s": length / speed + dwell * len(poses)}
    return poses


def plan_from_spec(strategy: str, scene: dict, n: int, cam: dict):
    """'planner' -> best 24 views; 'planner_budget<T>' -> flight-time budget of T seconds."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mesh = os.path.join(root, "data/gt_mesh.ply")
    if strategy.startswith("planner_budget"):
        return plan_budget(scene, float(strategy[len("planner_budget"):]), cam, mesh)
    return plan(scene, n, cam, mesh)


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
