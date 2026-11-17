"""Blender side of the pipeline: build the scene, render a flight, export ground truth.

Run as its own process (bpy keeps global state):
    python -m btp.render --blend MODEL.blend --poses poses.json --out runs/run_x
    python -m btp.render --blend MODEL.blend --scene-info out.json      # bbox only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import bpy
import bmesh
import mathutils
import numpy as np

from btp.geometry import pose_matrix

# Objects that make up the building in Blender_004.blend. Building_04 is an
# EMPTY parent: deleting it rescales its children, so it must be kept.
KEEP = {"Building_Base", "Details", "Glass", "Building_04"}
GROUND_RADIUS_MULT = 2.8          # ground plane sized from run_002's orbit; fixed for all runs


def load_scene(blend_path: str) -> dict:
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    for o in list(bpy.data.objects):
        if o.name not in KEEP:
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.context.view_layer.update()
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    pts = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    ctr = [(a + b) / 2 for a, b in zip(lo, hi)]
    struct_r = math.hypot(hi[0] - lo[0], hi[1] - lo[1]) / 2
    return {"bbox_lo": lo, "bbox_hi": hi, "centre": ctr,
            "struct_radius": struct_r, "height": hi[2] - lo[2]}


def add_environment(info: dict) -> None:
    """Ground plane, sun and sky — identical to the final Kaggle run_002 setup."""
    ctr, lo, hi = info["centre"], info["bbox_lo"], info["bbox_hi"]
    R = info["struct_radius"] * GROUND_RADIUS_MULT

    bpy.ops.mesh.primitive_plane_add(size=R * 8, location=(ctr[0], ctr[1], lo[2]))
    gnd = bpy.context.object
    gnd.name = "GroundPlane"
    mat = bpy.data.materials.new("ground")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 40.0
    nz.inputs["Detail"].default_value = 12.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.65
    ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.05, 1)
    ramp.color_ramp.elements[1].color = (0.35, 0.33, 0.30, 1)
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.95
    gnd.data.materials.append(mat)

    # Direct sun + diffuse sky. Sky light is what keeps the shadowed facade
    # textured; without it SfM lost the far side of the orbit (Kaggle run_002 v1).
    bpy.ops.object.light_add(type="SUN", location=(ctr[0] + R, ctr[1] - R, hi[2] + R))
    sun = bpy.context.object
    sun.data.energy = 3
    sun.data.angle = 0.1
    sun.rotation_euler = (math.radians(50), 0, math.radians(135))

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.35, 0.40, 0.48, 1.0)
    bg.inputs["Strength"].default_value = 1.2


def export_gt_mesh(path: str) -> tuple[int, int]:
    """Triangulated building mesh in world coordinates (ground plane excluded)."""
    verts, faces = [], []
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name == "GroundPlane":
            continue
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.transform(o.matrix_world)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.index_update()
        base = len(verts)
        verts += [tuple(v.co) for v in bm.verts]
        faces += [tuple(base + v.index for v in f.verts) for f in bm.faces]
        bm.free()
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                f"element face {len(faces)}\n"
                "property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    return len(verts), len(faces)


def render_flight(poses: list[dict], out: str, cam_cfg: dict, only: list[int] | None = None) -> list[dict]:
    os.makedirs(f"{out}/images", exist_ok=True)
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.data.lens = cam_cfg["lens_mm"]
    cam.data.sensor_width = cam_cfg["sensor_mm"]
    cam.data.clip_end = 5000
    sc = bpy.context.scene
    sc.camera = cam
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = cam_cfg["samples"]
    sc.render.resolution_x = cam_cfg["res_x"]
    sc.render.resolution_y = cam_cfg["res_y"]
    sc.render.resolution_percentage = 100
    # The model's 4K UDIM textures exceed this machine's 7 GB at full size.
    # A 2K cap is still finer than the render can resolve at this resolution.
    sc.render.use_simplify = True
    sc.cycles.texture_limit_render = cam_cfg.get("texture_limit", "2048")

    records = []
    idx = only if only is not None else range(len(poses))
    for k, i in enumerate(idx):
        p = poses[i]
        m = pose_matrix(p["position"], p["target"])
        cam.matrix_world = mathutils.Matrix(m.tolist())
        bpy.context.view_layer.update()
        name = f"img_{i:03d}.png"
        sc.render.filepath = f"{out}/images/{name}"
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        records.append({"file": name, "position": list(p["position"]),
                        "target": list(p["target"]),
                        "matrix_world": [list(r) for r in cam.matrix_world]})
        print(f"RENDERED {k + 1}/{len(idx)} {name} in {time.time() - t0:.1f}s", flush=True)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--poses")
    ap.add_argument("--out")
    ap.add_argument("--only", type=int, nargs="*", help="render only these pose indices")
    ap.add_argument("--scene-info", help="write scene bbox json and exit")
    a = ap.parse_args()

    info = load_scene(a.blend)
    if a.scene_info:
        os.makedirs(os.path.dirname(a.scene_info) or ".", exist_ok=True)
        export_gt_mesh(os.path.join(os.path.dirname(a.scene_info) or ".", "gt_mesh.ply"))
        json.dump(info, open(a.scene_info, "w"), indent=2)
        print("SCENE", json.dumps(info))
        return

    spec = json.load(open(a.poses))
    add_environment(info)
    os.makedirs(f"{a.out}/exports", exist_ok=True)
    nv, nf = export_gt_mesh(f"{a.out}/exports/gt_mesh.ply")
    print(f"GT MESH {nv} verts {nf} tris", flush=True)
    records = render_flight(spec["poses"], a.out, spec["camera"], a.only)
    if a.only is None:
        json.dump({**{k: v for k, v in spec.items() if k != "poses"},
                   "scene": info, "poses": records},
                  open(f"{a.out}/exports/rendered_poses.json", "w"), indent=2)
    print("RENDER DONE", flush=True)


if __name__ == "__main__":
    main()
