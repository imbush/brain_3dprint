"""Boolean-subtract cortical-area labels on each area's PIAL surface
(outside-facing centroid, not just dorsal). Uses the watertight
cortex from voxelize+remesh as the substrate so manifold3d boolean
ops work.

For each of the 43 isocortex areas:
1. Find vertices in this area on the OUTSIDE-facing side.
2. Compute mean outward normal → label face direction.
3. Pick the centroid vertex of the area's pial face.
4. Build vector text mesh via TextPath + extrude, oriented to the
   local tangent frame.
5. Concatenate all 43+ label meshes (bilateral = ~80).
6. One boolean diff subtract on the watertight cortex.
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.spatial import cKDTree
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
import pyvista as pv

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
WT_STL = OUT_DIR / 'stl' / 'other' / '688_isocortex_watertight.stl'

root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE
print(f'TARGET_MM = {TARGET_MM:.2f} mm ({SCALE_FACTOR}x)')

cortex = trimesh.load(str(WT_STL), force='mesh')
cortex.apply_scale(SHARED_SCALE)
cortex.apply_translation(-root_ctr)
print(f'cortex (print mm): {len(cortex.vertices):,} verts, '
      f'{len(cortex.faces):,} faces, wt={cortex.is_watertight}, '
      f'vol={cortex.is_volume}')

RES_UM = 25
rspc = ReferenceSpaceCache(resolution=RES_UM,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
ann, _ = rspc.get_annotation_volume()
tree = rspc.get_structure_tree()

# True Isocortex = 315
ISO_ID = 315
def is_layer_acro(a):
    return (a.endswith('2/3') or a.endswith('6a') or a.endswith('6b')
             or (len(a) >= 2 and a[-1].isdigit()))
iso_kid_set = set()
for sid in tree.descendant_ids([ISO_ID])[0]:
    if sid == ISO_ID: continue
    kids = tree.child_ids([int(sid)])[0]
    if not kids: continue
    if all(is_layer_acro(tree.get_structures_by_id([int(k)])[0]['acronym'])
            for k in kids):
        iso_kid_set.add(int(sid))
iso_kid_set.update({329, 1011, 894})  # SSp-bfd, AUDd, RSPagl
area_acro = {sid: tree.get_structures_by_id([sid])[0]['acronym']
              for sid in iso_kid_set}
print(f'{len(iso_kid_set)} cortical areas (true Isocortex)')

# Label each cortex vertex by area (inward sampling, fill unlabeled).
verts_um = (cortex.vertices + root_ctr) / SHARED_SCALE
INWARD_STEP_UM = 400.0
verts_inward = verts_um - INWARD_STEP_UM * cortex.vertex_normals
vox = np.clip(np.round(verts_inward / RES_UM).astype(int), 0,
               np.array(ann.shape) - 1)
leaf = ann[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.int64)
leaf_to_area = {0: 0}
for L in np.unique(leaf):
    if L == 0: continue
    try:
        path = tree.get_structures_by_id([int(L)])[0]['structure_id_path']
    except Exception:
        path = []
    match = [s for s in path if int(s) in iso_kid_set]
    leaf_to_area[int(L)] = int(match[0]) if match else 0
vert_area = np.array([leaf_to_area[int(l)] for l in leaf], dtype=np.int64)
labeled = vert_area != 0
if (~labeled).any() and labeled.any():
    kd = cKDTree(verts_um[labeled])
    _, ii = kd.query(verts_um[~labeled])
    vert_area[~labeled] = vert_area[np.where(labeled)[0][ii]]

# Mark outside-facing vertices
cortex_ctr = cortex.centroid
out_dir_v = cortex.vertices - cortex_ctr
out_dir_v /= (np.linalg.norm(out_dir_v, axis=1, keepdims=True) + 1e-12)
vn = cortex.vertex_normals
outside_mask = (vn * out_dir_v).sum(axis=1) > 0.3

# Text mesh helper
def text_to_mesh(text, height_mm, depth_mm, font_family='Helvetica'):
    fp = FontProperties(family=font_family, weight='bold')
    tp = TextPath((0, 0), text, size=height_mm, prop=fp)
    parts = []
    for ring in tp.to_polygons():
        if len(ring) < 3: continue
        try:
            p = Polygon(ring)
            if not p.is_valid: p = make_valid(p)
            if p.is_valid and p.area > 1e-9:
                parts.append(p)
        except Exception:
            pass
    if not parts: return None
    result = parts[0]
    for p in parts[1:]:
        result = result.symmetric_difference(p)
    polys = ([result] if isinstance(result, Polygon)
              else list(result.geoms) if isinstance(result, MultiPolygon)
              else [])
    meshes = []
    for p in polys:
        if not p.is_valid or p.area < 1e-9: continue
        try:
            meshes.append(trimesh.creation.extrude_polygon(p, height=depth_mm))
        except Exception: pass
    if not meshes: return None
    m = trimesh.util.concatenate(meshes)
    m.apply_translation(-m.bounding_box.centroid)   # center
    return m

def tangent_frame_for_normal(n, world_up=np.array([0, -1, 0])):
    n = n / np.linalg.norm(n)
    v = world_up - n * (world_up @ n)
    if np.linalg.norm(v) < 1e-3:
        v = np.array([1.0, 0.0, 0.0]) - n * (np.array([1.0, 0.0, 0.0]) @ n)
    v = v / np.linalg.norm(v)
    v = -v
    u = np.cross(n, v)
    u = u / np.linalg.norm(u)
    return u, v

# Place a label per area + hemisphere
LABEL_DEPTH_MM = 1.8
FIT_FRAC_W = 0.65
FIT_FRAC_H = 0.55
MIN_LETTER_MM = 1.6
MAX_LETTER_MM = 5.5

text_meshes = []
placed = []
for sid in sorted(iso_kid_set):
    acro = area_acro[sid]
    area_mask = vert_area == int(sid)
    if area_mask.sum() < 80: continue
    out_in_area = area_mask & outside_mask
    if out_in_area.sum() < 60: continue
    for hemi_name, hemi_sign in (('L', -1), ('R', +1)):
        hemi_mask = out_in_area & (cortex.vertices[:, 2] * hemi_sign > 0)
        if hemi_mask.sum() < 50: continue
        pts = cortex.vertices[hemi_mask]
        norms = vn[hemi_mask]
        # Mean outward normal — face direction
        mean_n = norms.mean(axis=0)
        mean_n /= (np.linalg.norm(mean_n) + 1e-12)
        # Filter to vertices aligned with mean_n
        align = norms @ mean_n
        keep = align > 0.5
        if keep.sum() < 20:
            keep = align > 0.3
        if keep.sum() < 10: continue
        pts_k = pts[keep]; norms_k = norms[keep]
        # Tangent frame from mean normal + PCA for orientation
        u0, v0 = tangent_frame_for_normal(mean_n)
        pts_uv = np.column_stack([pts_k @ u0, pts_k @ v0])
        uv_ctr = pts_uv.mean(axis=0); uv_c = pts_uv - uv_ctr
        cov = np.cov(uv_c.T) if len(uv_c) > 1 else np.eye(2)
        evals, evecs = np.linalg.eigh(cov)
        pc1 = evecs[:, 1].astype(float)
        proj1 = uv_c @ pc1; proj2 = uv_c @ evecs[:, 0]
        ext1 = float(proj1.max() - proj1.min())
        ext2 = float(proj2.max() - proj2.min())
        # Build reference text to measure its size, then scale to fit
        ref = text_to_mesh(acro, height_mm=1.0, depth_mm=LABEL_DEPTH_MM)
        if ref is None: continue
        ref_w, ref_h = ref.bounding_box.extents[:2]
        letter_mm = min(FIT_FRAC_W * ext1 / ref_w,
                         FIT_FRAC_H * ext2 / ref_h,
                         MAX_LETTER_MM)
        if letter_mm < MIN_LETTER_MM: continue
        # Build real text mesh
        m = text_to_mesh(acro, height_mm=letter_mm, depth_mm=LABEL_DEPTH_MM)
        if m is None: continue
        # Flip y so text reads upright in this frame
        m.vertices[:, 1] *= -1.0
        m.invert()
        # Site: vertex closest to (uv) centroid AND most aligned
        d_uv = np.linalg.norm(uv_c, axis=1)
        order = np.argsort(d_uv)[:min(80, len(pts_k))]
        best = order[np.argmax(align[keep][order])]
        site = pts_k[best]
        site_n = norms_k[best]
        # Rotated tangent frame at site by PC1
        pc1_3d = u0 * pc1[0] + v0 * pc1[1]
        u_ax = pc1_3d - site_n * (pc1_3d @ site_n)
        nu = np.linalg.norm(u_ax)
        if nu < 1e-6:
            u_ax, v_ax = tangent_frame_for_normal(site_n)
        else:
            u_ax = u_ax / nu
            v_ax = np.cross(site_n, u_ax)
            v_ax = v_ax / np.linalg.norm(v_ax)
            # Align v with world-up projection
            v_pref = np.array([0, -1, 0]) - site_n * (np.array([0, -1, 0]) @ site_n)
            if np.linalg.norm(v_pref) > 1e-3:
                v_pref = v_pref / np.linalg.norm(v_pref)
                if v_ax @ (-v_pref) < 0:
                    v_ax = -v_ax; u_ax = -u_ax
        R = np.column_stack([u_ax, v_ax, site_n])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = site
        m.apply_transform(T)
        text_meshes.append(m)
        placed.append((acro, hemi_name, letter_mm))

print(f'\nplaced {len(placed)} labels')
for name, h, lm in placed:
    print(f'  {name:>10s} {h}  letter={lm:.2f}mm')

text_union = trimesh.util.concatenate(text_meshes)
print(f'\nconcatenated text: {len(text_union.vertices):,} verts, '
      f'{len(text_union.faces):,} faces, wt={text_union.is_watertight}')

# Boolean subtraction
print('\n=== BOOLEAN SUBTRACT (manifold3d) ===')
tic = time.time()
labeled_cortex = cortex.difference(text_union, engine='manifold')
print(f'done in {time.time()-tic:.1f}s')
print(f'result: {len(labeled_cortex.vertices):,} verts, '
      f'{len(labeled_cortex.faces):,} faces, '
      f'wt={labeled_cortex.is_watertight}')

out_stl = OUT_DIR / 'stl' / 'cortex_hat' / f'allen_ccf_cortex_hat_pialboollabeled_{int(TARGET_MM)}mm.stl'
labeled_cortex.export(str(out_stl))
print(f'wrote {out_stl.name} ({out_stl.stat().st_size/1024/1024:.1f} MB)')

# Render — 4 views
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
d = labeled_cortex.bounding_box.extents.max() * 1.8

def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

views = [
    ('dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
    ('lateral R',  (0,  0, d), (0, 0, 0), (0, -1, 0)),
    ('lateral L',  (0,  0, -d), (0, 0, 0), (0, -1, 0)),
    ('ventral',    (0,  d, 0), (0, 0, 0), (1, 0, 0)),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for k, (vname, cam, focal, up) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    pl.add_mesh(make_poly(labeled_cortex), color=(0.85, 0.85, 0.85),
                 smooth_shading=True, specular=0.15, ambient=0.20,
                 diffuse=0.95)
    pl.add_text(vname, font_size=10)
    pl.camera_position = (cam, focal, up)
    pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'cortex_hat_pialboollabeled.png'))
pl.close()
print('wrote cortex_hat_pialboollabeled.png')
