"""Bilaterally-symmetric boolean labels.

The Allen CCF is approximately bilaterally symmetric: each area exists
in left and right hemisphere at mirror-image positions across the
midline (Z = ann.shape[2]/2 * 25 µm in input space, or Z = 0 in
centered print coords).

Instead of computing each hemisphere's label position independently
(which gives only approximately-symmetric placements), here we:
1. Compute the label site + orientation on the LEFT hemisphere only.
2. Mirror the site across the midline (z → -z).
3. Mirror the orientation: u_ax → flip Z, v_ax preserved, n → flip Z.
4. Build the same text mesh, transform with the mirrored frame.

This guarantees the two labels are exact mirror images of each other,
which feels right anatomically.
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

cortex = trimesh.load(str(WT_STL), force='mesh')
cortex.apply_scale(SHARED_SCALE)
cortex.apply_translation(-root_ctr)
print(f'cortex: {len(cortex.vertices):,} verts, '
      f'wt={cortex.is_watertight}, vol={cortex.is_volume}')

# midline plane: z = 0 in centered print coords (since root was centered)
# Verify: Allen midline is at ann.shape[2]/2 * 25 µm = ~5700 µm. After
# centering on root centroid (Allen brain is bilaterally symmetric so
# centroid lies on the midline), z=0 in print space corresponds to
# the midline. Good.

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
iso_kid_set.update({329, 1011, 894})
area_acro = {sid: tree.get_structures_by_id([sid])[0]['acronym']
              for sid in iso_kid_set}

# Per-vertex area labels
verts_um = (cortex.vertices + root_ctr) / SHARED_SCALE
INWARD = 400.0
v_in = verts_um - INWARD * cortex.vertex_normals
vox = np.clip(np.round(v_in / RES_UM).astype(int), 0,
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

vn = cortex.vertex_normals
cortex_ctr = cortex.centroid
out_dir = cortex.vertices - cortex_ctr
out_dir /= (np.linalg.norm(out_dir, axis=1, keepdims=True) + 1e-12)
outside_mask = (vn * out_dir).sum(axis=1) > 0.3


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
    m.apply_translation(-m.bounding_box.centroid)
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


LABEL_DEPTH_MM = 1.8
FIT_FRAC_W = 0.65
FIT_FRAC_H = 0.55
MIN_LETTER_MM = 1.6
MAX_LETTER_MM = 5.5

# For each area, compute label site/orientation on LEFT hemisphere
# (z < 0). Then mirror to right.
text_meshes = []
placed = []
for sid in sorted(iso_kid_set):
    acro = area_acro[sid]
    area_mask = vert_area == int(sid)
    if area_mask.sum() < 80: continue
    out_in_area = area_mask & outside_mask
    if out_in_area.sum() < 60: continue

    # Restrict to LEFT hemisphere (z < 0)
    left_mask = out_in_area & (cortex.vertices[:, 2] < 0)
    if left_mask.sum() < 50: continue
    pts = cortex.vertices[left_mask]
    norms = vn[left_mask]
    mean_n = norms.mean(axis=0)
    mean_n /= (np.linalg.norm(mean_n) + 1e-12)
    align = norms @ mean_n
    keep = align > 0.5
    if keep.sum() < 20:
        keep = align > 0.3
    if keep.sum() < 10: continue
    pts_k = pts[keep]; norms_k = norms[keep]
    u0, v0 = tangent_frame_for_normal(mean_n)
    pts_uv = np.column_stack([pts_k @ u0, pts_k @ v0])
    uv_ctr = pts_uv.mean(axis=0); uv_c = pts_uv - uv_ctr
    cov = np.cov(uv_c.T) if len(uv_c) > 1 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    pc1 = evecs[:, 1].astype(float)
    proj1 = uv_c @ pc1; proj2 = uv_c @ evecs[:, 0]
    ext1 = float(proj1.max() - proj1.min())
    ext2 = float(proj2.max() - proj2.min())
    ref = text_to_mesh(acro, height_mm=1.0, depth_mm=LABEL_DEPTH_MM)
    if ref is None: continue
    ref_w, ref_h = ref.bounding_box.extents[:2]
    letter_mm = min(FIT_FRAC_W * ext1 / ref_w,
                     FIT_FRAC_H * ext2 / ref_h,
                     MAX_LETTER_MM)
    if letter_mm < MIN_LETTER_MM: continue
    text_m = text_to_mesh(acro, height_mm=letter_mm,
                            depth_mm=LABEL_DEPTH_MM)
    if text_m is None: continue
    text_m.vertices[:, 1] *= -1.0
    text_m.invert()

    # LEFT-hemisphere site
    d_uv = np.linalg.norm(uv_c, axis=1)
    order = np.argsort(d_uv)[:min(80, len(pts_k))]
    best = order[np.argmax(align[keep][order])]
    site_L = pts_k[best]; n_L = norms_k[best]
    pc1_3d = u0 * pc1[0] + v0 * pc1[1]
    u_L = pc1_3d - n_L * (pc1_3d @ n_L)
    nu = np.linalg.norm(u_L)
    if nu < 1e-6:
        u_L, v_L = tangent_frame_for_normal(n_L)
    else:
        u_L = u_L / nu
        v_L = np.cross(n_L, u_L)
        v_L = v_L / np.linalg.norm(v_L)
        v_pref = np.array([0, -1, 0]) - n_L * (np.array([0, -1, 0]) @ n_L)
        if np.linalg.norm(v_pref) > 1e-3:
            v_pref = v_pref / np.linalg.norm(v_pref)
            if v_L @ (-v_pref) < 0:
                v_L = -v_L; u_L = -u_L

    # Place LEFT label
    R_L = np.column_stack([u_L, v_L, n_L])
    T_L = np.eye(4); T_L[:3, :3] = R_L; T_L[:3, 3] = site_L
    m_L = text_m.copy()
    m_L.apply_transform(T_L)
    text_meshes.append(m_L)

    # MIRROR to right hemisphere by reflecting z. Geometric mirror:
    # site_R = (x, y, -z), n_R = (n[0], n[1], -n[2]), u_R = mirror(u),
    # v_R = mirror(v). Flip the text reading direction (u) — but the
    # text mesh would then appear mirror-flipped. To keep text readable
    # on the right side, ALSO flip the bitmap (which we do by mirroring
    # u_R again). Net effect: text orientation stays text-readable on
    # both sides, just facing the other hemisphere's outside.
    site_R = site_L * np.array([1, 1, -1])
    n_R    = n_L    * np.array([1, 1, -1])
    # Mirror u and v across z, then flip u (so the text doesn't read
    # backwards on the right side).
    u_R_mirror = u_L * np.array([1, 1, -1])
    v_R_mirror = v_L * np.array([1, 1, -1])
    # Flip u so reading direction stays consistent (left-to-right
    # when looking at that hemisphere from outside).
    u_R = -u_R_mirror
    v_R = v_R_mirror
    # Make sure (u_R, v_R, n_R) is right-handed: cross(u_R, v_R) should
    # point in direction of n_R (outward).
    if np.cross(u_R, v_R) @ n_R < 0:
        v_R = -v_R
    R_R = np.column_stack([u_R, v_R, n_R])
    T_R = np.eye(4); T_R[:3, :3] = R_R; T_R[:3, 3] = site_R
    m_R = text_m.copy()
    m_R.apply_transform(T_R)
    text_meshes.append(m_R)

    placed.append((acro, letter_mm))

print(f'\nplaced {len(placed)} bilateral labels (= {2*len(placed)} total)')
for name, lm in placed:
    print(f'  {name:>10s}  letter={lm:.2f}mm')

text_union = trimesh.util.concatenate(text_meshes)
print(f'\nconcatenated: {len(text_union.vertices):,} verts, '
      f'{len(text_union.faces):,} faces, wt={text_union.is_watertight}')

print('\n=== BOOLEAN SUBTRACT ===')
tic = time.time()
labeled = cortex.difference(text_union, engine='manifold')
print(f'done in {time.time()-tic:.1f}s')
print(f'result: {len(labeled.vertices):,} verts, '
      f'{len(labeled.faces):,} faces, wt={labeled.is_watertight}')

out_stl = OUT_DIR / 'stl' / 'cortex_hat' / f'allen_ccf_cortex_hat_bilat_pialboollabeled_{int(TARGET_MM)}mm.stl'
labeled.export(str(out_stl))
print(f'wrote {out_stl.name} ({out_stl.stat().st_size/1024/1024:.1f} MB)')

# Render
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
d = labeled.bounding_box.extents.max() * 1.8

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
    pl.add_mesh(make_poly(labeled), color=(0.85, 0.85, 0.85),
                 smooth_shading=True, specular=0.15, ambient=0.20,
                 diffuse=0.95)
    pl.add_text(vname, font_size=10)
    pl.camera_position = (cam, focal, up)
    pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'cortex_hat_bilat_pialboollabeled.png'))
pl.close()
print('wrote cortex_hat_bilat_pialboollabeled.png')
