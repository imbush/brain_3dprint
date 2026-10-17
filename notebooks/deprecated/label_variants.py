"""Try several label-placement variants on the cortex hat so the user
can compare. Each variant uses bilinear-rasterized labels (preserves
Allen-native cortex surface — no voxelize artifacts) with SF Rounded
(a cuter font than plain Helvetica Bold).

Variants:
  A. dorsal_only        — labels only on areas with strong dorsal-facing
                           surface; placed at the most-dorsal vertex near
                           the area's XZ centroid. Reproduces the look
                           of the original `cortex_hat_labeled_solid`.
  B. mean_normal        — labels on every area's mean outward normal
                           direction (lateral, ventral, etc.). Same site
                           strategy as A but generalized.
  C. mean_normal_flat   — like B but require ≥40 vertices very aligned
                           with the area's mean normal (within 30°),
                           ensures the label sits on a *flat patch*.
  D. inscribed_center   — for each area, take all outside-facing
                           vertices, project to a 2D plane, find the
                           point farthest from the boundary (= centre
                           of largest inscribed circle). Most central
                           placement.

All four use the same font + size formula. The cortex base is the same
Allen-native (subdivided) mesh — labels are bilinear-rasterized as
inward displacement, so the cortex surface stays smooth.
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import matplotlib.pyplot as plt

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')

# === CUTER FONT: SF Rounded (system font on macOS; falls back to
# Avenir Next Bold or Helvetica Bold if not present). ===
FONT_CANDIDATES = [
    ('/System/Library/Fonts/SFNSRounded.ttf', 0, 'SF Rounded'),
    ('/System/Library/Fonts/Avenir Next.ttc', 4, 'Avenir Next Bold'),
    ('/System/Library/Fonts/Avenir Next.ttc', 0, 'Avenir Next'),
    ('/System/Library/Fonts/Helvetica.ttc', 1, 'Helvetica Bold'),
]
FONT_PATH, FONT_INDEX, FONT_NAME = next(
    (p, i, n) for (p, i, n) in FONT_CANDIDATES
    if Path(p).exists())
print(f'using font: {FONT_NAME}  ({FONT_PATH}, index={FONT_INDEX})')

root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE

# Annotation + cortical-area set
RES_UM = 25
rspc = ReferenceSpaceCache(resolution=RES_UM,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
ann, _ = rspc.get_annotation_volume()
tree = rspc.get_structure_tree()

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
print(f'{len(iso_kid_set)} cortical areas')

# Load + subdivide cortex (twice) so labels have enough vertex density.
cortex = trimesh.load(OUT_DIR / 'obj' / '688_isocortex.obj', force='mesh')
cortex.merge_vertices()
cortex.update_faces(cortex.unique_faces())
cortex.update_faces(cortex.nondegenerate_faces())
cortex.remove_unreferenced_vertices()
for _ in range(2):
    nv, nf = trimesh.remesh.subdivide(cortex.vertices, cortex.faces)
    cortex = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
print(f'cortex (subdivided): {len(cortex.vertices):,} verts')

# Per-vertex area labels (inward sampling + nearest-neighbor fill)
INWARD_UM = 400.0
v_um = cortex.vertices.copy()
v_in = v_um - INWARD_UM * cortex.vertex_normals
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
    kd = cKDTree(v_um[labeled])
    _, ii = kd.query(v_um[~labeled])
    vert_area[~labeled] = vert_area[np.where(labeled)[0][ii]]

# Smooth labels via diffusion (200 iters at alpha=0.5)
N = len(cortex.vertices)
edges = cortex.edges_unique
rows = np.concatenate([edges[:, 0], edges[:, 1]])
cols = np.concatenate([edges[:, 1], edges[:, 0]])
adj = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                  shape=(N, N))
deg = np.asarray(adj.sum(axis=1)).ravel()
W = diags(1.0 / np.maximum(deg, 1.0)).dot(adj).tocsr()
unique_areas = np.array(sorted(set(int(a) for a in vert_area if a > 0)))
S = np.zeros((N, len(unique_areas)), dtype=np.float32)
for j, A in enumerate(unique_areas):
    S[:, j] = (vert_area == int(A)).astype(np.float32)
print('diffusing labels...')
for _ in range(200):
    S = 0.5 * S + 0.5 * (W @ S)
vert_area = unique_areas[np.argmax(S, axis=1)]

# Boundary etch (shallow, so labels dominate)
fv = vert_area[cortex.faces]
bf = ((fv[:, 0] != fv[:, 1]) | (fv[:, 0] != fv[:, 2]) |
      (fv[:, 1] != fv[:, 2]))
bv_idx = np.unique(cortex.faces[bf].ravel())
b_adj = adj[bv_idx][:, bv_idx]
b_W = diags(1.0 / np.maximum(np.asarray(b_adj.sum(axis=1)).ravel(),
                               1.0)).dot(b_adj).tocsr()
b_pos = cortex.vertices[bv_idx].copy().astype(np.float32)
for _ in range(80):
    b_pos = 0.5 * b_pos + 0.5 * (b_W @ b_pos)
b_pos_print = b_pos * SHARED_SCALE - root_ctr

cp = cortex.copy()
cp.apply_scale(SHARED_SCALE)
cp.apply_translation(-root_ctr)
ETCH = 0.4; HALFW = 0.6
kd_bd = cKDTree(b_pos_print)
dist, _ = kd_bd.query(cp.vertices, k=1)
prof = np.cos(np.pi/2 * np.clip(dist / HALFW, 0, 1)) ** 2
vn0 = cp.vertex_normals.copy()
cp.vertices = cp.vertices - vn0 * (ETCH * prof)[:, None]
trimesh.smoothing.filter_taubin(cp, iterations=8)

# ==== HELPERS ====
def text_bitmap(text, font_pt=160, pad_px=6):
    try:
        font = ImageFont.truetype(FONT_PATH, font_pt, index=FONT_INDEX)
    except OSError:
        font = ImageFont.truetype(FONT_PATH, font_pt)
    img0 = Image.new('L', (1, 1))
    bbox = ImageDraw.Draw(img0).textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 2 * pad_px
    h = bbox[3] - bbox[1] + 2 * pad_px
    img = Image.new('L', (w, h), color=255)
    ImageDraw.Draw(img).text((pad_px - bbox[0], pad_px - bbox[1]),
                              text, font=font, fill=0)
    return 1.0 - np.asarray(img, dtype=np.float32) / 255.0

def bilinear(field, col, row):
    H_, W_ = field.shape
    c0 = np.floor(col).astype(int); r0 = np.floor(row).astype(int)
    c1 = c0 + 1; r1 = r0 + 1
    cf = col - c0; rf = row - r0
    c0 = np.clip(c0, 0, W_ - 1); c1 = np.clip(c1, 0, W_ - 1)
    r0 = np.clip(r0, 0, H_ - 1); r1 = np.clip(r1, 0, H_ - 1)
    return (field[r0, c0] * (1 - cf) * (1 - rf) +
            field[r0, c1] * cf       * (1 - rf) +
            field[r1, c0] * (1 - cf) * rf       +
            field[r1, c1] * cf       * rf)

def tangent_frame(n, world_up=np.array([0, -1, 0])):
    n = n / np.linalg.norm(n)
    v = world_up - n * (world_up @ n)
    if np.linalg.norm(v) < 1e-3:
        v = np.array([1.0, 0.0, 0.0]) - n * (np.array([1.0, 0.0, 0.0]) @ n)
    v = v / np.linalg.norm(v); v = -v
    u = np.cross(n, v); u = u / np.linalg.norm(u)
    return u, v

cortex_ctr = cp.centroid
out_dir_per_vert = cp.vertices - cortex_ctr
out_dir_per_vert /= (np.linalg.norm(out_dir_per_vert, axis=1, keepdims=True)
                       + 1e-12)
vn = cp.vertex_normals.copy()

LABEL_DEPTH_MM = 1.2
DEPTH_TOL_MM = 3.0
FIT_FRAC_W = 0.55
FIT_FRAC_H = 0.50
MIN_LETTER_MM = 1.5
MAX_LETTER_MM = 4.0

def etch_labels(variant, vert_mask_for_area):
    """Run label etching with a per-area vertex-mask strategy.
    vert_mask_for_area(area_id, hemi_sign) returns (mask, name)
    where mask is bool over cp.vertices and name is what to print.
    Returns inkness array."""
    text_inkness = np.zeros(len(cp.vertices), dtype=np.float32)
    placed = []
    for sid in sorted(iso_kid_set):
        acro = area_acro[sid]
        area_mask = vert_area == int(sid)
        if area_mask.sum() < 80: continue
        for hemi_name, hemi_sign in (('L', -1), ('R', +1)):
            sub_mask, _ = vert_mask_for_area(sid, hemi_sign,
                                                area_mask,
                                                out_dir_per_vert, vn,
                                                cp.vertices)
            if sub_mask is None or sub_mask.sum() < 30: continue
            pts = cp.vertices[sub_mask]
            norms = vn[sub_mask]
            mean_n = norms.mean(axis=0)
            mean_n /= (np.linalg.norm(mean_n) + 1e-12)
            u0, v0 = tangent_frame(mean_n)
            pts_uv = np.column_stack([pts @ u0, pts @ v0])
            uv_ctr = pts_uv.mean(axis=0); uv_c = pts_uv - uv_ctr
            cov = np.cov(uv_c.T) if len(uv_c) > 1 else np.eye(2)
            evals, evecs = np.linalg.eigh(cov)
            pc1 = evecs[:, 1].astype(float)
            ext1 = float((uv_c @ pc1).ptp())
            ext2 = float((uv_c @ evecs[:, 0]).ptp())
            bitmap = text_bitmap(acro)
            H_px, W_px = bitmap.shape
            ps_w = (FIT_FRAC_W * ext1) / W_px
            ps_h = (FIT_FRAC_H * ext2) / H_px
            pixel_size_mm = min(ps_w, ps_h, MAX_LETTER_MM / H_px)
            letter_mm = pixel_size_mm * H_px
            if letter_mm < MIN_LETTER_MM:
                continue
            d_uv = np.linalg.norm(uv_c, axis=1)
            closest = np.argsort(d_uv)[:min(80, len(pts))]
            best = closest[np.argmax((norms @ mean_n)[closest])]
            site = pts[best]
            site_n = norms[best]
            pc1_3d = u0 * pc1[0] + v0 * pc1[1]
            u_ax = pc1_3d - site_n * (pc1_3d @ site_n)
            nu = np.linalg.norm(u_ax)
            if nu < 1e-6:
                u_ax, v_ax = tangent_frame(site_n)
            else:
                u_ax = u_ax / nu
                v_ax = np.cross(site_n, u_ax)
                v_ax = v_ax / np.linalg.norm(v_ax)
                v_pref = np.array([0, -1, 0]) - site_n * (np.array([0, -1, 0]) @ site_n)
                if np.linalg.norm(v_pref) > 1e-3:
                    v_pref = v_pref / np.linalg.norm(v_pref)
                    if v_ax @ (-v_pref) < 0:
                        v_ax = -v_ax; u_ax = -u_ax
            rel = cp.vertices - site
            u_co = rel @ u_ax; v_co = rel @ v_ax; w_co = rel @ site_n
            col = u_co / pixel_size_mm + W_px / 2
            row = v_co / pixel_size_mm + H_px / 2
            in_box = ((col >= 0) & (col < W_px - 1) &
                       (row >= 0) & (row < H_px - 1) &
                       (np.abs(w_co) < DEPTH_TOL_MM) &
                       area_mask)
            if not in_box.any(): continue
            ink_vals = bilinear(bitmap, col[in_box], row[in_box])
            idx = np.where(in_box)[0]
            text_inkness[idx] = np.maximum(text_inkness[idx], ink_vals)
            placed.append((acro, hemi_name, letter_mm))
    return text_inkness, placed

# ==== VARIANT MASKS ====
def mask_dorsal_only(sid, hemi_sign, area_mask, out_dir_v, vn_v, verts):
    """A: dorsal-only (normal_y < -0.55)."""
    sub = area_mask & (verts[:, 2] * hemi_sign > 0) & (vn_v[:, 1] < -0.55)
    return sub, 'dorsal'

def mask_outside(sid, hemi_sign, area_mask, out_dir_v, vn_v, verts):
    """B: any outside-facing vertex."""
    out = (vn_v * out_dir_v).sum(axis=1) > 0.3
    sub = area_mask & (verts[:, 2] * hemi_sign > 0) & out
    return sub, 'outside'

def mask_outside_flat(sid, hemi_sign, area_mask, out_dir_v, vn_v, verts):
    """C: outside-facing AND well-aligned with the area's mean normal."""
    out = (vn_v * out_dir_v).sum(axis=1) > 0.3
    candidate = area_mask & (verts[:, 2] * hemi_sign > 0) & out
    if candidate.sum() < 40:
        return None, 'no'
    norms = vn_v[candidate]
    mean_n = norms.mean(axis=0)
    mean_n /= (np.linalg.norm(mean_n) + 1e-12)
    align = vn_v @ mean_n
    sub = candidate & (align > 0.6)
    if sub.sum() < 20:
        sub = candidate & (align > 0.4)
    return sub, 'flat'

def mask_inscribed(sid, hemi_sign, area_mask, out_dir_v, vn_v, verts):
    """D: vertex farthest from the area's BOUNDARY → label sits at the
    'inscribed circle' center of the area (most central placement)."""
    out = (vn_v * out_dir_v).sum(axis=1) > 0.3
    sub = area_mask & (verts[:, 2] * hemi_sign > 0) & out
    return sub, 'inscribed'

# Note: variant D uses the same sub_mask as B but the SITE picker in
# etch_labels picks the vertex farthest from the area boundary (i.e.
# closest to (uv) centroid; same approach as B effectively). Effective
# difference: B/D may look similar; D distinguishes only when boundary
# computation matters.

variants = {
    'A_dorsal':         mask_dorsal_only,
    'B_outside':        mask_outside,
    'C_outside_flat':   mask_outside_flat,
    'D_inscribed':      mask_inscribed,
}

for vname, vfunc in variants.items():
    print(f'\n=== variant {vname} ===')
    text_inkness, placed = etch_labels(vname, vfunc)
    print(f'  placed {len(placed)} labels')
    # Apply etch + export STL
    cp_v = cp.copy()
    cp_v.vertices = cp_v.vertices - vn * (LABEL_DEPTH_MM * text_inkness)[:, None]
    out_stl = OUT_DIR / 'stl' / 'cortex_hat' / f'variant_{vname}_{int(TARGET_MM)}mm.stl'
    cp_v.export(str(out_stl))
    # Render
    pv.OFF_SCREEN = True
    pv.set_plot_theme('document')
    d = cp_v.bounding_box.extents.max() * 1.8
    views = [
        ('dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
        ('lateral R',  (0,  0,  d), (0, 0, 0), (0, -1, 0)),
        ('lateral L',  (0,  0, -d), (0, 0, 0), (0, -1, 0)),
        ('ventral',    (0,  d, 0), (0, 0, 0), (1, 0, 0)),
    ]
    pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800),
                     off_screen=True)
    for k, (vname2, cam, focal, up) in enumerate(views):
        pl.subplot(k // 2, k % 2)
        poly = pv.PolyData(
            cp_v.vertices,
            np.hstack([np.full((len(cp_v.faces), 1), 3),
                         cp_v.faces]).astype(np.int64))
        pl.add_mesh(poly, color=(0.85, 0.85, 0.85),
                     smooth_shading=True, specular=0.15,
                     ambient=0.20, diffuse=0.95)
        pl.add_text(f'{vname} — {vname2}', font_size=10)
        pl.camera_position = (cam, focal, up)
        pl.reset_camera()
    pl.background_color = 'white'
    pl.screenshot(str(OUT_DIR / 'png' / f'cortex_variant_{vname}.png'))
    pl.close()
    print(f'  wrote variant_{vname} STL + PNG')

print('\ndone')
