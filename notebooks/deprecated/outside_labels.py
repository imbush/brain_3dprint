"""Bilinear-rasterized label etching that puts labels on each area's
OUTSIDE surface — wherever the area faces outward. Dorsal areas get
labels on the dorsal surface, lateral areas on the lateral surface,
ventral on the ventral, etc.

Builds on the existing labeled-cortex pipeline but drops the
"dorsal-only" filter and instead picks a label site per area based
on each area's actual mean outward normal.
"""
import sys, time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import matplotlib as mpl
from matplotlib.colors import to_rgb

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE
print(f'TARGET_MM = {TARGET_MM:.2f} mm ({SCALE_FACTOR}x)')

RES_UM = 25
rspc = ReferenceSpaceCache(resolution=RES_UM,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
ann, _ = rspc.get_annotation_volume()
tree = rspc.get_structure_tree()

ISO_ID = 315  # 315 = true Isocortex (688 was Cerebral cortex CTX, broader)
def is_layer_acro(a):
    return (a.endswith('2/3') or a.endswith('6a') or a.endswith('6b')
             or (len(a) >= 2 and a[-1].isdigit()))
# Strict heuristic + 3 manual additions for ontology quirks (SSp-bfd,
# AUDd, RSPagl each have a stray mis-parented child).
iso_kid_set = set()
for sid in tree.descendant_ids([ISO_ID])[0]:
    if sid == ISO_ID: continue
    kids = tree.child_ids([int(sid)])[0]
    if not kids: continue
    if all(is_layer_acro(tree.get_structures_by_id([int(k)])[0]['acronym'])
            for k in kids):
        iso_kid_set.add(int(sid))
iso_kid_set.update({329, 1011, 894})    # SSp-bfd, AUDd, RSPagl
print(f'{len(iso_kid_set)} cortical areas')

# Load + subdivide cortex twice for finer label resolution
cortex = trimesh.load(OUT_DIR / 'obj' / '688_isocortex.obj', force='mesh')
cortex.merge_vertices()
cortex.update_faces(cortex.unique_faces())
cortex.update_faces(cortex.nondegenerate_faces())
cortex.remove_unreferenced_vertices()
for _ in range(2):
    nv, nf = trimesh.remesh.subdivide(cortex.vertices, cortex.faces)
    cortex = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
print(f'cortex: {len(cortex.vertices):,} verts')

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

# Build adjacency + diffuse for smooth boundaries
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

# Boundary detection + smoothed phantom curve
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

# Scale + shallow groove etch
cp = cortex.copy()
cp.apply_scale(SHARED_SCALE)
cp.apply_translation(-root_ctr)
ETCH_DEPTH = 0.4
HALFW = 0.6
kd_bd = cKDTree(b_pos_print)
dist, _ = kd_bd.query(cp.vertices, k=1)
prof = np.cos(np.pi/2 * np.clip(dist / HALFW, 0, 1)) ** 2
vn0 = cp.vertex_normals.copy()
cp.vertices = cp.vertices - vn0 * (ETCH_DEPTH * prof)[:, None]
trimesh.smoothing.filter_taubin(cp, iterations=8)

# ==== LABELS ON OUTSIDE SURFACE (any orientation, not just dorsal) ====
# For each area:
# 1. Pick OUTSIDE-facing vertices: those whose normal points AWAY from
#    the cortex centroid (i.e. towards the brain envelope outside, not
#    into the hollow interior).
# 2. Compute the area's MEAN OUTWARD NORMAL (= label direction).
# 3. Filter to vertices closely aligned with the mean normal — these
#    sit on the "characteristic" outside face of the area.
# 4. Pick the centroid of that subset → label site.
# 5. Build tangent frame at the site, with v_ax preferring to point
#    along the area's elongation (PCA on the projection onto tangent
#    plane).
print('\n=== placing labels on each area\'s outside surface ===')
cortex_ctr = cp.centroid
out_dir_per_vert = cp.vertices - cortex_ctr
out_dir_per_vert /= (np.linalg.norm(out_dir_per_vert, axis=1, keepdims=True)
                       + 1e-12)
vn = cp.vertex_normals.copy()
# A vertex is OUTSIDE-facing if its normal points roughly outward.
outside_mask_global = (vn * out_dir_per_vert).sum(axis=1) > 0.3

# Skip the small, fragmented olfactory / amygdalar areas that look like
# spikes on the cortex hat (the "star"). These are anatomically
# correct but visually confusing as labeled patches.
SKIP_ACRO = {
    'NLOT', 'COAa', 'COApl', 'COApm', 'PAA',
    'TR', 'TTd', 'TTv', 'DP',
}

area_acro = {int(sid): tree.get_structures_by_id([int(sid)])[0]['acronym']
              for sid in iso_kid_set}

FONT_PATH = '/System/Library/Fonts/Helvetica.ttc'
FONT_INDEX = 1
FONT_PT = 160
PAD_PX = 6
LABEL_DEPTH_MM = 1.2
DEPTH_TOL_MM   = 3.0
FIT_FRAC_W = 0.55
FIT_FRAC_H = 0.50
MIN_LETTER_MM = 1.5
MAX_LETTER_MM = 4.0

def text_bitmap(text):
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_PT, index=FONT_INDEX)
    except OSError:
        font = ImageFont.truetype(FONT_PATH, FONT_PT)
    img0 = Image.new('L', (1, 1))
    bbox = ImageDraw.Draw(img0).textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 2 * PAD_PX
    h = bbox[3] - bbox[1] + 2 * PAD_PX
    img = Image.new('L', (w, h), color=255)
    ImageDraw.Draw(img).text((PAD_PX - bbox[0], PAD_PX - bbox[1]),
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

def tangent_frame_for_normal(n, world_up_pref=np.array([0, -1, 0])):
    """Build a 2D-readable tangent frame at a surface normal n.
    v_ax points 'down' in image (i.e. away from the world up direction
    projected onto the tangent plane). u_ax = v_ax × n.
    The convention makes text appear upright when the camera looks
    along -n with up=world_up_pref."""
    n = n / np.linalg.norm(n)
    # Project world_up onto the tangent plane and invert (image down).
    v = world_up_pref - n * (world_up_pref @ n)
    if np.linalg.norm(v) < 1e-3:
        # n is parallel to world_up; use arbitrary tangent
        v = np.array([1.0, 0.0, 0.0])
        v = v - n * (v @ n)
    v = v / np.linalg.norm(v)
    v = -v   # image down (away from up)
    u = np.cross(n, v)
    u = u / np.linalg.norm(u)
    return u, v

text_inkness = np.zeros(len(cp.vertices), dtype=np.float32)
placed = []
for area_id in sorted(iso_kid_set):
    acro = area_acro.get(int(area_id), '')
    if not acro or acro in SKIP_ACRO:
        continue
    area_mask = vert_area == int(area_id)
    if area_mask.sum() < 80:
        continue
    # Outside-facing vertices within this area
    out_mask = area_mask & outside_mask_global
    if out_mask.sum() < 50:
        continue
    pts = cp.vertices[out_mask]
    norms = vn[out_mask]
    # Area's mean outward normal
    mean_n = norms.mean(axis=0)
    mean_n /= (np.linalg.norm(mean_n) + 1e-12)

    # Keep vertices whose normal aligns with mean_n (so the label sits
    # on the AREA'S OWN outside face, not a curve-around edge)
    align = norms @ mean_n
    keep = align > 0.5
    if keep.sum() < 30:
        keep = align > 0.3
    if keep.sum() < 20:
        continue
    pts_k = pts[keep]; norms_k = norms[keep]

    # Bitmap
    bitmap = text_bitmap(acro)
    H_px, W_px = bitmap.shape

    # PCA on the (u, v) projection at this site to size the label
    # Project all area vertices onto a plane orthogonal to mean_n
    base_u, base_v = tangent_frame_for_normal(mean_n)
    pts_uv = np.column_stack([pts_k @ base_u, pts_k @ base_v])
    uv_ctr = pts_uv.mean(axis=0); uv_c = pts_uv - uv_ctr
    cov = np.cov(uv_c.T) if len(uv_c) > 1 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    pc1 = evecs[:, 1].astype(float)   # principal axis in (u, v) plane
    proj1 = uv_c @ pc1; proj2 = uv_c @ evecs[:, 0]
    ext1 = float(proj1.max() - proj1.min())
    ext2 = float(proj2.max() - proj2.min())
    ps_for_w = (FIT_FRAC_W * ext1) / W_px
    ps_for_h = (FIT_FRAC_H * ext2) / H_px
    pixel_size_mm = min(ps_for_w, ps_for_h, MAX_LETTER_MM / H_px)
    letter_mm = pixel_size_mm * H_px
    if letter_mm < MIN_LETTER_MM:
        continue

    # Site: most-aligned vertex closest to the (u, v) centroid
    d_uv = np.linalg.norm(uv_c, axis=1)
    closest = np.argsort(d_uv)[:min(80, len(pts_k))]
    best = closest[np.argmax(align[keep][closest])]
    site = pts_k[best]
    site_n = norms_k[best]

    # Build label-readable tangent frame at site, rotated by PC1
    pc1_3d = base_u * pc1[0] + base_v * pc1[1]
    u_ax = pc1_3d - site_n * (pc1_3d @ site_n)
    nu = np.linalg.norm(u_ax)
    if nu < 1e-6:
        u_ax, v_ax = tangent_frame_for_normal(site_n)
    else:
        u_ax = u_ax / nu
        v_ax = np.cross(site_n, u_ax)
        v_ax = v_ax / np.linalg.norm(v_ax)
        # Make sure v_ax aligns with the world-up projection so text
        # reads upright when looking at the area face-on
        v_pref = np.array([0, -1, 0]) - site_n * (np.array([0, -1, 0]) @ site_n)
        if np.linalg.norm(v_pref) > 1e-3:
            v_pref = v_pref / np.linalg.norm(v_pref)
            if v_ax @ (-v_pref) < 0:
                v_ax = -v_ax; u_ax = -u_ax

    # Project ALL cortex vertices into this area's (u, v, w) frame
    rel = cp.vertices - site
    u_co = rel @ u_ax
    v_co = rel @ v_ax
    w_co = rel @ site_n
    col = u_co / pixel_size_mm + W_px / 2
    row = v_co / pixel_size_mm + H_px / 2
    in_box = ((col >= 0) & (col < W_px - 1) &
               (row >= 0) & (row < H_px - 1) &
               (np.abs(w_co) < DEPTH_TOL_MM) &
               area_mask)
    if not in_box.any():
        continue
    ink_vals = bilinear(bitmap, col[in_box], row[in_box])
    idx = np.where(in_box)[0]
    text_inkness[idx] = np.maximum(text_inkness[idx], ink_vals)
    placed.append((acro, letter_mm, mean_n.copy()))

# Etch
cp.vertices = cp.vertices - vn * (LABEL_DEPTH_MM * text_inkness)[:, None]
print(f'\nplaced {len(placed)} labels:')
for acro, lm, n in placed:
    # describe direction
    d = ['dorsal' if n[1] < -0.5 else
         'lateral_R' if n[2] > 0.5 else
         'lateral_L' if n[2] < -0.5 else
         'anterior' if n[0] < -0.5 else
         'posterior' if n[0] > 0.5 else
         'ventral' if n[1] > 0.5 else
         'oblique'][0]
    print(f'  {acro:10s} {d:10s}  letter={lm:.1f}mm')

out_stl = OUT_DIR / 'stl' / 'cortex_hat' / f'allen_ccf_cortex_hat_outsidelabeled_{int(TARGET_MM)}mm.stl'
cp.export(str(out_stl))
print(f'\nwrote {out_stl.name} ({out_stl.stat().st_size/1024/1024:.1f} MB)')

# Render: dorsal + lateral_R + lateral_L + ventral
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
d = cp.bounding_box.extents.max() * 1.8

def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

views = [
    ('dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0), 1.0),
    ('lateral R',  (0,  0, d), (0, 0, 0), (0, -1, 0), 1.0),
    ('lateral L',  (0,  0, -d), (0, 0, 0), (0, -1, 0), 1.0),
    ('ventral',    (0,  d, 0), (0, 0, 0), (1, 0, 0), 1.0),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for k, (vname, cam, focal, up, zoom) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    pl.add_mesh(make_poly(cp), color=(0.85, 0.85, 0.85),
                 smooth_shading=True, specular=0.15, ambient=0.20,
                 diffuse=0.95)
    pl.add_text(vname, font_size=10)
    pl.camera_position = (cam, focal, up)
    pl.reset_camera(); pl.camera.zoom(zoom)
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'cortex_hat_outsidelabeled.png'))
pl.close()
print('wrote cortex_hat_outsidelabeled.png')
