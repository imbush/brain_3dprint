"""Cortex hat with etched cortical-area acronyms.

Fixes:
- Bilateral: each area gets a label on left AND right hemisphere
  (split at z = 0 in centered print coords).
- Per-area sizing: font scales to fit the area's local extent in
  each hemisphere.
- Containment: only mesh vertices BELONGING to the area being
  labeled get etched.
- Sharper edges: no diffusion-smoothing of the ink mask.
"""
import sys, time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
import pyvista as pv
import matplotlib as mpl
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path('/Users/inlebush/lab/ila/microns')
OUT_DIR = REPO_ROOT / 'data' / 'ccf_stl'
CCF_VERSION = ReferenceSpaceApi.CCF_2017

root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
# Print scale: 5x real-life mouse brain (real AP ≈ 13.2 mm → ≈ 66 mm print).
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE
print(f'TARGET_MM = {TARGET_MM:.2f} mm  ({SCALE_FACTOR}x real life)')

RES_UM = 25
rspc = ReferenceSpaceCache(resolution=RES_UM,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
ann, _ = rspc.get_annotation_volume()
tree = rspc.get_structure_tree()

ISO_ID = 688
def is_layer_acro(a):
    return (a.endswith('2/3') or a.endswith('6a') or a.endswith('6b')
             or (len(a) >= 2 and a[-1].isdigit()))
iso_kid_set = set()
for sid in tree.descendant_ids([ISO_ID])[0]:
    if sid == ISO_ID: continue
    kids = tree.child_ids([int(sid)])[0]
    if not kids: continue
    kid_acros = [tree.get_structures_by_id([int(k)])[0]['acronym']
                  for k in kids]
    if all(is_layer_acro(a) for a in kid_acros):
        iso_kid_set.add(int(sid))
print(f'{len(iso_kid_set)} cortical areas')

cortex = trimesh.load(OUT_DIR / 'obj' / '688_isocortex.obj', force='mesh')
cortex.merge_vertices()
cortex.update_faces(cortex.unique_faces())
cortex.update_faces(cortex.nondegenerate_faces())
cortex.remove_unreferenced_vertices()
for _ in range(2):
    nv, nf = trimesh.remesh.subdivide(cortex.vertices, cortex.faces)
    cortex = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
print(f'cortex: {len(cortex.vertices):,} verts, {len(cortex.faces):,} faces')

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
tic = time.time()
for _ in range(200):
    S = 0.5 * S + 0.5 * (W @ S)
vert_area = unique_areas[np.argmax(S, axis=1)]
print(f'  done in {time.time()-tic:.1f}s')

fv = vert_area[cortex.faces]
bf = (fv[:, 0] != fv[:, 1]) | (fv[:, 0] != fv[:, 2]) | \
     (fv[:, 1] != fv[:, 2])
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

# Very shallow area grooves so labels are the dominant feature.
ETCH_DEPTH = 0.4
HALFW = 0.6
kd_bd = cKDTree(b_pos_print)
dist, _ = kd_bd.query(cp.vertices, k=1)
prof = np.cos(np.pi/2 * np.clip(dist / HALFW, 0, 1)) ** 2
vn0 = cp.vertex_normals.copy()
cp.vertices = cp.vertices - vn0 * (ETCH_DEPTH * prof)[:, None]
trimesh.smoothing.filter_taubin(cp, iterations=8)

# ==== LABEL ETCHING (bilateral, per-area-sized, contained, sharp) ====
area_acro = {int(sid): tree.get_structures_by_id([int(sid)])[0]['acronym']
              for sid in iso_kid_set}
# Acronym → ontology id (only for dorsally-visible areas — anything
# not in DORSAL_AREAS gets skipped).
DORSAL_AREAS = {
    'FRP', 'MOp', 'MOs',
    'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul', 'SSp-un',
    'SSs',
    'VISp', 'VISal', 'VISam', 'VISpm', 'VISpl', 'VISl', 'VISrl', 'VISa',
    'AUDp',
    'ACAd', 'ACAv',
    'RSPd', 'RSPv',
    'ORBm',
}
dorsal_ids = {sid for sid, a in area_acro.items() if a in DORSAL_AREAS}
print(f'dorsal areas (curated): {len(dorsal_ids)} of {len(DORSAL_AREAS)} '
      f'requested')

FONT_PATH      = '/System/Library/Fonts/Helvetica.ttc'
FONT_INDEX     = 1                # 1 = Helvetica Bold
FONT_PT        = 160              # large render so anti-aliased edges
                                   # are smooth gradients, not a few pixels
PAD_PX         = 6
LABEL_DEPTH_MM = 1.2
DEPTH_TOL_MM   = 3.0
DORSAL_NORM_TH = -0.55
FIT_FRAC_W     = 0.55             # label W ≤ 55% of area extent along PC1
FIT_FRAC_H     = 0.50             # label H ≤ 50% of area extent along PC2
MIN_LETTER_MM  = 1.6
MAX_LETTER_MM  = 3.5              # tighter cap so labels never dominate areas

def text_bitmap(text):
    """Anti-aliased text bitmap: returns (H, W) float in [0, 1].
    1 = solid ink center, 0 = no ink, with smooth gradient at edges.
    The smooth gradient is the key to non-staircase letter curves on
    a polygonal mesh — bilinear sampling of this field gives the
    cortex vertices a continuous displacement that matches the
    vector-letter shape."""
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_PT, index=FONT_INDEX)
    except OSError:
        font = ImageFont.truetype(FONT_PATH, FONT_PT)
    img0 = Image.new('L', (1, 1))
    bbox = ImageDraw.Draw(img0).textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 2 * PAD_PX
    h = bbox[3] - bbox[1] + 2 * PAD_PX
    img = Image.new('L', (w, h), color=255)
    ImageDraw.Draw(img).text(
        (PAD_PX - bbox[0], PAD_PX - bbox[1]), text, font=font, fill=0)
    return 1.0 - np.asarray(img, dtype=np.float32) / 255.0

def bilinear(field, col, row):
    """Bilinearly sample `field` (H, W) at fractional (col, row)."""
    H_, W_ = field.shape
    c0 = np.floor(col).astype(int)
    r0 = np.floor(row).astype(int)
    c1 = c0 + 1
    r1 = r0 + 1
    cf = col - c0
    rf = row - r0
    c0 = np.clip(c0, 0, W_ - 1); c1 = np.clip(c1, 0, W_ - 1)
    r0 = np.clip(r0, 0, H_ - 1); r1 = np.clip(r1, 0, H_ - 1)
    return (field[r0, c0] * (1 - cf) * (1 - rf) +
            field[r0, c1] * cf       * (1 - rf) +
            field[r1, c0] * (1 - cf) * rf       +
            field[r1, c1] * cf       * rf)

WORLD_RIGHT = np.array([0.0, 0.0, 1.0])
WORLD_DOWN  = np.array([1.0, 0.0, 0.0])
def tangent_frame(normal):
    u = WORLD_RIGHT - normal * (WORLD_RIGHT @ normal)
    if np.linalg.norm(u) < 1e-3:
        u = np.cross(normal, WORLD_DOWN)
    u = u / np.linalg.norm(u)
    v = WORLD_DOWN - normal * (WORLD_DOWN @ normal) - u * (WORLD_DOWN @ u)
    if np.linalg.norm(v) < 1e-3:
        v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return u, v

vn = cp.vertex_normals.copy()
text_inkness = np.zeros(len(cp.vertices), dtype=np.float32)
placed = []  # (area_acro, hemisphere, letter_mm)
for area_id in sorted(iso_kid_set):
    if int(area_id) not in dorsal_ids:
        continue
    text = area_acro.get(int(area_id), '')
    if not text: continue
    bitmap = text_bitmap(text)        # float (H, W) in [0, 1]
    H_px, W_px = bitmap.shape

    area_mask_full = vert_area == int(area_id)
    if area_mask_full.sum() < 50: continue
    for hemi_name, hemi_sign in (('L', -1), ('R', +1)):
        hemi_mask = area_mask_full & (cp.vertices[:, 2] * hemi_sign > 0) \
                    & (vn[:, 1] < DORSAL_NORM_TH)
        if hemi_mask.sum() < 80: continue
        hpts = cp.vertices[hemi_mask]
        hnorms = vn[hemi_mask]

        # PCA on (X, Z) — text reading direction follows the area's
        # principal axis so an elongated area gets a rotated label.
        xz = hpts[:, [0, 2]]
        xz_ctr = xz.mean(axis=0)
        xz_c = xz - xz_ctr
        cov = np.cov(xz_c.T) if len(xz_c) > 1 else np.eye(2) * 1.0
        evals, evecs = np.linalg.eigh(cov)        # ascending
        pc1 = evecs[:, 1].astype(float)            # principal axis (X, Z)
        pc2 = evecs[:, 0].astype(float)            # perp axis
        proj1 = xz_c @ pc1; proj2 = xz_c @ pc2
        ext1 = float(proj1.max() - proj1.min())
        ext2 = float(proj2.max() - proj2.min())

        ps_for_w = (FIT_FRAC_W * ext1) / W_px
        ps_for_h = (FIT_FRAC_H * ext2) / H_px
        pixel_size_mm = min(ps_for_w, ps_for_h, MAX_LETTER_MM / H_px)
        letter_mm = pixel_size_mm * H_px
        if letter_mm < MIN_LETTER_MM:
            continue

        # Build u_ax along PC1, projected onto the local tangent plane.
        d_xz = np.linalg.norm(xz_c, axis=1)
        n_close = min(80, len(hpts))
        closest = np.argsort(d_xz)[:n_close]
        best = closest[np.argmin(hpts[closest, 1])]
        site = hpts[best]
        site_n = hnorms[best]

        pc1_3d = np.array([pc1[0], 0.0, pc1[1]])
        u_ax = pc1_3d - site_n * (pc1_3d @ site_n)
        nu = np.linalg.norm(u_ax)
        if nu < 1e-6:
            u_ax, _ = tangent_frame(site_n)
        else:
            u_ax = u_ax / nu
        v_ax = np.cross(site_n, u_ax)
        v_ax = v_ax / np.linalg.norm(v_ax)
        # Make the text right-side-up in the dorsal view:
        # v_ax (image down) should have a positive +X component
        # (posterior). If not, flip both axes (= 180° rotation).
        if v_ax[0] < 0:
            v_ax = -v_ax
            u_ax = -u_ax

        rel = cp.vertices - site
        u_co = rel @ u_ax
        v_co = rel @ v_ax
        w_co = rel @ site_n
        col = u_co / pixel_size_mm + W_px / 2
        row = v_co / pixel_size_mm + H_px / 2
        in_box = ((col >= 0) & (col < W_px - 1) &
                   (row >= 0) & (row < H_px - 1) &
                   (np.abs(w_co) < DEPTH_TOL_MM) &
                   area_mask_full)
        if not in_box.any():
            continue
        # Anti-aliased bilinear sampling — every vertex inside the
        # bitmap bounds gets a continuous depth weight ∈ [0, 1].
        ink_vals = bilinear(bitmap, col[in_box], row[in_box])
        idx = np.where(in_box)[0]
        text_inkness[idx] = np.maximum(text_inkness[idx], ink_vals)
        placed.append((text, hemi_name, letter_mm))

print(f'placed {len(placed)} labels')
for name, h, lm in placed[:20]:
    print(f'  {name:>10s} {h}  letter={lm:.2f}mm')
print('...' if len(placed) > 20 else '')

# NO diffusion smoothing — keep letter edges sharp.
print(f'inked vertices: {int((text_inkness > 0).sum())}')

cp.vertices = cp.vertices - vn * (LABEL_DEPTH_MM * text_inkness)[:, None]

# ==== Bregma + lambda landmarks (north star / Southern Cross) ====
# Allen CCF v3 stereotaxic dorsal-surface reference points (approx):
#   bregma ≈ (AP 5400, DV top, ML midline)
#   lambda ≈ (AP 9600, DV top, ML midline)  — ~4.2 mm posterior of bregma
# The volume midline is at ML ≈ 5700 µm.
BREGMA_CCF_UM = np.array([5400.0, 0.0, 5700.0])
LAMBDA_CCF_UM = np.array([9600.0, 0.0, 5700.0])
bregma_xz = (BREGMA_CCF_UM * SHARED_SCALE - root_ctr)[[0, 2]]
lambda_xz = (LAMBDA_CCF_UM * SHARED_SCALE - root_ctr)[[0, 2]]
print(f'bregma XZ (mm): {bregma_xz.round(2)}')
print(f'lambda XZ (mm): {lambda_xz.round(2)}')

# Refresh normals (mesh was just modified by text etching).
vn = cp.vertex_normals.copy()
xz_all = cp.vertices[:, [0, 2]]

def star_bitmap(size_px=512, n_points=5, r_in_frac=0.40):
    """N-pointed star, anti-aliased."""
    img = Image.new('L', (size_px, size_px), color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = size_px / 2, size_px / 2
    r_out = size_px * 0.46
    r_in  = r_out * r_in_frac
    pts = []
    for i in range(n_points * 2):
        ang = -np.pi / 2 + i * np.pi / n_points
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
    draw.polygon(pts, fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0

def crux_bitmap(size_px=512):
    """Southern Cross constellation — 4 main stars + 1 smaller offset
    (Epsilon Crucis). Each star is itself a 5-pointed star, sized to
    read clearly on a 16 mm bitmap projection."""
    out = np.zeros((size_px, size_px), dtype=np.float32)
    cx, cy = size_px / 2, size_px / 2
    R = size_px * 0.34
    # Each entry: (dx, dy, scale). Bottom Acrux is brightest.
    stars = [
        (0,         -R,         1.00),   # top    Gacrux
        ( R * 0.50,  R * 0.10,  0.90),   # right  δ Crucis
        (0,          R,         1.20),   # bottom Acrux
        (-R * 0.50,  R * 0.10,  0.90),   # left   Mimosa
        ( R * 0.22,  R * 0.40,  0.55),   # ε Crucis (smaller, offset)
    ]
    for dx, dy, scl in stars:
        s_px = max(24, int(size_px * 0.30 * scl))
        s = star_bitmap(size_px=s_px, r_in_frac=0.40)
        x0 = int(cx + dx - s_px / 2); y0 = int(cy + dy - s_px / 2)
        x1 = x0 + s_px;               y1 = y0 + s_px
        ax0 = max(0, x0); ay0 = max(0, y0)
        ax1 = min(size_px, x1); ay1 = min(size_px, y1)
        if ax1 <= ax0 or ay1 <= ay0: continue
        out[ay0:ay1, ax0:ax1] = np.maximum(
            out[ay0:ay1, ax0:ax1],
            s[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0])
    return out

MARKER_DEPTH_MM = 2.0
MARKER_SIZE_MM  = 16.0

# Lambda sits posterior to the cortex shell (it's over the cerebellum)
# so the anatomically-correct lambda is outside the cortex hat. Anchor
# the southern cross to the most-posterior dorsal cortex point on the
# midline so it still marks "back of the cortex" cleanly.
dorsal_top = vn[:, 1] < -0.5
near_midline = np.abs(cp.vertices[:, 2]) < 3.0
post_mask = dorsal_top & near_midline
if post_mask.any():
    post_idx = np.where(post_mask)[0]
    post_x = cp.vertices[post_idx, 0].max()
    # Place the cross 5 mm anterior of the posterior cortex edge.
    lambda_xz_anchored = np.array([post_x - 5.0, 0.0], dtype=np.float32)
else:
    lambda_xz_anchored = lambda_xz
print(f'lambda (anchored to cortex): XZ={lambda_xz_anchored.round(2)}')

landmarks = [
    ('north_star',     bregma_xz,           star_bitmap(size_px=512)),
    ('southern_cross', lambda_xz_anchored,  crux_bitmap(size_px=512)),
]

# Top-down projection: ignore cortex curvature, place the marker at
# the target (X, Z) and etch every dorsal-facing vertex underneath
# the bitmap. The cortex doesn't have a clean dorsal-midline surface
# (longitudinal fissure splits the two hemispheres), so a surface-
# tangent placement misfires. Top-down projection gives a marker
# centered on the geometric landmark regardless of the local
# medial-wall geometry — the marker spans both hemispheres.
landmark_inkness = np.zeros_like(text_inkness)
for name, target_xz, bm in landmarks:
    Hp, Wp = bm.shape
    pixel_mm = MARKER_SIZE_MM / max(Hp, Wp)
    col = (cp.vertices[:, 2] - target_xz[1]) / pixel_mm + Wp / 2
    row = (cp.vertices[:, 0] - target_xz[0]) / pixel_mm + Hp / 2
    in_box = ((col >= 0) & (col < Wp - 1) &
               (row >= 0) & (row < Hp - 1) &
               (vn[:, 1] < -0.25))
    if not in_box.any(): continue
    vals = bilinear(bm, col[in_box], row[in_box])
    idx = np.where(in_box)[0]
    landmark_inkness[idx] = np.maximum(landmark_inkness[idx], vals)
    print(f'  marker {name} @ XZ={target_xz.round(2)}: '
          f'inked {int((vals > 0.05).sum())} vertices')

cp.vertices = cp.vertices - vn * (MARKER_DEPTH_MM * landmark_inkness)[:, None]

cp.export(OUT_DIR / 'stl' / 'cortex_hat' / 'allen_ccf_cortex_hat_labeled_100mm.stl')
print(f'wrote cortex STL: {len(cp.faces):,} faces')

# ==== Render ====
face_area = fv[:, 0]
face_area = np.where((fv[:, 1] == fv[:, 2]) & (fv[:, 0] != fv[:, 1]),
                      fv[:, 1], face_area)
areas_present = sorted({int(a) for a in face_area if a > 0})
pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors) +
       list(mpl.colormaps['tab20c'].colors))
area_color = {sid: pal[i % len(pal)] for i, sid in enumerate(areas_present)}
face_rgb = np.array([area_color.get(int(a), (0.6, 0.6, 0.6))
                      for a in face_area])
face_rgb_solid = np.full_like(face_rgb, fill_value=0.83)

pv.OFF_SCREEN = True
pv.set_plot_theme('document')
d = cp.bounding_box.extents.max() * 1.8

views = [
    ('dorsal',                  (0, -d, 0), (0, 0, 0), (-1, 0, 0), 1.0),
    ('bregma (north star)',     (0, -d, 0),
                                  (float(bregma_xz[0]), 0, float(bregma_xz[1])),
                                  (-1, 0, 0), 5.0),
    ('lambda (southern cross)', (0, -d, 0),
                                  (float(lambda_xz_anchored[0]), 0,
                                   float(lambda_xz_anchored[1])),
                                  (-1, 0, 0), 5.0),
    ('oblique',                 (0.4*d, -0.7*d, 0.3*d), (0, 0, 0),
                                  (0, -1, 0), 1.0),
]
def render(face_colors, fname):
    pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800),
                     off_screen=True)
    for k, (vname, cam_pos, focal, up, zoom) in enumerate(views):
        pl.subplot(k // 2, k % 2)
        poly = pv.PolyData(
            cp.vertices,
            np.hstack([np.full((len(cp.faces), 1), 3),
                         cp.faces]).astype(np.int64))
        poly.cell_data['rgb'] = (face_colors * 255).astype(np.uint8)
        pl.add_mesh(poly, scalars='rgb', rgb=True,
                     smooth_shading=True, specular=0.15,
                     ambient=0.20, diffuse=0.95)
        pl.add_text(vname, font_size=10)
        pl.camera_position = (cam_pos, focal, up)
        pl.reset_camera()
        pl.camera.zoom(zoom)
    pl.background_color = 'white'
    pl.screenshot(str(fname))
    pl.close()
    print(f'wrote {fname.name}')

render(face_rgb_solid, OUT_DIR / 'png' / 'cortex_hat_labeled_solid.png')
render(face_rgb,       OUT_DIR / 'png' / 'cortex_hat_labeled_color.png')
print('done')
