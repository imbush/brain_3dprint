"""Cortex labels — fixed orientation + shrink-to-fit + skip report.

Fixes over label_variants.py:
1. ORIENTATION: tangent frame now satisfies u_ax × v_ax = -n, which is
   the right-handed screen frame seen from OUTSIDE the surface, so text
   reads forwards (not mirrored). The earlier code used
   v = cross(n, u) → u×v = +n → mirror-flipped text.
2. SHRINK-TO-FIT: after placing a label, measure what fraction of the
   inked vertices fall OUTSIDE the area. If > tolerance, shrink the
   font and retry until it fits (or drops below MIN_LETTER_MM).
3. SKIP REPORT: prints every area that got no label and why.
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')

FONT_CANDIDATES = [
    ('/System/Library/Fonts/SFNSRounded.ttf', 0, 'SF Rounded'),
    ('/System/Library/Fonts/Avenir Next.ttc', 4, 'Avenir Next Bold'),
    ('/System/Library/Fonts/Helvetica.ttc', 1, 'Helvetica Bold'),
]
FONT_PATH, FONT_INDEX, FONT_NAME = next(
    (p, i, n) for (p, i, n) in FONT_CANDIDATES if Path(p).exists())
print(f'font: {FONT_NAME}')

root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE

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

# diffuse smoothing
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
print('diffusing...')
for _ in range(200):
    S = 0.5 * S + 0.5 * (W @ S)
vert_area = unique_areas[np.argmax(S, axis=1)]

# boundary etch
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

# helpers
def text_bitmap(text, font_pt=160, pad_px=8):
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

def base_tangent(n, world_up=np.array([0, -1, 0])):
    n = n / np.linalg.norm(n)
    v = world_up - n * (world_up @ n)
    if np.linalg.norm(v) < 1e-3:
        v = np.array([1.0, 0.0, 0.0]) - n * (np.array([1.0, 0.0, 0.0]) @ n)
    v = v / np.linalg.norm(v); v = -v
    u = np.cross(v, n)              # u × v = -n (correct, non-mirrored)
    u = u / np.linalg.norm(u)
    return u, v

cortex_ctr = cp.centroid
out_dir = cp.vertices - cortex_ctr
out_dir /= (np.linalg.norm(out_dir, axis=1, keepdims=True) + 1e-12)
vn = cp.vertex_normals.copy()
outside_mask = (vn * out_dir).sum(axis=1) > 0.3

LABEL_DEPTH_MM = 1.2
DEPTH_TOL_MM = 3.0
FIT_FRAC_W = 0.62
FIT_FRAC_H = 0.50
MIN_LETTER_MM = 1.2
MAX_LETTER_MM = 4.0
OVERFLOW_TOL = 0.12   # ≤12% of ink may fall outside the area

text_inkness = np.zeros(len(cp.vertices), dtype=np.float32)
placed = []
skipped = []

for sid in sorted(iso_kid_set):
    acro = area_acro[sid]
    area_mask = vert_area == int(sid)
    n_area = int(area_mask.sum())
    if n_area < 60:
        skipped.append((acro, f'area too small on mesh ({n_area} verts)'))
        continue
    any_placed = False
    for hemi_name, hemi_sign in (('L', -1), ('R', +1)):
        sub = area_mask & (cp.vertices[:, 2] * hemi_sign > 0) & outside_mask
        if sub.sum() < 40:
            continue
        pts = cp.vertices[sub]; norms = vn[sub]
        mean_n = norms.mean(axis=0); mean_n /= (np.linalg.norm(mean_n) + 1e-12)
        # flat patch: keep aligned vertices
        align = norms @ mean_n
        flat = align > 0.6
        if flat.sum() < 25:
            flat = align > 0.4
        if flat.sum() < 15:
            continue
        fpts = pts[flat]; fnorms = norms[flat]
        u0, v0 = base_tangent(mean_n)
        uv = np.column_stack([fpts @ u0, fpts @ v0])
        uv_ctr = uv.mean(axis=0); uv_c = uv - uv_ctr
        cov = np.cov(uv_c.T) if len(uv_c) > 1 else np.eye(2)
        evals, evecs = np.linalg.eigh(cov)
        pc1 = evecs[:, 1].astype(float)
        ext1 = float((uv_c @ pc1).ptp())
        ext2 = float((uv_c @ evecs[:, 0]).ptp())
        # site = vertex closest to (uv) centroid, most aligned
        d_uv = np.linalg.norm(uv_c, axis=1)
        order = np.argsort(d_uv)[:min(80, len(fpts))]
        best = order[np.argmax((fnorms @ mean_n)[order])]
        site = fpts[best]; site_n = fnorms[best]
        # tangent frame at site, rotated to PC1, with u×v=-n
        pc1_3d = u0 * pc1[0] + v0 * pc1[1]
        u_ax = pc1_3d - site_n * (pc1_3d @ site_n)
        nu = np.linalg.norm(u_ax)
        if nu < 1e-6:
            u_ax, v_ax = base_tangent(site_n)
        else:
            u_ax = u_ax / nu
            v_ax = np.cross(u_ax, site_n)     # u × v = -n
            v_ax = v_ax / np.linalg.norm(v_ax)
            # up preference: v should point toward world-down projection
            v_pref = np.array([0, -1, 0]) - site_n * (np.array([0, -1, 0]) @ site_n)
            if np.linalg.norm(v_pref) > 1e-3:
                v_pref = v_pref / np.linalg.norm(v_pref)
                if v_ax @ (-v_pref) < 0:
                    v_ax = -v_ax; u_ax = -u_ax   # 180° rot keeps u×v=-n

        bitmap = text_bitmap(acro)
        H_px, W_px = bitmap.shape
        rel = cp.vertices - site
        u_co = rel @ u_ax; v_co = rel @ v_ax; w_co = rel @ site_n

        # SHRINK-TO-FIT loop
        ps = min((FIT_FRAC_W * ext1) / W_px,
                  (FIT_FRAC_H * ext2) / H_px,
                  MAX_LETTER_MM / H_px)
        success = False
        for _attempt in range(6):
            letter_mm = ps * H_px
            if letter_mm < MIN_LETTER_MM:
                break
            col = u_co / ps + W_px / 2
            row = v_co / ps + H_px / 2
            in_box = ((col >= 0) & (col < W_px - 1) &
                       (row >= 0) & (row < H_px - 1) &
                       (np.abs(w_co) < DEPTH_TOL_MM))
            if not in_box.any():
                ps *= 0.8; continue
            ink = bilinear(bitmap, col[in_box], row[in_box])
            inked = in_box.copy()
            inked[np.where(in_box)[0]] = ink > 0.3
            n_ink = int(inked.sum())
            if n_ink < 5:
                ps *= 0.8; continue
            # overflow = inked vertices that AREN'T in this area
            n_outside_area = int((inked & ~area_mask).sum())
            overflow = n_outside_area / max(n_ink, 1)
            if overflow <= OVERFLOW_TOL:
                # commit: only etch ink vertices inside the area
                commit = inked & area_mask
                vals = bilinear(bitmap,
                                 (u_co[commit] / ps + W_px / 2),
                                 (v_co[commit] / ps + H_px / 2))
                text_inkness[commit] = np.maximum(text_inkness[commit], vals)
                placed.append((acro, hemi_name, letter_mm))
                any_placed = True
                success = True
                break
            ps *= 0.82   # shrink and retry
        # end attempts
    if not any_placed:
        skipped.append((acro, 'no hemisphere fit (small/curved/medial)'))

print(f'\nplaced {len(placed)} labels')
print(f'skipped {len(skipped)} areas:')
for acro, reason in skipped:
    print(f'  {acro:10s}  {reason}')

cp.vertices = cp.vertices - vn * (LABEL_DEPTH_MM * text_inkness)[:, None]
out_stl = OUT_DIR / 'stl' / 'cortex_hat' / f'cortex_hat_fitted_{int(TARGET_MM)}mm.stl'
cp.export(str(out_stl))
print(f'\nwrote {out_stl.name}')

# render
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
d = cp.bounding_box.extents.max() * 1.8
def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))
views = [
    ('dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
    ('lateral R',  (0,  0,  d), (0, 0, 0), (0, -1, 0)),
    ('lateral L',  (0,  0, -d), (0, 0, 0), (0, -1, 0)),
    ('ventral',    (0,  d, 0), (0, 0, 0), (1, 0, 0)),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for k, (vn2, cam, focal, up) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    pl.add_mesh(make_poly(cp), color=(0.85, 0.85, 0.85),
                 smooth_shading=True, specular=0.15, ambient=0.20,
                 diffuse=0.95)
    pl.add_text(vn2, font_size=10)
    pl.camera_position = (cam, focal, up); pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'cortex_hat_fitted.png'))
pl.close()
print('wrote cortex_hat_fitted.png')
