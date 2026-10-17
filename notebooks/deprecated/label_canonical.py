"""Canonical cortex-hat labeling.

Base = Allen's native Isocortex (315) mesh — clean, smooth, watertight,
NO olfactory bulb / hippocampus (that was the 688 = "Cerebral cortex"
mistake). We subdivide it + smooth the area boundaries exactly like the
original `cortex_hat_labeled_solid` that looked good — NO voxelize, NO
union-of-shells, so the surface stays smooth (not wavy).

Area labels come from sampling the annotation volume per vertex (rolled
up to the cortical-area level), then diffused for smooth boundaries.

Label orientation FIX: reading direction u_ax is sign-disambiguated
against a screen-right reference so text always reads forwards.

Shrink-to-fit: labels shrink until they don't overflow the area
boundary. Skip report lists unlabeled areas + reasons.
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

# True Isocortex = 315 (NOT 688 = CTX which has the olfactory bulb)
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
area_acro = {sid: tree.get_structures_by_id([int(sid)])[0]['acronym']
              for sid in iso_kid_set}
print(f'{len(iso_kid_set)} cortical areas')

# === Clean Allen Isocortex (315) mesh — smooth, subdivided twice. ===
cortex = trimesh.load(str(OUT_DIR / 'obj' / '315_Isocortex.obj'),
                       force='mesh')
cortex.merge_vertices()
cortex.update_faces(cortex.unique_faces())
cortex.update_faces(cortex.nondegenerate_faces())
cortex.remove_unreferenced_vertices()
for _ in range(2):
    nv, nf = trimesh.remesh.subdivide(cortex.vertices, cortex.faces)
    cortex = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
print(f'cortex (315, subdivided): {len(cortex.vertices):,} verts')

# Per-vertex area labels via annotation sampling (inward) + NN fill
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

# Build adjacency + diffuse the labels (smooth area boundaries)
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

# Scale to print + center
cp = cortex.copy()
cp.apply_scale(SHARED_SCALE)
cp.apply_translation(-root_ctr)

fv = vert_area[cp.faces]
bf = ((fv[:, 0] != fv[:, 1]) | (fv[:, 0] != fv[:, 2]) |
      (fv[:, 1] != fv[:, 2]))
bv_idx = np.unique(cp.faces[bf].ravel())
if len(bv_idx) > 0:
    b_adj = adj[bv_idx][:, bv_idx]
    b_W = diags(1.0 / np.maximum(np.asarray(b_adj.sum(axis=1)).ravel(),
                                   1.0)).dot(b_adj).tocsr()
    b_pos = cp.vertices[bv_idx].copy().astype(np.float32)
    for _ in range(60):
        b_pos = 0.5 * b_pos + 0.5 * (b_W @ b_pos)
    kd_bd = cKDTree(b_pos)
    dist, _ = kd_bd.query(cp.vertices, k=1)
    prof = np.cos(np.pi/2 * np.clip(dist / 0.6, 0, 1)) ** 2
    cp.vertices = cp.vertices - cp.vertex_normals * (0.4 * prof)[:, None]
    trimesh.smoothing.filter_taubin(cp, iterations=6)

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

vn = cp.vertex_normals.copy()
cortex_ctr = cp.centroid
out_dir = cp.vertices - cortex_ctr
out_dir /= (np.linalg.norm(out_dir, axis=1, keepdims=True) + 1e-12)
outside_mask = (vn * out_dir).sum(axis=1) > 0.2

def reading_frame(site_n, pc1_3d=None):
    """Return a SCREEN-ALIGNED tangent frame (u_ax, v_ax) so text always
    reads horizontally and upright when the area is viewed from outside.
    No PCA rotation — that caused unstable/backwards orientation on
    AP-elongated areas (MOp/MOs) because the principal axis was nearly
    perpendicular to screen-right.

    - screen_up  = dorsal (-Y) projected onto the tangent plane
                   (anterior (-X) fallback for near-horizontal surfaces)
    - u_ax (reading right) = screen_up × n   →  guarantees u×v = -n
    - v_ax (down) = -screen_up"""
    n = site_n / np.linalg.norm(site_n)
    up_world = np.array([0.0, -1.0, 0.0])           # dorsal
    su = up_world - n * (up_world @ n)
    if np.linalg.norm(su) < 0.35:                   # surface ~horizontal
        up_world = np.array([-1.0, 0.0, 0.0])       # use anterior as up
        su = up_world - n * (up_world @ n)
    su = su / np.linalg.norm(su)
    u = np.cross(su, n)                             # reading-right
    u = u / np.linalg.norm(u)
    v = -su                                         # screen-down
    return u, v

LABEL_DEPTH_MM = 1.2
DEPTH_TOL_MM = 3.0
FIT_FRAC_W = 0.62
FIT_FRAC_H = 0.50
MIN_LETTER_MM = 1.0
MAX_LETTER_MM = 4.0
OVERFLOW_TOL = 0.15

text_inkness = np.zeros(len(cp.vertices), dtype=np.float32)
placed = []
skipped = []
for sid in sorted(iso_kid_set):
    acro = area_acro[sid]
    area_mask = vert_area == int(sid)
    if area_mask.sum() < 60:
        skipped.append((acro, f'small ({int(area_mask.sum())} verts)'))
        continue
    any_placed = False
    for hemi_sign in (-1, +1):
        sub = area_mask & (cp.vertices[:, 2] * hemi_sign > 0) & outside_mask
        if sub.sum() < 40:
            continue
        pts = cp.vertices[sub]; norms = vn[sub]
        mean_n = norms.mean(axis=0); mean_n /= (np.linalg.norm(mean_n) + 1e-12)
        align = norms @ mean_n
        flat = align > 0.5
        if flat.sum() < 20:
            flat = align > 0.3
        if flat.sum() < 12:
            continue
        fpts = pts[flat]; fnorms = norms[flat]
        # Site = vertex closest to the flat-patch centroid, most aligned
        # with the mean normal (a stable, central spot).
        pc = fpts - fpts.mean(axis=0)
        d_ctr = np.linalg.norm(pc, axis=1)
        order = np.argsort(d_ctr)[:min(80, len(fpts))]
        best = order[np.argmax((fnorms @ mean_n)[order])]
        site = fpts[best]; site_n = fnorms[best]
        # Screen-aligned reading frame (no PCA → always upright/forward)
        u_ax, v_ax = reading_frame(site_n)

        bitmap = text_bitmap(acro)
        H_px, W_px = bitmap.shape
        rel = cp.vertices - site
        u_co = rel @ u_ax; v_co = rel @ v_ax; w_co = rel @ site_n
        # Size to the area's extent ALONG the actual label axes (u, v),
        # measured over the flat-patch vertices.
        f_u = (fpts - site) @ u_ax
        f_v = (fpts - site) @ v_ax
        ext_u = float(f_u.ptp())
        ext_v = float(f_v.ptp())
        ps = min((FIT_FRAC_W * ext_u) / W_px,
                  (FIT_FRAC_H * ext_v) / H_px,
                  MAX_LETTER_MM / H_px)
        for _att in range(7):
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
            inked = np.zeros(len(cp.vertices), dtype=bool)
            inked[np.where(in_box)[0]] = ink > 0.3
            n_ink = int(inked.sum())
            if n_ink < 5:
                ps *= 0.8; continue
            overflow = int((inked & ~area_mask).sum()) / max(n_ink, 1)
            if overflow <= OVERFLOW_TOL:
                commit = inked & area_mask
                vals = bilinear(bitmap,
                                 u_co[commit] / ps + W_px / 2,
                                 v_co[commit] / ps + H_px / 2)
                text_inkness[commit] = np.maximum(text_inkness[commit], vals)
                placed.append((acro, 'L' if hemi_sign < 0 else 'R',
                                letter_mm))
                any_placed = True
                break
            ps *= 0.82
    if not any_placed:
        skipped.append((acro, 'no fit (mostly medial / curved)'))

print(f'\nplaced {len(placed)} labels')
print(f'skipped {len(skipped)}:')
for acro, why in skipped:
    print(f'  {acro:10s} {why}')

cp.vertices = cp.vertices - vn * (LABEL_DEPTH_MM * text_inkness)[:, None]
out_stl = OUT_DIR / 'stl' / 'cortex_hat' / f'cortex_hat_canonical_{int(TARGET_MM)}mm.stl'
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
    ('anterior',   (-d, 0, 0), (0, 0, 0), (0, -1, 0)),
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
pl.screenshot(str(OUT_DIR / 'png' / 'cortex_hat_canonical.png'))
pl.close()

# dorsal zoom to verify orientation
poly = make_poly(cp)
pl = pv.Plotter(shape=(1, 2), window_size=(2400, 1100), off_screen=True)
for k, focal in enumerate([(20, 0, 0), (-25, 0, 0)]):
    pl.subplot(0, k)
    pl.add_mesh(poly, color=(0.85,)*3, smooth_shading=True,
                 specular=0.15, ambient=0.2, diffuse=0.95)
    pl.add_text('dorsal zoom', font_size=11)
    pl.camera_position = ((0, -d, 0), focal, (-1, 0, 0))
    pl.reset_camera(); pl.camera.zoom(3.0)
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'cortex_hat_canonical_zoom.png'))
pl.close()
print('wrote cortex_hat_canonical.png + _zoom.png')
