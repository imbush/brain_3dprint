"""Label each of the 9 major-division meshes (CNU, HPF, CTXsp, OLF,
TH, HY, MB, HB, CB) with its acronym on its outside surface.

Renders all 9 overlaid in their colors with labels engraved on each.
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from scipy.sparse import csr_matrix, diags
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')

# Use the same regions + same colors as _per_region_test
REGIONS = [
    ('CNU',   'Cerebral nuclei',         '#5dd39e'),
    ('HPF',   'Hippocampal formation',   '#ff9f1c'),
    ('CTXsp', 'Cortical subplate',       '#bdb2ff'),
    ('TH',    'Thalamus',                '#cf6cf6'),
    ('HY',    'Hypothalamus',            '#f08080'),
    ('MB',    'Midbrain',                '#5a9bff'),
    ('HB',    'Hindbrain',               '#118ab2'),
    ('CB',    'Cerebellum',              '#fa8072'),
    ('OLF',   'Olfactory areas',         '#ffd166'),
]

FONT_CANDIDATES = [
    ('/System/Library/Fonts/SFNSRounded.ttf', 0),
    ('/System/Library/Fonts/Avenir Next.ttc', 4),
    ('/System/Library/Fonts/Helvetica.ttc', 1),
]
FONT_PATH, FONT_INDEX = next((p, i) for (p, i) in FONT_CANDIDATES
                              if Path(p).exists())
print(f'using font: {FONT_PATH}, idx={FONT_INDEX}')

# Already-loaded per-region STLs (at 5x = 66 mm)
TARGET_MM = 66.05
loaded = {}
for acro, name, color in REGIONS:
    p = OUT_DIR / 'stl' / 'regions' / f'allen_ccf_region_{acro}_{int(TARGET_MM)}mm.stl'
    if not p.exists():
        print(f'  ✗ missing {p.name}')
        continue
    m = trimesh.load(str(p), force='mesh')
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    loaded[acro] = (color, name, m)
    print(f'  {acro:6s}  {len(m.vertices):>6,} verts, {len(m.faces):>7,} faces')

print(f'\nloaded {len(loaded)} region meshes')

# Subdivide each mesh once for finer label resolution
for acro in list(loaded.keys()):
    color, name, m = loaded[acro]
    nv, nf = trimesh.remesh.subdivide(m.vertices, m.faces)
    loaded[acro] = (color, name, trimesh.Trimesh(vertices=nv, faces=nf,
                                                    process=False))

# ===  Label helpers ===
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

LABEL_DEPTH_MM = 1.5
DEPTH_TOL_MM = 4.0
FIT_FRAC_W = 0.55
FIT_FRAC_H = 0.45
MIN_LETTER_MM = 2.0
MAX_LETTER_MM = 6.0

# Etch each region's acronym on its own surface
labeled_meshes = {}
for acro, (color, name, m) in loaded.items():
    vn = m.vertex_normals
    # Outside: every vertex of an isolated region is outside-facing
    # (no enclosed cavity), so use ALL vertices.
    pts = m.vertices
    mean_n = vn.mean(axis=0)
    mean_n /= (np.linalg.norm(mean_n) + 1e-12)
    # If the region has no clear mean direction (e.g. CB wraps), use
    # 0,-1,0 (dorsal) as default for the label
    if np.linalg.norm(mean_n) < 0.1:
        mean_n = np.array([0, -1, 0])

    # For bilateral regions (HPF, CTXsp, OLF), label each hemisphere
    # separately. Detect bilateral by checking if pts span both z<0
    # and z>0 with significant mass on each side.
    z_pts = pts[:, 2]
    bilateral = (z_pts < -1).any() and (z_pts > 1).any()

    text_inkness = np.zeros(len(m.vertices), dtype=np.float32)
    for hemi_name, hemi_sign in (('L', -1), ('R', +1)) if bilateral \
                                  else (('whole', 0),):
        if hemi_sign != 0:
            mask = pts[:, 2] * hemi_sign > 0
        else:
            mask = np.ones(len(pts), dtype=bool)
        sub_pts = pts[mask]; sub_norms = vn[mask]
        if len(sub_pts) < 50: continue
        # Find the outward-facing centroid
        # Filter: top-half-most-outward-facing vertices
        out_dir = sub_pts - m.centroid
        out_dir /= np.linalg.norm(out_dir, axis=1, keepdims=True) + 1e-12
        outness = (sub_norms * out_dir).sum(axis=1)
        keep = outness > np.percentile(outness, 60)
        if keep.sum() < 30: continue
        pts_k = sub_pts[keep]; norms_k = sub_norms[keep]
        mean_n_h = norms_k.mean(axis=0)
        mean_n_h /= (np.linalg.norm(mean_n_h) + 1e-12)

        u0, v0 = tangent_frame(mean_n_h)
        pts_uv = np.column_stack([pts_k @ u0, pts_k @ v0])
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
        closest = np.argsort(d_uv)[:min(80, len(pts_k))]
        best = closest[np.argmax((norms_k @ mean_n_h)[closest])]
        site = pts_k[best]; site_n = norms_k[best]
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
        rel = m.vertices - site
        u_co = rel @ u_ax; v_co = rel @ v_ax; w_co = rel @ site_n
        col = u_co / pixel_size_mm + W_px / 2
        row = v_co / pixel_size_mm + H_px / 2
        in_box = ((col >= 0) & (col < W_px - 1) &
                   (row >= 0) & (row < H_px - 1) &
                   (np.abs(w_co) < DEPTH_TOL_MM))
        if not in_box.any(): continue
        ink_vals = bilinear(bitmap, col[in_box], row[in_box])
        idx = np.where(in_box)[0]
        text_inkness[idx] = np.maximum(text_inkness[idx], ink_vals)

    m_l = m.copy()
    m_l.vertices = m_l.vertices - vn * (LABEL_DEPTH_MM * text_inkness)[:, None]
    labeled_meshes[acro] = (color, name, m_l)
    print(f'  {acro:6s}  inked vertices: {int((text_inkness > 0).sum())}')

# Render combined
pv.OFF_SCREEN = True
pv.set_plot_theme('document')

def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

all_v = np.vstack([m.vertices for _, _, m in labeled_meshes.values()])
d = (all_v.max(0) - all_v.min(0)).max() * 1.8

views = [
    ('dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
    ('lateral R',  (0,  0,  d), (0, 0, 0), (0, -1, 0)),
    ('caudal',     (d,  0,  0), (0, 0, 0), (0, -1, 0)),
    ('oblique',    (0.6*d, -0.5*d, 0.6*d), (0, 0, 0), (0, -1, 0)),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for k, (vname, cam, focal, up) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    for acro, (color, name, m) in labeled_meshes.items():
        pl.add_mesh(make_poly(m), color=color,
                     smooth_shading=True, specular=0.20,
                     ambient=0.25, diffuse=0.90)
    pl.add_text(vname, font_size=12)
    pl.camera_position = (cam, focal, up)
    pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'per_region_labeled.png'))
pl.close()
print('\nwrote per_region_labeled.png')

# Save each labeled region STL (replaces the unlabeled originals)
for acro, (color, name, m) in labeled_meshes.items():
    p = OUT_DIR / 'stl' / 'regions' / f'allen_ccf_region_{acro}_labeled_{int(TARGET_MM)}mm.stl'
    m.export(str(p))
print('saved labeled per-region STLs')
