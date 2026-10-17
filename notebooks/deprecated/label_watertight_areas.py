"""Engrave each area's acronym onto its own watertight area piece.

Loads stl/cortical_areas/wt_<acro>_66mm.stl (already watertight, print
scale, shared frame). For each connected component (= hemisphere),
places the acronym on the outside-facing centroid using the
screen-aligned reading frame (forward + upright) + shrink-to-fit, then
bilinear-engraves. Saves wtl_<acro>_66mm.stl (watertight + labeled).
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import matplotlib as mpl

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
TARGET_MM = 66  # files are named *_66mm

FONT_CANDIDATES = [
    ('/System/Library/Fonts/SFNSRounded.ttf', 0),
    ('/System/Library/Fonts/Avenir Next.ttc', 4),
    ('/System/Library/Fonts/Helvetica.ttc', 1),
]
FONT_PATH, FONT_INDEX = next((p, i) for (p, i) in FONT_CANDIDATES
                              if Path(p).exists())

rspc = ReferenceSpaceCache(resolution=25,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
tree = rspc.get_structure_tree()

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
    return (field[r0, c0]*(1-cf)*(1-rf) + field[r0, c1]*cf*(1-rf) +
            field[r1, c0]*(1-cf)*rf + field[r1, c1]*cf*rf)

def reading_frame(n):
    n = n / np.linalg.norm(n)
    up = np.array([0.0, -1.0, 0.0])
    su = up - n * (up @ n)
    if np.linalg.norm(su) < 0.35:
        up = np.array([-1.0, 0.0, 0.0])
        su = up - n * (up @ n)
    su = su / np.linalg.norm(su)
    u = np.cross(su, n); u = u / np.linalg.norm(u)
    v = -su
    return u, v

LABEL_DEPTH_MM = 1.2
DEPTH_TOL_MM = 4.0
FIT_FRAC_W = 0.62
FIT_FRAC_H = 0.50
MIN_LETTER_MM = 1.0
MAX_LETTER_MM = 5.0
OVERFLOW_TOL = 0.15

files = sorted((OUT_DIR / 'stl' / 'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl'))
print(f'{len(files)} watertight area pieces to label')

pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors) +
       list(mpl.colormaps['tab20c'].colors))
labeled_meshes = {}
placed_total = 0
for fi, f in enumerate(files):
    acro = f.stem[len('wt_'):-len(f'_{TARGET_MM}mm')]
    m = trimesh.load(str(f), force='mesh')
    # subdivide for label resolution
    nv, nf = trimesh.remesh.subdivide(m.vertices, m.faces)
    m = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
    vn = m.vertex_normals
    ctr = m.centroid
    out_dir = m.vertices - ctr
    out_dir /= (np.linalg.norm(out_dir, axis=1, keepdims=True) + 1e-12)
    outside = (vn * out_dir).sum(axis=1) > 0.2

    text_inkness = np.zeros(len(m.vertices), dtype=np.float32)
    # Label each hemisphere component
    for hemi_sign in (-1, +1):
        hemi = outside & (m.vertices[:, 2] * hemi_sign > 0)
        if hemi.sum() < 40:
            continue
        pts = m.vertices[hemi]; norms = vn[hemi]
        mean_n = norms.mean(axis=0); mean_n /= (np.linalg.norm(mean_n)+1e-12)
        align = norms @ mean_n
        flat = align > 0.5
        if flat.sum() < 20: flat = align > 0.3
        if flat.sum() < 12: continue
        fpts = pts[flat]; fnorms = norms[flat]
        pc = fpts - fpts.mean(axis=0)
        d_ctr = np.linalg.norm(pc, axis=1)
        order = np.argsort(d_ctr)[:min(80, len(fpts))]
        best = order[np.argmax((fnorms @ mean_n)[order])]
        site = fpts[best]; site_n = fnorms[best]
        u_ax, v_ax = reading_frame(site_n)
        bitmap = text_bitmap(acro)
        H_px, W_px = bitmap.shape
        rel = m.vertices - site
        u_co = rel @ u_ax; v_co = rel @ v_ax; w_co = rel @ site_n
        f_u = (fpts - site) @ u_ax; f_v = (fpts - site) @ v_ax
        ps = min((FIT_FRAC_W * f_u.ptp()) / W_px,
                  (FIT_FRAC_H * f_v.ptp()) / H_px,
                  MAX_LETTER_MM / H_px)
        for _ in range(7):
            if ps * H_px < MIN_LETTER_MM: break
            col = u_co / ps + W_px / 2
            row = v_co / ps + H_px / 2
            in_box = ((col >= 0) & (col < W_px-1) & (row >= 0) &
                       (row < H_px-1) & (np.abs(w_co) < DEPTH_TOL_MM))
            if not in_box.any(): ps *= 0.8; continue
            ink = bilinear(bitmap, col[in_box], row[in_box])
            inked = np.zeros(len(m.vertices), dtype=bool)
            inked[np.where(in_box)[0]] = ink > 0.3
            n_ink = int(inked.sum())
            if n_ink < 5: ps *= 0.8; continue
            # overflow = inked verts that face away (off the piece edge)
            overflow = int((inked & ~outside).sum()) / max(n_ink, 1)
            if overflow <= OVERFLOW_TOL:
                commit = inked & outside
                vals = bilinear(bitmap, u_co[commit]/ps + W_px/2,
                                 v_co[commit]/ps + H_px/2)
                text_inkness[commit] = np.maximum(text_inkness[commit], vals)
                placed_total += 1
                break
            ps *= 0.82
    m.vertices = m.vertices - vn * (LABEL_DEPTH_MM * text_inkness)[:, None]
    labeled_meshes[acro] = (pal[fi % len(pal)], m)
    m.export(str(OUT_DIR / 'stl' / 'cortical_areas' /
                  f'wtl_{acro}_{TARGET_MM}mm.stl'))

print(f'engraved {placed_total} labels across {len(labeled_meshes)} pieces')

# Render combined (colored, labels engraved)
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))
all_v = np.vstack([m.vertices for _, m in labeled_meshes.values()])
d = (all_v.max(0) - all_v.min(0)).max() * 1.8
views = [
    ('dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
    ('lateral R',  (0,  0,  d), (0, 0, 0), (0, -1, 0)),
    ('lateral L',  (0,  0, -d), (0, 0, 0), (0, -1, 0)),
    ('ventral',    (0,  d, 0), (0, 0, 0), (1, 0, 0)),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for k, (vname, cam, focal, up) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    for acro, (color, m) in labeled_meshes.items():
        pl.add_mesh(make_poly(m), color=color, smooth_shading=True,
                     specular=0.2, ambient=0.25, diffuse=0.9)
    pl.add_text(vname, font_size=12)
    pl.camera_position = (cam, focal, up); pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'watertight_areas_labeled.png'))
pl.close()
print('wrote watertight_areas_labeled.png')
