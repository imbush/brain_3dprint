"""Engrave the labels you placed (data/ccf_brain/label_positions.json)
with the smooth boolean method. Honors per-label 'text' override (used
to engrave numbers on super-small areas), per-label angle + letter_mm,
and a selectable FONT.

Usage:
    python boolean_area_labels_manual.py [font]
    font in {sf, arial, comic}  (default sf)

Outputs:
  stl/cortical_areas/boollabeled_<font>/wtl_<acro>_66mm.stl
  png/prototype_<font>.png
"""
import sys, json
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
from scipy.spatial import cKDTree
import pyvista as pv
import matplotlib as mpl

import label_common as lc

OUT_DIR = lc.OUT_DIR
TARGET_MM = lc.TARGET_MM
POS_JSON = OUT_DIR / 'label_positions.json'

FONTS = {
    'sf':    ('/System/Library/Fonts/SFNSRounded.ttf', 0),
    'arial': ('/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf', 0),
    'comic': ('/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf', 0),
}
font_key = sys.argv[1] if len(sys.argv) > 1 else 'sf'
fp, fi = FONTS[font_key]
lc.set_font(fp, fi)
print(f'font = {font_key} ({fp})')

lc.MIN_LETTER_MM = 0.7
lc.OVERFLOW_TOL = 0.22

positions = json.loads(POS_JSON.read_text())
print(f'{sum(len(v) for v in positions.values())} anchors across '
      f'{len(positions)} areas')

def repair(m):
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64),
                                            m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    for _ in range(3):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=30000)
        except Exception: pass
    cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

pieces = lc.load_pieces()
odir = OUT_DIR / 'stl' / 'cortical_areas' / f'boollabeled_{font_key}'
odir.mkdir(parents=True, exist_ok=True)

pal = (list(mpl.colormaps['tab20'].colors) + list(mpl.colormaps['tab20b'].colors)
       + list(mpl.colormaps['tab20c'].colors))
labeled = {}
n_placed = 0; n_pieces = 0; n_fixed = 0
for acro, color, m_orig, m_sub in pieces:
    out = lc.outside_mask(m_sub)
    out_idx = np.where(out)[0]
    kd = cKDTree(m_sub.vertices[out_idx])

    solids = []
    for e in positions.get(acro, []):
        site = m_sub.vertices[out_idx[kd.query(np.asarray(e['point'], float))[1]]]
        vi = out_idx[kd.query(site)[1]]
        site = m_sub.vertices[vi]; site_n = m_sub.vertex_normals[vi]
        angle = float(e.get('angle', 0.0))
        text = str(e.get('text', acro))           # number override for tiny areas
        letter_mm = e.get('letter_mm', None)
        if letter_mm is None:
            sign = 1 if site[2] > 0 else -1
            sub = out & (m_sub.vertices[:, 2] * sign > 0)
            letter_mm = lc.fit_label(text, m_sub, site, site_n, sub, out)
            if letter_mm is None:
                print(f'  {acro}: did not fit — skipped'); continue
        sol = lc.build_text_world(text, site, site_n, float(letter_mm), angle)
        if sol is not None:
            solids.append(sol)

    if solids:
        nv, nf = trimesh.remesh.subdivide(m_sub.vertices, m_sub.faces)
        m_eng = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
    else:
        m_eng = m_sub
    mesh, ok = lc.engrave(m_eng, solids)
    if ok:
        n_placed += len(solids); n_pieces += 1
    # ensure printable: watertight + single volume
    if not (all(c.is_watertight for c in mesh.split(only_watertight=False))
            and mesh.is_volume):
        mesh = repair(mesh); n_fixed += 1
    labeled[acro] = (color, mesh)
    mesh.export(str(odir / f'wtl_{acro}_{TARGET_MM}mm.stl'))

n_wt = sum(1 for _, (c, m) in labeled.items()
           if all(cc.is_watertight for cc in m.split(only_watertight=False)))
print(f'engraved {n_placed} labels on {n_pieces} pieces; '
      f'repaired {n_fixed}; watertight {n_wt}/{len(labeled)}')

# render colored prototype
pv.OFF_SCREEN = True; pv.set_plot_theme('document')
def mp(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))
all_v = np.vstack([m.vertices for _, m in labeled.values()])
ctr = all_v.mean(0); dd = (all_v.max(0) - all_v.min(0)).max()
views = [('dorsal', (0, -1.7*dd, 0), (-1, 0, 0)),
         ('lateral R', (0, 0, 1.7*dd), (0, -1, 0)),
         ('ventral', (0, 1.7*dd, 0), (1, 0, 0))]
pl = pv.Plotter(shape=(1, 3), window_size=(2700, 1100), off_screen=True)
for k, (vn, cam, up) in enumerate(views):
    pl.subplot(0, k)
    for acro, (color, m) in labeled.items():
        pl.add_mesh(mp(m), color=color, smooth_shading=True,
                     specular=0.2, ambient=0.25, diffuse=0.9)
    pl.add_text(f'{font_key} — {vn}', font_size=12)
    pl.camera_position = (tuple(np.array(cam)), tuple(ctr*0), up); pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / f'prototype_{font_key}.png'))
pl.close()
print(f'wrote prototype_{font_key}.png')
