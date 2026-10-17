"""Cortex ONE-piece via a HIDDEN buried connector (sheets), letters last.

Pipeline:
  1. plain watertight area pieces (no labels)
  2. add an L4-depth thin connector shell (overlaps mid-cortex of each area)
     + a deep midline bridge in the fissure
  3. union -> one fused solid
  4. THEN subtract the labels (clean letters)

Usage: python build_onepiece_sheets.py
Output: stl/FINAL_onepiece_sheets_arial.stl + png/FINAL_onepiece_sheets_arial.png
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
from scipy.spatial import cKDTree
import pyvista as pv
import label_common as lc

O = lc.OUT_DIR
TARGET_MM = 66
lc.set_font('/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf', 0)

def ensure_volume(m):
    if m.is_volume: return m
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces(); ms.meshing_remove_unreferenced_vertices()
    for _ in range(4):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=30000)
        except Exception: pass
    ms.meshing_re_orient_faces_coherently()
    cm = ms.current_mesh(); r = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
    if r.is_volume: return r
    diag = float(np.linalg.norm(m.bounding_box.extents))
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.generate_alpha_wrap(alpha=ml.PureValue(diag*0.010), offset=ml.PureValue(diag*0.003))
    cm = ms.current_mesh(); return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

# isocortex -> L4 connector + bridge
root = trimesh.load(O/'obj'/'997_root.obj', force='mesh')
SH = (5.0*root.bounding_box.extents.max()/1000.0)/root.bounding_box.extents.max()
CTR = root.bounding_box.centroid*SH
iso = trimesh.load(O/'obj'/'315_Isocortex.obj', force='mesh')
iso.merge_vertices(); iso.apply_scale(SH); iso.apply_translation(-CTR)
ic = iso.centroid; R = np.linalg.norm(iso.vertices - ic, axis=1).max()
def shrink(fr): c = iso.copy(); c.vertices = ic + (c.vertices-ic)*fr; return c
l4 = ensure_volume(shrink(1.0-1.3/R)).difference(ensure_volume(shrink(1.0-2.8/R)), engine='manifold')

files = sorted((O/'stl'/'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl'))
pieces = [ensure_volume(trimesh.load(str(f), force='mesh')) for f in files]

# CORTEX FOOTPRINT MASK: union of the areas slightly inflated (1.2 mm) so it
# fills the inter-area gaps AND fuses across the midline. Clipping the
# connector to this keeps it bridging gaps but trims it at the cortex rim
# (no peek) and gives a conformal midline join (no box).
def infl(m, d):
    c = m.centroid; r = np.linalg.norm(m.vertices-c, axis=1).mean()
    g = m.copy(); g.vertices = c + (g.vertices-c)*(1+d/r); return g
print('building cortex footprint mask...')
mask = trimesh.boolean.union([ensure_volume(infl(trimesh.load(str(f), force='mesh'), 1.2))
                              for f in files], engine='manifold')
mask = max(mask.split(only_watertight=False), key=lambda c: len(c.faces))

# clip the connector to the footprint -> no rim peek
l4c = ensure_volume(l4).intersection(mask, engine='manifold')
# conformal midline join: the thin central slice of the mask (follows the
# medial areas, fuses L<->R), trimmed below the pia so it stays hidden.
zslab = trimesh.creation.box(extents=[200, 200, 5.0])   # |z|<2.5 slab
midjoin = mask.intersection(zslab, engine='manifold')
print(f'{len(pieces)} plain pieces + clipped connector + conformal midline join; unioning...')
fused = trimesh.boolean.union(pieces + [l4c, midjoin], engine='manifold')
fused = max(fused.split(only_watertight=False), key=lambda c: len(c.faces))
print(f'fused: faces={len(fused.faces):,}')

# engrave labels last
pos = json.loads((O/'label_positions.json').read_text())
kd = cKDTree(fused.vertices); fn = fused.vertex_normals
solids = []
for acro, lst in pos.items():
    for e in lst:
        _, j = kd.query(np.asarray(e['point'], float))
        s = lc.build_text_world(str(e.get('text', acro)), fused.vertices[j], fn[j],
                                float(e.get('letter_mm', 2.5)), float(e.get('angle', 0.0)))
        if s is not None: solids.append(s)
print(f'subtracting {len(solids)} labels...')
labeled = fused.difference(trimesh.util.concatenate(solids), engine='manifold')
labeled = max(labeled.split(only_watertight=False), key=lambda c: len(c.faces))
print(f'final: faces={len(labeled.faces):,}, components={len(labeled.split(only_watertight=False))}, watertight={labeled.is_watertight}')
out = O/'stl'/'FINAL_onepiece_sheets_arial.stl'; labeled.export(str(out)); print('wrote', out.name)

pv.OFF_SCREEN = True; pv.set_plot_theme('document')
def poly(m): return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64))
from PIL import Image
TMP = O/'png'/'_ops'; TMP.mkdir(exist_ok=True, parents=True)
dd = labeled.bounding_box.extents.max()
views=[('dorsal',(0,-1.7*dd,0),(-1,0,0)),('ventral',(0,1.7*dd,0),(1,0,0)),
       ('lateral R',(0,0,1.7*dd),(0,-1,0)),('front',(-1.7*dd,0,0),(0,-1,0))]
panels=[]
for nm,cam,up in views:
    pl=pv.Plotter(window_size=(1200,1050),off_screen=True)
    pl.add_mesh(poly(labeled),color=(0.80,0.80,0.83),smooth_shading=True,specular=0.35,specular_power=18,ambient=0.15,diffuse=0.9)
    pl.camera_position=(cam,(0,0,0),up); pl.reset_camera(); pl.enable_anti_aliasing('ssaa')
    pl.background_color='white'; pl.add_text(nm,font_size=13)
    f=TMP/f'{nm}.png'.replace(' ',''); pl.screenshot(str(f)); pl.close(); panels.append(str(f))
imgs=[Image.open(p) for p in panels]; w,h=imgs[0].size
grid=Image.new('RGB',(w*2,h*2),'white')
for i,im in enumerate(imgs): grid.paste(im,((i%2)*w,(i//2)*h))
grid.save(str(O/'png'/'FINAL_onepiece_sheets_arial.png')); print('wrote FINAL_onepiece_sheets_arial.png')
print('done')
