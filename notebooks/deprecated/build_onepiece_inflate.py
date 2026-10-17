"""Cortex ONE-piece by minimal inflation, letters engraved LAST.

Pipeline:
  1. take the PLAIN watertight area pieces (no labels)
  2. inflate each by the smallest amount that closes the gaps to neighbors
     (affine scale about the piece centroid -> never self-intersects)
  3. union -> one fused cortex solid (closes inter-area gaps + dorsal lip)
  4. THEN boolean-subtract the acronym/number labels on the fused surface,
     so the letters stay clean and undistorted.

Usage: python build_onepiece_inflate.py [inflate_mm]   (default 0.6)
Output: stl/FINAL_onepiece_inflated_arial.stl + png/FINAL_onepiece_inflated_arial.png
"""
import sys, json
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
from scipy.spatial import cKDTree
import pyvista as pv
import label_common as lc

O = lc.OUT_DIR
TARGET_MM = 66
DIL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
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

def inflate(m, d):
    """Affine grow about the piece centroid so ~d mm is added near the
    surface; affine => no self-intersection."""
    c = m.centroid
    r = np.linalg.norm(m.vertices - c, axis=1).mean()
    s = 1.0 + d / max(r, 1e-6)
    g = m.copy(); g.vertices = c + (g.vertices - c) * s; return g

# 1-2. inflate plain pieces. At ~1.2 mm the medial areas (ACAv/RSPv ...)
# overlap across the midline and fuse the two hemispheres on their own --
# a conformal, fully-hidden join along the bordering areas, no bridge.
files = sorted((O/'stl'/'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl'))
pieces = [ensure_volume(inflate(trimesh.load(str(f), force='mesh'), DIL)) for f in files]
print(f'{len(pieces)} plain pieces inflated ~{DIL} mm (no bridge); unioning...')

# 3. union
fused = trimesh.boolean.union(pieces, engine='manifold')
comps = sorted(fused.split(only_watertight=False), key=lambda c: -len(c.faces))
print(f'fused components={len(comps)} (top {[len(c.faces) for c in comps[:4]]}), watertight={fused.is_watertight}')
fused = comps[0]                      # keep main body (drop boolean slivers)

# 4. engrave labels on the fused surface
pos = json.loads((O/'label_positions.json').read_text())
kd = cKDTree(fused.vertices); fn = fused.vertex_normals
solids = []
for acro, lst in pos.items():
    for e in lst:
        _, j = kd.query(np.asarray(e['point'], float))   # snap to fused surface
        site = fused.vertices[j]; nrm = fn[j]
        text = str(e.get('text', acro)); ang = float(e.get('angle', 0.0))
        lm = float(e.get('letter_mm', 2.5))
        s = lc.build_text_world(text, site, nrm, lm, ang)
        if s is not None: solids.append(s)
print(f'subtracting {len(solids)} label solids...')
text_union = trimesh.util.concatenate(solids)
labeled = fused.difference(text_union, engine='manifold')
comps = sorted(labeled.split(only_watertight=False), key=lambda c: -len(c.faces))
labeled = comps[0]
print(f'final: faces={len(labeled.faces):,}, components(after keep)={len(labeled.split(only_watertight=False))}, '
      f'watertight={labeled.is_watertight}')

out = O/'stl'/'FINAL_onepiece_inflated_arial.stl'
labeled.export(str(out)); print('wrote', out.name)

# render
pv.OFF_SCREEN = True; pv.set_plot_theme('document')
def poly(m): return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64))
from PIL import Image
TMP = O/'png'/'_opi'; TMP.mkdir(exist_ok=True, parents=True)
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
grid.save(str(O/'png'/'FINAL_onepiece_inflated_arial.png')); print('wrote FINAL_onepiece_inflated_arial.png')
print('done')
