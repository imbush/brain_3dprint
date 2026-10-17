"""Cortex ONE-piece: hidden SHEET fuses areas within each hemisphere
(built from the area footprint, shrunk inward -> no rim peek, union-only
-> watertight), and short STICKS (data/ccf_brain/stick_positions.json,
placed in the app) connect the two hemispheres. Letters engraved last.

If no sticks file is present the footprint sheets alone are used (they may
already join the hemispheres at the medial wall).
Output: stl/FINAL_onepiece_hybrid_arial.stl + png/FINAL_onepiece_hybrid_arial.png
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
from scipy.spatial import cKDTree
import pyvista as pv
import label_common as lc

O = lc.OUT_DIR; TM = 66
lc.set_font('/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf', 0)

def ev(m):
    if m.is_volume: return m
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces(); ms.meshing_remove_unreferenced_vertices()
    for _ in range(3):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=30000)
        except Exception: pass
    ms.meshing_re_orient_faces_coherently(); cm = ms.current_mesh()
    r = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
    if r.is_volume: return r
    d = float(np.linalg.norm(m.bounding_box.extents)); ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.generate_alpha_wrap(alpha=ml.PureValue(d*0.01), offset=ml.PureValue(d*0.003)); cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

def infl(m, d):
    c = m.centroid; r = np.linalg.norm(m.vertices-c, axis=1).mean()
    g = m.copy(); g.vertices = c + (g.vertices-c)*(1+d/r); return g

files = sorted((O/'stl'/'cortical_areas').glob(f'wt_*_{TM}mm.stl'))
comps = []
for f in files:
    for c in trimesh.load(str(f), force='mesh').split(only_watertight=False):
        if len(c.faces) >= 50: comps.append(c)

# intra-hemisphere SHEET = the isocortex shrunk 15% (a solid, NOT a thin
# shell -> unions cleanly). This is the "old sheet" look. It sits under the
# pia and overlaps the deep part of every area, fusing them.
root = trimesh.load(O/'obj'/'997_root.obj', force='mesh')
SH = (5*root.bounding_box.extents.max()/1000)/root.bounding_box.extents.max()
CTR = root.bounding_box.centroid*SH
iso = trimesh.load(O/'obj'/'315_Isocortex.obj', force='mesh')
iso.merge_vertices(); iso.apply_scale(SH); iso.apply_translation(-CTR)
ic = iso.centroid
sheet = iso.copy(); sheet.vertices = ic + (sheet.vertices-ic)*0.85
print('built solid shrunk-isocortex sheet')

parts = [ev(c) for c in comps] + [ev(sheet)]

# midline STICKS from the app (true coords)
SJ = O/'stick_positions.json'
nstick = 0
if SJ.exists():
    for e in json.loads(SJ.read_text()):
        A = np.array(e['A'], float); B = np.array(e['B'], float)
        u = B-A; L = np.linalg.norm(u)
        if L < 1e-6: continue
        u /= L
        parts.append(trimesh.creation.cylinder(radius=1.0, segment=np.array([A-1.8*u, B+1.8*u]), sections=12))
        nstick += 1
print(f'+ {nstick} midline sticks; unioning...')

fused = trimesh.boolean.union(parts, engine='manifold')
fused = max(fused.split(only_watertight=False), key=lambda c: len(c.faces))

# engrave labels last
pos = json.loads((O/'label_positions.json').read_text())
kd = cKDTree(fused.vertices); fn = fused.vertex_normals
solids = []
for acro, lst in pos.items():
    for e in lst:
        _, jj = kd.query(np.asarray(e['point'], float))
        s = lc.build_text_world(str(e.get('text', acro)), fused.vertices[jj], fn[jj],
                                float(e.get('letter_mm', 2.5)), float(e.get('angle', 0.0)))
        if s is not None: solids.append(s)
labeled = fused.difference(trimesh.util.concatenate(solids), engine='manifold')
labeled = max(labeled.split(only_watertight=False), key=lambda c: len(c.faces))
labeled.merge_vertices()
out = O/'stl'/'FINAL_onepiece_hybrid_arial.stl'; labeled.export(str(out))
rl = trimesh.load(str(out), force='mesh'); rl.merge_vertices()
print(f'RELOAD: faces={len(rl.faces):,}, components={len(rl.split(only_watertight=False))}, watertight={rl.is_watertight}')

pv.OFF_SCREEN = True; pv.set_plot_theme('document')
def poly(m): return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64))
from PIL import Image
TMP = O/'png'/'_oph'; TMP.mkdir(exist_ok=True, parents=True)
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
grid.save(str(O/'png'/'FINAL_onepiece_hybrid_arial.png')); print('wrote FINAL_onepiece_hybrid_arial.png')
print('done')
