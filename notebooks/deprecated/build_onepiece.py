"""Fuse everything into ONE watertight printable volume (single color):
  43 labeled cortex pieces (arial)
  + isocortex sheets shrunk to overlap every area's deep side (frac 0.92)
  + rest-of-brain body to bridge the two hemispheres
-> one solid with engraved area labels on the surface.

Outputs stl/FINAL_onepiece_arial.stl + png/FINAL_onepiece_arial.png
"""
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv

def ensure_volume(m):
    """Manifold boolean needs every input to be a closed volume; repair
    any that aren't (close holes; alpha-wrap as last resort)."""
    if m.is_volume:
        return m
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64),
                                           m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    for _ in range(4):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=30000)
        except Exception: pass
    try: ms.meshing_re_orient_faces_coherently()
    except Exception: pass
    cm = ms.current_mesh()
    r = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
    if r.is_volume:
        return r
    diag = float(np.linalg.norm(m.bounding_box.extents))
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.generate_alpha_wrap(alpha=ml.PureValue(diag*0.010), offset=ml.PureValue(diag*0.003))
    cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

O = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
TARGET_MM = 66

root = trimesh.load(O/'obj'/'997_root.obj', force='mesh')
SH = (5.0*root.bounding_box.extents.max()/1000.0)/root.bounding_box.extents.max()
CTR = root.bounding_box.centroid*SH
root_print = root.copy(); root_print.apply_scale(SH); root_print.apply_translation(-CTR)
# whole-brain core, made watertight then shrunk so its surface sits ~1.7 mm
# under the pia: it OVERLAPS the deep part of every cortex piece (welds them)
# and bridges the two hemispheres into one solid. Overlap (not coincident)
# is what lets the boolean union actually merge them.
ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(root_print.vertices.astype(np.float64), root_print.faces.astype(np.int32)))
ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces(); ms.meshing_remove_unreferenced_vertices()
for _ in range(4):
    try: ms.meshing_repair_non_manifold_edges()
    except Exception: pass
    try: ms.meshing_close_holes(maxholesize=100000)
    except Exception: pass
ms.meshing_re_orient_faces_coherently()
cm = ms.current_mesh(); root_wt = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
rc = root_wt.centroid
R = np.linalg.norm(root_wt.vertices - rc, axis=1).max()
core = root_wt.copy(); core.vertices = rc + (core.vertices - rc) * (1.0 - 1.7/R)

pieces = [trimesh.load(str(f), force='mesh') for f in
          sorted((O/'stl'/'cortical_areas'/'boollabeled_arial').glob(f'wtl_*_{TARGET_MM}mm.stl'))]
print(f'{len(pieces)} labeled pieces + shrunk-brain core')

parts = [ensure_volume(p) for p in pieces] + [ensure_volume(core)]
print(f'all volumes: {all(p.is_volume for p in parts)}; unioning (manifold)...')
one = trimesh.boolean.union(parts, engine='manifold')
comps = one.split(only_watertight=False)
print(f'result: {len(one.vertices):,} verts, {len(one.faces):,} faces, '
      f'components={len(comps)}, watertight={one.is_watertight}, volume={one.is_volume}')

out = O/'stl'/f'FINAL_onepiece_arial.stl'
one.export(str(out))
print(f'wrote {out.name} ({out.stat().st_size/1024/1024:.1f} MB)')

# render
pv.OFF_SCREEN = True; pv.set_plot_theme('document')
def poly(m): return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64))
from PIL import Image
TMP = O/'png'/'_op'; TMP.mkdir(exist_ok=True, parents=True)
dd = one.bounding_box.extents.max()
views=[('dorsal',(0,-1.7*dd,0),(-1,0,0)),('ventral',(0,1.7*dd,0),(1,0,0)),
       ('lateral R',(0,0,1.7*dd),(0,-1,0)),('lateral L',(0,0,-1.7*dd),(0,-1,0))]
panels=[]
for nm,cam,up in views:
    pl=pv.Plotter(window_size=(1200,1050),off_screen=True)
    pl.add_mesh(poly(one),color=(0.80,0.80,0.83),smooth_shading=True,specular=0.35,specular_power=18,ambient=0.15,diffuse=0.9)
    pl.camera_position=(cam,(0,0,0),up); pl.reset_camera(); pl.enable_anti_aliasing('ssaa')
    pl.background_color='white'; pl.add_text(nm,font_size=13)
    f=TMP/f'{nm}.png'.replace(' ',''); pl.screenshot(str(f)); pl.close(); panels.append(str(f))
imgs=[Image.open(p) for p in panels]; w,h=imgs[0].size
grid=Image.new('RGB',(w*2,h*2),'white')
for i,im in enumerate(imgs): grid.paste(im,((i%2)*w,(i//2)*h))
grid.save(str(O/'png'/'FINAL_onepiece_arial.png')); print('wrote FINAL_onepiece_arial.png')
print('done')
