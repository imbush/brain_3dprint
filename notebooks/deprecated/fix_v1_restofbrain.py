"""Rebuild V1 rest-of-brain so the cortex AREA pieces sit proud on it.
Carve the actual area-piece footprint out of the watertight brain (instead
of the isocortex mesh, which was misaligned and left the areas buried).
"""
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv
import matplotlib as mpl

OUT = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
CONN = OUT / 'stl' / 'connectors'
TARGET_MM = 66

root_raw = trimesh.load(OUT / 'obj' / '997_root.obj', force='mesh')
SHARED = (5.0 * root_raw.bounding_box.extents.max()/1000.0)/root_raw.bounding_box.extents.max()
ROOT_CTR = root_raw.bounding_box.centroid * SHARED
def to_print(m):
    m = m.copy(); m.apply_scale(SHARED); m.apply_translation(-ROOT_CTR); return m
def close_holes(m):
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    for _ in range(4):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=100000)
        except Exception: pass
    try: ms.meshing_re_orient_faces_coherently()
    except Exception: pass
    cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

body = close_holes(to_print(root_raw))
print('root watertight', body.is_watertight)

# carve each cortex area out of the body (areas already in print frame)
files = sorted((OUT/'stl'/'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl'))
for i, f in enumerate(files):
    a = trimesh.load(str(f), force='mesh')
    try:
        nb = body.difference(a, engine='manifold')
        if nb is not None and len(nb.faces) > 0:
            body = nb
    except Exception as e:
        print('skip', f.stem, e)
    if (i+1) % 10 == 0: print(f'  carved {i+1}/{len(files)}')
body.export(str(CONN / f'connector_restofbrain_{TARGET_MM}mm.stl'))
print(f'rest-of-brain: {len(body.vertices):,} verts, watertight={body.is_watertight}')

# how proud are the areas now?
for a in ['MOs', 'VISp', 'SSp-bfd']:
    m = trimesh.load(str(OUT/'stl'/'cortical_areas'/f'wt_{a}_{TARGET_MM}mm.stl'), force='mesh')
    outside = (~body.contains(m.vertices)).mean()*100
    print(f'  {a}: {outside:.0f}% of verts proud of body')

# render labeled + plain bundles
pv.OFF_SCREEN = True; pv.set_plot_theme('document')
pal = (list(mpl.colormaps['tab20'].colors)+list(mpl.colormaps['tab20b'].colors)+list(mpl.colormaps['tab20c'].colors))
def poly(m): return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64))
def render(area_dir, glob, fname, title):
    files = sorted((OUT/area_dir).glob(glob)); meshes=[trimesh.load(str(f),force='mesh') for f in files]
    allv=np.vstack([m.vertices for m in meshes]); dd=(allv.max(0)-allv.min(0)).max()
    pl=pv.Plotter(window_size=(1500,1300),off_screen=True)
    for i,m in enumerate(meshes): pl.add_mesh(poly(m),color=pal[i%len(pal)],smooth_shading=True,specular=0.2,ambient=0.25,diffuse=0.9)
    pl.add_mesh(poly(body),color=(0.5,0.5,0.52),smooth_shading=True)
    pl.add_text(title,font_size=12)
    pl.camera_position=((0,0,1.7*dd),(0,0,0),(0,-1,0)); pl.reset_camera()
    pl.enable_anti_aliasing('ssaa'); pl.background_color='white'
    pl.screenshot(str(OUT/'png'/fname)); pl.close(); print('wrote',fname)
render('stl/cortical_areas/boollabeled_manual', f'wtl_*_{TARGET_MM}mm.stl',
       'bundle_restofbrain_labeled.png', 'V1 rest-of-brain + labels')
render('stl/cortical_areas', f'wt_*_{TARGET_MM}mm.stl',
       'bundle_restofbrain_plain.png', 'V1 rest-of-brain (no labels)')
print('done')
