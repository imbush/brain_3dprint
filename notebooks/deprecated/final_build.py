"""Final prototype bundles: 2 connectors x 3 fonts (+ plain, no labels).

Connectors:
  rest-of-brain : connector_restofbrain_66mm.stl (cortex carved out so the
                  area pieces sit proud on the subcortical body)
  sheets        : connector_sheets_66mm.stl (isocortex shrunk deeper so the
                  two hemisphere shells stay hidden under the pia)

For each font (sf/arial/comic) renders the labeled cortex on each connector
(connector translucent so the cortex reads), plus one plain (unlabeled) set.
"""
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import pyvista as pv
import matplotlib as mpl

OUT = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
CONN = OUT / 'stl' / 'connectors'
TARGET_MM = 66
pal = (list(mpl.colormaps['tab20'].colors) + list(mpl.colormaps['tab20b'].colors)
       + list(mpl.colormaps['tab20c'].colors))

# ---- re-shrink the sheets deeper so they hide under the pia ----
root_raw = trimesh.load(OUT/'obj'/'997_root.obj', force='mesh')
SH = (5.0*root_raw.bounding_box.extents.max()/1000.0)/root_raw.bounding_box.extents.max()
CTR = root_raw.bounding_box.centroid*SH
iso = trimesh.load(OUT/'obj'/'315_Isocortex.obj', force='mesh')
iso.merge_vertices(); iso.update_faces(iso.unique_faces())
iso.update_faces(iso.nondegenerate_faces()); iso.remove_unreferenced_vertices()
iso.apply_scale(SH); iso.apply_translation(-CTR)
ic = iso.centroid
FRAC = 0.80                       # deeper than before (was 0.85)
sheets = iso.copy(); sheets.vertices = ic + (sheets.vertices-ic)*FRAC
sheets.export(str(CONN/f'connector_sheets_{TARGET_MM}mm.stl'))
for c, side in zip(sorted(sheets.split(only_watertight=False), key=lambda c: c.centroid[2]), ['L', 'R']):
    c.export(str(CONN/f'connector_sheet_{side}_{TARGET_MM}mm.stl'))
print(f'sheets reshrunk frac={FRAC}')

restofbrain = trimesh.load(str(CONN/f'connector_restofbrain_{TARGET_MM}mm.stl'), force='mesh')

# ---- verify each connector connects every area + sheets stay hidden ----
def connect_stats(conn):
    kd = cKDTree(conn.vertices); files = sorted((OUT/'stl'/'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl'))
    n = 0
    for f in files:
        a = trimesh.load(str(f), force='mesh')
        try:
            if conn.contains(a.vertices).any(): n += 1; continue
        except Exception: pass
        if kd.query(a.vertices)[0].min() < 1.0: n += 1
    return n, len(files)
ns = connect_stats(sheets); nr = connect_stats(restofbrain)
print(f'sheets connect {ns[0]}/{ns[1]}   rest-of-brain connect {nr[0]}/{nr[1]}')

# sheets hidden? max radial extent vs the area pieces' pia
av = np.vstack([trimesh.load(str(f), force='mesh').vertices
                for f in (OUT/'stl'/'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl')])
amax = np.linalg.norm(av - av.mean(0), axis=1).max()
smax = np.linalg.norm(sheets.vertices - av.mean(0), axis=1).max()
print(f'sheets max radius {smax:.1f}mm vs cortex max radius {amax:.1f}mm (hidden if smaller)')

# ---- render bundles ----
pv.OFF_SCREEN = True; pv.set_plot_theme('document')
def poly(m): return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64))
def bundle(area_dir, glob, conn, fname, title, conn_op=0.45):
    files = sorted((OUT/area_dir).glob(glob)); meshes=[trimesh.load(str(f),force='mesh') for f in files]
    colored = '_plain' not in fname
    allv = np.vstack([m.vertices for m in meshes]); ctr=allv.mean(0); dd=(allv.max(0)-allv.min(0)).max()
    views = [('dorsal',(0,-1.7*dd,0),(-1,0,0)), ('lateral R',(0,0,1.7*dd),(0,-1,0))]
    pl = pv.Plotter(shape=(1,2), window_size=(2400,1100), off_screen=True)
    for k,(vn,cam,up) in enumerate(views):
        pl.subplot(0,k)
        for i,m in enumerate(meshes):
            pl.add_mesh(poly(m), color=pal[i%len(pal)] if colored else (0.82,0.82,0.85),
                        smooth_shading=True, specular=0.2, ambient=0.25, diffuse=0.9)
        if conn is not None:
            pl.add_mesh(poly(conn), color=(0.5,0.5,0.52), opacity=conn_op, smooth_shading=True)
        pl.add_text(f'{title} — {vn}', font_size=11)
        pl.camera_position=(cam,(0,0,0),up); pl.reset_camera()
    pl.background_color='white'
    pl.screenshot(str(OUT/'png'/fname)); pl.close(); print('wrote', fname)

for fk in ['sf', 'arial', 'comic']:
    lab = (f'stl/cortical_areas/boollabeled_{fk}', f'wtl_*_{TARGET_MM}mm.stl')
    bundle(*lab, restofbrain, f'bundle_restofbrain_{fk}.png', f'rest-of-brain + {fk}')
    bundle(*lab, sheets,      f'bundle_sheets_{fk}.png',      f'sheets + {fk}')
# plain (no labels), font-independent
plain = ('stl/cortical_areas', f'wt_*_{TARGET_MM}mm.stl')
bundle(*plain, restofbrain, 'bundle_restofbrain_plain.png', 'rest-of-brain (no labels)')
bundle(*plain, sheets,      'bundle_sheets_plain.png',      'sheets (no labels)')
print('done')
