"""Build two hidden/structural connectors so the 43 cortical-area pieces
print as one holdable object instead of a loose jigsaw, and repair the
one non-manifold labeled piece (RSPagl).

Connectors (each overlaps the DEEP side of every area piece so the set
is physically joined; colors stay separate because bodies aren't fused):

  V1  rest-of-brain  : root brain made watertight, minus the isocortex
                       shell -> a solid subcortical body the cortex areas
                       sit on. Doubles as the "inner brain + base".
                       -> stl/connectors/connector_restofbrain_66mm.stl
  V2  two sheets     : the isocortex shell shrunk a few mm inward, kept as
                       its two natural hemisphere components -> two thin
                       shells that sit UNDER the pia, invisible from
                       outside, joining every area in each hemisphere.
                       -> stl/connectors/connector_sheets_66mm.stl
                          (+ _L / _R split files)

Each connector pairs with either the labeled area set
(stl/cortical_areas/boollabeled_manual/wtl_*) or the plain set
(stl/cortical_areas/wt_*) -> 4 printable bundles. Renders all 4.
"""
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
from scipy.spatial import cKDTree
import pyvista as pv

OUT = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
CONN = OUT / 'stl' / 'connectors'; CONN.mkdir(parents=True, exist_ok=True)
TARGET_MM = 66

root_raw = trimesh.load(OUT / 'obj' / '997_root.obj', force='mesh')
SCALE = 5.0
SHARED = (SCALE * root_raw.bounding_box.extents.max() / 1000.0) / \
         root_raw.bounding_box.extents.max()
ROOT_CTR = root_raw.bounding_box.centroid * SHARED

def to_print(m):
    m = m.copy(); m.apply_scale(SHARED); m.apply_translation(-ROOT_CTR); return m

def close_holes(m):
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
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

def shrink(m, frac, ctr):
    c = m.copy(); c.vertices = ctr + (c.vertices - ctr) * frac; return c

# ---------- (a) repair RSPagl labeled piece ----------
rp = OUT / 'stl' / 'cortical_areas' / 'boollabeled_manual' / f'wtl_RSPagl_{TARGET_MM}mm.stl'
if rp.exists():
    m = trimesh.load(str(rp), force='mesh')
    fixed = close_holes(m)
    ok = all(c.is_watertight for c in fixed.split(only_watertight=False)) and fixed.is_volume
    if not ok:
        diag = float(np.linalg.norm(m.bounding_box.extents))
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
        ms.generate_alpha_wrap(alpha=ml.PureValue(diag*0.010),
                                offset=ml.PureValue(diag*0.003))
        cm = ms.current_mesh()
        fixed = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
    fixed.export(str(rp))
    print(f'RSPagl repaired: watertight={fixed.is_watertight} volume={fixed.is_volume}')

# ---------- isocortex (print frame) ----------
iso = trimesh.load(OUT / 'obj' / '315_Isocortex.obj', force='mesh')
iso.merge_vertices(); iso.update_faces(iso.unique_faces())
iso.update_faces(iso.nondegenerate_faces()); iso.remove_unreferenced_vertices()
iso = to_print(iso)
iso_ctr = iso.centroid
diag = float(np.linalg.norm(iso.bounding_box.extents))
avg_r = diag / 4.0
FRAC = max(0.85, 1.0 - 3.5 / avg_r)     # ~3.5 mm inward -> under the pia
print(f'iso bbox(mm)={iso.bounding_box.extents.round(1).tolist()} shrink_frac={FRAC:.3f}')

# ---------- V2: two hemisphere sheets ----------
sheets = shrink(iso, FRAC, iso_ctr)
sheets.export(str(CONN / f'connector_sheets_{TARGET_MM}mm.stl'))
comps = sorted(sheets.split(only_watertight=False),
               key=lambda c: c.centroid[2])     # -z first
for c, side in zip(comps, ['L', 'R'] if len(comps) == 2 else
                   [str(i) for i in range(len(comps))]):
    c.export(str(CONN / f'connector_sheet_{side}_{TARGET_MM}mm.stl'))
print(f'V2 sheets: {len(comps)} components, '
      f'{len(sheets.vertices):,} verts')

# ---------- V1: rest-of-brain ----------
root_wt = close_holes(to_print(root_raw))
print(f'root watertight={root_wt.is_watertight} volume={root_wt.is_volume}')
iso_sub = iso   # subtract the FULL cortex shell -> areas seat on the body, fully visible
restofbrain = None
try:
    restofbrain = root_wt.difference(iso_sub, engine='manifold')
    if restofbrain is None or len(restofbrain.faces) == 0:
        restofbrain = None
except Exception as e:
    print('rest-of-brain boolean failed:', e)
if restofbrain is not None:
    restofbrain.export(str(CONN / f'connector_restofbrain_{TARGET_MM}mm.stl'))
    print(f'V1 rest-of-brain: {len(restofbrain.vertices):,} verts, '
          f'watertight={restofbrain.is_watertight}')

# ---------- verify each area piece is connected ----------
def connects(connector):
    kd = cKDTree(connector.vertices)
    files = sorted((OUT / 'stl' / 'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl'))
    n = 0
    for f in files:
        a = trimesh.load(str(f), force='mesh')
        try:
            if connector.contains(a.vertices).any():
                n += 1; continue
        except Exception:
            pass
        # fallback: any area vertex within 1 mm of the connector surface
        if kd.query(a.vertices)[0].min() < 1.0:
            n += 1
    return n, len(files)

ns = connects(sheets)
print(f'V2 sheets connect {ns[0]}/{ns[1]} area pieces')
if restofbrain is not None:
    nr = connects(restofbrain)
    print(f'V1 rest-of-brain connects {nr[0]}/{nr[1]} area pieces')

# ---------- render the 4 bundles ----------
pv.OFF_SCREEN = True; pv.set_plot_theme('document')
def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))
import matplotlib as mpl
pal = (list(mpl.colormaps['tab20'].colors) + list(mpl.colormaps['tab20b'].colors)
       + list(mpl.colormaps['tab20c'].colors))

def render(area_dir, area_glob, connector, conn_opacity, title, fname):
    files = sorted((OUT / area_dir).glob(area_glob))
    meshes = [trimesh.load(str(f), force='mesh') for f in files]
    allv = np.vstack([m.vertices for m in meshes]); dd = (allv.max(0)-allv.min(0)).max()
    pl = pv.Plotter(window_size=(1500, 1300), off_screen=True)
    for i, m in enumerate(meshes):
        pl.add_mesh(poly(m), color=pal[i % len(pal)], smooth_shading=True,
                    specular=0.2, ambient=0.25, diffuse=0.9)
    if connector is not None:
        pl.add_mesh(poly(connector), color=(0.45, 0.45, 0.48),
                    opacity=conn_opacity, smooth_shading=True)
    pl.add_text(title, font_size=12)
    pl.camera_position = ((0, 0, 1.7*dd), (0, 0, 0), (0, -1, 0)); pl.reset_camera()
    pl.background_color = 'white'
    pl.screenshot(str(OUT / 'png' / fname)); pl.close()
    print('wrote', fname)

LAB = ('stl/cortical_areas/boollabeled_manual', f'wtl_*_{TARGET_MM}mm.stl')
PLAIN = ('stl/cortical_areas', f'wt_*_{TARGET_MM}mm.stl')
if restofbrain is not None:
    render(*LAB, restofbrain, 1.0, 'V1 rest-of-brain + labels', 'bundle_restofbrain_labeled.png')
    render(*PLAIN, restofbrain, 1.0, 'V1 rest-of-brain (no labels)', 'bundle_restofbrain_plain.png')
render(*LAB, sheets, 0.5, 'V2 sheets + labels (connector translucent)', 'bundle_sheets_labeled.png')
render(*PLAIN, sheets, 0.5, 'V2 sheets (no labels, connector translucent)', 'bundle_sheets_plain.png')
print('done')
