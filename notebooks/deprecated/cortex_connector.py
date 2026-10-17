"""Build a hidden inner connector that joins all 43 cortical-area
pieces so they print as ONE object without fusing their colors.

Idea: the 43 area pieces are chunks of the cortical ribbon (pia
outside, white-matter inside). They tile the cortex but don't touch
(86 disconnected components). A single inner shell, conformal to the
cortex and sitting BELOW the pia (at/under the white-matter depth),
overlaps the deep side of every area piece — so when all bodies are
printed together they're physically joined, while the connector stays
hidden inside.

We make the connector from the Isocortex (315) mesh:
  connector = 315 solid scaled inward about its centroid by CONNECT_FRAC
This sits inside the pia surface everywhere. Overlap with each area's
white-matter side is what fuses them.

Outputs:
  stl/cortex_hat/cortex_connector_66mm.stl
  png/cortex_connector_check.png  (areas + translucent connector)
"""
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv
import matplotlib as mpl

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE

# Isocortex 315 (already watertight + smooth) → scale to print frame
iso = trimesh.load(str(OUT_DIR / 'obj' / '315_Isocortex.obj'), force='mesh')
iso.merge_vertices()
iso.update_faces(iso.unique_faces())
iso.update_faces(iso.nondegenerate_faces())
iso.remove_unreferenced_vertices()
iso.apply_scale(SHARED_SCALE)
iso.apply_translation(-root_ctr)
print(f'isocortex (print): {len(iso.vertices):,} verts, '
      f'bbox(mm)={iso.bounding_box.extents.round(1).tolist()}')

# Connector = isocortex shrunk toward its own centroid. The cortical
# ribbon is ~1 mm thick in real life → ~5 mm at 5×. Shrinking by a few
# % moves the surface inward by a few mm, landing within the ribbon
# (overlaps each area's deep side) while staying under the pia.
iso_ctr = iso.centroid
def shrink(frac):
    c = iso.copy()
    c.vertices = iso_ctr + (c.vertices - iso_ctr) * frac
    return c

# Pick a shrink that gives ~3-4 mm inward offset on average.
diag = np.linalg.norm(iso.bounding_box.extents)
# average radius ~ diag/4; want ~4mm inward → frac = 1 - 4/(diag/4)
avg_r = diag / 4.0
CONNECT_FRAC = max(0.80, 1.0 - 4.0 / avg_r)
connector = shrink(CONNECT_FRAC)
print(f'connector: shrink_frac={CONNECT_FRAC:.3f}, '
      f'{len(connector.vertices):,} verts, wt={connector.is_watertight}')
connector.export(str(OUT_DIR / 'stl' / 'cortex_hat' /
                      f'cortex_connector_{int(TARGET_MM)}mm.stl'))

# Check overlap: for each area piece, does the connector reach into it?
from scipy.spatial import cKDTree
kd = cKDTree(connector.vertices)
files = sorted((OUT_DIR / 'stl' / 'cortical_areas').glob(f'wt_*_{int(TARGET_MM)}mm.stl'))
area_meshes = {}
pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors) +
       list(mpl.colormaps['tab20c'].colors))
n_overlap = 0
for fi, f in enumerate(files):
    acro = f.stem[len('wt_'):-len(f'_{int(TARGET_MM)}mm')]
    m = trimesh.load(str(f), force='mesh')
    area_meshes[acro] = (pal[fi % len(pal)], m)
    # does connector contain any of this area's deep vertices?
    inside = connector.contains(m.vertices)
    if inside.any():
        n_overlap += 1
print(f'{n_overlap}/{len(files)} area pieces overlap the connector')

# Render: areas colored + connector translucent gray, to confirm it's
# hidden under the pia.
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
def mp(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))
all_v = np.vstack([m.vertices for _, m in area_meshes.values()])
d = (all_v.max(0) - all_v.min(0)).max() * 1.8
views = [
    ('areas + connector — dorsal',  (0, -d, 0), (0, 0, 0), (-1, 0, 0), True),
    ('connector only — dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0), False),
    ('areas + connector — lateral',  (0, 0, d), (0, 0, 0), (0, -1, 0), True),
    ('cut: connector inside',        (0.4*d, -0.3*d, 0.5*d), (0,0,0), (0,-1,0), True),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for k, (vname, cam, focal, up, show_areas) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    if show_areas:
        for acro, (color, m) in area_meshes.items():
            pl.add_mesh(mp(m), color=color, smooth_shading=True,
                         specular=0.2, ambient=0.25, diffuse=0.9,
                         opacity=0.55)
    pl.add_mesh(mp(connector), color=(0.4, 0.4, 0.42),
                 smooth_shading=True, specular=0.1, ambient=0.3,
                 diffuse=0.8)
    pl.add_text(vname, font_size=11)
    pl.camera_position = (cam, focal, up); pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'cortex_connector_check.png'))
pl.close()
print('wrote cortex_connector_check.png')
