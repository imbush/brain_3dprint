"""Does the cortex hat fit cleanly onto the per-region core?

Loads cortex_hat_etched + the 9 region STLs, overlays them, and checks:
- Whether they overlap (cortex sits inside region territory)
- Whether there are gaps (open volume between cortex and regions)
- Renders the assembled set + cross-sections for inspection
"""
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import pyvista as pv
from matplotlib.colors import to_rgb

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')

# Reload everything fresh and apply the SAME scale to all 10 pieces.
root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE

REGIONS = [
    (623,  'CNU',   '#5dd39e'),
    (1089, 'HPF',   '#ff9f1c'),
    (703,  'CTXsp', '#bdb2ff'),
    (549,  'TH',    '#cf6cf6'),
    (1097, 'HY',    '#f08080'),
    (313,  'MB',    '#5a9bff'),
    (1065, 'HB',    '#118ab2'),
    (512,  'CB',    '#fa8072'),
    (698,  'OLF',   '#ffd166'),
]
def load_and_scale(obj_name):
    m = trimesh.load(OUT_DIR / obj_name, force='mesh')
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.apply_scale(SHARED_SCALE)
    m.apply_translation(-root_ctr)
    return m

cortex = load_and_scale('688_isocortex.obj')
print(f'cortex: {len(cortex.vertices):,} verts, {len(cortex.faces):,} faces')
regions = {}
for sid, acro, color in REGIONS:
    m = load_and_scale(f'{sid}_{acro}.obj')
    regions[acro] = (color, m)
    print(f'{acro}: {len(m.vertices):,} verts')

# === Geometric fit check ===
# For each region vertex, find distance to closest cortex vertex.
# - If positive (region OUTSIDE cortex but close): clean fit
# - If region vertices are INSIDE cortex volume: overlap
# - If far away (gap): bad fit

# Bounding box check — does cortex envelope contain all region pieces?
cortex_bb = cortex.bounding_box
print('\nbbox check (1 if region bbox inside cortex bbox):')
for acro, (color, m) in regions.items():
    rb = m.bounding_box
    inside = (rb.bounds[0] >= cortex_bb.bounds[0] - 0.5).all() and \
             (rb.bounds[1] <= cortex_bb.bounds[1] + 0.5).all()
    print(f'  {acro}: bbox_inside={inside}  '
          f'ext={[round(x,1) for x in rb.extents.tolist()]}')

# For each region, check inside/outside ratio relative to cortex
# trimesh has contains() but it requires watertight mesh. Use raycasting.
ray = trimesh.ray.ray_pyembree.RayMeshIntersector(cortex) \
        if trimesh.ray.has_embree else trimesh.ray.ray_triangle.RayMeshIntersector(cortex)
print('\nfit metrics per region (cortex hat as reference):')
print(f'  cortex hat bbox extents: '
      f'{[round(x, 2) for x in cortex.bounding_box.extents.tolist()]} mm')
kd_cortex = cKDTree(cortex.vertices)
for acro, (color, m) in regions.items():
    # Sample region vertices and measure distance to cortex surface
    d, _ = kd_cortex.query(m.vertices, k=1)
    print(f'  {acro:6s}  '
          f'min_d_to_cortex={d.min():.2f}mm  '
          f'mean={d.mean():.2f}mm  '
          f'max={d.max():.2f}mm  '
          f'region_volume={m.volume:.0f}mm^3')

# === Render combined ===
pv.OFF_SCREEN = True
pv.set_plot_theme('document')

def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

# Get bbox for camera
all_v = np.vstack([cortex.vertices] +
                   [m.vertices for _, m in regions.values()])
d = (all_v.max(0) - all_v.min(0)).max() * 1.8

views = [
    ('full dorsal',                      (0, -d, 0),       (0,0,0), (-1,0,0)),
    ('full oblique',                     (0.6*d, -0.5*d, 0.6*d), (0,0,0), (0,-1,0)),
    ('cortex translucent — dorsal',      (0, -d, 0),       (0,0,0), (-1,0,0)),
    ('cortex translucent — oblique',     (0.6*d, -0.5*d, 0.6*d), (0,0,0), (0,-1,0)),
    ('cortex sliced off (top) — dorsal', (0, -d, 0),       (0,0,0), (-1,0,0)),
    ('cortex sliced off — oblique',      (0.6*d, -0.5*d, 0.6*d), (0,0,0), (0,-1,0)),
]
pl = pv.Plotter(shape=(3, 2), window_size=(2200, 2400), off_screen=True)
for k, (vname, cam, focal, up) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    # Decide rendering mode based on panel title
    if 'sliced' in vname:
        # Hide cortex, show regions
        for acro, (color, m) in regions.items():
            pl.add_mesh(make_poly(m), color=color,
                         smooth_shading=True, specular=0.20,
                         ambient=0.25, diffuse=0.90)
    elif 'translucent' in vname:
        # Translucent cortex, opaque regions
        for acro, (color, m) in regions.items():
            pl.add_mesh(make_poly(m), color=color,
                         smooth_shading=True, specular=0.20,
                         ambient=0.25, diffuse=0.90)
        pl.add_mesh(make_poly(cortex), color=(0.83, 0.78, 0.70),
                     smooth_shading=True, specular=0.15,
                     ambient=0.30, diffuse=0.65, opacity=0.35)
    else:
        # Opaque cortex hat (the regions are hidden behind)
        for acro, (color, m) in regions.items():
            pl.add_mesh(make_poly(m), color=color,
                         smooth_shading=True, specular=0.20,
                         ambient=0.25, diffuse=0.90)
        pl.add_mesh(make_poly(cortex), color=(0.88, 0.82, 0.74),
                     smooth_shading=True, specular=0.15,
                     ambient=0.30, diffuse=0.80, opacity=0.85)
    pl.add_text(vname, font_size=10)
    pl.camera_position = (cam, focal, up)
    pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'fit_check_assembly.png'))
pl.close()
print('wrote fit_check_assembly.png')

# === Coronal cross-section to see the cortex/region interface ===
slab = cortex.bounding_box.centroid[0]   # cut at AP midpoint
# half-slabs for everything
def slice_half(m, z_world, normal):
    try:
        h = m.copy().slice_plane(plane_origin=z_world,
                                   plane_normal=normal, cap=True)
        return h if h is not None and len(h.vertices) > 0 else None
    except Exception:
        return None
half_cortex = slice_half(cortex, [slab, 0, 0], [-1, 0, 0])
half_regions = {acro: (color, slice_half(m, [slab, 0, 0], [-1, 0, 0]))
                 for acro, (color, m) in regions.items()}

pl = pv.Plotter(shape=(1, 2), window_size=(2200, 1100), off_screen=True)
for k, (vname, cam) in enumerate([
    ('coronal cut (anterior half)', (d, 0, 0)),
    ('coronal oblique',              (0.3*d, -0.3*d, 0.9*d)),
]):
    pl.subplot(0, k)
    for acro, (color, h) in half_regions.items():
        if h is None: continue
        pl.add_mesh(make_poly(h), color=color,
                     smooth_shading=True, specular=0.15,
                     ambient=0.30, diffuse=0.85)
    if half_cortex is not None:
        pl.add_mesh(make_poly(half_cortex), color=(0.83, 0.78, 0.70),
                     smooth_shading=True, specular=0.10,
                     ambient=0.30, diffuse=0.75)
    pl.add_text(vname, font_size=10)
    pl.camera_position = (cam, (0, 0, 0), (0, -1, 0))
    pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'fit_check_coronal.png'))
pl.close()
print('wrote fit_check_coronal.png')

print('done')
