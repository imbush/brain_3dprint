"""Build the canonical cortex base from the UNION of the 43
per-cortical-area meshes (= the per_cortical_area_combined map).

This replaces the old 688 ("Cerebral cortex" = CTX, which wrongly
included olfactory bulb + hippocampus) base. The union of the true
isocortex areas is the correct cortex envelope.

Pipeline:
1. Load all per-area OBJ meshes (already downloaded by per_cortical_area.py)
2. Concatenate → voxelize at 30 µm → marching cubes → watertight
3. Save as the canonical base: obj/cortex_base_union.stl (input µm)
   and a per-vertex area-label array (saved as .npy) so downstream
   label scripts don't have to re-derive it.
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.spatial import cKDTree
import pyvista as pv

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
rspc = ReferenceSpaceCache(resolution=25,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
tree = rspc.get_structure_tree()

ISO_ID = 315
def is_layer_acro(a):
    return (a.endswith('2/3') or a.endswith('6a') or a.endswith('6b')
             or (len(a) >= 2 and a[-1].isdigit()))
iso_kid_set = set()
for sid in tree.descendant_ids([ISO_ID])[0]:
    if sid == ISO_ID: continue
    kids = tree.child_ids([int(sid)])[0]
    if not kids: continue
    if all(is_layer_acro(tree.get_structures_by_id([int(k)])[0]['acronym'])
            for k in kids):
        iso_kid_set.add(int(sid))
iso_kid_set.update({329, 1011, 894})
area_acro = {sid: tree.get_structures_by_id([sid])[0]['acronym']
              for sid in iso_kid_set}

# Load all per-area OBJ meshes (in input µm). Track which area each
# source mesh belongs to so we can label voxels by nearest source.
area_meshes = []
area_ids_per_mesh = []
for sid in sorted(iso_kid_set):
    acro = area_acro[sid]
    p = OUT_DIR / 'obj' / f'area_{sid}_{acro}.obj'
    if not p.exists():
        continue
    m = trimesh.load(str(p), force='mesh')
    if len(m.faces) == 0:
        continue
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    # keep components ≥5% (drops floaters; keeps bilateral pairs)
    comps = m.split(only_watertight=False)
    if len(comps) > 1:
        sizes = np.array([len(c.faces) for c in comps])
        keep = [c for c, k in zip(comps, sizes / sizes.sum() >= 0.05) if k]
        if not keep:
            keep = [comps[int(np.argmax(sizes))]]
        m = trimesh.util.concatenate(keep)
    area_meshes.append(m)
    area_ids_per_mesh.append(sid)
print(f'loaded {len(area_meshes)} per-area meshes for the union')

union = trimesh.util.concatenate(area_meshes)
print(f'union: {len(union.vertices):,} verts, {len(union.faces):,} faces, '
      f'bbox(mm)={(union.bounding_box.extents/1000).round(2).tolist()}')

# Voxelize + marching cubes → watertight
PITCH = 30.0
print(f'voxelize @ {PITCH} µm + marching cubes...')
tic = time.time()
v = union.voxelized(pitch=PITCH).fill()
base = v.marching_cubes
base.apply_scale(PITCH)   # voxel-index → µm
base.merge_vertices()
base.update_faces(base.unique_faces())
base.update_faces(base.nondegenerate_faces())
base.remove_unreferenced_vertices()
trimesh.smoothing.filter_taubin(base, iterations=18)
print(f'  base: {len(base.vertices):,} verts, {len(base.faces):,} faces  '
      f'wt={base.is_watertight}  vol={base.is_volume}  '
      f'({time.time()-tic:.1f}s)')

base.export(str(OUT_DIR / 'obj' / 'cortex_base_union.stl'))
print('wrote obj/cortex_base_union.stl')

# Label each base vertex by NEAREST per-area source mesh. Build a KDTree
# of all source-mesh vertices tagged with their area id.
src_pts = []
src_ids = []
for sid, m in zip(area_ids_per_mesh, area_meshes):
    src_pts.append(m.vertices)
    src_ids.append(np.full(len(m.vertices), sid, dtype=np.int64))
src_pts = np.vstack(src_pts)
src_ids = np.concatenate(src_ids)
kd = cKDTree(src_pts)
_, ii = kd.query(base.vertices)
base_vert_area = src_ids[ii]
np.save(OUT_DIR / 'obj' / 'cortex_base_union_vert_area.npy', base_vert_area)
print(f'labeled {len(base.vertices):,} base verts by nearest area; '
      f'{len(set(base_vert_area.tolist()))} areas present')

# Quick render to confirm the olfactory bulb is gone
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
import matplotlib as mpl
pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors) +
       list(mpl.colormaps['tab20c'].colors))
present = sorted(set(base_vert_area.tolist()))
cmap = {sid: pal[i % len(pal)] for i, sid in enumerate(present)}
# Face color = first vertex's area
fa = base_vert_area[base.faces[:, 0]]
face_rgb = np.array([cmap[int(a)] for a in fa])
poly = pv.PolyData(
    base.vertices,
    np.hstack([np.full((len(base.faces), 1), 3), base.faces]).astype(np.int64))
poly.cell_data['rgb'] = (face_rgb * 255).astype(np.uint8)
d = base.bounding_box.extents.max() * 1.9
views = [
    ('dorsal',    (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
    ('lateral R', (0,  0, d), (0, 0, 0), (0, -1, 0)),
    ('anterior',  (-d, 0, 0), (0, 0, 0), (0, -1, 0)),
    ('ventral',   (0,  d, 0), (0, 0, 0), (1, 0, 0)),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2000, 1700), off_screen=True)
for k, (vname, cam, focal, up) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    pl.add_mesh(poly, scalars='rgb', rgb=True, smooth_shading=True,
                 specular=0.2, ambient=0.25, diffuse=0.9)
    pl.add_text(vname, font_size=11)
    pl.camera_position = (cam, focal, up); pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'cortex_base_union.png'))
pl.close()
print('wrote cortex_base_union.png')
