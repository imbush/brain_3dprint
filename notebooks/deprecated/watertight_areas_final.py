"""Final: each of the 43 cortical-area meshes is ALREADY a smooth
watertight volume from Allen. We just:
  1. clean (merge dupes, drop degenerate faces)
  2. keep components ≥5% (drops floaters, keeps bilateral L+R pairs)
  3. if not watertight → pymeshlab close_holes + repair (rare)
  4. save scaled-to-print STL
This preserves Allen's smooth shape exactly (voxelize/union made it wavy
— avoided entirely).

Outputs:
  stl/cortical_areas/wt_<acro>_<TARGET_MM>mm.stl  (watertight, smooth)
  png/watertight_areas_combined.png               (all overlaid)
"""
import time
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
from allensdk.core.reference_space_cache import ReferenceSpaceCache
import pyvista as pv
import matplotlib as mpl

OUT_DIR = Path(__file__).resolve().parent.parent / 'data'
rspc = ReferenceSpaceCache(resolution=25,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
tree = rspc.get_structure_tree()

root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE

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
area_acro = {sid: tree.get_structures_by_id([int(sid)])[0]['acronym']
              for sid in iso_kid_set}

def ms_close_holes(m):
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64),
                         m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    for _ in range(4):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=30000)
        except Exception: pass
    try: ms.meshing_re_orient_faces_coherently()
    except Exception: pass
    cm = ms.current_mesh()
    return trimesh.Trimesh(vertices=cm.vertex_matrix(),
                            faces=cm.face_matrix(), process=True)

pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors) +
       list(mpl.colormaps['tab20c'].colors))

processed = {}
not_wt = []
for k, sid in enumerate(sorted(iso_kid_set)):
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
    comps = m.split(only_watertight=False)
    if len(comps) > 1:
        sizes = np.array([len(c.faces) for c in comps])
        keep = [c for c, kk in zip(comps, sizes/sizes.sum() >= 0.05) if kk]
        if not keep: keep = [comps[int(np.argmax(sizes))]]
        m = trimesh.util.concatenate(keep)
        m.merge_vertices()
        m.update_faces(m.unique_faces())
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
    # If a kept component isn't watertight, close holes (preserves shape)
    if not all(c.is_watertight for c in m.split(only_watertight=False)):
        m = ms_close_holes(m)
    # scale to print + center (shared frame)
    m.apply_scale(SHARED_SCALE)
    m.apply_translation(-root_ctr)
    wt_all = all(c.is_watertight for c in m.split(only_watertight=False))
    if not wt_all:
        not_wt.append(acro)
    processed[acro] = (pal[k % len(pal)], m)
    stl = OUT_DIR / 'stl' / 'cortical_areas' / f'wt_{acro}_{int(TARGET_MM)}mm.stl'
    m.export(str(stl))

n_wt = sum(1 for acro, (c, m) in processed.items()
            if all(cc.is_watertight for cc in m.split(only_watertight=False)))
print(f'processed {len(processed)} areas; '
      f'{n_wt} fully watertight; not-wt: {not_wt}')

# Render combined
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))
all_v = np.vstack([m.vertices for _, m in processed.values()])
d = (all_v.max(0) - all_v.min(0)).max() * 1.8
views = [
    ('dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
    ('lateral R',  (0,  0,  d), (0, 0, 0), (0, -1, 0)),
    ('lateral L',  (0,  0, -d), (0, 0, 0), (0, -1, 0)),
    ('ventral',    (0,  d, 0), (0, 0, 0), (1, 0, 0)),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for kk, (vname, cam, focal, up) in enumerate(views):
    pl.subplot(kk // 2, kk % 2)
    for acro, (color, m) in processed.items():
        pl.add_mesh(make_poly(m), color=color, smooth_shading=True,
                     specular=0.2, ambient=0.25, diffuse=0.9)
    pl.add_text(vname, font_size=12)
    pl.camera_position = (cam, focal, up); pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'watertight_areas_combined.png'))
pl.close()
print('wrote watertight_areas_combined.png')
