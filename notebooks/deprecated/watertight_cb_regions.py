"""Apply the same trimesh-fill watertighting to the cerebellum lobules
and the 9 major divisions (no voxelize/union — preserves Allen's smooth
shape). Reports watertight status for each.
"""
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE

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

def process(src_obj, dst_stl):
    m = trimesh.load(str(src_obj), force='mesh')
    if len(m.faces) == 0:
        return None
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    comps = m.split(only_watertight=False)
    if len(comps) > 1:
        sizes = np.array([len(c.faces) for c in comps])
        keep = [c for c, k in zip(comps, sizes/sizes.sum() >= 0.05) if k]
        if not keep: keep = [comps[int(np.argmax(sizes))]]
        m = trimesh.util.concatenate(keep)
        m.merge_vertices()
        m.update_faces(m.unique_faces())
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
    if not all(c.is_watertight for c in m.split(only_watertight=False)):
        m2 = ms_close_holes(m)
        if all(c.is_watertight for c in m2.split(only_watertight=False)):
            m = m2
        else:
            # close-holes insufficient (complex topology) → alpha-wrap.
            # ~1% shape change but guarantees a single watertight solid.
            ms = ml.MeshSet()
            ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64),
                                 m.faces.astype(np.int32)))
            diag = float(np.linalg.norm(m.bounding_box.extents))
            ms.generate_alpha_wrap(alpha=ml.PureValue(diag*0.010),
                                    offset=ml.PureValue(diag*0.003))
            cm = ms.current_mesh()
            m = trimesh.Trimesh(vertices=cm.vertex_matrix(),
                                 faces=cm.face_matrix(), process=True)
    m.apply_scale(SHARED_SCALE)
    m.apply_translation(-root_ctr)
    m.export(str(dst_stl))
    wt = all(c.is_watertight for c in m.split(only_watertight=False))
    return wt, len(m.vertices)

# 9 major divisions (Allen native region OBJs)
REGIONS = ['CNU', 'HPF', 'CTXsp', 'TH', 'HY', 'MB', 'HB', 'CB', 'OLF']
REGION_SID = {'CNU': 623, 'HPF': 1089, 'CTXsp': 703, 'TH': 549,
               'HY': 1097, 'MB': 313, 'HB': 1065, 'CB': 512, 'OLF': 698}
print('=== 9 major divisions ===')
for acro in REGIONS:
    sid = REGION_SID[acro]
    src = OUT_DIR / 'obj' / f'{sid}_{acro}.obj'
    if not src.exists():
        print(f'  {acro:6s} (no obj)')
        continue
    dst = OUT_DIR / 'stl' / 'regions' / f'wt_{acro}_{int(TARGET_MM)}mm.stl'
    r = process(src, dst)
    if r:
        print(f'  {acro:6s} watertight={r[0]}  verts={r[1]:,}')

# cerebellum lobules (cb_<sid>_<acro>.obj)
print('\n=== cerebellum lobules ===')
cb_files = sorted((OUT_DIR / 'obj').glob('cb_*.obj'))
nwt = 0
for f in cb_files:
    acro = f.stem.split('_', 2)[2]
    dst = OUT_DIR / 'stl' / 'cerebellum_areas' / f'wt_{acro}_{int(TARGET_MM)}mm.stl'
    r = process(f, dst)
    if r:
        nwt += int(r[0])
        print(f'  {acro:8s} watertight={r[0]}  verts={r[1]:,}')
print(f'\ncerebellum: {nwt}/{len(cb_files)} watertight')
print('done')
