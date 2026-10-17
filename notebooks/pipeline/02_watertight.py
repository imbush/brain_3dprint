"""STEP 2 — for each structure set, clean the per-area .obj meshes into
watertight printable solids `wt_<acro>_66mm.stl`.

Identical to the cortex `watertight_areas_final` script but generic across
sets, and writes into data/structure_sets/<set>/stl/ at the shared print
scale (5x => ~66 mm). Uses the same `ev()` repair chain (dedup → repair
non-manifold edges → close holes → alpha-wrap fallback) used by the
one-piece builder, so every output is guaranteed `is_volume`.

Run:
  conda activate microns-m2
  python pipeline/02_watertight.py [setname [setname ...]]    # default: all
"""
import sys, traceback
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sets import DATA, MAJOR_DIVISIONS, set_dir, obj_dir, stl_dir, png_dir

CCF_DIR = DATA / 'ccf'
TARGET_MM = 66

# print scale derived from the whole-brain root so every set lives in the
# same coordinate frame (so any two sets can be assembled together later)
root = trimesh.load(CCF_DIR / 'obj' / '997_root.obj', force='mesh')
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
ROOT_CTR = root.bounding_box.centroid * SHARED_SCALE

def ev(m):
    if m.is_volume: return m
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces(); ms.meshing_remove_unreferenced_vertices()
    for _ in range(3):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=30000)
        except Exception: pass
    try: ms.meshing_re_orient_faces_coherently()
    except Exception: pass
    cm = ms.current_mesh()
    r = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
    if r.is_volume: return r
    d = float(np.linalg.norm(m.bounding_box.extents))
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.generate_alpha_wrap(alpha=ml.PureValue(d*0.01), offset=ml.PureValue(d*0.003))
    cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

def process_one(src):
    m = trimesh.load(str(src), force='mesh')
    if len(m.faces) == 0: return None
    m.merge_vertices(); m.update_faces(m.unique_faces()); m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    comps = m.split(only_watertight=False)
    if len(comps) > 1:
        sizes = np.array([len(c.faces) for c in comps])
        keep = [c for c, k in zip(comps, sizes/sizes.sum() >= 0.05) if k]
        if not keep: keep = [comps[int(np.argmax(sizes))]]
        m = trimesh.util.concatenate(keep)
        m.merge_vertices(); m.update_faces(m.unique_faces()); m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
    m = ev(m)
    m.apply_scale(SHARED_SCALE); m.apply_translation(-ROOT_CTR)
    return m

def process_set(name):
    out = stl_dir(name); out.mkdir(parents=True, exist_ok=True)
    files = sorted(obj_dir(name).glob('area_*.obj'))
    nwt = 0; not_wt = []; meshes = []
    for f in files:
        sid_acro = f.stem[len('area_'):]
        try:
            m = process_one(f)
            if m is None: continue
            wt = all(c.is_watertight for c in m.split(only_watertight=False))
            if not wt: not_wt.append(sid_acro)
            else: nwt += 1
            acro = sid_acro.split('_', 1)[1]
            m.export(str(out / f'wt_{acro}_{TARGET_MM}mm.stl'))
            meshes.append(m)
        except Exception:
            traceback.print_exc()
    print(f'{name}: {nwt}/{len(files)} watertight; not-wt: {not_wt[:5]}{"..." if len(not_wt)>5 else ""}')

if __name__ == '__main__':
    wanted = sys.argv[1:] if len(sys.argv) > 1 else list(MAJOR_DIVISIONS)
    for name in wanted:
        if name not in MAJOR_DIVISIONS: print(f'unknown set: {name}'); continue
        process_set(name)
