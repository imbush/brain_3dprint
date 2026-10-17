"""Compare approaches to make a single cortical-area mesh watertight
while keeping it smooth + shape-faithful.

Tests on 3 representative areas:
  VISp     — compact blob
  SSp-bfd  — large
  RSPd     — thin curved strip (hardest)

Approaches (all via pymeshlab unless noted):
  A trimesh_fill   — trimesh fill_holes + fix_normals
  B close_holes    — pymeshlab iterative close_holes + repair
  C alpha_wrap     — CGAL alpha-wrapping (shape-preserving, watertight)
  D resample_sdf   — SDF uniform resampling (marching cubes on signed
                     distance field) + Taubin
  E poisson        — screened Poisson surface reconstruction
  F voxel_mc       — naive voxelize+marching cubes+Taubin (the wavy one)

For each: report watertight / is_volume / verts / Hausdorff-to-original
(shape fidelity, lower = better) and render a comparison grid.
"""
import time
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
TEST = [('VISp', 385), ('SSp-bfd', 329), ('RSPd', 879)]

def trimesh_to_ms(m):
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64),
                         m.faces.astype(np.int32)))
    return ms

def ms_to_trimesh(ms):
    cm = ms.current_mesh()
    return trimesh.Trimesh(vertices=cm.vertex_matrix(),
                            faces=cm.face_matrix(), process=True)

def hausdorff_pct(orig, new):
    """Max surface deviation as % of bbox diagonal (sample-based)."""
    from scipy.spatial import cKDTree
    kd = cKDTree(orig.vertices)
    d, _ = kd.query(new.vertices)
    diag = np.linalg.norm(orig.bounding_box.extents)
    return 100.0 * d.max() / diag, 100.0 * d.mean() / diag

def load_clean(acro, sid):
    p = OUT_DIR / 'obj' / f'area_{sid}_{acro}.obj'
    m = trimesh.load(str(p), force='mesh')
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    # keep components ≥5% (bilateral pairs kept, floaters dropped)
    comps = m.split(only_watertight=False)
    if len(comps) > 1:
        sizes = np.array([len(c.faces) for c in comps])
        keep = [c for c, k in zip(comps, sizes/sizes.sum() >= 0.05) if k]
        if not keep: keep = [comps[int(np.argmax(sizes))]]
        m = trimesh.util.concatenate(keep)
    return m

# ===== approaches =====
def A_trimesh_fill(m):
    out = m.copy()
    trimesh.repair.fill_holes(out)
    trimesh.repair.fix_normals(out)
    trimesh.repair.fix_winding(out)
    out.merge_vertices(); out.process()
    return out

def B_close_holes(m):
    ms = trimesh_to_ms(m)
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
    return ms_to_trimesh(ms)

def C_alpha_wrap(m, alpha_frac=0.012, offset_frac=0.004):
    ms = trimesh_to_ms(m)
    diag = np.linalg.norm(m.bounding_box.extents)
    ms.generate_alpha_wrap(alpha=ml.PureValue(diag*alpha_frac),
                            offset=ml.PureValue(diag*offset_frac))
    return ms_to_trimesh(ms)

def D_resample_sdf(m, cell_frac=0.012):
    ms = trimesh_to_ms(m)
    diag = np.linalg.norm(m.bounding_box.extents)
    ms.generate_resampled_uniform_mesh(
        cellsize=ml.PureValue(diag*cell_frac),
        offset=ml.PercentageValue(50.0), mergeclosevert=True)
    out = ms_to_trimesh(ms)
    trimesh.smoothing.filter_taubin(out, iterations=12)
    return out

def E_poisson(m, depth=8):
    ms = trimesh_to_ms(m)
    # need vertex normals
    try:
        ms.compute_normal_for_point_clouds()
    except Exception:
        pass
    try:
        ms.generate_surface_reconstruction_screened_poisson(
            depth=depth, samplespernode=2.0, pointweight=4.0)
    except Exception as e:
        print('   poisson err', e)
        return None
    return ms_to_trimesh(ms)

def F_voxel_mc(m, pitch_frac=0.012):
    diag = np.linalg.norm(m.bounding_box.extents)
    v = m.voxelized(pitch=diag*pitch_frac).fill()
    out = v.marching_cubes
    out.apply_scale(diag*pitch_frac)
    out.merge_vertices()
    trimesh.smoothing.filter_taubin(out, iterations=15)
    return out

APPROACHES = {
    'A_trimesh_fill': A_trimesh_fill,
    'B_close_holes':  B_close_holes,
    'C_alpha_wrap':   C_alpha_wrap,
    'D_resample_sdf': D_resample_sdf,
    'E_poisson':      E_poisson,
    'F_voxel_mc':     F_voxel_mc,
}

results = {}
for acro, sid in TEST:
    orig = load_clean(acro, sid)
    print(f'\n##### {acro} (orig {len(orig.vertices)} v, '
          f'wt={orig.is_watertight}) #####')
    results[acro] = {'orig': orig}
    for name, fn in APPROACHES.items():
        try:
            t0 = time.time()
            out = fn(orig)
            dt = time.time() - t0
            if out is None or len(out.faces) == 0:
                print(f'  {name:16s} FAILED')
                continue
            hmax, hmean = hausdorff_pct(orig, out)
            print(f'  {name:16s} wt={str(out.is_watertight):5s} '
                  f'vol={str(out.is_volume):5s} '
                  f'v={len(out.vertices):>7,} '
                  f'Hmax={hmax:5.2f}% Hmean={hmean:4.2f}% '
                  f'({dt:.1f}s)')
            results[acro][name] = out
        except Exception as e:
            print(f'  {name:16s} ERROR: {e}')

# Render comparison grid: rows = areas, cols = approaches (+orig)
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

col_order = ['orig'] + list(APPROACHES.keys())
for acro, _ in TEST:
    r = results[acro]
    cols = [c for c in col_order if c in r]
    pl = pv.Plotter(shape=(1, len(cols)),
                     window_size=(420*len(cols), 460), off_screen=True)
    om = r['orig']
    d = om.bounding_box.extents.max() * 2.0
    ctr = om.bounding_box.centroid
    for k, c in enumerate(cols):
        pl.subplot(0, k)
        m = r[c]
        wt = '✓wt' if m.is_watertight else '✗'
        pl.add_mesh(make_poly(m), color=(0.7, 0.8, 0.95),
                     smooth_shading=True, specular=0.2, ambient=0.25,
                     diffuse=0.9)
        pl.add_text(f'{c}\n{wt}', font_size=8)
        pl.camera_position = ((ctr[0]+0.5*d, ctr[1]-0.4*d, ctr[2]+0.6*d),
                               tuple(ctr), (0, -1, 0))
        pl.reset_camera()
    pl.background_color = 'white'
    pl.screenshot(str(OUT_DIR / 'png' / f'watertight_compare_{acro}.png'))
    pl.close()
    print(f'wrote watertight_compare_{acro}.png')
print('done')
