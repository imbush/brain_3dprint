"""Build a single brain-shaped STL with the Bambu Lab kit shell as an
interior cavity. The Bambu kit case (already containing the electronics)
slides UP into this brain from below; the brain is a decorative outer
cover.

Inputs:
  data/structure_sets/<set>/stl/<set>_engraved_66mm.stl   (engraved set)
  data/ccf/obj/<id>_<acro>.obj                            (parent envelope fallback)
  data/mouse_kit/extracted/05_*hardware_shell*.stl        (Bambu kit shell)
  data/mouse_brain_assembly/brain_in_shell.json           (placement transform)

Output:
  data/mouse_brain_assembly/brain_over_shell.stl
  data/mouse_brain_assembly/brain_over_shell_preview.png
"""
from pathlib import Path
import json
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
CCF  = DATA / 'ccf' / 'obj'
SETS = DATA / 'structure_sets'
KIT  = DATA / 'mouse_kit' / 'extracted'
ASM  = DATA / 'mouse_brain_assembly'

CLEARANCE_MM      = 1.0     # gap between Bambu shell and brain cavity wall
BOTTOM_OPEN_MM    = 1.5     # how much above the cut plane the cavity rises
FLAT_BOTTOM_MARGIN = 2.0    # brain extends this far below the shell bottom

# ---- mesh helpers -------------------------------------------------------
def ev(m):
    if m is None or len(m.faces) == 0: return m
    if m.is_volume: return m
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
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
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.generate_alpha_wrap(alpha=ml.PureValue(d*0.01), offset=ml.PureValue(d*0.003))
    cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

def voxel_dilate(m, mm, pitch=0.6):
    iters = max(1, int(round(mm / pitch)))
    vox = m.voxelized(pitch=pitch).fill()
    pad = np.zeros([s + 2 * iters for s in vox.matrix.shape], dtype=bool)
    pad[iters:-iters, iters:-iters, iters:-iters] = vox.matrix
    grown = binary_dilation(pad, iterations=iters)
    T = vox.transform.copy()
    T[:3, 3] = T[:3, 3] - iters * np.diag(T)[:3]
    out = trimesh.voxel.VoxelGrid(grown, transform=T).marching_cubes
    out.apply_transform(T)
    out.merge_vertices()
    return out

# ---- load whole brain ----------------------------------------------------
TM_REF = 66
root_ccf = trimesh.load(str(CCF/'997_root.obj'), force='mesh')
SHARED_SCALE = TM_REF / root_ccf.bounding_box.extents.max()
ROOT_CTR = root_ccf.bounding_box.centroid * SHARED_SCALE

def load_set(label, engraved_path=None, parent_obj=None):
    if engraved_path and engraved_path.exists():
        m = trimesh.load(str(engraved_path), force='mesh')
        return label, m
    if parent_obj and (CCF/parent_obj).exists():
        m = trimesh.load(str(CCF/parent_obj), force='mesh')
        m.apply_scale(SHARED_SCALE); m.apply_translation(-ROOT_CTR)
        return label, m
    return None

print('=== loading whole brain ===')
loaded = list(filter(None, [
    load_set('isocortex',    SETS/'isocortex/stl/FINAL_onepiece_voxsheet_arial.stl'),
    load_set('cerebellum',   SETS/'cerebellum/stl/cerebellum_engraved_66mm.stl'),
    load_set('olfactory',    SETS/'olfactory/stl/olfactory_engraved_66mm.stl'),
    load_set('cnu_ctxsp',    SETS/'cnu_ctxsp/stl/cnu_ctxsp_engraved_66mm.stl'),
    load_set('thalamus',     parent_obj='549_TH.obj'),
    load_set('hypothalamus', parent_obj='1097_HY.obj'),
    load_set('midbrain',     parent_obj='313_MB.obj'),
    load_set('hindbrain',    parent_obj='1065_HB.obj'),
    load_set('hippocampal',  parent_obj='1089_HPF.obj'),
]))
for label, m in loaded:
    print(f'  {label:14s}  bbox={[round(x,1) for x in m.bounding_box.extents]}')

# Same axis flip as the placement tool (so olfactory faces shell's front)
FLIP_BRAIN_AP = np.diag([-1.0, 1.0, -1.0, 1.0])
for _, m in loaded:
    m.apply_transform(FLIP_BRAIN_AP)

# Apply the user's saved placement
cfg = json.loads((ASM/'brain_in_shell.json').read_text())
print(f'\nuser transform: scale={cfg["scale"]:.3f}  tilt={cfg["tilt_deg"]:+.1f}°  '
      f'tx={cfg["tx"]:+.2f}  ty={cfg["ty"]:+.2f}')

all_v = np.vstack([m.vertices for _, m in loaded])
brain_centroid = (all_v.min(0) + all_v.max(0)) / 2

def brain_xform(cfg):
    T = np.eye(4); T[:3, 3] = -brain_centroid
    S = np.eye(4) * cfg['scale']; S[3, 3] = 1.0
    th = np.deg2rad(cfg['tilt_deg'])
    c, s = np.cos(th), np.sin(th)
    R = np.eye(4)
    R[0, 0] = c;  R[0, 1] = -s
    R[1, 0] = s;  R[1, 1] =  c
    Tr = np.eye(4); Tr[:3, 3] = [cfg['tx'], cfg['ty'], 0.0]
    return Tr @ R @ S @ T
BX = brain_xform(cfg)

for _, m in loaded:
    m.apply_transform(BX)

# ---- fuse brain pieces into one solid (ev each first) ------------------
print('\n=== fusing brain pieces ===')
fixed = []
for label, m in loaded:
    fm = ev(m)
    fixed.append(fm)
    print(f'  {label:14s}  ev() -> volume_ok={fm.is_volume}  vol={fm.volume:.0f}mm³')
brain = trimesh.boolean.union(fixed, engine='manifold')
brain_comps = sorted(brain.split(only_watertight=False), key=lambda c: -c.volume)
print(f'  fused: {len(brain_comps)} components, total vol {brain.volume:.0f}mm³')

# ---- load + orient Bambu kit shell -------------------------------------
print('\n=== loading Bambu kit shell ===')
SHELL_STL = KIT / '05_3D_model_for_Wireless_Mouse_Kit-002__hardware_shell____.stl'
shell = trimesh.load(str(SHELL_STL), force='mesh')
shell.apply_translation(-shell.bounding_box.centroid)
R_SHELL = np.array([[0, 1, 0, 0],
                    [0, 0,-1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]], dtype=float)
shell.apply_transform(R_SHELL)
shell_comps = [c for c in shell.split(only_watertight=False) if c.volume > 100]
# Use the two case halves merged into one solid envelope
shell_solid = trimesh.util.concatenate(shell_comps)
shell_solid = ev(shell_solid)
print(f'  bambu shell solid: vol {shell_solid.volume:.0f}mm³  '
      f'bbox {[round(x,1) for x in shell_solid.bounding_box.extents]}')

# ---- cavity = convex hull of (shell + clearance) so the shell drops in
# cleanly without snagging on internal pockets/ribs of the Bambu case.
hull_input = voxel_dilate(shell_solid, CLEARANCE_MM)
cavity = hull_input.convex_hull
cavity = ev(cavity)
print(f'  cavity = convex hull of (shell + {CLEARANCE_MM}mm clearance): '
      f'vol {cavity.volume:.0f}mm³  bbox {[round(x,1) for x in cavity.bounding_box.extents]}')

# ---- flat-cut the brain bottom so the shell can slide in from below ----
brain_bounds = brain.bounds
shell_bounds = shell_solid.bounds
flat_y = shell_bounds[1][1] + FLAT_BOTTOM_MARGIN     # just below shell bottom
print(f'\nflat-cut brain bottom at y={flat_y:.1f}  '
      f'(shell bottom at y={shell_bounds[1][1]:.1f})')
brain_cut = brain.slice_plane(plane_origin=[0, flat_y, 0],
                              plane_normal=[0, -1, 0], cap=True)
brain_cut = ev(brain_cut)
print(f'  brain after flat-cut: vol {brain_cut.volume:.0f}mm³')

# ---- subtract cavity from brain ----------------------------------------
print('\n=== carving shell-shaped cavity ===')
# extend cavity downward through cut plane so a clear opening remains
# Extend the cavity through the brain's flat bottom so the shell can
# actually slide IN from below (otherwise a thin wall remains and the shell
# is trapped). ext_floor spans from the cavity's lower edge to a few mm
# past the brain's flat bottom.
cb = cavity.bounds
ext_floor = trimesh.creation.box(extents=(
    cb[1,0]-cb[0,0] + 2,
    (flat_y + 5) - cb[0,1],
    cb[1,2]-cb[0,2] + 2,
))
ext_floor.apply_translation([
    (cb[0,0]+cb[1,0])/2,
    (cb[0,1] + flat_y + 5)/2,         # spans cavity-top to past brain bottom
    (cb[0,2]+cb[1,2])/2,
])
cavity_open = ev(trimesh.boolean.union([cavity, ext_floor], engine='manifold'))
print(f'  cavity_open: vol {cavity_open.volume:.0f}mm³  '
      f'bbox {[round(x,1) for x in cavity_open.bounding_box.extents]}')
brain_over_shell = brain_cut.difference(cavity_open, engine='manifold')
brain_over_shell = ev(brain_over_shell)
comps = sorted(brain_over_shell.split(only_watertight=False), key=lambda c: -c.volume)
# Keep every substantial piece (e.g. both olfactory bulbs survive even if
# they're not connected to the main shell). Threshold = 50 mm³ so we drop
# only true boolean debris.
keep = [c for c in comps if c.volume > 50]
if len(keep) < len(comps):
    print(f'  components: {len(comps)} -> {len(keep)} kept ('
          f'dropped {len(comps)-len(keep)} debris < 50mm³)')
brain_over_shell = trimesh.util.concatenate(keep)
brain_over_shell.merge_vertices()
print(f'  final: vol {brain_over_shell.volume:.0f}mm³  faces {len(brain_over_shell.faces):,}  '
      f'components={len(keep)}')

# ---- write ---------------------------------------------------------------
OUT = ASM / 'brain_over_shell.stl'
brain_over_shell.export(str(OUT))
print(f'\nwrote {OUT.relative_to(DATA.parent)}')

# ---- preview ------------------------------------------------------------
def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

views = [('dorsal (above)',  ( 0, -1,  0), (-1,  0,  0)),
         ('lateral right',   ( 0,  0,  1), ( 0, -1,  0)),
         ('front (rostral)', (-1,  0,  0), ( 0, -1,  0)),
         ('exploded',        ( 0,  0,  1), ( 0, -1,  0))]
pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1900, 1500))
for i, (name, view, up) in enumerate(views):
    pl.subplot(i // 2, i % 2)
    pl.add_mesh(poly(brain_over_shell), color=(0.92, 0.72, 0.72),
                smooth_shading=True, opacity=0.55,
                specular=0.3, ambient=0.35, diffuse=0.85)
    sh = shell_solid.copy()
    if name == 'exploded': sh.apply_translation([0, 25, 0])  # drop down for visibility
    pl.add_mesh(poly(sh), color=(0.6, 0.6, 0.7),
                smooth_shading=True, opacity=0.95,
                specular=0.3, ambient=0.3, diffuse=0.85)
    pl.add_text(name, font_size=10, color='black', position='upper_left')
    pl.background_color = 'white'
    pl.view_vector(view, up); pl.reset_camera()
png = ASM / 'brain_over_shell_preview.png'
pl.screenshot(str(png))
print(f'preview -> {png.relative_to(DATA.parent)}')
