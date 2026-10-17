"""Mouse-brain MOUSE — 3D-printable computer-mouse shaped like the mouse
brain, sized for the Bambu Lab Wireless Mouse Components Kit-002.

Outputs under data/mouse_brain_assembly/:
  body.stl        main hollow shell (cortex + cerebellum + olfactory),
                  flat-cut bottom, internal cavity for the kit chassis,
                  shallow pocket on top where the cortex buttons sit,
                  midline slot for the scroll wheel.
  cortex_L.stl    left  hemisphere cap = left-click button.
  cortex_R.stl    right hemisphere cap = right-click button.
  base_plate.stl  flat closure for the bottom opening; the kit electronics
                  slide in from below before this plate is screwed on.
  preview.png     4-up assembled / exploded render.

Geometry conventions (all in mm, brain frame):
  x = anterior(-) / posterior(+)        (AP — mouse length axis = ~112)
  y = dorsal(-)   / ventral(+)          (DV — top of brain is small y)
  z = left(-)     / right(+)            (LR — width = ~88)

Run:
    python notebooks/build/mouse_brain_assembly.py
"""
from pathlib import Path
import sys, json
import numpy as np
import trimesh
import pymeshlab as ml
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
CCF  = DATA / 'ccf' / 'obj'
SETS = DATA / 'structure_sets'
KIT  = DATA / 'mouse_kit' / 'extracted'
OUT  = DATA / 'mouse_brain_assembly'
OUT.mkdir(parents=True, exist_ok=True)

# ---- tunables -----------------------------------------------------------
MOUSE_LENGTH_MM   = 110.0    # AP extent of final shell
FLAT_BOTTOM_MM    = 45.0     # height kept above the cut plane
WALL_MM           = 2.5      # shell wall thickness after hollowing
CAVITY_MARGIN_MM  = 1.5      # clearance around every chassis component
COMP_VOLUME_MIN   = 100      # mm³ — components below this are screws / springs
CORTEX_CAP_MM     = 8.0      # how thick the click-button caps are. Must be
                             # > WALL_MM so the pocket carve doesn't punch
                             # through the body wall in the cortex region.
BUTTON_GAP_MM     = 0.4      # clearance between cap and body in the pocket
WHEEL_SLOT_W      = 25.0     # mm LR — wheel disk dia + clearance
WHEEL_SLOT_AP     = 22.0     # mm AP — slot length
WHEEL_SLOT_DROP   = 18.0     # mm DV — how deep the slot cuts down from cortex top
BASE_PLATE_MM     = 2.5      # base-plate thickness
BOSS_OUTER_R      = 2.4      # M2 screw boss outer radius (mm)
BOSS_INNER_R      = 0.9      # M2 pilot hole (self-tap fits)
BOSS_HEIGHT       = 6.0      # post height up from cut plane

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

def voxel_offset(m, mm, pitch=0.6):
    """Minkowski offset by `mm` (positive=outward, negative=inward)."""
    iters = max(1, int(round(abs(mm) / pitch)))
    vox = m.voxelized(pitch=pitch).fill()
    T = vox.transform.copy()
    if mm > 0:
        pad = np.zeros([s + 2 * iters for s in vox.matrix.shape], dtype=bool)
        pad[iters:-iters, iters:-iters, iters:-iters] = vox.matrix
        out_grid = binary_dilation(pad, iterations=iters)
        T[:3, 3] = T[:3, 3] - iters * np.diag(T)[:3]
    else:
        out_grid = binary_erosion(vox.matrix, iterations=iters)
    out = trimesh.voxel.VoxelGrid(out_grid, transform=T).marching_cubes
    out.apply_transform(T)
    out.merge_vertices()
    return out

def union(parts):
    parts = [p for p in parts if p is not None and len(p.faces) > 0]
    if not parts: return None
    if len(parts) == 1: return parts[0]
    return trimesh.boolean.union(parts, engine='manifold')

def biggest(m):
    comps = m.split(only_watertight=False)
    return max(comps, key=lambda c: c.volume) if comps else m

# ---- 1. brain pieces -----------------------------------------------------
print('=== loading outer-surface pieces ===')
cortex     = ev(trimesh.load(str(SETS/'isocortex/stl/FINAL_onepiece_voxsheet_arial.stl'), force='mesh'))
cerebellum = ev(biggest(trimesh.load(str(SETS/'cerebellum/stl/cerebellum_engraved_66mm.stl'), force='mesh')))
olfactory  = ev(biggest(trimesh.load(str(SETS/'olfactory/stl/olfactory_engraved_66mm.stl'), force='mesh')))
for label, m in [('cortex', cortex), ('cerebellum', cerebellum), ('olfactory', olfactory)]:
    print(f'  {label:12s}  bbox={[round(x,1) for x in m.bounding_box.extents]}  vol={m.volume:.0f}mm³')

# scale all three so AP extent = MOUSE_LENGTH_MM
all_v = np.vstack([cortex.vertices, cerebellum.vertices, olfactory.vertices])
SCALE = MOUSE_LENGTH_MM / (all_v[:, 0].max() - all_v[:, 0].min())
for m in (cortex, cerebellum, olfactory):
    m.apply_scale(SCALE)
all_v = np.vstack([cortex.vertices, cerebellum.vertices, olfactory.vertices])
brain_min = all_v.min(0); brain_max = all_v.max(0)
print(f'  scaled: scale={SCALE:.3f}  bbox={[round(x,1) for x in brain_max-brain_min]}  '
      f'x range {brain_min[0]:.1f}..{brain_max[0]:.1f}')

# ---- 2. fuse + flat-cut bottom ------------------------------------------
print('\n=== fuse + flat-cut bottom ===')
shell_raw = ev(union([cortex, cerebellum, olfactory]))
cut_y = brain_min[1] + FLAT_BOTTOM_MM
shell_cut = ev(biggest(shell_raw.slice_plane(plane_origin=[0, cut_y, 0],
                                              plane_normal=[0, -1, 0], cap=True)))
print(f'  cut at y={cut_y:.1f}, kept dorsal {FLAT_BOTTOM_MM}mm: vol {shell_cut.volume:.0f}mm³')

# ---- 3. hollow interior ------------------------------------------------
print('\n=== hollow ===')
interior = ev(voxel_offset(shell_cut, -WALL_MM))
interior = interior.slice_plane(plane_origin=[0, cut_y - 0.1, 0],
                                plane_normal=[0, -1, 0], cap=True)
interior = ev(interior)
shell_hollow = ev(shell_cut.difference(interior, engine='manifold'))
print(f'  hollow vol {shell_hollow.volume:.0f}mm³  (interior carved out {interior.volume:.0f}mm³)')

# ---- 4. load kit, split components (chassis vs screws) -----------------
print('\n=== kit components ===')
kit_raw = trimesh.load(str(KIT / '02_3D_model_for_Wireless_Mouse_Kit-002__hardware_components______.stl'), force='mesh')
kit_raw.apply_translation(-kit_raw.bounding_box.centroid)
comps_all = sorted(kit_raw.split(only_watertight=False), key=lambda c: -c.volume)
chassis_comps = [c for c in comps_all if c.volume >= COMP_VOLUME_MIN]
screw_comps   = [c for c in comps_all if c.volume <  COMP_VOLUME_MIN]
print(f'  chassis  components: {len(chassis_comps)} (vol >= {COMP_VOLUME_MIN}mm³)')
print(f'  screw    components: {len(screw_comps)}  (vol <  {COMP_VOLUME_MIN}mm³)')

# Identify the wheel: cylinder-like, ~22mm diameter. Bbox closest to (21,22,22).
def is_wheel(c):
    b = sorted(c.bounding_box.extents)
    return abs(b[0]-21) < 3 and abs(b[1]-22) < 3 and abs(b[2]-22) < 3
wheel_comp = next((c for c in chassis_comps if is_wheel(c)), None)
print(f'  wheel    identified at kit-local centroid '
      f'{[round(x,1) for x in wheel_comp.bounding_box.centroid]}'
      if wheel_comp is not None else '  wheel not auto-identified')

# Concatenate chassis as a single mesh (multi-component is fine for boolean)
kit_chassis = trimesh.util.concatenate(chassis_comps)

# ---- 5. orient + position the kit --------------------------------------
# Kit native frame: x=20 (wheel axis), y=105 (long, PCB length), z=27 (height).
# Brain frame:      x=AP (110), y=DV (45), z=LR (88).
# Map:  brain_x  <-  kit_y          (long axis -> mouse length)
#       brain_y  <- -kit_z + offset (kit up -> brain dorsal = small y)
#       brain_z  <-  kit_x          (wheel axis -> LR)
print('\n=== orient + position kit ===')
# R = axis swap; FLIP = 180° about brain-y so the wheel (at kit +y end) ends up
# at brain -x (= front), and the PCB / battery extend toward brain +x (= back).
R    = np.array([[0, 1, 0, 0], [0, 0,-1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
FLIP = np.array([[-1,0, 0, 0], [0, 1, 0, 0], [0, 0,-1, 0], [0, 0, 0, 1]], dtype=float)
RT   = FLIP @ R
kit_chassis.apply_transform(RT)
if wheel_comp is not None:
    wheel = wheel_comp.copy(); wheel.apply_transform(RT)
else:
    wheel = None

# Position: wheel at midline (z=0), at AP=0.3 from the front of the shell;
# kit ventral surface near the cut plane.
shell_min = shell_hollow.bounds[0]; shell_max = shell_hollow.bounds[1]
WHEEL_FROM_FRONT_FRAC = 0.30                            # 0=rostral pole, 1=caudal pole
WHEEL_POKE_UP_MM      = 3.0                              # wheel top pokes up by this
wheel_radius          = 11.0                             # 22 mm wheel / 2
wheel_target_x = shell_min[0] + WHEEL_FROM_FRONT_FRAC * (shell_max[0] - shell_min[0])
wheel_target_y = brain_min[1] + wheel_radius - WHEEL_POKE_UP_MM   # CENTER inside the brain
target_wheel_pos = np.array([wheel_target_x, wheel_target_y, 0.0])
print(f'  wheel target ({wheel_target_x:.1f}, {wheel_target_y:.1f}, 0.0)  '
      f'(rostral pole x={shell_min[0]:.1f}, caudal pole x={shell_max[0]:.1f})')
if wheel is not None:
    src_wheel_pos = wheel.bounding_box.centroid
    translate_kit = target_wheel_pos - src_wheel_pos
else:
    translate_kit = np.array([
        (shell_min[0] + shell_max[0]) / 2 - kit_chassis.bounding_box.centroid[0],
        cut_y - WALL_MM - kit_chassis.bounds[1][1],
        0.0 - kit_chassis.bounding_box.centroid[2],
    ])

# Allow override from interactive tool
kt_path = OUT / 'kit_transform.json'
if kt_path.exists():
    T_kit = np.asarray(json.loads(kt_path.read_text())['matrix'], float)
    kit_chassis = trimesh.load(str(KIT / '02_3D_model_for_Wireless_Mouse_Kit-002__hardware_components______.stl'), force='mesh')
    kit_chassis.apply_translation(-kit_chassis.bounding_box.centroid)
    kit_chassis = trimesh.util.concatenate(
        [c for c in kit_chassis.split(only_watertight=False) if c.volume >= COMP_VOLUME_MIN])
    kit_chassis.apply_transform(T_kit)
    print(f'  kit placed via {kt_path.name}')
else:
    kit_chassis.apply_translation(translate_kit)
    if wheel is not None: wheel.apply_translation(translate_kit)
    print(f'  default pose: wheel at ({target_wheel_pos[0]:.1f}, midline z=0), '
          f'kit bbox center {[round(x,1) for x in kit_chassis.bounding_box.centroid]}')
print(f'  kit bbox after place: {[round(x,1) for x in kit_chassis.bounding_box.extents]}')

# ---- 6. carve cavity for chassis ---------------------------------------
print('\n=== carve cavity (chassis only) ===')
cavity = ev(voxel_offset(kit_chassis, CAVITY_MARGIN_MM))
shell_with_cavity = ev(shell_hollow.difference(cavity, engine='manifold'))
print(f'  shell after cavity: vol {shell_with_cavity.volume:.0f}mm³ '
      f'(removed {shell_hollow.volume - shell_with_cavity.volume:.0f}mm³ for electronics)')

# ---- 7. body = full brain shell (no cortex pocket — keeps cerebellum) --
# Earlier versions subtracted the cortex cap from the body to make room
# for click buttons, but that also ate the cerebellum (cortex overlaps
# cerebellum geometrically). V3: keep the body as the WHOLE brain shell
# (cortex+cerebellum+olfactory all visible). Cortex L/R hemispheres are
# exported as separate pieces — for V1 they are decorative covers; later
# they become proper hinged buttons.
print('\n=== body = whole brain shell (no cortex pocket) ===')
body = shell_with_cavity
print(f'  body kept whole: vol {body.volume:.0f}mm³')

cortex_cut = ev(biggest(cortex.slice_plane(plane_origin=[0, cut_y, 0],
                                            plane_normal=[0, -1, 0], cap=True)))

def split_half(m, side):
    sign = +1 if side == 'R' else -1
    half = m.slice_plane(plane_origin=[0, 0, 0], plane_normal=[0, 0, sign], cap=True)
    return ev(biggest(half)) if half is not None else None

cortex_L = split_half(cortex_cut, 'L')
cortex_R = split_half(cortex_cut, 'R')
print(f'  cortex_L vol {cortex_L.volume:.0f}mm³  cortex_R vol {cortex_R.volume:.0f}mm³  '
      f'(full hemispheres, for separate printing)')

# ---- 8. wheel slot through the cortex midline --------------------------
# A rectangular cutout that runs LR (z) across the midline so the wheel can
# pop up. Placed at the wheel's xy position; spans full LR width of slot.
print('\n=== wheel slot through cortex midline ===')
if wheel is not None:
    wx = wheel.bounding_box.centroid[0]
else:
    wx = wheel_target_x
slot = trimesh.creation.box(extents=(WHEEL_SLOT_AP, WHEEL_SLOT_DROP, WHEEL_SLOT_W))
# Slot runs from above the cortex top down through it: y_top sits above
# brain_min[1] so the slot definitely breaches the dorsal surface, y_bottom
# reaches a bit past the wheel center so the wheel can poke through cleanly.
slot_y_center = brain_min[1] - 4 + WHEEL_SLOT_DROP / 2
slot.apply_translation([wx, slot_y_center, 0.0])
print(f'  slot bbox {[round(x,1) for x in slot.bounding_box.extents]} '
      f'centered at ({wx:.1f}, {brain_min[1]+WHEEL_SLOT_DROP/2-4:.1f}, 0)')
cortex_L = ev(cortex_L.difference(slot, engine='manifold'))
cortex_R = ev(cortex_R.difference(slot, engine='manifold'))
body     = ev(body.difference(slot, engine='manifold'))
body     = ev(biggest(body))
print(f'  after slot: cortex_L {cortex_L.volume:.0f}  cortex_R {cortex_R.volume:.0f}  body {body.volume:.0f}')

# ---- 9. base plate matching the full SHELL footprint at the cut plane --
# Use the un-carved shell_cut so the plate covers the entire underside,
# including over the cerebellum/olfactory + cortex regions.
print('\n=== base plate ===')
shell_min, shell_max = shell_cut.bounds
base_extents = (shell_max[0]-shell_min[0]+4, BASE_PLATE_MM,
                shell_max[2]-shell_min[2]+4)
base = trimesh.creation.box(extents=base_extents)
base.apply_translation([(shell_min[0]+shell_max[0])/2,
                        cut_y + BASE_PLATE_MM/2 + 0.2,
                        (shell_min[2]+shell_max[2])/2])
print(f'  base plate {base_extents[0]:.1f} x {base_extents[2]:.1f} x {BASE_PLATE_MM} mm '
      f'(matches full shell footprint)')

# ---- 10. write outputs --------------------------------------------------
print('\n=== writing outputs ===')
parts = [('body', body), ('cortex_L', cortex_L), ('cortex_R', cortex_R),
         ('base_plate', base)]
for name, mesh in parts:
    p = OUT / f'{name}.stl'
    mesh.export(str(p))
    print(f'  {p.relative_to(DATA.parent)}  faces={len(mesh.faces):,}  '
          f'vol={mesh.volume:.0f}mm³  watertight={mesh.is_watertight}')

(OUT / 'manifest.json').write_text(json.dumps({
    'mouse_length_mm': MOUSE_LENGTH_MM,
    'flat_bottom_mm':  FLAT_BOTTOM_MM,
    'wall_mm':         WALL_MM,
    'cortex_cap_mm':   CORTEX_CAP_MM,
    'wheel_slot':      [WHEEL_SLOT_AP, WHEEL_SLOT_DROP, WHEEL_SLOT_W],
    'scale':           SCALE,
    'cut_plane_y':     cut_y,
    'wheel_xy_brain':  [float(wx), 0.0],
    'chassis_components_used': len(chassis_comps),
    'screw_components_dropped': len(screw_comps),
}, indent=2))
print(f'  manifest.json')

# ---- 11. preview render -------------------------------------------------
print('\n=== preview render ===')
import pyvista as pv
def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

# pyvista view_vector(vec, up): camera at (focal + vec), looking toward
# focal. So `vec` points FROM focal TO camera.  In CCF: -y=dorsal/top.
# Dorsal view  -> camera above brain (small y) -> vec = (0, -1, 0).
# Ventral view -> camera below brain (large y) -> vec = (0, +1, 0).
# Lateral R    -> camera at +z (right side)    -> vec = (0, 0, +1).
# Front view   -> camera at -x (rostral pole)  -> vec = (-1, 0, 0).
views = [('dorsal (above)',     ( 0, -1,  0), (-1,  0,  0), 0),
         ('lateral right',      ( 0,  0,  1), ( 0, -1,  0), 0),
         ('front (rostral)',    (-1,  0,  0), ( 0, -1,  0), 0),
         ('exploded - lateral', ( 0,  0,  1), ( 0, -1,  0), 22)]
pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1900, 1500))
for i, (name, view, up, lift) in enumerate(views):
    pl.subplot(i // 2, i % 2)
    pl.add_mesh(poly(body), color=(0.78, 0.78, 0.82),
                smooth_shading=True, specular=0.25, ambient=0.3, diffuse=0.85)
    cL = cortex_L.copy(); cL.apply_translation([0, -lift, 0])
    cR = cortex_R.copy(); cR.apply_translation([0, -lift, 0])
    bb_low = base.copy();  bb_low.apply_translation([0, +lift, 0])
    pl.add_mesh(poly(cL),     color=(0.85, 0.40, 0.45), smooth_shading=True,
                specular=0.3, ambient=0.35, diffuse=0.85)
    pl.add_mesh(poly(cR),     color=(0.45, 0.55, 0.85), smooth_shading=True,
                specular=0.3, ambient=0.35, diffuse=0.85)
    pl.add_mesh(poly(bb_low), color=(0.45, 0.85, 0.55), smooth_shading=True,
                specular=0.3, ambient=0.35, diffuse=0.85)
    pl.add_mesh(poly(kit_chassis), color=(0.95, 0.85, 0.55), smooth_shading=True,
                opacity=0.95, specular=0.4, ambient=0.4, diffuse=0.8)
    pl.add_text(name, font_size=10, color='black', position='upper_left')
    pl.background_color = 'white'
    pl.view_vector(view, up); pl.reset_camera()
png = OUT / 'preview.png'
pl.screenshot(str(png))
print(f'  {png.relative_to(DATA.parent)}')
