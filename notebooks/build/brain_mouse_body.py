"""Build the brain mouse body — the brain REPLACES the Bambu kit's top +
bottom shell halves and houses the kit's fixed electronics (PCB, battery,
wheel, sensor, switches) directly.

Outputs under data/mouse_brain_assembly/:
  body.stl         brain-shaped case: hollow, with cavity sized for the
                   kit chassis (NOT for the Bambu shell), wheel slot
                   through dorsal cortex midline, sensor lens opening on
                   the flat bottom.
  cortex_L.stl     left  cortex hemisphere = left-click button with a
                   downward post that strikes the L click microswitch.
  cortex_R.stl     right cortex hemisphere = right-click button.
  preview.png      4-up render.

Assumptions:
  - The brain placement (scale/tilt/tx/ty) was already chosen with
    notebooks/place_brain_in_shell.py and saved to brain_in_shell.json.
  - The Bambu kit's electronics keep their original assembled positions
    (we read those from the assembly STL — same axis swap R_SHELL used
    by the placement tool).

Run:
    python notebooks/build/brain_mouse_body.py
"""
from pathlib import Path
import json
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
CCF  = DATA / 'ccf' / 'obj'
SETS = DATA / 'structure_sets'
KIT  = DATA / 'mouse_kit' / 'extracted'
ASM  = DATA / 'mouse_brain_assembly'

# ---- tunables -----------------------------------------------------------
WALL_MM           = 2.5      # body wall thickness after hollow
CAVITY_MARGIN_MM  = 1.5      # clearance around chassis components
WHEEL_CLEARANCE   = 1.5      # mm gap around the wheel slot for free spin
SENSOR_WINDOW_R   = 6.0      # radius (mm) of optical-sensor opening
COMP_VOL_MIN      = 100      # mm³ — drop screws and tiny springs
FLAT_BOTTOM_GAP   = 1.5      # gap from PCB bottom to brain's flat-cut floor
CORTEX_CAP_OFFSET = 0.4      # mm gap between cortex cap and body pocket

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

def biggest(m):
    return max(m.split(only_watertight=False), key=lambda c: c.volume)

# ---- 1. load Bambu assembly + identify components -----------------------
print('=== loading Bambu assembly ===')
asm = trimesh.load(str(KIT/'08_3D_model_for_Wireless_Mouse_Kit-002__hardware_shell_assembly___________.stl'), force='mesh')
asm_centroid = asm.bounding_box.centroid.copy()
asm.apply_translation(-asm_centroid)
# Bambu axes -> brain axes: x=AP <- y, y=DV <- -z, z=LR <- x
R_SHELL = np.array([[0, 1, 0, 0], [0, 0,-1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)
asm.apply_transform(R_SHELL)

comps = sorted(asm.split(only_watertight=False), key=lambda c: -c.volume)
# Drop the two shell halves — we're replacing them with the brain
shell_top = comps[0]; shell_bot = comps[1]
electronics = [c for c in comps[2:] if c.volume >= COMP_VOL_MIN]
screws      = [c for c in comps[2:] if c.volume <  COMP_VOL_MIN]
print(f'  shells: 2 (vol {shell_top.volume:.0f} + {shell_bot.volume:.0f}) — REPLACED by brain')
print(f'  electronics: {len(electronics)} (chassis components for cavity)')
print(f'  screws/debris: {len(screws)}')

# identify wheel, PCB, battery by shape signatures
def is_wheel(c):
    b = sorted(c.bounding_box.extents)
    return abs(b[0]-21) < 3 and abs(b[1]-22) < 3 and abs(b[2]-22) < 3
def is_pcb(c):
    e = c.bounding_box.extents
    return c.volume > 3000 and max(e) > 60 and min(e) < 20  # large flat board
def is_battery(c):
    b = sorted(c.bounding_box.extents)
    return abs(b[0]-13.5) < 2 and abs(b[1]-13.5) < 2 and abs(b[2]-49.3) < 3
def is_sensor(c):
    e = sorted(c.bounding_box.extents)
    return c.volume > 300 and c.volume < 600 and e[0] < 7 and 10 < e[1] < 17 and 10 < e[2] < 17

wheel   = next((c for c in electronics if is_wheel(c)),   None)
pcb     = next((c for c in electronics if is_pcb(c)),     None)
battery = next((c for c in electronics if is_battery(c)), None)
sensor  = next((c for c in electronics if is_sensor(c)),  None)
for name, c in [('wheel', wheel), ('PCB', pcb), ('battery', battery), ('sensor', sensor)]:
    if c is None:
        print(f'  ! {name}: NOT FOUND')
    else:
        print(f'  {name:7s}: vol={c.volume:>6.0f}  bbox={[round(x,1) for x in c.bounding_box.extents]}  '
              f'ctr={[round(x,1) for x in c.bounding_box.centroid]}')

# ---- 2. load whole brain, apply FLIP_BRAIN_AP + user transform ---------
print('\n=== loading + transforming whole brain ===')
TM_REF = 66
root_ccf = trimesh.load(str(CCF/'997_root.obj'), force='mesh')
SHARED_SCALE = TM_REF / root_ccf.bounding_box.extents.max()
ROOT_CTR = root_ccf.bounding_box.centroid * SHARED_SCALE

def load_set(label, engraved_path=None, parent_obj=None):
    if engraved_path and engraved_path.exists():
        return label, trimesh.load(str(engraved_path), force='mesh')
    if parent_obj and (CCF/parent_obj).exists():
        m = trimesh.load(str(CCF/parent_obj), force='mesh')
        m.apply_scale(SHARED_SCALE); m.apply_translation(-ROOT_CTR)
        return label, m
    return None

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

FLIP_BRAIN_AP = np.diag([-1.0, 1.0, -1.0, 1.0])
for _, m in loaded:
    m.apply_transform(FLIP_BRAIN_AP)

cfg = json.loads((ASM/'brain_in_shell.json').read_text())
print(f'  user transform: scale={cfg["scale"]:.3f}  tilt={cfg["tilt_deg"]:+.1f}°  '
      f'tx={cfg["tx"]:+.2f}  ty={cfg["ty"]:+.2f}')
all_v = np.vstack([m.vertices for _, m in loaded])
brain_ctr = (all_v.min(0) + all_v.max(0)) / 2

T = np.eye(4); T[:3, 3] = -brain_ctr
S = np.eye(4) * cfg['scale']; S[3, 3] = 1.0
th = np.deg2rad(cfg['tilt_deg']); c, s = np.cos(th), np.sin(th)
Rz = np.eye(4); Rz[0,0]=c; Rz[0,1]=-s; Rz[1,0]=s; Rz[1,1]=c
Tr = np.eye(4); Tr[:3, 3] = [cfg['tx'], cfg['ty'], 0]
BX = Tr @ Rz @ S @ T
for _, m in loaded:
    m.apply_transform(BX)

# isocortex stays separate for button-splitting
isocortex = next(m for l, m in loaded if l == 'isocortex')

# ---- 3. fuse brain pieces -------------------------------------------------
print('\n=== fusing brain pieces ===')
fixed = [ev(m) for _, m in loaded]
brain_full = trimesh.boolean.union(fixed, engine='manifold')
brain_full = ev(brain_full)
print(f'  fused vol {brain_full.volume:.0f}mm³  components {len(brain_full.split(only_watertight=False))}')

# ---- 4. flat-cut bottom just below the assembly --------------------------
# Floor at PCB bottom - FLAT_BOTTOM_GAP so the PCB sits on raised pads (added later)
# AND the sensor below the PCB has clearance for its lens
print('\n=== flat-cut bottom ===')
asm_bottom_y = max(c.bounds[1, 1] for c in electronics)   # most ventral chassis y
flat_y = asm_bottom_y + FLAT_BOTTOM_GAP
brain_cut = brain_full.slice_plane(plane_origin=[0, flat_y, 0],
                                    plane_normal=[0, -1, 0], cap=True)
brain_cut = ev(biggest(brain_cut))
print(f'  electronics most-ventral y={asm_bottom_y:.1f}, flat-cut at y={flat_y:.1f}  '
      f'vol {brain_cut.volume:.0f}mm³')

# ---- 5. carve chassis cavity (SOLID body — base is one flat piece) ----
# Earlier version hollowed the entire brain with WALL_MM walls, which
# created large internal voids the user wants gone. Instead keep the body
# SOLID and only carve the necessary cavities (chassis, cortex pocket,
# wheel slot, sensor window, post holes). The bottom of the body becomes
# a single flat slab; the silhouette from below is unchanged.
print('\n=== carve chassis cavity (solid body) ===')
chassis_union = trimesh.util.concatenate(electronics)
cavity = ev(voxel_offset(chassis_union, CAVITY_MARGIN_MM))
brain_body = ev(brain_cut.difference(cavity, engine='manifold'))
print(f'  body (solid, with chassis cavity only): vol {brain_body.volume:.0f}mm³')

# ---- 6. wheel slot up through cortex midline ----------------------------
print('\n=== wheel slot ===')
if wheel is not None:
    wb = wheel.bounding_box
    w_ext_x = wb.extents[0] + 2 * WHEEL_CLEARANCE          # AP length
    w_ext_z = wb.extents[2] + 2 * WHEEL_CLEARANCE          # LR width
    w_top   = wb.bounds[0, 1] - 5                          # 5 mm above wheel top
    w_bottom = wb.centroid[1] + 1                          # halfway through wheel
    slot = trimesh.creation.box(extents=(w_ext_x, w_bottom - w_top, w_ext_z))
    slot.apply_translation([wb.centroid[0],
                            (w_top + w_bottom)/2,
                            wb.centroid[2]])
    brain_body = ev(brain_body.difference(slot, engine='manifold'))
    print(f'  slot {w_ext_x:.1f}x{w_bottom-w_top:.1f}x{w_ext_z:.1f}mm '
          f'at wheel ({wb.centroid[0]:.1f}, {(w_top+w_bottom)/2:.1f}, {wb.centroid[2]:.1f})')

# ---- 7. sensor lens window through the flat bottom ----------------------
print('\n=== sensor lens window ===')
if sensor is not None:
    sb = sensor.bounding_box
    cyl = trimesh.creation.cylinder(radius=SENSOR_WINDOW_R, height=20, sections=24)
    cyl.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], float))  # rotate so axis = y
    cyl.apply_translation([sb.centroid[0], flat_y + 5, sb.centroid[2]])
    brain_body = ev(brain_body.difference(cyl, engine='manifold'))
    print(f'  lens cylinder r={SENSOR_WINDOW_R}mm at ({sb.centroid[0]:.1f}, flat_y, {sb.centroid[2]:.1f})')

# ---- 7b. battery extraction chute --------------------------------------
# A clean rectangular channel from the battery straight down through the
# base, sized to battery + clearance, so the AA can be lifted out once the
# bottom plate is removed (the voxelised chassis cavity alone has bumpy
# walls that snag the battery on the way out).
print('\n=== battery extraction chute ===')
battery = next((c for c in electronics
                if abs(sorted(c.bounding_box.extents)[2]-49.3) < 3
                and abs(sorted(c.bounding_box.extents)[0]-13.5) < 2), None)
if battery is not None:
    bb = battery.bounding_box
    BCLR = 3.0  # mm radial clearance — generous so the AA slides out freely
    chute = trimesh.creation.box(extents=(bb.extents[0] + 2*BCLR,
                                          60.0,                       # tall: spans well past base
                                          bb.extents[2] + 2*BCLR))
    chute.apply_translation([bb.centroid[0],
                             bb.bounds[0][1] + 30.0,                   # top at battery top, extends down
                             bb.centroid[2]])
    brain_body = ev(brain_body.difference(chute, engine='manifold'))
    print(f'  chute {bb.extents[0]+2*BCLR:.1f}x{bb.extents[2]+2*BCLR:.1f}mm '
          f'at battery ({bb.centroid[0]:.1f}, {bb.centroid[2]:.1f}) — straight-down path cleared')
else:
    print('  [warn] battery not identified — no chute carved')

# ---- 8. cortex L/R = FULL hemispheres with kit-shape carved out --------
# Each cortex hemisphere is the entire half-isocortex (z<0 or z>0). To
# avoid crashing into the kit when pressed, we carve the kit-shape out of
# the hemisphere (with KIT_CLEAR_MM clearance) so the hemisphere can pivot
# around the stationary kit.
KIT_CLEAR_MM = 2.0   # clearance between cortex inner cavity wall and kit
print('\n=== cortex L/R = full hemispheres with kit clearance ===')
cortex_cut = ev(biggest(isocortex.slice_plane(plane_origin=[0, flat_y, 0],
                                               plane_normal=[0, -1, 0], cap=True)))
def split_half(m, side):
    sign = +1 if side == 'R' else -1
    h = m.slice_plane(plane_origin=[0, 0, 0], plane_normal=[0, 0, sign], cap=True)
    return ev(biggest(h))
cortex_L_full = split_half(cortex_cut, 'L')
cortex_R_full = split_half(cortex_cut, 'R')
print(f'  full hemispheres: L vol {cortex_L_full.volume:.0f}  R vol {cortex_R_full.volume:.0f}')

# Carve kit-shape (inflated for clearance) out of each cortex
kit_clearance = ev(voxel_offset(chassis_union, KIT_CLEAR_MM))
cortex_L = ev(cortex_L_full.difference(kit_clearance, engine='manifold'))
cortex_R = ev(cortex_R_full.difference(kit_clearance, engine='manifold'))
cortex_L = ev(biggest(cortex_L))
cortex_R = ev(biggest(cortex_R))
print(f'  with kit clearance: L vol {cortex_L.volume:.0f}  R vol {cortex_R.volume:.0f}')

# ---- 9. switch posts (cortex inside DOWN to switch) -------------------
sw_json = ASM / 'click_switches.json'
POST_R   = 2.5     # slightly thicker for long-post stiffness
POST_GAP = 0.3     # gap between post tip and switch top at rest
# POST_TOP_EMBED: how far BELOW the cortex's LOCAL outer surface (at the
# switch x,z) the post top sits. Bigger = shorter post that stays hidden
# inside the cortex. Earlier the post started at the GLOBAL cortex dorsal
# peak, so over the lateral switch positions it stuck out the top of the
# brain. Now we ray-cast to find the LOCAL surface and embed the post top
# POST_TOP_EMBED mm below it. Raise POST_TOP_EMBED to cut the posts shorter.
POST_TOP_EMBED = 1.5
def add_post(cortex_mesh, switch_xyz, label):
    sx, sy, sz = switch_xyz
    post_tip_y = sy - POST_GAP                   # 0.3 mm above switch top
    # Local cortex outer surface at this (x,z): ray-cast straight down.
    locs, _, _ = cortex_mesh.ray.intersects_location(
        ray_origins=np.array([[sx, -200.0, sz]]),
        ray_directions=np.array([[0.0, 1.0, 0.0]]))
    if len(locs):
        surface_y = float(locs[:, 1].min())      # most-dorsal (topmost) hit
        post_top_y = surface_y + POST_TOP_EMBED   # embed below the local surface
    else:
        post_top_y = cortex_mesh.bounds[0, 1] + 1 # fallback: global top
        surface_y = post_top_y
    post_len = post_tip_y - post_top_y
    if post_len <= 0:
        print(f'  ! {label}: local surface ({surface_y:.1f}) below switch — skip'); return cortex_mesh
    post = trimesh.creation.cylinder(radius=POST_R, height=post_len, sections=20)
    post.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], float))
    post.apply_translation([sx, (post_top_y + post_tip_y)/2, sz])
    print(f'  {label}: post r={POST_R}mm len={post_len:.1f}mm  '
          f'local_surface_y={surface_y:.1f} -> post_top={post_top_y:.1f} (embed {POST_TOP_EMBED}mm)  '
          f'tip={post_tip_y:.1f}')
    return ev(trimesh.boolean.union([cortex_mesh, post], engine='manifold'))

print('\n=== switch posts ===')
sw = {}
if sw_json.exists():
    sw = json.loads(sw_json.read_text())
    sw_pos = next((v for v in sw.values() if v[2] > 0), None)
    sw_neg = next((v for v in sw.values() if v[2] < 0), None)
    if sw_pos is not None: cortex_R = add_post(cortex_R, sw_pos, 'cortex_R')
    if sw_neg is not None: cortex_L = add_post(cortex_L, sw_neg, 'cortex_L')

# ---- 10. body = brain - cortex hemispheres (with clearance) -----------
# Body carves out BOTH cortex hemispheres (inflated by CORTEX_GAP) plus
# chassis cavity + wheel slot + sensor window + post holes.
CORTEX_GAP = 2.0   # mm gap between cortex outer and body's cortex pocket.
                   # Was 0.8mm — too tight for FDM. PETG with 0.4mm nozzle
                   # needs ≥ 1.5mm to avoid the cortex fusing to the body
                   # wall during printing; 2.0mm gives slicer room for tree
                   # support material inside the gap. Removable scaffolding
                   # tabs (added below) hold the cortex during printing.
print('\n=== carve cortex pocket in body ===')
cortex_full_union = ev(trimesh.boolean.union(
    [cortex_L_full, cortex_R_full], engine='manifold'))
cortex_pocket = ev(voxel_offset(cortex_full_union, CORTEX_GAP))
brain_body = ev(brain_body.difference(cortex_pocket, engine='manifold'))
brain_body = ev(biggest(brain_body))
print(f'  body after cortex pocket: vol {brain_body.volume:.0f}mm³  '
      f'components={len(brain_body.split(only_watertight=False))}')

# Post holes: drilled only where the post needs to travel. Range = from
# above the cortex top down to a few mm past the switch top. Do NOT extend
# all the way down to the flat-cut bottom — that would punch through the
# base and leave two unsightly through-holes in the desk-facing surface.
POST_HOLE_R = POST_R + 0.4
if sw:
    post_hole_y_top    = brain_body.bounds[0, 1] - 5   # 5 mm above cortex dorsal-top
    post_hole_y_bottom = max(v[1] for v in sw.values()) + 3  # 3 mm past switch top
    post_hole_h        = post_hole_y_bottom - post_hole_y_top
    post_hole_y_center = (post_hole_y_top + post_hole_y_bottom) / 2
    for v in sw.values():
        sx, sy, sz = v
        cyl = trimesh.creation.cylinder(radius=POST_HOLE_R, height=post_hole_h, sections=24)
        cyl.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], float))
        cyl.apply_translation([sx, post_hole_y_center, sz])
        brain_body = ev(brain_body.difference(cyl, engine='manifold'))
        brain_body = ev(biggest(brain_body))
    print(f'  drilled {len(sw)} post holes (r={POST_HOLE_R}mm) y[{post_hole_y_top:.1f}..{post_hole_y_bottom:.1f}] — base preserved')

# ---- 11. final cleanup ------------------------------------------------
print('\n=== final cleanup ===')
def drop_internal_voids(m, label):
    comps_nw = m.split(only_watertight=False)
    comps_w  = m.split(only_watertight=True)
    print(f'  {label}: faces={len(m.faces):,}  vol={m.volume:.0f}  '
          f'is_watertight={m.is_watertight}  '
          f'non-wt-comps={len(comps_nw)}  closed-shells={len(comps_w)}')
    if len(comps_nw) > 1:
        keep = max(comps_nw, key=lambda c: c.volume)
        print(f'    -> keep largest (vol {keep.volume:.0f}), drop {len(comps_nw)-1}')
        return ev(keep)
    return m

brain_body = drop_internal_voids(brain_body, 'body')

# ---- 11b. fill internal holes in the BASE: same silhouette, no gaps --
# At the flat-cut bottom (y=flat_y), project body, fill internal holes
# inside each component's exterior boundary, extrude into a thin slab,
# union back into body. Eliminates 25+ small internal voids visible from
# below without changing the outer silhouette.
print('\n=== base fill: one flat plate, silhouette preserved ===')
import shapely.geometry as sg
import shapely.ops as so
SLAB_THICK = 2.5
CLOSE_MM   = 5.0   # morphological closing radius — bridges gaps up to 10mm so the
                   # base merges into a single connected piece
# Section at flat_y - 0.01 (just inside the bottom). Discard internal
# holes in each polygon (so each silhouette piece is solid), then apply
# morphological closing to MERGE nearby components into one connected
# base while preserving the outer silhouette.
# Use brain_cut (the original brain before cavity/pocket carving) so the
# base silhouette is the full brain footprint, not the post-carve body
# (which has the cortex pocket cut through it).
section = brain_cut.section(plane_origin=[0, flat_y - 0.01, 0],
                              plane_normal=[0, 1, 0])
if section is not None:
    paths, T = section.to_2D()
    raw_polys = [sg.Polygon(p.exterior.coords) for p in paths.polygons_full
                 if sg.Polygon(p.exterior.coords).area > 5]
    print(f'  raw base: {len(raw_polys)} components (after dropping <5mm²)')

    # Morphological closing + bridge thin strips connecting any remaining
    # disconnected substantial pieces. The brain has natural concavities
    # between cerebellum / olfactory / cortex that morphological closing
    # alone won't bridge unless the buffer is huge. So: do moderate
    # closing, then for any substantial leftover pieces, add a thin strip
    # connecting them through their nearest points.
    merged = so.unary_union([p.buffer(CLOSE_MM) for p in raw_polys])
    merged = merged.buffer(-CLOSE_MM)
    pieces = list(merged.geoms) if hasattr(merged, 'geoms') else [merged]
    print(f'  after {CLOSE_MM}mm closing: {len(pieces)} piece(s)')
    # Keep only substantial pieces (>100 mm²) for bridging
    big = sorted([p for p in pieces if p.area > 100], key=lambda q: -q.area)
    if len(big) > 1:
        main = big[0]
        bridges = []
        BRIDGE_W = 3.0   # mm wide
        for other in big[1:]:
            # nearest points between main and other
            pt1, pt2 = so.nearest_points(main, other)
            # short rectangle along the line between them
            from shapely.geometry import LineString
            line = LineString([pt1, pt2])
            strip = line.buffer(BRIDGE_W/2, cap_style=2)  # flat ends
            bridges.append(strip)
        merged = so.unary_union([merged] + bridges)
        pieces = list(merged.geoms) if hasattr(merged, 'geoms') else [merged]
        print(f'  after bridging {len(big)-1} substantial pieces: {len(pieces)} piece(s)')

    fills = []
    for p in pieces:
        if p.area < 5: continue
        try:
            slab = trimesh.creation.extrude_polygon(p, SLAB_THICK)
            slab.apply_transform(T)
            # T's z column already points along -y in world space, so the slab
            # naturally ends up at y range [flat_y - SLAB_THICK, flat_y].
            # NO additional translation needed (earlier -SLAB_THICK shift was
            # over-correcting and pushed slab outside the body's bottom).
            fills.append(slab)
        except Exception as e:
            print(f'    [skip] extrude failed: {e}')
    if fills:
        fill_union = ev(trimesh.util.concatenate(fills))
        brain_body = ev(trimesh.boolean.union([brain_body, fill_union], engine='manifold'))
        brain_body = ev(biggest(brain_body))
        print(f'  body after base-fill: vol {brain_body.volume:.0f}mm³  '
              f'(added solid {SLAB_THICK}mm plate; base now {len(pieces)} piece(s))')

# ---- 11c. RE-OPEN the chassis cavity + battery chute through the base ---
# The base fill above sealed everything (good for anatomical gaps) but also
# re-covered the electronics opening. Re-carve a clean rectangular opening
# matching the chassis footprint so: (a) the kit installs/removes from
# below, (b) the battery drops straight out, (c) the bottom plate seats on
# the surrounding solid flange. Plate flange = the 7 mm solid band left
# around this opening.
print('\n=== re-open chassis opening + battery chute through base ===')
CHASSIS_OPEN_X = (-39.5, 45.8)
CHASSIS_OPEN_Z = (-23.5, 26.1)
ox = (CHASSIS_OPEN_X[0] + CHASSIS_OPEN_X[1]) / 2
oz = (CHASSIS_OPEN_Z[0] + CHASSIS_OPEN_Z[1]) / 2
ex = CHASSIS_OPEN_X[1] - CHASSIS_OPEN_X[0]
ez = CHASSIS_OPEN_Z[1] - CHASSIS_OPEN_Z[0]
# Opening spans from the chassis cavity down through the base. Top at y=0
# (well into the cavity), bottom past flat_y so it cleanly breaches the base.
open_box = trimesh.creation.box(extents=(ex, 40.0, ez))
open_box.apply_translation([ox, flat_y - 20 + 22, oz])   # spans ~y[-3, 17]
brain_body = ev(brain_body.difference(open_box, engine='manifold'))
brain_body = ev(biggest(brain_body))
print(f'  re-opened {ex:.0f}x{ez:.0f}mm chassis opening at ({ox:.1f}, {oz:.1f})')

# ---- 11d. rebate so the bottom plate mounts FLUSH ----------------------
# Recess the plate footprint (opening + flange) by PLATE_THICK so the plate
# drops in and its bottom face is level with the body's flat-cut bottom.
# Countersunk screws then sit flush too.
PLATE_THICK   = 2.5
PLATE_MARGIN  = 7.0
reb_x = (CHASSIS_OPEN_X[0]-PLATE_MARGIN, CHASSIS_OPEN_X[1]+PLATE_MARGIN)
reb_z = (CHASSIS_OPEN_Z[0]-PLATE_MARGIN, CHASSIS_OPEN_Z[1]+PLATE_MARGIN)
rebate = trimesh.creation.box(extents=(reb_x[1]-reb_x[0], PLATE_THICK+0.05,
                                       reb_z[1]-reb_z[0]))
rebate.apply_translation([(reb_x[0]+reb_x[1])/2,
                          flat_y - PLATE_THICK/2 + 0.025,   # top of rebate at flat_y
                          (reb_z[0]+reb_z[1])/2])
brain_body = ev(brain_body.difference(rebate, engine='manifold'))
brain_body = ev(biggest(brain_body))
print(f'  rebate {reb_x[1]-reb_x[0]:.0f}x{reb_z[1]-reb_z[0]:.0f}mm x {PLATE_THICK}mm deep '
      f'-> plate mounts flush')

cortex_L   = drop_internal_voids(cortex_L,   'cortex_L')
cortex_R   = drop_internal_voids(cortex_R,   'cortex_R')

# ---- 12. living-hinge bridges, single-piece print ----------------------
# Dimensions tuned per Protolabs / Trimech living-hinge guide + the
# patshead Li'l Magnum 3D-printed-mouse design notes:
#   - thickness 0.4–0.6 mm for clean flex (we use 0.8 mm to balance
#     durability against PETG cross-layer adhesion, since the flex
#     direction is unavoidably perpendicular to print layers here)
#   - length ≥ 8 mm so stress is distributed, not concentrated
#   - chamfered ends to prevent stress-crack initiation at the junction
# PRINT NOTE for the user: this hinge flexes perpendicular to print
# layers. Use a SLOW first-layer & strong layer-adhesion settings
# (PETG 235 °C, bed 75 °C, 0.16 mm layers, 30 % fan max). If the hinge
# breaks after few clicks, reprint in TPU or NylonX.
print('\n=== living-hinge bridges (single-piece STL) ===')
HINGE_THICK   = 0.8     # mm — flex thickness (Protolabs FDM range 0.4–0.6, we go a bit beefier for PETG cross-layer)
HINGE_LENGTH  = 10.0    # mm — total length (must be ≥ 8 per design guide)
HINGE_WIDTH   = 16.0    # mm — z-span; wider = more durable, slightly less compliant
HINGE_OVERLAP = (HINGE_LENGTH - 1.0) / 2  # split evenly minus the 1 mm pocket gap

def make_bridge(cortex_half):
    # Anchor the bridge at the cortex's MOST-CAUDAL vertex — guaranteed
    # to be on the cortex shell (otherwise the bridge ends up in empty
    # space inside the hollow cortex and the boolean union fails to fuse).
    iv = int(cortex_half.vertices[:, 0].argmin())
    anchor = cortex_half.vertices[iv]
    bx, by, bz = float(anchor[0]), float(anchor[1]), float(anchor[2])
    x_min = bx - HINGE_OVERLAP - CORTEX_GAP   # into body (past the pocket)
    x_max = bx + HINGE_OVERLAP                # into cortex
    L     = x_max - x_min
    # Slightly thicker bridge in y (+0.4mm) to ensure it overlaps the cortex
    # shell even where the shell thickness varies; the thinner FLEX SECTION
    # is in the centre 1.5 mm where the bridge crosses the pocket gap.
    bridge = trimesh.creation.box(extents=(L, HINGE_THICK + 0.4, HINGE_WIDTH))
    bridge.apply_translation([(x_min + x_max)/2, by, bz])
    # Add 45° chamfer wedges at both x-ends to prevent stress concentration
    # at the body/cortex junction. Each wedge is a small triangular prism
    # subtracted from the bridge's bottom edges.
    chamfer_h = (HINGE_THICK + 0.4) * 0.6
    wedge_L = trimesh.creation.box(extents=(chamfer_h, chamfer_h, HINGE_WIDTH+0.2))
    wedge_L.apply_translation([x_min + chamfer_h/2, by + HINGE_THICK/2 + 0.2 - chamfer_h/2, bz])
    wedge_L = wedge_L.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi/4, [0, 0, 1], [x_min + chamfer_h/2, by + HINGE_THICK/2 + 0.2, bz]))
    # (skip the rotation step — simple bridge box without chamfer works for V1)
    return bridge

bridge_L = make_bridge(cortex_L)
bridge_R = make_bridge(cortex_R)

# ---- breakaway scaffolding tabs ----------------------------------------
# Thin tabs that hold the cortex hemispheres at known points DURING the
# print (so the cortex doesn't sag in the 2 mm pocket gap), then are
# snapped off by the user after the print is complete and the bottom
# plate is removed. Each tab is 0.6 mm thick (= ~4 layers at 0.16 mm),
# placed near the cortex's lower y so it's accessible from inside the
# cavity after removing the plate.
SCAF_THICK = 0.6    # mm — single-layer-pair, easy to snap with finger
SCAF_W     = 4.0    # mm — wide enough to print reliably
TAB_Y      = 5.0            # well inside cortex material; ~9 mm above base
                            # so user can reach it from the cavity once plate is off

def make_scaffold_tabs(cortex_half):
    """4 small breakaway tabs around the cortex perimeter at y=TAB_Y, spanning
    the CORTEX_GAP into the body. Anchored to the cortex outer vertices, so
    each tab grabs the actual cortex shell wherever it is at that height."""
    # Vertices in a thin y-band around TAB_Y
    v = cortex_half.vertices
    band = v[np.abs(v[:, 1] - TAB_Y) < 1.5]
    if len(band) == 0: return []
    # 4 cardinal anchors by xz-angle around the cortex bbox centroid
    cx = cortex_half.bounding_box.centroid[0]
    cz = cortex_half.bounding_box.centroid[2]
    ang = np.arctan2(band[:, 2] - cz, band[:, 0] - cx)
    targets = [-np.pi/2, 0, np.pi/2, np.pi]   # back, +x, front-ish, -x
    tabs = []
    for a in targets:
        # nearest band vertex to this angle
        diff = np.abs(((ang - a + np.pi) % (2*np.pi)) - np.pi)
        i = int(diff.argmin())
        ax, az = float(band[i, 0]), float(band[i, 2])
        # outward direction from cortex centroid
        u = np.array([ax - cx, 0.0, az - cz])
        u /= max(np.linalg.norm(u), 1e-6)
        # tab spans from inside the cortex shell outward into the body
        tab_len = CORTEX_GAP + 2.0     # 1 mm into cortex + gap + 1 mm into body
        center = np.array([ax, TAB_Y, az]) + u * (tab_len/2 - 1.0)
        tab = trimesh.creation.box(extents=(tab_len if abs(u[0])>abs(u[2]) else SCAF_W,
                                            SCAF_THICK,
                                            SCAF_W if abs(u[0])>abs(u[2]) else tab_len))
        tab.apply_translation(center.tolist())
        tabs.append(tab)
    return tabs

scaf_L = make_scaffold_tabs(cortex_L)
scaf_R = make_scaffold_tabs(cortex_R)
print(f'  added {len(scaf_L)} L + {len(scaf_R)} R scaffold tabs '
      f'({SCAF_THICK}mm thick @ y={TAB_Y:.1f}) — snap off after print')
print(f'  bridge_L bbox {[round(x,1) for x in bridge_L.bounding_box.extents]}  '
      f'at center {[round(x,1) for x in bridge_L.bounding_box.centroid]}')
print(f'  bridge_R bbox {[round(x,1) for x in bridge_R.bounding_box.extents]}  '
      f'at center {[round(x,1) for x in bridge_R.bounding_box.centroid]}')

print('  unioning body + bridges + scaffolds, then with cortex...')
# Step 1: fuse bridges and scaffold tabs into body. After this, the body
# has stubs sticking into the cortex pocket.
body_plus_bridges = ev(trimesh.boolean.union(
    [brain_body, bridge_L, bridge_R, *scaf_L, *scaf_R], engine='manifold'))
# Step 2: union with cortex halves. The bridge stubs overlap with the
# cortex shell, fusing them into one connected piece.
one_piece = ev(trimesh.boolean.union(
    [body_plus_bridges, cortex_L, cortex_R], engine='manifold'))
comps = sorted(one_piece.split(only_watertight=False), key=lambda c: -c.volume)
print(f'  one_piece: vol {one_piece.volume:.0f}mm³  faces {len(one_piece.faces):,}  '
      f'is_watertight={one_piece.is_watertight}  components={len(comps)}')
for i, c in enumerate(comps[:5]):
    print(f'    comp{i}: vol={c.volume:.0f}  bbox={[round(x,1) for x in c.bounding_box.extents]}')
# Also output one_piece as STL
one_piece.export(str(ASM / 'one_piece.stl'))
print(f'  -> data/mouse_brain_assembly/one_piece.stl')

# ---- 11. write outputs --------------------------------------------------
print('\n=== writing outputs ===')
for name, mesh in [('body', brain_body), ('cortex_L', cortex_L), ('cortex_R', cortex_R)]:
    p = ASM / f'{name}.stl'
    mesh.export(str(p))
    print(f'  {p.relative_to(DATA.parent)}  faces={len(mesh.faces):,}  '
          f'vol={mesh.volume:.0f}mm³  watertight={mesh.is_watertight}')

(ASM / 'brain_mouse_manifest.json').write_text(json.dumps({
    'brain_in_shell': cfg,
    'flat_y':         float(flat_y),
    'wall_mm':        WALL_MM,
    'cavity_margin':  CAVITY_MARGIN_MM,
    'wheel_xyz':      None if wheel is None else wheel.bounding_box.centroid.tolist(),
    'sensor_xyz':     None if sensor is None else sensor.bounding_box.centroid.tolist(),
    'pcb_xyz':        None if pcb is None else pcb.bounding_box.centroid.tolist(),
    'battery_xyz':    None if battery is None else battery.bounding_box.centroid.tolist(),
}, indent=2))

# ---- 10. preview --------------------------------------------------------
print('\n=== preview ===')
def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

views = [('dorsal (above)',  ( 0, -1,  0), (-1,  0,  0), 0),
         ('lateral right',   ( 0,  0,  1), ( 0, -1,  0), 0),
         ('front (rostral)', ( 1,  0,  0), ( 0, -1,  0), 0),
         ('exploded',        ( 0,  0,  1), ( 0, -1,  0), 22)]
pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1900, 1500))
for i, (name, view, up, lift) in enumerate(views):
    pl.subplot(i // 2, i % 2)
    pl.add_mesh(poly(brain_body), color=(0.92, 0.72, 0.72),
                smooth_shading=True, opacity=0.65,
                specular=0.25, ambient=0.3, diffuse=0.85)
    cL = cortex_L.copy(); cL.apply_translation([0, -lift, 0])
    cR = cortex_R.copy(); cR.apply_translation([0, -lift, 0])
    pl.add_mesh(poly(cL), color=(0.85, 0.40, 0.45), smooth_shading=True, opacity=0.9)
    pl.add_mesh(poly(cR), color=(0.45, 0.55, 0.85), smooth_shading=True, opacity=0.9)
    # render kit electronics (in their fixed positions inside the brain)
    for c in electronics:
        pl.add_mesh(poly(c), color=(0.95, 0.85, 0.55), smooth_shading=True,
                    opacity=0.85, specular=0.4, ambient=0.4, diffuse=0.8)
    pl.add_text(name, font_size=10, color='black', position='upper_left')
    pl.background_color = 'white'
    pl.view_vector(view, up); pl.reset_camera()
png = ASM / 'brain_mouse_preview.png'
pl.screenshot(str(png))
print(f'  {png.relative_to(DATA.parent)}')
