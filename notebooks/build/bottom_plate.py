"""Screw-on bottom plate for the brain-mouse body.

Generates:
  bottom_plate.stl       brain-silhouette plate (2.5 mm) with countersunk
                         screw holes, optical-sensor lens window, and 4
                         skate-pad recesses on the underside.
  body_with_bosses.stl   body.stl + 4 M2 screw bosses (cylinders
                         with pilot holes) the plate screws into.
  bottom_plate_preview.png

Screw layout (brain frame, measured from the kit):
  back-left  (-52,  +12)   back-right (-52, -14)
  front-left ( 50,  +10)   front-right( 50,  -8)

Run:
    python notebooks/build/bottom_plate.py
"""
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv
import shapely.geometry as sg

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
ASM  = DATA / 'mouse_brain_assembly'

# ---- tunables -----------------------------------------------------------
PLATE_THICK     = 2.5
PLATE_CLEARANCE = 0.6     # plate is this much SMALLER than the rebate per side
                          # (clearance fit so it drops out; 0.6 mm suits PETG)
PLATE_MARGIN    = 7.0     # rebate flange (must match brain_mouse_body.py)
SCREW_PILOT_R   = 0.85    # M2 self-tap pilot radius (boss tapped hole)
SCREW_CLEAR_R   = 1.2     # M2 clearance radius (plate through-hole)
SCREW_HEAD_R    = 2.1     # countersink head radius
BOSS_OUTER_R    = 2.5     # screw boss outer radius (reduced so the boss fits
                          # fully in the flange without overflowing the opening)
BOSS_HEIGHT     = 9.0     # boss height up from base
SENSOR_XZ       = (4.3, -9.4)   # optical sensor lens (brain frame) — vol=479 module
SENSOR_WIN_R    = 6.0
SKATE_R         = 5.0     # PTFE skate recess radius
SKATE_DEPTH     = 0.6
# Screw positions (x,z): boss centre held >= BOSS_OUTER_R from the chassis
# opening edge so the boss sits fully in SOLID body (pulled back into the
# body, not overflowing the opening).
SCREWS = [(-42.5, -26.5), (-42.5, 29.0), (48.5, -26.5), (48.5, 29.0)]

def ev(m):
    if m is None or len(m.faces) == 0: return m
    if m.is_volume: return m
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    for _ in range(3):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=30000)
        except Exception: pass
    cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

def cyl_y(radius, height, x, y_center, z, sections=32):
    """Cylinder with its axis along world y."""
    c = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    c.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], float))
    c.apply_translation([x, y_center, z])
    return c

print('=== load body (MST variant) ===')
body = trimesh.load(str(ASM/'body.stl'), force='mesh')
flat_y = body.bounds[1, 1]                 # base plane (ventral-most)
print(f'  body bbox {[round(x,1) for x in body.bounding_box.extents]}  flat_y={flat_y:.1f}')

# ---- SMALLER plate covering just the chassis opening (inset like a real
#      computer mouse, NOT the full brain footprint) -----------------------
# Chassis opening (brain frame, measured): x[-39.5, 45.8], z[-23.5, 26.1].
# Plate = that opening + PLATE_MARGIN, rounded corners.
import shapely.ops as so
from shapely.geometry import box
CHASSIS_X = (-39.5, 45.8)
CHASSIS_Z = (-23.5, 26.1)
CORNER_R     = 6.0
section = body.section(plane_origin=[0, flat_y - 0.5, 0], plane_normal=[0, 1, 0])
_, T = section.to_2D()   # only need the 3D transform
probe = trimesh.transformations.transform_points(
    np.array([[0,0,0],[1,0,0],[0,1,0]]), T)
u_axis = np.argmax(np.abs(probe[1]-probe[0]))
v_axis = np.argmax(np.abs(probe[2]-probe[0]))
print(f'  section T: path-u -> world axis {u_axis}, path-v -> world axis {v_axis}')
# Plate footprint = rebate (CHASSIS ± PLATE_MARGIN) shrunk by PLATE_CLEARANCE
# per side so it slides out of the rebate cleanly.
eff_margin = PLATE_MARGIN - PLATE_CLEARANCE
xr = (CHASSIS_X[0]-eff_margin, CHASSIS_X[1]+eff_margin)
zr = (CHASSIS_Z[0]-eff_margin, CHASSIS_Z[1]+eff_margin)
# path-u -> world x, path-v -> world z  (typical); fall back if swapped
if u_axis == 0:
    sil = box(xr[0], zr[0], xr[1], zr[1])
else:
    sil = box(zr[0], xr[0], zr[1], xr[1])
sil = sil.buffer(CORNER_R, join_style=1).buffer(-0.0)   # rounded corners
print(f'  plate footprint {xr[1]-xr[0]:.0f} x {zr[1]-zr[0]:.0f}mm  area {sil.area:.0f}mm² '
      f'(brain base is 148x116 — plate is inset)')

# ---- build the plate (FLUSH: sits in the body's rebate) ----------------
# Body has a PLATE_THICK-deep rebate at the base. The plate's bottom face
# is flush with the body bottom (y=flat_y); plate top rests on the rebate
# ledge at y=flat_y-PLATE_THICK.
plate = trimesh.creation.extrude_polygon(sil, PLATE_THICK)
plate.apply_transform(T)
print(f'  plate raw: y range [{plate.bounds[0,1]:.1f}, {plate.bounds[1,1]:.1f}]')
# place plate at y[flat_y - PLATE_THICK, flat_y]  (inside the rebate, flush)
plate_lo = plate.bounds[0,1]
shift = (flat_y - PLATE_THICK) - plate_lo
plate.apply_translation([0, shift, 0])
print(f'  plate placed (flush in rebate): y range [{plate.bounds[0,1]:.1f}, {plate.bounds[1,1]:.1f}]')
plate = ev(plate)

plate_center_y = flat_y - PLATE_THICK/2     # mid-plane of the plate
plate_bottom_y = flat_y                       # flush with body bottom
# screw through-holes + countersink (head flush with the plate bottom)
for (sx, sz) in SCREWS:
    thru = cyl_y(SCREW_CLEAR_R, PLATE_THICK + 2, sx, plate_center_y, sz)
    plate = ev(plate.difference(thru, engine='manifold'))
    csk = trimesh.creation.cone(radius=SCREW_HEAD_R, height=SCREW_HEAD_R, sections=24)
    csk.apply_transform(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]], float))
    csk.apply_translation([sx, plate_bottom_y - SCREW_HEAD_R/2 + 0.1, sz])
    plate = ev(plate.difference(csk, engine='manifold'))
print(f'  drilled {len(SCREWS)} countersunk screw holes (heads flush at y={flat_y:.1f})')

# sensor lens window
win = cyl_y(SENSOR_WIN_R, PLATE_THICK + 2, SENSOR_XZ[0], plate_center_y, SENSOR_XZ[1])
plate = ev(plate.difference(win, engine='manifold'))
print(f'  sensor lens window r={SENSOR_WIN_R}mm at {SENSOR_XZ}')

# PCB retention bumps on the plate TOP face. Without these the PCB has
# 2.2 mm vertical play (plate top y=11.7, PCB bottom y=9.5) and rattles.
# Bumps push the PCB up against the cavity ceiling so it's clamped.
# Positions avoid the sensor lens (4.3, -9.4) and the L/R click switch
# bumps (39, ±17) on the PCB top. (-15,±15) and (15,±15) hit either the
# PCB directly or the battery cradle that sits on it.
PCB_BUMP_R = 2.5
PCB_BUMP_H = 2.6   # plate-to-PCB gap = 2.2mm, +0.4mm compression
plate_top_y = plate.bounds[0, 1]    # most-dorsal face of plate (= rebate ledge)
for (bx, bz) in [(-15, -15), (-15, 15), (15, -15), (15, 15)]:
    bump = trimesh.creation.cylinder(radius=PCB_BUMP_R, height=PCB_BUMP_H, sections=20)
    bump.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], float))
    # cylinder extends symmetrically about its center; place center so its
    # base sits at plate_top_y and it extends PCB_BUMP_H upward (negative y)
    bump.apply_translation([bx, plate_top_y - PCB_BUMP_H/2, bz])
    plate = ev(trimesh.boolean.union([plate, bump], engine='manifold'))
print(f'  added 4 PCB retention bumps (r={PCB_BUMP_R}mm, h={PCB_BUMP_H}mm) — '
      f'clamp PCB when plate is screwed in')

# 4 skate recesses on the underside (corners of the silhouette bbox, inset)
minx, _, minz = plate.bounds[0]; maxx, _, maxz = plate.bounds[1]
inset = 10
skate_xz = [(minx+inset, minz+inset), (minx+inset, maxz-inset),
            (maxx-inset, minz+inset), (maxx-inset, maxz-inset)]
for (sx, sz) in skate_xz:
    rec = cyl_y(SKATE_R, SKATE_DEPTH*2, sx, plate_bottom_y - SKATE_DEPTH + 0.01, sz)
    plate = ev(plate.difference(rec, engine='manifold'))
print(f'  4 skate recesses r={SKATE_R}mm depth={SKATE_DEPTH}mm')

plate.export(str(ASM/'bottom_plate.stl'))
print(f'  -> bottom_plate.stl  vol {plate.volume:.0f}mm³  watertight={plate.is_watertight}')

# ---- screw bosses on the body ------------------------------------------
print('\n=== screw bosses ===')
# Bosses sit ABOVE the rebate ledge (y < flat_y - PLATE_THICK) so they
# don't collide with the seated plate. Screw threads engage here.
boss_top_y = flat_y - PLATE_THICK          # rebate ledge (plate top)
bosses = []
for (sx, sz) in SCREWS:
    by = boss_top_y - BOSS_HEIGHT/2
    boss = cyl_y(BOSS_OUTER_R, BOSS_HEIGHT, sx, by, sz)
    pilot = cyl_y(SCREW_PILOT_R, BOSS_HEIGHT + 2, sx, by, sz)
    boss = ev(boss.difference(pilot, engine='manifold'))
    bosses.append(boss)
boss_union = trimesh.util.concatenate(bosses)
body_with_bosses = trimesh.util.concatenate([body, boss_union])
body_with_bosses.export(str(ASM/'body_with_bosses.stl'))
print(f'  -> body_with_bosses.stl  ({len(SCREWS)} bosses OD={BOSS_OUTER_R*2}mm, '
      f'pilot {SCREW_PILOT_R*2}mm, height {BOSS_HEIGHT}mm)')

# ---- preview ------------------------------------------------------------
print('\n=== preview ===')
def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))
views = [('assembled - lateral', (10,-15,290), (10,-15,0), (0,-1,0), 0),
         ('exploded - plate down', (150,-110,230), (10,-15,0), (0,-1,0), 30),
         ('plate underside', (10, 200, 0), (10, 14, 0), (1,0,0), 30)]
pl = pv.Plotter(shape=(1,3), off_screen=True, window_size=(2400,1000))
for i,(name,pos,foc,up,drop) in enumerate(views):
    pl.subplot(0,i)
    pl.add_mesh(poly(body), color=(0.85,0.78,0.78), smooth_shading=True, opacity=0.55)
    for b in bosses:
        pl.add_mesh(poly(b), color=(0.5,0.7,0.9), smooth_shading=True)
    pd = plate.copy(); pd.apply_translation([0, drop, 0])
    pl.add_mesh(poly(pd), color=(0.5,0.85,0.55), smooth_shading=True, opacity=0.95)
    pl.camera_position = [pos, foc, up]
    pl.background_color = 'white'
    pl.add_text(name, font_size=11, color='black', position='upper_left')
png = ASM/'bottom_plate_preview.png'
pl.screenshot(str(png))
print(f'  -> {png.relative_to(DATA.parent)}')

# ---- battery removal simulation ----------------------------------------
# With the plate OFF, can the AA battery be extracted? Test two motions:
#   (a) straight down (+y) out through the cavity opening
#   (b) slide along -x (axial, toward the back) then down
# At each step check battery-vertices-inside-body (collision with walls).
print('\n=== battery removal simulation ===')
KIT = DATA / 'mouse_kit' / 'extracted'
asm = trimesh.load(str(KIT/'08_3D_model_for_Wireless_Mouse_Kit-002__hardware_shell_assembly___________.stl'), force='mesh')
asm.apply_translation(-asm.bounding_box.centroid)
Rk = np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]], float)
asm.apply_transform(Rk)
elec = [c for c in sorted(asm.split(only_watertight=False), key=lambda c:-c.volume)[2:]
        if c.volume>100]
def is_batt(c):
    b=sorted(c.bounding_box.extents); return abs(b[2]-49.3)<3 and abs(b[0]-13.5)<2
battery = next(c for c in elec if is_batt(c))
body_ev_sim = ev(body)   # need a volume for .contains
print(f'  battery start bbox y[{battery.bounds[0,1]:.1f},{battery.bounds[1,1]:.1f}]')

def collide_count(mesh):
    try:
        return int(body_ev_sim.contains(mesh.vertices).sum())
    except Exception:
        return -1

rest = collide_count(battery)
print(f'  rest: {rest} battery verts inside body (baseline)')

print('\n  (a) straight-DOWN extraction (+y):')
print(f'  {"drop":>6s}  {"verts in body":>14s}  status')
clear_down = True
for dy in np.arange(0, 30.1, 2.0):
    m = battery.copy(); m.apply_translation([0, dy, 0])
    c = collide_count(m)
    delta = c - rest
    status = 'CLEAR' if delta <= 2 else f'BLOCKED (+{delta})'
    if delta > 2: clear_down = False
    print(f'  {dy:>5.1f}mm  {c:>10d}     {status}')
    if dy > 18 and delta <= 2:
        print('  -> battery fully exits the base by 18mm drop'); break

print('\n  (b) axial slide (-x toward back) then drop:')
clear_axial = True
for dx in np.arange(0, 30.1, 3.0):
    m = battery.copy(); m.apply_translation([-dx, 5, 0])  # slight drop + slide
    c = collide_count(m)
    delta = c - rest
    status = 'CLEAR' if delta <= 2 else f'BLOCKED (+{delta})'
    if delta > 2: clear_axial = False
    print(f'  -x {dx:>4.1f}mm + 5mm drop:  {c:>6d} verts  {status}')

print(f'\n  VERDICT: straight-down extraction {"WORKS" if clear_down else "BLOCKED"}; '
      f'axial slide {"WORKS" if clear_axial else "BLOCKED"}')

# Render the extraction sequence
print('\n  rendering extraction frames...')
pl = pv.Plotter(off_screen=True, window_size=(1200, 800))
pl.background_color = 'white'
pl.camera_position = [(150,-110,230),(10,-15,0),(0,-1,0)]
pl.open_gif(str(ASM/'battery_removal.gif'), fps=6)
for dy in list(np.arange(0, 26, 2.0)) + list(np.arange(26, -0.1, -2.0)):
    pl.clear_actors()
    pl.add_mesh(poly(body), color=(0.85,0.78,0.78), smooth_shading=True, opacity=0.35)
    bm = battery.copy(); bm.apply_translation([0, dy, 0])
    pl.add_mesh(poly(bm), color=(0.2,0.8,0.3), smooth_shading=True)
    # other electronics for context
    for c in elec:
        if c is battery: continue
        pl.add_mesh(poly(c), color=(0.95,0.85,0.55), smooth_shading=True, opacity=0.5)
    pl.add_text(f'battery drop: {dy:.0f}mm', font_size=14, color='black', position='lower_left')
    pl.write_frame()
pl.close()
print(f'  -> data/mouse_brain_assembly/battery_removal.gif')
