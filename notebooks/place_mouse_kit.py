"""Interactive: position the Bambu Lab Wireless Mouse Kit-002 electronics
inside the brain. Saved transform is consumed by mouse_brain_assembly.py
when it carves the cavity.

Shows:
  - the brain shell (translucent gray)
  - the cortex hemispheres (translucent left/right)
  - the kit electronics (opaque yellow)
The scroll wheel is part of the kit STL — line it up so it pokes UP between
the two cortex hemispheres at the midline.

Keys:
  Translate
    Left / Right  : ±x   (anterior / posterior in brain frame)
    Up / Down     : ±y   (dorsal / ventral; smaller y = dorsal)
    1 / 2         : ±z   (left / right)

  Rotate (about kit centroid)
    x / X         : ±roll   about brain x-axis
    y / Y         : ±yaw    about brain y-axis
    z / Z         : ±pitch  about brain z-axis
    f             : flip 180° about y (common when kit is backwards)
    .             : snap 90° about y

  Misc
    g             : toggle large/small translation step (5mm <-> 0.5mm)
    h             : toggle hide/show cortex hemispheres
    o             : toggle brain opacity (low <-> high)
    r             : reset all
    s             : save -> data/mouse_brain_assembly/kit_transform.json
    q             : quit
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
ASM  = DATA / 'mouse_brain_assembly'
KIT  = DATA / 'mouse_kit' / 'extracted'

# inputs from the assembly builder
SHELL_STL    = ASM / 'shell.stl'
CORTEX_L_STL = ASM / 'cortex_L.stl'
CORTEX_R_STL = ASM / 'cortex_R.stl'
KIT_STL      = KIT / '02_3D_model_for_Wireless_Mouse_Kit-002__hardware_components______.stl'
OUT          = ASM / 'kit_transform.json'

for p in (SHELL_STL, CORTEX_L_STL, CORTEX_R_STL, KIT_STL):
    if not p.exists():
        raise SystemExit(f'missing: {p}')

# ---- load -------------------------------------------------------------
shell    = trimesh.load(str(SHELL_STL),    force='mesh')
cortex_L = trimesh.load(str(CORTEX_L_STL), force='mesh')
cortex_R = trimesh.load(str(CORTEX_R_STL), force='mesh')
kit_base = trimesh.load(str(KIT_STL),      force='mesh')
# centre the kit on its own bounding-box centroid so transforms rotate
# about a sensible pivot. The kit STL ships offset by huge translations.
kit_base.apply_translation(-kit_base.bounding_box.centroid)

print(f'shell     bbox {[round(x,1) for x in shell.bounding_box.extents]}    '
      f'centroid {[round(x,1) for x in shell.bounding_box.centroid]}')
print(f'cortex L  bbox {[round(x,1) for x in cortex_L.bounding_box.extents]}')
print(f'cortex R  bbox {[round(x,1) for x in cortex_R.bounding_box.extents]}')
print(f'kit (centred)  bbox {[round(x,1) for x in kit_base.bounding_box.extents]}')

# default transform: place kit centered in the brain bbox, then user adjusts
brain_center = (shell.bounding_box.centroid +
                cortex_L.bounding_box.centroid +
                cortex_R.bounding_box.centroid) / 3
default_T = np.eye(4)
default_T[:3, 3] = brain_center

state = {'matrix': default_T.tolist()}
if OUT.exists():
    state = json.loads(OUT.read_text())
    print(f'loaded existing transform from {OUT.name}')
T = np.asarray(state['matrix'], float)

STEP_LARGE = 5.0; STEP_SMALL = 0.5
step = [STEP_LARGE]
ROT_DEG = 5.0
SHOW_CORTEX = [True]
BRAIN_OPACITY_LO, BRAIN_OPACITY_HI = 0.25, 0.85
brain_opacity = [BRAIN_OPACITY_LO]

def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

pl = pv.Plotter(window_size=(1700, 1100))
pv.set_new_attribute(pl, 'pickpoint', None)

shell_actor  = pl.add_mesh(poly(shell),    color=(0.75, 0.75, 0.80),
                            smooth_shading=True, opacity=brain_opacity[0])
cl_actor     = pl.add_mesh(poly(cortex_L), color=(0.85, 0.45, 0.45),
                            smooth_shading=True, opacity=brain_opacity[0])
cr_actor     = pl.add_mesh(poly(cortex_R), color=(0.45, 0.55, 0.85),
                            smooth_shading=True, opacity=brain_opacity[0])
kit_actor    = {'a': None}

def build_kit():
    m = kit_base.copy(); m.apply_transform(T); return m

def redraw():
    if kit_actor['a'] is not None: pl.remove_actor(kit_actor['a'])
    m = build_kit()
    kit_actor['a'] = pl.add_mesh(poly(m), color=(0.95, 0.80, 0.30),
                                  smooth_shading=True, opacity=0.95,
                                  reset_camera=False)
    bb = m.bounding_box
    t = T[:3, 3]
    print(f'  translate=({t[0]:+6.1f},{t[1]:+6.1f},{t[2]:+6.1f})  '
          f'kit bbox center=({bb.centroid[0]:+6.1f},{bb.centroid[1]:+6.1f},{bb.centroid[2]:+6.1f})')

redraw()

def translate(axis, sign):
    T[axis, 3] += sign * step[0]; redraw()
def rotate(axis, deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.eye(4)
    if axis == 0:
        R[1, 1] = c; R[1, 2] = -s; R[2, 1] = s; R[2, 2] = c
    elif axis == 1:
        R[0, 0] = c; R[0, 2] = s;  R[2, 0] = -s; R[2, 2] = c
    else:
        R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s;  R[1, 1] = c
    # rotate about the kit's current centroid (not the world origin)
    pivot = (kit_base.copy() if T[:3, 3].sum() == 0 else build_kit()).bounding_box.centroid
    P = np.eye(4); P[:3, 3] = -pivot
    Pinv = np.eye(4); Pinv[:3, 3] = pivot
    T[:] = Pinv @ R @ P @ T
    redraw()

def toggle_step():
    step[0] = STEP_SMALL if step[0] == STEP_LARGE else STEP_LARGE
    print(f'step = {step[0]}mm')

def toggle_cortex():
    SHOW_CORTEX[0] = not SHOW_CORTEX[0]
    cl_actor.SetVisibility(SHOW_CORTEX[0])
    cr_actor.SetVisibility(SHOW_CORTEX[0])
    pl.render()
    print(f'cortex visible: {SHOW_CORTEX[0]}')

def toggle_opacity():
    brain_opacity[0] = BRAIN_OPACITY_HI if brain_opacity[0] == BRAIN_OPACITY_LO else BRAIN_OPACITY_LO
    for a in (shell_actor, cl_actor, cr_actor):
        a.prop.opacity = brain_opacity[0]
    pl.render()
    print(f'brain opacity: {brain_opacity[0]}')

def reset():
    T[:] = default_T; redraw()
    print('reset to default')

def save():
    OUT.write_text(json.dumps({'matrix': T.tolist()}, indent=2))
    print(f'>> saved {OUT.relative_to(DATA.parent)}')

def flip_y(): rotate(1, 180.0)
def snap_y(): rotate(1, 90.0)

for keys, fn in [
    (('Left',),  lambda: translate(0, -1)),
    (('Right',), lambda: translate(0, +1)),
    (('Up',),    lambda: translate(1, -1)),     # smaller y = dorsal in CCF
    (('Down',),  lambda: translate(1, +1)),
    (('1',),     lambda: translate(2, -1)),
    (('2',),     lambda: translate(2, +1)),
    (('x',),     lambda: rotate(0, -ROT_DEG)),
    (('X',),     lambda: rotate(0, +ROT_DEG)),
    (('y',),     lambda: rotate(1, -ROT_DEG)),
    (('Y',),     lambda: rotate(1, +ROT_DEG)),
    (('z',),     lambda: rotate(2, -ROT_DEG)),
    (('Z',),     lambda: rotate(2, +ROT_DEG)),
    (('f',),     flip_y),
    (('period',), snap_y),
    (('g',),     toggle_step),
    (('h',),     toggle_cortex),
    (('o',),     toggle_opacity),
    (('r',),     reset),
    (('s',),     save),
]:
    for k in keys:
        pl.add_key_event(k, fn)

pl.add_text(
    'BRAIN (translucent) + YELLOW kit electronics\n'
    'arrows: x/y translate    1/2: z translate    g: step 5mm<->0.5mm\n'
    'x/X y/Y z/Z: rotate 5°    f: flip 180° y    .: snap 90° y\n'
    'h: hide cortex    o: brain opacity    r: reset    s: save    q: quit',
    font_size=10, position='upper_left')
pl.background_color = 'white'
print('window opening — position the kit (wheel up through midline)')
pl.show()
