"""Interactive: click on the Bambu Lab kit's PCB to mark the locations of
the L click and R click microswitches. The saved positions are read by
the brain-mouse build to place the cortex-button switch-contact posts.

Workflow:
  1. Press 'l' to arm picking the LEFT click switch, then click on the PCB
     where the L switch lives.
  2. Press 'r' to arm picking the RIGHT click switch, then click on it.
  3. Press 'm' (mute) or 'w' (wheel-click) to mark those too if you want
     (optional — they will be added to the JSON).
  4. 's' saves -> data/mouse_brain_assembly/click_switches.json
  5. 'q' quits.

Coloured markers appear where you've clicked. Click again with the same
'l'/'r' arming to overwrite.
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
KIT  = DATA / 'mouse_kit' / 'extracted'
ASM  = DATA / 'mouse_brain_assembly'
ASM.mkdir(parents=True, exist_ok=True)
OUT  = ASM / 'click_switches.json'

# ---- load assembly + axis-swap (same R_SHELL as build script) -----------
asm = trimesh.load(str(KIT/'08_3D_model_for_Wireless_Mouse_Kit-002__hardware_shell_assembly___________.stl'), force='mesh')
asm.apply_translation(-asm.bounding_box.centroid)
R_SHELL = np.array([[0, 1, 0, 0], [0, 0,-1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)
asm.apply_transform(R_SHELL)

comps = sorted(asm.split(only_watertight=False), key=lambda c: -c.volume)
shells = comps[:2]                          # bottom + top shell halves
electronics = comps[2:14]                    # the chassis components

# identify the PCB to colour it differently
def is_pcb(c):
    e = c.bounding_box.extents
    return c.volume > 3000 and max(e) > 60 and min(e) < 20
pcb = next((c for c in electronics if is_pcb(c)), None)

print('=== components (clickable surfaces) ===')
for i, c in enumerate(electronics):
    e = c.bounding_box.extents
    ctr = c.bounding_box.centroid
    print(f'  comp{i:2d}: vol={c.volume:>6.0f}  bbox={[round(x,1) for x in e]}  '
          f'ctr={[round(x,1) for x in ctr]}')

# ---- pyvista scene ------------------------------------------------------
def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

pl = pv.Plotter(window_size=(1700, 1100))
pv.set_new_attribute(pl, 'pickpoint', None)

# shells translucent for context
for s in shells:
    pl.add_mesh(poly(s), color=(0.7, 0.7, 0.75), opacity=0.18,
                smooth_shading=True)
# PCB highlighted
if pcb is not None:
    pl.add_mesh(poly(pcb), color=(0.30, 0.55, 0.95), opacity=0.95,
                smooth_shading=True, specular=0.4, ambient=0.3, diffuse=0.85)
# other electronics in light yellow
for c in electronics:
    if c is pcb: continue
    pl.add_mesh(poly(c), color=(0.95, 0.85, 0.55), opacity=0.95,
                smooth_shading=True, specular=0.4, ambient=0.4, diffuse=0.8)

# state ------------------------------------------------------------------
arm = ['']           # current arming target: 'L', 'R', 'M', 'W' or ''
saved = {}           # name -> (x, y, z)
if OUT.exists():
    raw = json.loads(OUT.read_text())
    for k, v in raw.items():
        saved[k] = tuple(v)
    print(f'loaded existing switches: {saved}')
marker_actors = {}

MARKER_R = 1.4
MARKER_COL = {
    'L_click':    (0.95, 0.20, 0.20),
    'R_click':    (0.20, 0.20, 0.95),
    'mute':       (0.85, 0.60, 0.10),
    'wheel_click':(0.20, 0.85, 0.40),
}

def draw_marker(name, pos):
    if name in marker_actors:
        pl.remove_actor(marker_actors[name])
    sph = pv.Sphere(radius=MARKER_R, center=pos)
    marker_actors[name] = pl.add_mesh(sph, color=MARKER_COL[name],
        smooth_shading=True, reset_camera=False)

for name, p in saved.items():
    draw_marker(name, p)

def arm_(target):
    arm[0] = target
    print(f'ARMED: click on the kit to set {target}')

def on_click(point, *args):
    if not arm[0]:
        return
    name = {'L':'L_click', 'R':'R_click', 'M':'mute', 'W':'wheel_click'}[arm[0]]
    p = tuple(float(x) for x in point)
    saved[name] = p
    draw_marker(name, p)
    print(f'  {name} = ({p[0]:+.1f}, {p[1]:+.1f}, {p[2]:+.1f})')
    arm[0] = ''

pl.enable_surface_point_picking(callback=on_click, show_point=False,
    left_clicking=True, show_message=False, tolerance=0.01)

def save_():
    OUT.write_text(json.dumps({k: list(v) for k, v in saved.items()}, indent=2))
    print(f'>> saved {OUT.relative_to(DATA.parent)}: {list(saved.keys())}')

def reset_():
    for a in marker_actors.values(): pl.remove_actor(a)
    marker_actors.clear(); saved.clear()
    print('reset')

for k, fn in [('l', lambda: arm_('L')),
              ('r', lambda: arm_('R')),
              ('m', lambda: arm_('M')),
              ('w', lambda: arm_('W')),
              ('s', save_),
              ('R', reset_)]:
    pl.add_key_event(k, fn)

pl.add_text(
    'Bambu kit electronics — pick switch positions\n'
    'BLUE board = PCB, YELLOW = other electronics, grey = shell\n'
    'l: arm L-click pick    r: arm R-click pick\n'
    'm: arm mute pick       w: arm wheel-click pick\n'
    's: save    R: reset    q: quit\n'
    'After arming, LEFT-CLICK the spot on the PCB',
    font_size=10, position='upper_left')
pl.background_color = 'white'
print('window opening — arm with l/r and click PCB to mark switches')
pl.show()
