"""Interactive: position the whole mouse brain inside the Bambu Lab kit's
hardware shell. Saved transform is consumed by the mouse-brain assembly
builder to make the printable shell that contains every structure set.

Brain motions allowed (midline-plane only):
  - translate forward / backward  (x)
  - translate up / down            (y)
  - scale uniform
  - tilt back / forward            (rotation about z, the LR axis)

Locked:
  - no z translation (always at midline)
  - no rotation about x or y (no roll, no yaw)

Keys:
  Up    / Down   : translate y (-/+)        [smaller y = dorsal/top]
  Left  / Right  : translate x (-/+)        [smaller x = rostral/front]
  + / =          : scale up                 [SCALE_STEP per press]
  - / _          : scale down
  t              : tilt nose-down (rotate +deg about z)
  T              : tilt nose-up   (rotate -deg about z)
  g              : toggle step  (large 2mm/3° <-> fine 0.2mm/0.5°)
  o              : toggle brain opacity (translucent <-> solid)
  h              : toggle Bambu-shell visibility
  r              : reset
  s              : save -> data/mouse_brain_assembly/brain_in_shell.json
  q              : quit
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CCF  = DATA / 'ccf' / 'obj'
SETS = DATA / 'structure_sets'
KIT  = DATA / 'mouse_kit' / 'extracted'
ASM  = DATA / 'mouse_brain_assembly'
ASM.mkdir(parents=True, exist_ok=True)
OUT  = ASM / 'brain_in_shell.json'

# ---- Bambu hardware shell as the target envelope -----------------------
SHELL_STL = KIT / '05_3D_model_for_Wireless_Mouse_Kit-002__hardware_shell____.stl'
shell_raw = trimesh.load(str(SHELL_STL), force='mesh')
shell_raw.apply_translation(-shell_raw.bounding_box.centroid)
# Bambu axes: x=width(LR), y=length(AP, ~110mm), z=height(DV). Reorient to
# brain CCF: x=AP, y=DV (smaller=dorsal), z=LR.
R_SHELL = np.array([[0, 1, 0, 0],
                    [0, 0,-1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]], dtype=float)
shell_raw.apply_transform(R_SHELL)
print(f'Bambu shell (reoriented to brain frame):')
print(f'  bbox extents {shell_raw.bounding_box.extents}')
print(f'  bounds  x[{shell_raw.bounds[0,0]:6.1f},{shell_raw.bounds[1,0]:6.1f}]  '
      f'y[{shell_raw.bounds[0,1]:6.1f},{shell_raw.bounds[1,1]:6.1f}]  '
      f'z[{shell_raw.bounds[0,2]:6.1f},{shell_raw.bounds[1,2]:6.1f}]')

# split into top + bottom shells (the two halves of the case)
shell_comps = [c for c in shell_raw.split(only_watertight=False) if c.volume > 100]
print(f'  shell components: {len(shell_comps)}')

# ---- whole brain = every structure set we have -------------------------
# Use engraved per-set unioned STL where available; otherwise Allen parent
# envelope (single piece, clean shell) scaled to the same 66mm whole-brain
# frame used by the rest of the pipeline.
TM_REF = 66
root_ccf = trimesh.load(str(CCF/'997_root.obj'), force='mesh')
SHARED_SCALE = TM_REF / root_ccf.bounding_box.extents.max()
ROOT_CTR = root_ccf.bounding_box.centroid * SHARED_SCALE

def load_set(label, engraved_path=None, parent_obj=None):
    if engraved_path and engraved_path.exists():
        m = trimesh.load(str(engraved_path), force='mesh')
        print(f'  + {label:14s}  engraved  bbox={[round(x,1) for x in m.bounding_box.extents]}')
        return label, m
    if parent_obj and (CCF/parent_obj).exists():
        m = trimesh.load(str(CCF/parent_obj), force='mesh')
        m.apply_scale(SHARED_SCALE); m.apply_translation(-ROOT_CTR)
        print(f'  + {label:14s}  parent    bbox={[round(x,1) for x in m.bounding_box.extents]}')
        return label, m
    print(f'  - {label}: missing')
    return None

print('\n=== loading whole brain (every set) ===')
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

# Flip every brain piece 180° about the dorsoventral (y) axis so the
# olfactory bulb points toward the shell's front. CCF convention has the
# rostral pole at small x; the Bambu shell's "front" sits at small x too
# after R_SHELL, but the brain was pointing the wrong way relative to it.
FLIP_BRAIN_AP = np.diag([-1.0, 1.0, -1.0, 1.0])    # negate x and z (180° about y)
for _, m in loaded:
    m.apply_transform(FLIP_BRAIN_AP)

# single brain colour — pinkish, like a real brain
BRAIN_COLOR = (0.92, 0.72, 0.72)

# All-brain bbox for default-scale fit
all_v = np.vstack([m.vertices for _, m in loaded])
brain_centroid = (all_v.min(0) + all_v.max(0)) / 2
brain_extents  = all_v.max(0) - all_v.min(0)
shell_extents  = shell_raw.bounding_box.extents
# default scale: fit brain AP (x) inside shell AP * 0.95 for clearance
default_scale = (shell_extents[0] * 0.95) / brain_extents[0]
print(f'\nbrain bbox {[round(x,1) for x in brain_extents]}')
print(f'shell bbox {[round(x,1) for x in shell_extents]}')
print(f'default uniform scale {default_scale:.3f}')

# transform state
state = {'scale': default_scale, 'tilt_deg': 0.0, 'tx': 0.0, 'ty': 0.0}
if OUT.exists():
    state = {**state, **json.loads(OUT.read_text())}
    print(f'loaded existing state: {state}')

STEP_LARGE_T = 5.0;   STEP_FINE_T = 0.5
STEP_LARGE_R = 5.0;   STEP_FINE_R = 1.0
STEP_LARGE_S = 1.08;  STEP_FINE_S = 1.01
step_idx = [0]   # 0 = large, 1 = fine
def step_t(): return STEP_LARGE_T if step_idx[0] == 0 else STEP_FINE_T
def step_r(): return STEP_LARGE_R if step_idx[0] == 0 else STEP_FINE_R
def step_s(): return STEP_LARGE_S if step_idx[0] == 0 else STEP_FINE_S

def brain_transform():
    """Build the 4x4 transform: center brain at origin first, then scale,
    tilt-about-z, translate (x, y, 0)."""
    T = np.eye(4)
    T[:3, 3] = -brain_centroid                          # center
    S = np.eye(4) * state['scale']; S[3, 3] = 1.0       # scale
    th = np.deg2rad(state['tilt_deg'])
    c, s = np.cos(th), np.sin(th)
    R = np.eye(4)
    R[0, 0] = c;  R[0, 1] = -s
    R[1, 0] = s;  R[1, 1] =  c
    Tr = np.eye(4); Tr[:3, 3] = [state['tx'], state['ty'], 0.0]
    return Tr @ R @ S @ T

def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

pl = pv.Plotter(window_size=(1700, 1100))
pv.set_new_attribute(pl, 'pickpoint', None)

# Bambu shell actors — fully opaque grey; toggle with 'h' to see inside
shell_actors = []
for c in shell_comps:
    a = pl.add_mesh(poly(c), color=(0.7, 0.7, 0.75),
                    smooth_shading=True, opacity=1.0,
                    specular=0.3, ambient=0.3, diffuse=0.85,
                    style='surface', show_edges=False)
    shell_actors.append(a)

brain_actors = {}   # label -> actor
brain_op = [0.95]
def redraw():
    T = brain_transform()
    for label, a in list(brain_actors.items()):
        pl.remove_actor(a)
    brain_actors.clear()
    for label, m in loaded:
        mm = m.copy(); mm.apply_transform(T)
        brain_actors[label] = pl.add_mesh(poly(mm), color=BRAIN_COLOR,
            smooth_shading=True, opacity=brain_op[0],
            specular=0.25, ambient=0.3, diffuse=0.85, reset_camera=False)
    bbox = np.array([m.bounds for _, m in
                     [(l, mm) for l, mm in loaded
                      for mm in [m.copy()]]
                     for m in [mm.apply_transform(T)]])
    # report brain bbox in shell frame
    tv = np.vstack([m.copy().apply_transform(T).vertices for _, m in loaded])
    bbx = tv.max(0) - tv.min(0)
    print(f'  scale={state["scale"]:.3f}  tilt={state["tilt_deg"]:+.1f}°  '
          f'tx={state["tx"]:+.2f}  ty={state["ty"]:+.2f}  '
          f'brain extents {[round(x,1) for x in bbx]}  '
          f'fits shell {[round(x,1) for x in shell_extents]}')

redraw()
pl.add_axes()

# ---- key handlers -------------------------------------------------------
def tx_pos(): state['tx'] += step_t(); redraw()
def tx_neg(): state['tx'] -= step_t(); redraw()
def ty_pos(): state['ty'] += step_t(); redraw()
def ty_neg(): state['ty'] -= step_t(); redraw()
def scale_up():   state['scale'] *= step_s(); redraw()
def scale_down(): state['scale'] /= step_s(); redraw()
def tilt_pos(): state['tilt_deg'] += step_r(); redraw()
def tilt_neg(): state['tilt_deg'] -= step_r(); redraw()
def toggle_step():
    step_idx[0] = 1 - step_idx[0]
    print(f'step: {"FINE" if step_idx[0] else "LARGE"}')
def toggle_op():
    brain_op[0] = 0.35 if brain_op[0] >= 0.9 else 1.0
    redraw()
    print(f'brain opacity: {brain_op[0]}')
def toggle_shell():
    for a in shell_actors:
        a.SetVisibility(not a.GetVisibility())
    pl.render()
def reset():
    state.update({'scale': default_scale, 'tilt_deg': 0.0, 'tx': 0.0, 'ty': 0.0})
    redraw()
def save():
    OUT.write_text(json.dumps(state, indent=2))
    print(f'>> saved {OUT.relative_to(DATA.parent)}: {state}')

for keys, fn in [
    (('Left',),     tx_neg), (('Right',), tx_pos),
    (('Up',),       ty_neg), (('Down',),  ty_pos),
    (('plus','equal'),   scale_up),
    (('minus','underscore'), scale_down),
    (('t',),        tilt_pos), (('T',), tilt_neg),
    (('g',),        toggle_step),
    (('o',),        toggle_op),
    (('h',),        toggle_shell),
    (('r',),        reset), (('s',), save),
]:
    for k in keys:
        pl.add_key_event(k, fn)

pl.add_text(
    'Translucent grey = Bambu kit hardware shell (target envelope)\n'
    'Coloured       = whole brain (cortex, cerebellum, olfactory,\n'
    '                 cnu_ctxsp, thalamus, hypothalamus, midbrain,\n'
    '                 hindbrain, hippocampal)\n'
    'arrows: x/y translate    +/-: scale    t/T: tilt about LR axis\n'
    'g: step LARGE/fine    o: brain opacity    h: hide shell\n'
    'r: reset    s: save    q: quit',
    font_size=10, position='upper_left')
pl.background_color = 'white'
print('\nwindow opening — position brain inside the Bambu shell')
pl.show()
