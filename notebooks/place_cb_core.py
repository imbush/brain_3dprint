"""Interactive: shrink + position the Allen `CB` parent mesh as an internal
core connector for the cerebellum one-piece. Saved state is consumed by
notebooks/build/cerebellum_engrave.py.

The core mesh starts at SCALE0 of the CCF-scale CB envelope, centred on it.
Adjust live, then 's' to save -> data/structure_sets/cerebellum/cb_core.json.

Keys:
  + / =        : scale up by SCALE_STEP
  - / _        : scale down by SCALE_STEP
  Up / Down    : translate +/- y
  Left / Right : translate +/- x
  1 / 2        : translate -/+ z   (replaces PageDn/PageUp; works on laptops)
  t            : toggle area opacity (translucent <-> solid)
  c            : cut mode — click 2 points to slice the core with the plane
                 through (camera, click1, click2). Side kept = the larger
                 chunk. Cuts accumulate; saved with the state.
  u            : undo last cut
  r            : reset scale + translate (cuts kept)
  R            : reset everything including cuts
  s            : save state
  q            : quit
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv

DATA = Path(__file__).resolve().parent.parent / 'data'
CCF  = DATA / 'ccf'
SET  = DATA / 'structure_sets' / 'cerebellum'
STL  = SET / 'stl'
TM   = 66
OUT  = SET / 'cb_core.json'

SCALE0 = 0.75       # starting fraction of the CB envelope
SCALE_STEP = 1.03
TRANS_STEP = 0.5    # mm per key press

# shared scale (matches step 2 watertight): map CCF -> 66 mm whole-brain frame
root = trimesh.load(CCF / 'obj' / '997_root.obj', force='mesh')
SHARED_SCALE = TM / root.bounding_box.extents.max()
ROOT_CTR = root.bounding_box.centroid * SHARED_SCALE

cb_raw = trimesh.load(CCF / 'obj' / '512_CB.obj', force='mesh')
cb_raw.apply_scale(SHARED_SCALE)
cb_raw.apply_translation(-ROOT_CTR)
CB_CTR = cb_raw.centroid.copy()
print(f'CB envelope: vol={cb_raw.volume:.0f}mm³  bbox={[round(x,1) for x in cb_raw.bounding_box.extents]}  centroid={[round(x,2) for x in CB_CTR]}')

# load engraved cerebellum areas for context (translucent backdrop)
areas = []   # list of (acronym, mesh)
for f in sorted(STL.glob(f'engraved_*_{TM}mm.stl')):
    acro = f.stem[len('engraved_'):-len(f'_{TM}mm')]
    m = trimesh.load(str(f), force='mesh')
    areas.append((acro, m))
print(f'loaded {len(areas)} engraved cerebellum areas')

def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

state = {'scale': SCALE0, 'translate': [0.0, 0.0, 0.0], 'cuts': []}
if OUT.exists():
    state = {**state, **json.loads(OUT.read_text())}
    state.setdefault('cuts', [])
    print(f'loaded existing state: scale={state["scale"]:.3f}  '
          f'translate={state["translate"]}  cuts={len(state["cuts"])}')

def build_core():
    """Apply scale + translate to the CB envelope, then each saved cut as a
    plane slice (keep the +n side). `cuts` is a list of
    {origin: [x,y,z], normal: [x,y,z]}."""
    m = cb_raw.copy()
    s = state['scale']
    m.vertices = (m.vertices - CB_CTR) * s + CB_CTR + np.asarray(state['translate'], float)
    for cut in state['cuts']:
        m = m.slice_plane(np.asarray(cut['origin'], float),
                          np.asarray(cut['normal'], float), cap=True)
        if m is None or len(m.faces) == 0:
            print('  WARN: a cut removed the entire core'); return cb_raw.copy()
    return m

pl = pv.Plotter(window_size=(1600, 1100))
# pyvista 0.47's surface picker assigns self.pickpoint, which the new
# attribute-guard otherwise rejects — register it up front.
pv.set_new_attribute(pl, 'pickpoint', None)

area_actors = []      # (actor, acro, mesh) so cut/hide can touch them
hidden = set()        # acronyms currently hidden
OPACITY_LO, OPACITY_HI = 0.35, 1.0
opacity = [OPACITY_LO]
for acro, m in areas:
    a = pl.add_mesh(poly(m), color=(0.8, 0.8, 0.85), smooth_shading=True,
                    opacity=opacity[0], specular=0.15, ambient=0.3, diffuse=0.7,
                    pickable=True)
    area_actors.append((a, acro, m))
core_actor = {'a': None}
def redraw():
    if core_actor['a'] is not None:
        pl.remove_actor(core_actor['a'])
    m = build_core()
    core_actor['a'] = pl.add_mesh(poly(m), color=(0.95, 0.25, 0.25),
                                  smooth_shading=True, opacity=0.85,
                                  specular=0.3, ambient=0.35, diffuse=0.8,
                                  reset_camera=False)
    t = state['translate']
    print(f'  scale={state["scale"]:.3f}  translate=({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f})  '
          f'vol={m.volume:.0f}mm³')
redraw()

def scale_up():   state['scale'] *= SCALE_STEP; redraw()
def scale_down(): state['scale'] /= SCALE_STEP; redraw()
def tx_pos():     state['translate'][0] += TRANS_STEP; redraw()
def tx_neg():     state['translate'][0] -= TRANS_STEP; redraw()
def ty_pos():     state['translate'][1] += TRANS_STEP; redraw()
def ty_neg():     state['translate'][1] -= TRANS_STEP; redraw()
def tz_pos():     state['translate'][2] += TRANS_STEP; redraw()
def tz_neg():     state['translate'][2] -= TRANS_STEP; redraw()
def reset():
    state['scale'] = SCALE0
    state['translate'][:] = [0.0, 0.0, 0.0]
    redraw()
def reset_all():
    state['scale'] = SCALE0
    state['translate'][:] = [0.0, 0.0, 0.0]
    state['cuts'].clear()
    redraw()
    print('reset everything (including cuts)')
def save():
    OUT.write_text(json.dumps(state, indent=2))
    print(f'>> saved {OUT.relative_to(DATA.parent)} '
          f'({len(state["cuts"])} cuts)')

# ---- opacity toggle ------------------------------------------------------
def toggle_opacity():
    opacity[0] = OPACITY_HI if opacity[0] == OPACITY_LO else OPACITY_LO
    for act, _, _ in area_actors:
        act.prop.opacity = opacity[0]
    print(f'area opacity -> {opacity[0]}')

# ---- cut mode: slice the CORE mesh with a plane through (camera, click1, click2)
cut_mode   = [False]
cut_clicks = []

def toggle_cut_mode():
    cut_mode[0] = not cut_mode[0]
    cut_clicks.clear()
    print('CUT MODE ' + ('ON — click 2 points to slice the core'
                         if cut_mode[0] else 'OFF'))

def _apply_cut(p1, p2):
    cam = np.asarray(pl.camera.position, float)
    n = np.cross(np.asarray(p1) - cam, np.asarray(p2) - cam)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        print('  cut: degenerate plane (clicks collinear with camera)'); return
    n /= nn
    origin = (np.asarray(p1) + np.asarray(p2)) / 2.0   # any point on the plane
    # Decide which side to KEEP: try both halves of the current core, keep
    # the larger one. This matches the intuition "cut off a slice."
    core = build_core()
    half_pos = core.slice_plane(origin, n, cap=True)
    half_neg = core.slice_plane(origin, -n, cap=True)
    v_pos = half_pos.volume if (half_pos is not None and len(half_pos.faces)) else 0
    v_neg = half_neg.volume if (half_neg is not None and len(half_neg.faces)) else 0
    if max(v_pos, v_neg) < 1e-6:
        print('  cut: plane misses the core'); return
    keep_n = n if v_pos >= v_neg else -n
    state['cuts'].append({'origin': origin.tolist(), 'normal': keep_n.tolist()})
    redraw()
    print(f'  cut #{len(state["cuts"])}: removed {min(v_pos, v_neg):.0f}mm³')

def undo_cut():
    if not state['cuts']: print('no cuts to undo'); return
    state['cuts'].pop()
    redraw()
    print(f'undo: {len(state["cuts"])} cuts remain')

def on_click(point, *args):
    if not cut_mode[0]: return
    cut_clicks.append(np.asarray(point, float))
    print(f'  cut click {len(cut_clicks)}: '
          f'({point[0]:.1f},{point[1]:.1f},{point[2]:.1f})')
    if len(cut_clicks) == 2:
        _apply_cut(*cut_clicks)
        cut_clicks.clear()
        cut_mode[0] = False
        print('CUT MODE OFF (press c to cut again)')

pl.enable_surface_point_picking(callback=on_click, show_point=False,
                                left_clicking=True, show_message=False,
                                tolerance=0.01)

for keys, fn in [(('plus', 'equal'), scale_up),
                 (('minus', 'underscore'), scale_down),
                 (('Right',), tx_pos), (('Left',), tx_neg),
                 (('Up',), ty_pos),    (('Down',), ty_neg),
                 (('1',), tz_neg),     (('2',), tz_pos),
                 (('t',), toggle_opacity), (('c',), toggle_cut_mode),
                 (('u',), undo_cut),
                 (('r',), reset), (('R',), reset_all), (('s',), save)]:
    for k in keys:
        pl.add_key_event(k, fn)

pl.add_text('Translucent gray = engraved cerebellar areas\n'
            'RED = shrunken CB core (the connector)\n'
            '+/-: scale   arrows: x/y translate   1/2: z translate\n'
            't toggle opacity   c cut mode (click 2 pts to slice core)   u undo\n'
            'r reset xform   R reset all   s save   q quit',
            font_size=10, position='upper_left')
pl.background_color = 'white'
print('window opening — adjust scale + translation; press s to save, q to quit')
pl.show()
