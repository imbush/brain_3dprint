"""Interactive stick placer: the two hemispheres are drawn SPREAD APART
(along the L-R axis) so you can see the medial walls, and you click pairs
of points to define short connecting sticks.

Click a point on one piece, then a point on another -> a stick is drawn
between them (a cylinder). Repeat for as many connections as you want
(across the midline to join the hemispheres, or anywhere). Points are
stored in the TRUE (un-spread) coordinates.

Keys:
  (click) (click)  place a stick between the two clicked points
  u  undo last stick      r  reset all
  s  save -> data/ccf_brain/stick_positions.json
  q  quit

Then build_onepiece_sticks.py (manual mode) unions the pieces + your
sticks (+ a minimal hidden intra-hemisphere spanning set) into one solid.

Run from a terminal WITH a display:
  conda activate microns-m2
  python notebooks/ccf_brain/place_sticks_interactive.py
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv
import matplotlib as mpl

OUT = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
TARGET_MM = 66
SPREAD = 14.0          # mm each hemisphere is shifted away from the midline
STICK_JSON = OUT / 'stick_positions.json'

pal = (list(mpl.colormaps['tab20'].colors) + list(mpl.colormaps['tab20b'].colors)
       + list(mpl.colormaps['tab20c'].colors))

def poly(m):
    return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces), 1), 3),
                                              m.faces]).astype(np.int64))

pl = pv.Plotter(window_size=(1700, 1200))
pv.set_new_attribute(pl, 'pickpoint', None)   # 0.47 left-click picker fix

files = sorted((OUT/'stl'/'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl'))
for fi, f in enumerate(files):
    m = trimesh.load(str(f), force='mesh')
    for c in m.split(only_watertight=False):
        if len(c.faces) < 50: continue
        off = np.array([0.0, 0.0, SPREAD if c.centroid[2] > 0 else -SPREAD])
        d = c.copy(); d.apply_translation(off)
        pl.add_mesh(poly(d), color=pal[fi % len(pal)], smooth_shading=True,
                    specular=0.2, ambient=0.25, diffuse=0.9, pickable=True)

# Un-spread reliably from the DISPLAY z-sign: every R piece is shifted +SPREAD
# and every L piece -SPREAD, and the two never overlap in display z, so the
# sign of the clicked point's z tells us its hemisphere.
def to_true(p):
    hemi = 'R' if p[2] > 0 else 'L'
    return p - np.array([0.0, 0.0, SPREAD if hemi == 'R' else -SPREAD]), hemi

pending = [None]                 # (display_point, true_point, hemi)
sticks = []                      # (A_true, B_true, hemiA, hemiB, [actors])
pend_actor = [None]

def click(point, *args):
    p = np.asarray(point, float)
    p_true, hemi = to_true(p)
    if pending[0] is None:
        pending[0] = (p, p_true, hemi)
        pend_actor[0] = pl.add_mesh(pv.Sphere(radius=1.2, center=p),
                                    color='yellow', pickable=False, reset_camera=False)
        print(f'point A ({hemi}) at {p_true.round(1).tolist()} — click point B')
    else:
        a_disp, a_true, a_h = pending[0]
        act = pl.add_mesh(pv.Line(a_disp, p).tube(radius=1.0), color='red',
                          pickable=False, reset_camera=False)
        if pend_actor[0] is not None:
            pl.remove_actor(pend_actor[0]); pend_actor[0] = None
        sticks.append((a_true, p_true, a_h, hemi, [act]))
        pending[0] = None
        print(f'+ stick {len(sticks)}: {a_h}{a_true.round(1).tolist()} -> {hemi}{p_true.round(1).tolist()}')

def undo():
    if pending[0] is not None:
        if pend_actor[0] is not None: pl.remove_actor(pend_actor[0]); pend_actor[0]=None
        pending[0] = None; print('cleared pending point'); return
    if not sticks: print('nothing to undo'); return
    _, _, acts = sticks.pop()
    for a in acts: pl.remove_actor(a)
    print(f'- undo stick (total {len(sticks)})')

def reset():
    for _, _, acts in sticks:
        for a in acts: pl.remove_actor(a)
    sticks.clear()
    if pend_actor[0] is not None: pl.remove_actor(pend_actor[0]); pend_actor[0]=None
    pending[0]=None; print('reset')

def save():
    STICK_JSON.write_text(json.dumps([{'A': a.tolist(), 'B': b.tolist()}
                                      for a, b, _ in sticks], indent=2))
    print(f'>> saved {STICK_JSON.name} ({len(sticks)} sticks)')

# preload existing
if STICK_JSON.exists():
    for e in json.loads(STICK_JSON.read_text()):
        a_true = np.array(e['A']); b_true = np.array(e['B'])
        a_disp = a_true + np.array([0,0, SPREAD if a_true[2] > 0 else -SPREAD])
        b_disp = b_true + np.array([0,0, SPREAD if b_true[2] > 0 else -SPREAD])
        act = pl.add_mesh(pv.Line(a_disp, b_disp).tube(radius=1.0), color='red', pickable=False)
        sticks.append((a_true, b_true, [act]))
    print(f'loaded {len(sticks)} existing sticks')

pl.enable_surface_point_picking(callback=click, show_point=False,
                                left_clicking=True, show_message=False, tolerance=0.01)
pl.add_key_event('u', undo); pl.add_key_event('r', reset); pl.add_key_event('s', save)
pl.add_text('Hemispheres SPREAD apart. Click point A then point B to place a stick.\n'
            'u undo   r reset   s save   q quit', font_size=11, position='upper_left')
pl.background_color = 'white'
print('window opening — click pairs to connect; s to save, q to quit')
pl.show()
