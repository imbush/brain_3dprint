"""Strut pruning viewer. Shows the solid cortex (gray) with the L4/5
struts drawn as RED cylinders. Struts that bow out of the cortex are
visible sticking out. LEFT-CLICK a strut to delete it. Save to update
struts.json, then rebuild with build_onepiece_sticks.py.

Keys:  (click a strut) delete it   u undo   s save   q quit

Run:
  conda activate microns-m2
  python notebooks/ccf_brain/prune_struts_interactive.py
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv

OUT = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
TARGET_MM = 66
SJ = OUT / 'struts.json'
R_STICK = 0.9; POKE = 1.2

if not SJ.exists():
    raise SystemExit('no struts.json — run build_onepiece_sticks.py first')
struts = json.loads(SJ.read_text())
print(f'{len(struts)} struts')

def poly(m):
    return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces), 1), 3),
                                              m.faces]).astype(np.int64))

pl = pv.Plotter(window_size=(1700, 1200))
pv.set_new_attribute(pl, 'pickpoint', None)

# cortex solid (gray, not pickable) so struts that poke out are visible
for f in sorted((OUT/'stl'/'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl')):
    pl.add_mesh(poly(trimesh.load(str(f), force='mesh')), color=(0.82, 0.82, 0.85),
                smooth_shading=True, specular=0.2, ambient=0.25, diffuse=0.9,
                opacity=1.0, pickable=False)

# struts as red cylinders (pickable). map actor -> strut index
strut_actor = {}
removed = set()
def cyl(a, b):
    a = np.array(a, float); b = np.array(b, float)
    u = b-a; L = np.linalg.norm(u); u = u/L if L > 1e-6 else u
    seg = np.array([a-POKE*u, b+POKE*u])
    m = trimesh.creation.cylinder(radius=R_STICK, segment=seg, sections=12)
    return poly(m)
for i, e in enumerate(struts):
    a = pl.add_mesh(cyl(e['A'], e['B']), color='red', pickable=True)
    strut_actor[id(a)] = (i, a)

undo_stack = []
def on_pick(*args):
    pa = getattr(pl, 'picked_actor', None)
    if pa is None: return
    hit = strut_actor.get(id(pa))
    if hit is None: return
    i, act = hit
    if i in removed: return
    removed.add(i); pl.remove_actor(act); undo_stack.append(i)
    print(f'deleted strut {i}  ({len(struts)-len(removed)} remain)')

def undo():
    if not undo_stack: print('nothing to undo'); return
    i = undo_stack.pop(); removed.discard(i)
    e = struts[i]
    a = pl.add_mesh(cyl(e['A'], e['B']), color='red', pickable=True)
    strut_actor[id(a)] = (i, a)
    print(f'restored strut {i}')

def save():
    keep = [e for i, e in enumerate(struts) if i not in removed]
    SJ.write_text(json.dumps(keep, indent=2))
    print(f'>> saved struts.json ({len(keep)} kept, {len(removed)} pruned)')

# only struts are pickable; clicking one resolves via pl.picked_actor
pl.enable_surface_point_picking(callback=lambda *a: on_pick(), show_point=False,
                                left_clicking=True, show_message=False, tolerance=0.01)
pl.add_key_event('u', undo); pl.add_key_event('s', save)
pl.add_text('Click a RED strut that pokes out to delete it.  u undo   s save   q quit',
            font_size=11, position='upper_left')
pl.background_color = 'white'
print('window opening — click pokers to delete, s to save, q to quit')
pl.show()
