"""Rotatable viewer of the final one-piece cortex. CLICK each spot where a
strut pokes out -> a red dot appears (visible confirmation) and the spot is
recorded. Press s to save -> data/ccf_brain/poke_marks.json, which I use to
reshape/remove the offending struts. u undo, r reset, q quit.
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv

OUT = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
MARKS = OUT / 'poke_marks.json'
m = trimesh.load(str(OUT/'stl'/'FINAL_onepiece_sticks_arial.stl'), force='mesh')
print(f'{len(m.faces):,} faces, watertight={m.is_watertight}')

pl = pv.Plotter(window_size=(1700, 1200))
pv.set_new_attribute(pl, 'pickpoint', None)
poly = pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces), 1), 3),
                                          m.faces]).astype(np.int64))
pl.add_mesh(poly, color=(0.82, 0.82, 0.85), smooth_shading=True,
            specular=0.3, ambient=0.2, diffuse=0.9, pickable=True)

marks = []     # (point, actor)
def click(point, *args):
    p = np.asarray(point, float)
    a = pl.add_mesh(pv.Sphere(radius=1.3, center=p), color='red',
                    pickable=False, reset_camera=False)
    marks.append((p, a))
    print(f'marked {p.round(1).tolist()}  (total {len(marks)})')

def undo():
    if not marks: return
    p, a = marks.pop(); pl.remove_actor(a); print('undo')

def reset():
    for _, a in marks: pl.remove_actor(a)
    marks.clear(); print('reset')

def save():
    MARKS.write_text(json.dumps([p.tolist() for p, _ in marks], indent=2))
    print(f'>> saved {MARKS.name} ({len(marks)} marks)')

pl.enable_surface_point_picking(callback=click, show_point=False,
                                left_clicking=True, show_message=False, tolerance=0.01)
pl.add_key_event('u', undo); pl.add_key_event('r', reset); pl.add_key_event('s', save)
pl.add_text('CLICK each strut that pokes out (red dot appears).\n'
            'u undo   r reset   s save   q quit', font_size=11, position='upper_left')
pl.background_color = 'white'
print('window opening — click pokers (red dots), s to save, q to quit')
pl.show()
