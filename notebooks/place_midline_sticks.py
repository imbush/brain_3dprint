"""Position the midline sticks by hand. The cortex is shown semi-transparent
so you can see the sticks in the fissure. Select a stick and translate it in
the sagittal (midline) plane, tilt it, and change its length/width until it's
tucked inside and not poking out.

Keys (act on the ACTIVE stick, drawn yellow):
  n / p            select next / previous stick
  Left / Right     translate along AP (x)
  Up / Down        translate along DV / depth (y)
  q / e            tilt about AP axis (rotate dir about x)
  z / c            tilt about DV axis (rotate dir about y)
  - / =            shorten / lengthen
  [ / ]            thinner / thicker
  v                save -> data/ccf_brain/midline_sticks.json
  r                reload from json (discard edits)
  o                toggle cortex opacity
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv

OUT = Path(__file__).resolve().parent.parent / 'data' / 'structure_sets' / 'isocortex'
TM = 66
MSJ = OUT / 'midline_sticks.json'
if not MSJ.exists():
    raise SystemExit('no midline_sticks.json — run build_onepiece_voxsheet.py first')

def poly(m):
    return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces), 1), 3),
                                              m.faces]).astype(np.int64))
def Rx(t): c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(t): c,s=np.cos(t),np.sin(t); return np.array([[c,0,s],[0,1,0],[-s,0,c]])

pl = pv.Plotter(window_size=(1700, 1200))
pv.set_new_attribute(pl, 'pickpoint', None)
cortex_op = [0.5]
cortex_actors = []
for f in sorted((OUT/'stl').glob(f'wt_*_{TM}mm.stl')):
    a = pl.add_mesh(poly(trimesh.load(str(f), force='mesh')), color=(0.8,0.8,0.85),
                    opacity=cortex_op[0], smooth_shading=True, pickable=False)
    cortex_actors.append(a)

sticks = json.loads(MSJ.read_text())   # list of {C,D,H,r}
for s in sticks:
    s['C'] = np.asarray(s['C'], float); s['D'] = np.asarray(s['D'], float)
active = [0]; actors = [None]*len(sticks)

def draw(i):
    if actors[i] is not None: pl.remove_actor(actors[i])
    s = sticks[i]; C, D, H, r = s['C'], s['D']/np.linalg.norm(s['D']), s['H'], s['r']
    line = pv.Line(C - H*D, C + H*D)
    actors[i] = pl.add_mesh(line.tube(radius=r), color=('yellow' if i == active[0] else 'red'),
                            pickable=False, reset_camera=False)
def redraw_all():
    for i in range(len(sticks)): draw(i)
def set_active(i):
    prev = active[0]; active[0] = i % len(sticks)
    draw(prev); draw(active[0])
    s = sticks[active[0]]
    print(f'stick {active[0]}: C={s["C"].round(1).tolist()} H={s["H"]:.1f} r={s["r"]:.2f}')

def edit(dx=0,dy=0,rx=0,ry=0,dH=0,dr=0):
    s = sticks[active[0]]
    s['C'] = s['C'] + np.array([dx, dy, 0.0])
    if rx: s['D'] = Rx(np.radians(rx)) @ s['D']
    if ry: s['D'] = Ry(np.radians(ry)) @ s['D']
    s['H'] = max(2.0, s['H']+dH); s['r'] = max(0.3, s['r']+dr)
    draw(active[0])

def save():
    out = [{'C': s['C'].tolist(), 'D': (s['D']/np.linalg.norm(s['D'])).tolist(),
            'H': float(s['H']), 'r': float(s['r'])} for s in sticks]
    MSJ.write_text(json.dumps(out, indent=2)); print(f'>> saved {len(out)} sticks')

def toggle_op():
    cortex_op[0] = 1.0 if cortex_op[0] < 0.9 else 0.4
    for a in cortex_actors: a.prop.opacity = cortex_op[0]

redraw_all(); set_active(0)
for k, fn in [('n', lambda: set_active(active[0]+1)), ('p', lambda: set_active(active[0]-1)),
              ('Left', lambda: edit(dx=-1)), ('Right', lambda: edit(dx=1)),
              ('Up', lambda: edit(dy=-1)), ('Down', lambda: edit(dy=1)),
              ('f', lambda: edit(rx=-6)), ('g', lambda: edit(rx=6)),
              ('z', lambda: edit(ry=-6)), ('c', lambda: edit(ry=6)),
              ('minus', lambda: edit(dH=-1)), ('equal', lambda: edit(dH=1)),
              ('bracketleft', lambda: edit(dr=-0.1)), ('bracketright', lambda: edit(dr=0.1)),
              ('v', save), ('r', lambda: (redraw_all(), set_active(active[0]))),
              ('o', toggle_op)]:
    pl.add_key_event(k, fn)
pl.add_text('Position midline sticks.  n/p select   arrows translate (AP / depth)\n'
            'f/g tilt-AP   z/c tilt-DV   -/= length   [/] width   v save   o opacity   q quit',
            font_size=10, position='upper_left')
pl.background_color = 'white'
print('window opening — adjust sticks, v to save')
pl.show()
