"""Simulate the cortex-button click mechanics.

For each angle in a sweep, pivot the cortex hemisphere about its hinge
(z-axis at the bridge), then check:
  - Does the cortex hemisphere intersect the body's cortex pocket wall?
    (binding — bad)
  - Does the post tip reach the switch top?  (engagement — needed)
  - Does the post bind in the post hole?  (lateral motion — small)

Sweep 0–3° in 0.25° steps. Report binding angles and engagement angles.

Run:
    python notebooks/build/click_simulation.py
"""
from pathlib import Path
import json
import numpy as np
import trimesh
from trimesh.collision import CollisionManager

ROOT = Path(__file__).resolve().parent.parent.parent
ASM  = ROOT / 'data' / 'mouse_brain_assembly'

print('=== loading parts ===')
body     = trimesh.load(str(ASM/'body.stl'), force='mesh')
cortex_L = trimesh.load(str(ASM/'cortex_L.stl'), force='mesh')
cortex_R = trimesh.load(str(ASM/'cortex_R.stl'), force='mesh')
sw       = json.loads((ASM/'click_switches.json').read_text())
print(f'  body vol {body.volume:.0f}mm³  bbox {[round(x,1) for x in body.bounding_box.extents]}')
print(f'  cortex_L vol {cortex_L.volume:.0f}mm³  cortex_R vol {cortex_R.volume:.0f}mm³')
print(f'  switches: {sw}')

# Hinge axis = z-axis at the back of each cortex hemisphere
# (location: cortex's most-caudal vertex, used by the build script)
def hinge_xyz(cortex):
    iv = int(cortex.vertices[:, 0].argmin())
    v = cortex.vertices[iv]
    return float(v[0]), float(v[1]), float(v[2])

# Post tip = the most-ventral (largest y) vertex INSIDE a small radius
# around the switch xz position
def post_tip_y(cortex, sx, sz, r=3.0):
    d2 = (cortex.vertices[:, 0] - sx)**2 + (cortex.vertices[:, 2] - sz)**2
    candidates = cortex.vertices[d2 < r*r]
    if len(candidates) == 0: return None
    return float(candidates[:, 1].max())

def rotate_about_z_axis(mesh, pivot_xyz, angle_deg):
    """Rotate mesh by angle_deg about the z-axis through pivot."""
    px, py, pz = pivot_xyz
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.eye(4)
    R[0, 0] = c;  R[0, 1] = -s
    R[1, 0] = s;  R[1, 1] =  c
    T = np.eye(4); T[:3, 3] = [-px, -py, -pz]
    Tinv = np.eye(4); Tinv[:3, 3] = [px, py, pz]
    M = Tinv @ R @ T
    m = mesh.copy(); m.apply_transform(M); return m

def collision_count(rotated_cortex, body):
    """How many vertices of rotated cortex are INSIDE the body's solid
    material? At rest the cortex sits in the body's pocket (empty space, not
    solid material) so this should be ~0. If the cortex rotates and binds
    into the body wall, vertices land inside body material -> count > 0."""
    try:
        inside = body.contains(rotated_cortex.vertices)
        return int(inside.sum())
    except Exception as e:
        return -1   # error sentinel

# match switches to hemispheres by sign of z
sw_pos = next(v for v in sw.values() if v[2] > 0)   # R hemisphere switch
sw_neg = next(v for v in sw.values() if v[2] < 0)   # L hemisphere switch

for label, cortex, switch in [('L', cortex_L, sw_neg), ('R', cortex_R, sw_pos)]:
    print(f'\n=== simulating cortex_{label} click ===')
    hinge = hinge_xyz(cortex)
    rest_tip = post_tip_y(cortex, switch[0], switch[2])
    sw_top = switch[1] - 0.3   # POST_GAP = 0.3 mm
    print(f'  hinge at ({hinge[0]:+.1f},{hinge[1]:+.1f},{hinge[2]:+.1f})')
    print(f'  switch xyz=({switch[0]:+.1f},{switch[1]:+.1f},{switch[2]:+.1f})  '
          f'switch top (post target)={sw_top:+.2f}')
    print(f'  rest post-tip y = {rest_tip:+.2f}  -> initial gap = {switch[1]-rest_tip:.2f} mm')
    print(f'  {"angle":>6s}  {"front-travel":>13s}  {"tip y":>8s}  {"engaged?":>10s}  '
          f'{"verts in body":>14s}')
    rest_count = collision_count(cortex, body)
    print(f'  (rest baseline: {rest_count} cortex verts inside body material)')
    for ang in np.arange(0.0, 3.01, 0.25):
        rotated = rotate_about_z_axis(cortex, hinge, ang)
        tip_y = post_tip_y(rotated, switch[0], switch[2])
        # front travel = how much the most-ventral cortex point (front-bottom) moves
        front_travel = rotated.bounds[1, 1] - cortex.bounds[1, 1]
        engaged = '✔ press' if tip_y is not None and tip_y > sw_top else ''
        cnt = collision_count(rotated, body)
        delta = cnt - rest_count
        warn = ' BIND' if delta > 50 else ''
        tip_str = f'{tip_y:+.2f}' if tip_y is not None else 'NA'
        print(f'  {ang:>5.2f}°  {front_travel:>12.2f}mm  {tip_str:>8s}  {engaged:>10s}  '
              f'{cnt:>8d} (+{delta}){warn}')
        if delta > 200:
            print(f'    severe binding (>200 extra verts) — stop')
            break

print('\n=== done ===')
