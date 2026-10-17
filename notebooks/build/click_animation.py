"""Animate the click mechanic by rendering frames at successive cortex
pivot angles and stitching them into a GIF.

For each side (L, R), and each rotation angle in a press cycle, render
the body + cortex (rotated) + the kit electronics (so the post-switch
relation is visible).

Output:
  data/mouse_brain_assembly/click_animation_lateral.gif
  data/mouse_brain_assembly/click_animation_cutaway.gif

Run:
    python notebooks/build/click_animation.py
"""
from pathlib import Path
import json
import numpy as np
import trimesh
import pyvista as pv
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
ASM  = DATA / 'mouse_brain_assembly'
KIT  = DATA / 'mouse_kit' / 'extracted'

print('=== loading parts ===')
body     = trimesh.load(str(ASM/'body.stl'), force='mesh')
cortex_L = trimesh.load(str(ASM/'cortex_L.stl'), force='mesh')
cortex_R = trimesh.load(str(ASM/'cortex_R.stl'), force='mesh')
sw       = json.loads((ASM/'click_switches.json').read_text())
assembly = trimesh.load(str(KIT/'08_3D_model_for_Wireless_Mouse_Kit-002__hardware_shell_assembly___________.stl'), force='mesh')
assembly.apply_translation(-assembly.bounding_box.centroid)
R_SHELL = np.array([[0, 1, 0, 0], [0, 0,-1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)
assembly.apply_transform(R_SHELL)
comps = sorted(assembly.split(only_watertight=False), key=lambda c: -c.volume)
# drop shell halves; keep chassis only
electronics = [c for c in comps[2:] if c.volume > 100]
chassis = trimesh.util.concatenate(electronics)

# Hinge per cortex = most-caudal vertex
def hinge_xyz(cortex):
    iv = int(cortex.vertices[:, 0].argmin())
    v = cortex.vertices[iv]
    return float(v[0]), float(v[1]), float(v[2])

def rotate_about_z(mesh, pivot, deg):
    px, py, pz = pivot
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.eye(4)
    R[0,0] = c; R[0,1] = -s; R[1,0] = s; R[1,1] = c
    T = np.eye(4); T[:3,3] = [-px, -py, -pz]
    Tinv = np.eye(4); Tinv[:3,3] = [px, py, pz]
    m = mesh.copy(); m.apply_transform(Tinv @ R @ T); return m

def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

hinge_L = hinge_xyz(cortex_L)
hinge_R = hinge_xyz(cortex_R)
sw_neg  = next(v for v in sw.values() if v[2] < 0)
sw_pos  = next(v for v in sw.values() if v[2] > 0)

print(f'  hinge L at {[round(x,1) for x in hinge_L]}')
print(f'  hinge R at {[round(x,1) for x in hinge_R]}')
print(f'  switch L (z<0): {[round(x,1) for x in sw_neg]}')
print(f'  switch R (z>0): {[round(x,1) for x in sw_pos]}')

# Press cycle: 0 -> 0.6 -> 0 -> 0.6 -> 0 (two clicks). 0.6° is past engagement
ANGLES = list(np.linspace(0.0, 0.6, 12)) + list(np.linspace(0.6, 0.0, 12))[1:]
ANGLES = ANGLES + ANGLES   # two click cycles

# Camera presets — framed to show the WHOLE mouse. Body center ~ (10,-15,0),
# body diagonal ~ 200 mm, so place cameras ~280 mm out.
FOCAL = (10, -15, 0)
VIEWS = {
    'lateral': dict(
        position=(10, -15, 290),     # straight side view (camera on +z)
        focal=FOCAL,
        up=(0, -1, 0),
    ),
    'cutaway': dict(
        position=(150, -110, 230),   # 3/4 angled view from front-above
        focal=FOCAL,
        up=(0, -1, 0),
    ),
}

def render_cycle(L_pattern, R_pattern, view_cfg, fname,
                 transparent_body=False, show_kit=True, size=(1100, 800)):
    """Render an animation by REUSING a single plotter (pyvista doesn't
    like opening/closing per frame). Each frame clears the scene and
    re-adds meshes at the new pivot angle."""
    pl = pv.Plotter(off_screen=True, window_size=size)
    pv.set_new_attribute(pl, 'pickpoint', None)
    pl.background_color = 'white'
    pl.camera_position = [view_cfg['position'], view_cfg['focal'], view_cfg['up']]
    body_op = 0.35 if transparent_body else 0.85
    out = ASM / fname
    pl.open_gif(str(out), fps=12)
    for i, (aL, aR) in enumerate(zip(L_pattern, R_pattern)):
        pl.clear_actors()
        pl.add_mesh(poly(body), color=(0.85, 0.78, 0.78), smooth_shading=True,
                    opacity=body_op, specular=0.2, ambient=0.3, diffuse=0.85)
        cL = rotate_about_z(cortex_L, hinge_L, aL)
        cR = rotate_about_z(cortex_R, hinge_R, aR)
        pl.add_mesh(poly(cL), color=(0.90, 0.45, 0.45), smooth_shading=True,
                    opacity=0.95, specular=0.3, ambient=0.35, diffuse=0.85)
        pl.add_mesh(poly(cR), color=(0.45, 0.55, 0.90), smooth_shading=True,
                    opacity=0.95, specular=0.3, ambient=0.35, diffuse=0.85)
        if show_kit:
            pl.add_mesh(poly(chassis), color=(0.95, 0.85, 0.45),
                        smooth_shading=True, opacity=0.85, specular=0.4, ambient=0.4)
            for sw_xyz, col in [(sw_neg, (0.95, 0.25, 0.25)),
                                (sw_pos, (0.25, 0.45, 0.95))]:
                pl.add_mesh(pv.Sphere(radius=1.6, center=sw_xyz), color=col,
                            smooth_shading=True)
        pl.add_text(f'L={aL:.2f}°   R={aR:.2f}°',
                    font_size=14, color='black', position='lower_left')
        pl.write_frame()
        if i % 8 == 0:
            print(f'  frame {i+1}/{len(L_pattern)}: L={aL:.2f}°  R={aR:.2f}°')
    pl.close()
    print(f'  -> {out.relative_to(DATA.parent)}')

# Left clicks: L cycles 0->0.5°->0, R stays 0; then R cycles
print('\n=== rendering ===')
L_pattern = ANGLES + [0.0]*len(ANGLES)        # 48 frames: L click, then R click
R_pattern = [0.0]*len(ANGLES) + ANGLES

render_cycle(L_pattern, R_pattern, VIEWS['lateral'],
             'click_animation_lateral.gif', transparent_body=False)
render_cycle(L_pattern, R_pattern, VIEWS['cutaway'],
             'click_animation_cutaway.gif', transparent_body=True)
print('\n=== done ===')
