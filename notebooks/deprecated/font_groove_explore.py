"""Compare label FONT and GROOVE PROFILE on real area pieces, rendered
obliquely (so the engraving actually catches light).

Grid per area: rows = fonts (SF Rounded / Arial Rounded / Comic Sans Bold),
cols = groove profile (straight / conical / spherical).
  straight   = vertical walls, flat bottom
  conical    = walls slope inward to a V
  spherical  = rounded U bottom
Outputs png/fontgroove_<area>.png for VISp (clear) and MOs (as requested).
"""
from pathlib import Path
import numpy as np
import trimesh
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
import pyvista as pv
from PIL import Image
import label_common as lc

OUT = lc.OUT_DIR; TARGET_MM = 66
TMP = OUT / 'png' / '_fg'; TMP.mkdir(parents=True, exist_ok=True)
FONTS = {'SF Rounded': '/System/Library/Fonts/SFNSRounded.ttf',
         'Arial Rounded': '/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf',
         'Comic Sans Bold': '/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf'}
PROFILES = ['straight', 'conical', 'spherical']
DEPTH = 1.6

def region_of(fp, text, tol=0.12):
    tp = TextPath((0, 0), text, size=100.0, prop=FontProperties(fname=fp))
    C = []; cur = []; pos = None
    for verts, code in tp.iter_segments(curves=True):
        v = np.asarray(verts, float).reshape(-1, 2)
        if code == MplPath.MOVETO:
            if len(cur) >= 3: C.append(np.array(cur))
            pos = v[0]; cur = [pos]
        elif code == MplPath.LINETO: pos = v[0]; cur.append(pos)
        elif code == MplPath.CURVE3: lc._flatten_quad(pos, v[0], v[1], tol, cur); pos = v[1]
        elif code == MplPath.CURVE4: lc._flatten_cubic(pos, v[0], v[1], v[2], tol, cur); pos = v[2]
        elif code == MplPath.CLOSEPOLY:
            if len(cur) >= 3: C.append(np.array(cur))
            cur = []
    if len(cur) >= 3: C.append(np.array(cur))
    P = []
    for r in C:
        try:
            p = Polygon(r)
            if not p.is_valid: p = make_valid(p)
            if p.is_valid and p.area > 1e-9: P.append(p)
        except Exception: pass
    if not P: return None
    res = P[0]
    for p in P[1:]: res = res.symmetric_difference(p)
    return res

def text_solid(region, height_mm, profile, depth=DEPTH, nl=10):
    minx, miny, maxx, maxy = region.bounds
    s = height_mm / (maxy - miny); max_off = 0.5 / s
    if profile == 'straight':
        polys = [region] if isinstance(region, Polygon) else list(region.geoms)
        m = trimesh.util.concatenate(
            [trimesh.creation.extrude_polygon(p, height=(depth+1.0)/s) for p in polys])
    else:
        slabs = []
        for k in range(nl):
            t = (k+0.5)/nl
            off = max_off*t if profile == 'conical' else max_off*(1-np.cos(t*np.pi/2))
            er = region.buffer(-off)
            if er.is_empty: continue
            dz = (depth/s)/nl
            for p in ([er] if isinstance(er, Polygon) else list(er.geoms)):
                if p.area < 1e-9: continue
                sl = trimesh.creation.extrude_polygon(p, height=dz*1.1)
                sl.apply_translation([0, 0, -dz*(k+1)]); slabs.append(sl)
        # add a thin full-size cap at the top so the groove opens flush
        polys = [region] if isinstance(region, Polygon) else list(region.geoms)
        for p in polys:
            cap = trimesh.creation.extrude_polygon(p, height=1.0/s)
            slabs.append(cap)
        try: m = trimesh.boolean.union(slabs, engine='manifold')
        except Exception: m = trimesh.util.concatenate(slabs)
    m.apply_scale(s)
    m.apply_translation([-m.bounding_box.centroid[0], -m.bounding_box.centroid[1], 0])
    m.apply_translation([0, 0, -m.vertices[:, 2].max()])      # top at z=0
    return m

def engrave(area, fp, word, profile):
    m = trimesh.load(str(OUT/'stl'/'cortical_areas'/f'wt_{area}_{TARGET_MM}mm.stl'), force='mesh')
    for _ in range(2):
        nv, nf = trimesh.remesh.subdivide(m.vertices, m.faces)
        m = trimesh.Trimesh(nv, nf, process=False)
    o = lc.outside_mask(m); hemi = np.where(o & (m.vertices[:, 2] > 0))[0]
    c = m.vertices[hemi].mean(0)
    si = hemi[np.argmin(np.linalg.norm(m.vertices[hemi]-c, axis=1))]
    site = m.vertices[si]; n = m.vertex_normals[si]
    reg = region_of(fp, word)
    # fit height to ~50% of patch v-extent
    u, v = lc.reading_frame(n); up = -v
    pv_ = (m.vertices[hemi]-site) @ up
    H = float(np.clip(0.5*(pv_.max()-pv_.min()), 2.0, 6.0))
    sol = text_solid(reg, height_mm=H, profile=profile)
    R = np.column_stack([u, up, n]); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = site
    sol.apply_translation([0, 0, 0.2]); sol.apply_transform(T)
    try:
        cut = m.difference(sol, engine='manifold')
        if cut is not None and len(cut.faces): m = cut
    except Exception: pass
    return m, site, n, up

def poly(m): return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64))
pv.OFF_SCREEN = True; pv.set_plot_theme('document')

for area in ['VISp', 'MOs']:
    panels = []
    for fname, fp in FONTS.items():
        for prof in PROFILES:
            m, site, n, up = engrave(area, fp, area, prof)
            dist = m.bounding_box.extents.max()*1.05
            pl = pv.Plotter(window_size=(720, 620), off_screen=True)
            pl.add_mesh(poly(m), color=(0.84, 0.84, 0.86), smooth_shading=False,
                        specular=0.5, specular_power=22, ambient=0.12, diffuse=0.9)
            pl.camera_position = (tuple(site+n*dist), tuple(site), tuple(up))
            pl.reset_camera(); pl.camera.zoom(1.55)
            pl.enable_anti_aliasing('ssaa'); pl.background_color = 'white'
            pl.add_text(f'{fname} / {prof}', font_size=11)
            f = TMP/f'{area}_{fname}_{prof}.png'.replace(' ', '')
            pl.screenshot(str(f)); pl.close(); panels.append(str(f))
            print(area, fname, prof)
    imgs = [Image.open(p) for p in panels]; w, h = imgs[0].size
    grid = Image.new('RGB', (w*3, h*3), 'white')
    for i, im in enumerate(imgs): grid.paste(im, ((i % 3)*w, (i//3)*h))
    grid.save(str(OUT/'png'/f'fontgroove_{area}.png')); print('wrote fontgroove_%s.png' % area)
print('done')
