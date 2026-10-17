"""cnu_ctxsp (cerebral nuclei + cortical subplate): per-area engraved STLs
+ one fused multi-component STL.

Same recipe as the cerebellum builder (ev() repair -> subdivide once ->
manifold boolean difference). No connector — these structures already abut
each other tightly so the boolean union of all engraved pieces produces a
single watertight piece (or near-single, with at most one or two small
detached components).

Run via:
    python notebooks/pipeline/04_build_set.py cnu_ctxsp
"""
import json, sys
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv
import matplotlib as mpl
from scipy.spatial import cKDTree
import label_common as lc

def ev(m):
    if m.is_volume: return m
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    for _ in range(3):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=30000)
        except Exception: pass
    try: ms.meshing_re_orient_faces_coherently()
    except Exception: pass
    cm = ms.current_mesh()
    r = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
    if r.is_volume: return r
    d = float(np.linalg.norm(m.bounding_box.extents))
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.generate_alpha_wrap(alpha=ml.PureValue(d*0.01), offset=ml.PureValue(d*0.003))
    cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

O = lc.SET_DIR; TM = lc.TARGET_MM
STL = O / 'stl'
POS = O / 'label_positions.json'
LEG = O / 'number_legend.json'

if not POS.exists():
    print(f'no labels at {POS} — run step 3 first'); sys.exit(1)

# Small / thin areas: use a shallower cut so the label doesn't slice through.
# All numbered structures here are <~3 mm thick (FS, MA, SH, MS, AAA, TRS,
# NDB, SF, LSr); use shallow depth across that group.
SHALLOW_DEPTH = 0.5
SHALLOW_MARGIN = 0.3
SHALLOW_AREAS = {'FS', 'MA', 'SH', 'MS', 'AAA', 'TRS', 'NDB', 'SF', 'LSr',
                 'BAC', 'BA', 'IA', 'PA'}

def build_solid(text, site, normal, letter_mm, angle, acro):
    if acro in SHALLOW_AREAS:
        d, mo = SHALLOW_DEPTH, SHALLOW_MARGIN
    else:
        d, mo = lc.LABEL_DEPTH_MM, lc.MARGIN_OUT_MM
    return lc.oriented_text(text, site, normal, letter_mm, d + mo, -d, angle)

labels = json.loads(POS.read_text())
print(f'cnu_ctxsp: {len(labels)} areas labeled, '
      f'{sum(len(v) for v in labels.values())} labels total')

lc.set_font('/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf', 0)
lc.TEXT_CHORD_PT = 0.08

files = sorted(STL.glob(f'wt_*_{TM}mm.stl'))
if not files:
    print(f'no watertight stls in {STL}'); sys.exit(1)

engraved_meshes = []
n_ok = n_skip = n_fail = 0
for f in files:
    acro = f.stem[len('wt_'):-len(f'_{TM}mm')]
    m_raw = trimesh.load(str(f), force='mesh')
    entries = labels.get(acro, [])
    if not entries:
        out = STL / f'engraved_{acro}_{TM}mm.stl'
        m_raw.export(str(out))
        engraved_meshes.append((acro, m_raw))
        n_skip += 1
        print(f'  {acro:>6s}: no label — copied through')
        continue
    m_clean = ev(m_raw)
    nv, nf = trimesh.remesh.subdivide(m_clean.vertices, m_clean.faces)
    m_fine = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
    kd = cKDTree(m_fine.vertices); fn = m_fine.vertex_normals
    solids = []
    for e in entries:
        text = e.get('text') or acro
        _, jj = kd.query(np.asarray(e['point'], float))
        solid = build_solid(
            text, m_fine.vertices[jj], fn[jj],
            float(e.get('letter_mm', 2.5)),
            float(e.get('angle', 0.0)), acro,
        )
        if solid is not None:
            solids.append(solid)
    cut, ok = lc.engrave(m_fine, solids)
    if ok:
        comps = cut.split(only_watertight=False)
        if len(comps) > 1:
            tot = sum(len(c.faces) for c in comps)
            comps = [c for c in comps if len(c.faces) / tot >= 0.005]
            cut = trimesh.util.concatenate(comps)
        cut.merge_vertices()
    out = STL / f'engraved_{acro}_{TM}mm.stl'
    cut.export(str(out))
    engraved_meshes.append((acro, cut))
    if ok:
        n_ok += 1
        print(f'  {acro:>6s}: engraved {len(solids)} label(s) -> {out.name}')
    else:
        n_fail += 1
        print(f'  {acro:>6s}: BOOLEAN FAILED ({len(solids)} solids) — wrote original')

print(f'\nsummary: {n_ok} engraved, {n_skip} no-label pass-through, '
      f'{n_fail} boolean-fail')

# ---- fuse into one multi-component STL ----------------------------------
print('\nfusing via manifold3d...')
parts = [m for _, m in engraved_meshes if len(m.faces) > 0]
fused = trimesh.boolean.union(parts, engine='manifold')
comps = fused.split(only_watertight=False)
if len(comps) > 1:
    keep = [c for c in comps if c.volume > 0.5]
    fused = trimesh.util.concatenate(keep)
    print(f'  fused: {len(comps)} components -> {len(keep)} kept')
fused.merge_vertices()
fused_out = STL / f'cnu_ctxsp_engraved_{TM}mm.stl'
fused.export(str(fused_out))
print(f'fused -> {fused_out.relative_to(O.parent.parent)}  '
      f'({len(fused.faces):,} faces, vol={fused.volume:.0f}mm³, '
      f'components={len(fused.split(only_watertight=False))})')

# ---- preview render: every area in its own color, 4 views ---------------
pal = (list(mpl.colormaps['tab20'].colors) + list(mpl.colormaps['tab20b'].colors))
def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

views = [('dorsal',   ( 0,  0,  1), ( 0, -1,  0)),
         ('ventral',  ( 0,  0, -1), ( 0, -1,  0)),
         ('lateral',  ( 0,  1,  0), ( 0,  0,  1)),
         ('rostral',  (-1,  0,  0), ( 0,  0,  1))]
pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1600, 1600))
for i, (name, view, up) in enumerate(views):
    pl.subplot(i // 2, i % 2)
    for j, (acro, m) in enumerate(engraved_meshes):
        pl.add_mesh(poly(m), color=pal[j % len(pal)], smooth_shading=True,
                    specular=0.2, ambient=0.3, diffuse=0.85, opacity=0.85)
    pl.add_text(name, font_size=8, color='black', position='upper_left')
    pl.background_color = 'white'
    pl.view_vector(view, up); pl.reset_camera()
png = O / 'png' / 'engraved.png'
pl.screenshot(str(png))
print(f'preview -> {png.relative_to(O.parent.parent)}')

if LEG.exists():
    leg = json.loads(LEG.read_text())
    print('\nnumber legend:')
    for n, a in leg.items():
        print(f'  {n}. {a}')
