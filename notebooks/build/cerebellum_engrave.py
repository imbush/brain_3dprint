"""Cerebellum: per-area engraved (label-subtracted) STLs.

Loads each `wt_<acro>_66mm.stl`, builds text/number solids from
`label_positions.json` (one per placed label, using its site/normal/angle/
letter_mm and the optional `text` override from `number_legend.json`), and
boolean-subtracts them with manifold3d. Outputs `engraved_<acro>_66mm.stl`
per area + a 4-up preview PNG.

Run via the pipeline:
    python notebooks/pipeline/04_build_set.py cerebellum
"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv
from scipy.spatial import cKDTree
import label_common as lc

def ev(m):
    """ev() repair chain — same as the cortex builder + step-2 watertight:
    dedup -> repair non-manifold -> close holes -> reorient. Alpha-wrap as
    last-resort fallback. Returns an `is_volume` trimesh."""
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

# Per-area depth override. Default = lc.LABEL_DEPTH_MM (1.5mm).
# Thin midline lobules (vermis) + the slender VeCB are < ~3 mm thick at the
# label site — a 1.5 mm cut goes most of the way through. Shallower depth
# keeps the structure intact.
SHALLOW_DEPTH = 0.5      # mm into surface
SHALLOW_MARGIN = 0.3     # mm out of surface (for boolean robustness)
SHALLOW_AREAS = {
    'LING', 'CENT2', 'CENT3', 'CUL4, 5',
    'DEC', 'FOTU', 'PYR', 'UVU', 'NOD',
    'VeCB',
}

def build_solid(text, site, normal, letter_mm, angle, acro):
    if acro in SHALLOW_AREAS:
        d, mo = SHALLOW_DEPTH, SHALLOW_MARGIN
    else:
        d, mo = lc.LABEL_DEPTH_MM, lc.MARGIN_OUT_MM
    return lc.oriented_text(text, site, normal, letter_mm, d + mo, -d, angle)

labels = json.loads(POS.read_text())
print(f'cerebellum: {len(labels)} areas labeled, '
      f'{sum(len(v) for v in labels.values())} labels total')

# Optional finer font for crisper letter curves on small numbers
lc.set_font('/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf', 0)
lc.TEXT_CHORD_PT = 0.08

files = sorted(STL.glob(f'wt_*_{TM}mm.stl'))
if not files:
    print(f'no watertight stls in {STL}'); sys.exit(1)

engraved_meshes = []      # for preview render
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
        print(f'  {acro:>10s}: no label — copied through')
        continue
    # exactly the cortex recipe: ev() repair -> subdivide once -> boolean
    # subtract via manifold3d. The subdivision is what makes the groove rims
    # crisp (otherwise the boolean inherits the source mesh's coarse edges).
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
        # KEEP every component — bilateral areas have two hemispheres and we
        # need both. Just drop tiny boolean-debris pieces (< 0.5% of faces).
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
        print(f'  {acro:>10s}: engraved {len(solids)} label(s) -> {out.name}')
    else:
        n_fail += 1
        print(f'  {acro:>10s}: BOOLEAN FAILED ({len(solids)} solids) — wrote original')

print(f'\nsummary: {n_ok} engraved, {n_skip} no-label pass-through, '
      f'{n_fail} boolean-fail')

# ---- internal "CB core" connector: shrunken Allen CB envelope -----------
# Replaces the earlier per-nucleus struts. The user adjusts scale+position
# interactively in notebooks/place_cb_core.py and saves to cb_core.json;
# we rebuild that core mesh here and union it into the engraved areas. The
# core touches every lobule (lobules tile around it) and reaches each deep
# nucleus, fusing everything into one watertight piece.
CORE_JSON = O / 'cb_core.json'
core_parts = []
if CORE_JSON.exists():
    cfg = json.loads(CORE_JSON.read_text())
    root_mesh = trimesh.load(lc.CCF_DIR / 'obj' / '997_root.obj', force='mesh')
    shared = TM / root_mesh.bounding_box.extents.max()
    root_ctr = root_mesh.bounding_box.centroid * shared
    cb = trimesh.load(lc.CCF_DIR / 'obj' / '512_CB.obj', force='mesh')
    cb.apply_scale(shared); cb.apply_translation(-root_ctr)
    ctr = cb.centroid.copy()
    cb.vertices = (cb.vertices - ctr) * cfg['scale'] + ctr + np.asarray(cfg['translate'], float)
    cuts = cfg.get('cuts', [])
    for cut in cuts:
        cb = cb.slice_plane(np.asarray(cut['origin'], float),
                            np.asarray(cut['normal'], float), cap=True)
    cb = ev(cb)   # CB mesh isn't a volume out of the box -> manifold boolean rejects it
    core_parts.append(cb)
    print(f'\nCB core: scale={cfg["scale"]:.3f}  translate={cfg["translate"]}  '
          f'cuts={len(cuts)}  vol={cb.volume:.0f}mm³  volume_ok={cb.is_volume}')
else:
    print(f'\n[warn] no {CORE_JSON.name} — run notebooks/place_cb_core.py first; '
          f'fusion will likely leave deep nuclei disconnected.')

# ---- final fuse: lobules + deep nuclei + CB core -> one watertight STL ----
print(f'\nfusing {len(engraved_meshes)} engraved areas + {len(core_parts)} core...')
parts = [m for _, m in engraved_meshes if len(m.faces) > 0] + core_parts
fused = trimesh.boolean.union(parts, engine='manifold')
comps = fused.split(only_watertight=False)
if len(comps) > 1:
    # keep anything > 0.5 mm³ (smaller than VeCB's ~3 mm³ per side);
    # drop only true boolean debris (~zero-volume sliver components).
    keep = [c for c in comps if c.volume > 0.5]
    fused = trimesh.util.concatenate(keep)
    print(f'  fused: {len(comps)} components -> {len(keep)} kept '
          f'(lobule shell + bilateral deep nuclei)')
fused.merge_vertices()
fused_out = STL / f'cerebellum_engraved_{TM}mm.stl'
fused.export(str(fused_out))
print(f'fused -> {fused_out.relative_to(O.parent.parent)}  '
      f'({len(fused.faces):,} faces, vol={fused.volume:.0f}mm³)')

# ---- preview render: all engraved areas together, 4 views ----
import matplotlib as mpl
pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors))
def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

# Render deep nuclei as opaque coloured solids in front of the semi-transparent
# lobule shell so the user can actually see them. Without this the lobules
# (which form a closed outer surface) hide the deep nuclei from every angle.
DEEP_NUCLEI = {'FN': (0.95, 0.25, 0.25), 'IP': (0.95, 0.55, 0.10),
               'DN': (0.30, 0.60, 0.95), 'VeCB': (0.55, 0.30, 0.85)}
deep_by_acro = {acro: m for acro, m in engraved_meshes if acro in DEEP_NUCLEI}
lobule_meshes = [m for acro, m in engraved_meshes if acro not in DEEP_NUCLEI]
lobules_only = trimesh.util.concatenate(lobule_meshes)

views = [('dorsal',   ( 0,  0,  1), ( 0, -1,  0)),
         ('ventral',  ( 0,  0, -1), ( 0, -1,  0)),
         ('lateral',  ( 0,  1,  0), ( 0,  0,  1)),
         ('caudal',   ( 1,  0,  0), ( 0,  0,  1))]
pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1600, 1600))
for i, (name, view, up) in enumerate(views):
    pl.subplot(i // 2, i % 2)
    pl.add_mesh(poly(lobules_only), color=(0.78, 0.78, 0.82),
                smooth_shading=True, specular=0.2, ambient=0.3, diffuse=0.8,
                opacity=0.35)
    for acro, m in deep_by_acro.items():
        pl.add_mesh(poly(m), color=DEEP_NUCLEI[acro], smooth_shading=True,
                    specular=0.3, ambient=0.4, diffuse=0.85)
    pl.add_text(name, font_size=8, color='black', position='upper_left')
    pl.background_color = 'white'
    pl.view_vector(view, up)
    pl.reset_camera()
png = O / 'png' / 'engraved.png'
pl.screenshot(str(png))
print(f'preview -> {png.relative_to(O.parent.parent)}  '
      f'(deep nuclei colored: red=FN, orange=IP, blue=DN, purple=VeCB)')

if LEG.exists():
    leg = json.loads(LEG.read_text())
    print('\nnumber legend:')
    for n, a in leg.items():
        print(f'  {n}. {a}')
