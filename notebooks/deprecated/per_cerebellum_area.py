"""Same per-region pipeline but for the CEREBELLUM (CB, id=512) and
its subdivisions: lobules, cortical layers, deep nuclei.

Cerebellum sub-areas in Allen (canonical lobules):
- LING (Lingula I)
- CENT (Central lobule)
- CUL (Culmen)
- DEC (Declive VI)
- FOTU (Folium-tuber vermis VII)
- PYR (Pyramus VIII)
- UVU (Uvula IX)
- NOD (Nodulus X)
- SIM (Simple lobule)
- ANcr1, ANcr2 (Ansiform Crus 1, 2)
- PRM (Paramedian lobule)
- COPY (Copula pyramidis)
- PFL (Paraflocculus)
- FL (Flocculus)
+ deep cerebellar nuclei (DN, IP, FN)
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
import pyvista as pv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
CCF_VERSION = ReferenceSpaceApi.CCF_2017
rsa = ReferenceSpaceApi()
rspc = ReferenceSpaceCache(resolution=25,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
tree = rspc.get_structure_tree()

root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE

# Cerebellum CB = 512
CB_ID = 512

# Find sub-areas: descendants of CB whose children are all "molecular
# layer / Purkinje / granular" (CB has layer-like substructure but
# the lobules are the more useful level).
# Allen's CB tree:
#  CB
#  ├── CBX (Cerebellar cortex)
#  │     ├── VERM (vermal regions: LING, CENT, CUL, DEC, FOTU, PYR, UVU, NOD)
#  │     │     ├── LING ──> LINGmo, LINGpu, LINGgr
#  │     │     ├── ...
#  │     └── HEM (hemispheric: SIM, ANcr1, ANcr2, PRM, COPY, PFL, FL)
#  └── CBN (Cerebellar nuclei: FN, IP, DN, VeCB)
def is_cb_layer(a):
    # CB cortex sublayer suffixes: mo (molecular), pu (Purkinje), gr (granular)
    return a.endswith('mo') or a.endswith('pu') or a.endswith('gr')

cb_lobule_set = set()
for sid in tree.descendant_ids([CB_ID])[0]:
    if sid == CB_ID: continue
    s = tree.get_structures_by_id([sid])[0]
    kids = tree.child_ids([int(sid)])[0]
    if not kids:
        continue
    kid_acros = [tree.get_structures_by_id([int(k)])[0]['acronym']
                  for k in kids]
    if all(is_cb_layer(a) for a in kid_acros):
        cb_lobule_set.add(int(sid))

# Also include deep nuclei (which have no layer kids — they're nuclei)
NUCLEI_ACRO = {'FN', 'IP', 'DN', 'VeCB'}
for sid in tree.descendant_ids([CB_ID])[0]:
    s = tree.get_structures_by_id([sid])[0]
    if s['acronym'] in NUCLEI_ACRO:
        cb_lobule_set.add(int(sid))

print(f'detected {len(cb_lobule_set)} cerebellum sub-regions')

# Curated colour palette — distinct for each lobule
pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors))

cb_acro = {sid: tree.get_structures_by_id([sid])[0]['acronym']
            for sid in cb_lobule_set}
cb_name = {sid: tree.get_structures_by_id([sid])[0]['name']
            for sid in cb_lobule_set}

meshes = {}
for k, sid in enumerate(sorted(cb_lobule_set)):
    acro = cb_acro[sid]
    obj_path = OUT_DIR / 'obj' / f'cb_{sid}_{acro}.obj'
    if not obj_path.exists():
        try:
            rsa.download_structure_mesh(structure_id=sid,
                                          ccf_version=CCF_VERSION,
                                          file_name=str(obj_path))
        except Exception as e:
            print(f'  ✗ {acro}: {e}')
            continue
    try:
        m = trimesh.load(obj_path, force='mesh')
    except Exception as e:
        print(f'  ✗ {acro} (load): {e}')
        continue
    if len(m.faces) == 0:
        continue
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    # Drop small floaters
    comps = m.split(only_watertight=False)
    if len(comps) > 1:
        sizes = np.array([len(c.faces) for c in comps])
        keep_mask = sizes / sizes.sum() >= 0.05
        kept = [c for c, kk in zip(comps, keep_mask) if kk]
        if not kept:
            kept = [comps[int(np.argmax(sizes))]]
        m = trimesh.util.concatenate(kept)
        m.merge_vertices()
        m.update_faces(m.unique_faces())
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
    m.apply_scale(SHARED_SCALE)
    m.apply_translation(-root_ctr)
    meshes[acro] = (sid, pal[k % len(pal)], cb_name[sid], m)
    print(f'  {acro:>8s}  {len(m.vertices):>6,} verts')

# Save STLs
for acro, (sid, color, name, m) in meshes.items():
    stl = OUT_DIR / 'stl' / 'cerebellum_areas' / f'allen_ccf_cb_{acro}_{int(TARGET_MM)}mm.stl'
    m.export(stl)
print(f'\nsaved {len(meshes)} cerebellum sub-area STLs')

# Render
pv.OFF_SCREEN = True
pv.set_plot_theme('document')

def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

all_v = np.vstack([m.vertices for _, _, _, m in meshes.values()])
d = (all_v.max(0) - all_v.min(0)).max() * 1.8

views = [
    ('caudal',     (d, 0, 0), (0, 0, 0), (0, -1, 0)),
    ('dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
    ('lateral R',  (0,  0, d), (0, 0, 0), (0, -1, 0)),
    ('oblique',    (0.5*d, -0.4*d, 0.5*d), (0, 0, 0), (0, -1, 0)),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for k, (vname, cam, focal, up) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    for acro, (sid, color, name, m) in meshes.items():
        pl.add_mesh(make_poly(m), color=color,
                     smooth_shading=True, specular=0.20,
                     ambient=0.25, diffuse=0.90)
    pl.add_text(vname, font_size=12)
    pl.camera_position = (cam, focal, up)
    pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'per_cerebellum_combined.png'))
pl.close()
print('wrote per_cerebellum_combined.png')

# Legend
fig = plt.figure(figsize=(10, max(3, int(np.ceil(len(meshes) / 4)))),
                  dpi=140, facecolor='white')
ax = fig.add_subplot(111); ax.axis('off')
ncol = 3; nrow = int(np.ceil(len(meshes) / ncol))
for i, (acro, (sid, color, name, m)) in enumerate(meshes.items()):
    col = i % ncol; row = i // ncol
    x = col / ncol
    y = 1.0 - (row + 0.5) / nrow
    ax.add_patch(plt.Rectangle((x, y - 0.04), 0.015, 0.04,
                                 facecolor=color, edgecolor='k', lw=0.4,
                                 transform=ax.transAxes))
    short_name = name[:36] + ('...' if len(name) > 36 else '')
    ax.text(x + 0.02, y - 0.02, f'{acro} — {short_name}',
             transform=ax.transAxes, va='center', fontsize=8)
fig.suptitle(f'Cerebellum sub-regions — {len(meshes)}', fontsize=12)
fig.savefig(OUT_DIR / 'png' / 'per_cerebellum_legend.png', dpi=140,
             bbox_inches='tight')
plt.close(fig)
print('wrote per_cerebellum_legend.png')

# Save table
md = ['# Cerebellum sub-regions',
       '',
       '| sid | acronym | full name |',
       '|---:|---|---|']
for acro in sorted(meshes.keys()):
    sid, color, name, m = meshes[acro]
    md.append(f'| {sid} | **{acro}** | {name} |')
(OUT_DIR / 'cerebellum_areas_table.md').write_text('\n'.join(md))
print('wrote cerebellum_areas_table.md')
