"""Download Allen's per-cortical-area structure meshes (~50 areas)
and render them as an exploded view + a combined overlay (each area
in its own color, mimicking Wang-2020-style 2D area maps but in 3D).
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
import pyvista as pv
import matplotlib as mpl
from matplotlib.colors import to_rgb

OUT_DIR = Path(__file__).resolve().parent.parent / 'data'
CCF_VERSION = ReferenceSpaceApi.CCF_2017
rsa = ReferenceSpaceApi()

root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE
print(f'TARGET_MM = {TARGET_MM:.2f} mm')

rspc = ReferenceSpaceCache(resolution=25,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
tree = rspc.get_structure_tree()

ISO_ID = 315  # 315 = true Isocortex (688 was Cerebral cortex CTX, broader)
def is_layer_acro(a):
    return (a.endswith('2/3') or a.endswith('6a') or a.endswith('6b')
             or (len(a) >= 2 and a[-1].isdigit()))
# Strict heuristic: all children are layer-suffix. This catches most
# areas correctly but misses 3 that have a single mis-parented stray
# kid (Allen ontology quirk): SSp-bfd (329), AUDd (1011), RSPagl (894).
# Add those by ID explicitly.
iso_kid_set = set()
for sid in tree.descendant_ids([ISO_ID])[0]:
    if sid == ISO_ID: continue
    kids = tree.child_ids([int(sid)])[0]
    if not kids: continue
    if all(is_layer_acro(tree.get_structures_by_id([int(k)])[0]['acronym'])
            for k in kids):
        iso_kid_set.add(int(sid))
EXTRA_AREAS = {329, 1011, 894}    # SSp-bfd, AUDd, RSPagl
iso_kid_set.update(EXTRA_AREAS)
print(f'detected {len(iso_kid_set)} cortical areas '
      f'(added {len(EXTRA_AREAS)} manually for ontology quirks)')
# Skip the "star" areas (olfactory cortex fragments) for clarity.
SKIP_ACRO = {'NLOT', 'COAa', 'COApl', 'COApm', 'PAA',
             'TR', 'TTd', 'TTv', 'DP'}
area_acro = {int(sid): tree.get_structures_by_id([int(sid)])[0]['acronym']
              for sid in iso_kid_set}
keep_ids = [sid for sid, acro in area_acro.items() if acro not in SKIP_ACRO]
print(f'cortical areas to download: {len(keep_ids)}')

# Tab40-ish colormap for ~50 distinct hues
pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors) +
       list(mpl.colormaps['tab20c'].colors))

meshes = {}
for k, sid in enumerate(sorted(keep_ids)):
    acro = area_acro[sid]
    obj_path = OUT_DIR / 'obj' / f'area_{sid}_{acro}.obj'
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

    # Drop small floaters: keep components ≥ 5% of total
    components = m.split(only_watertight=False)
    if len(components) > 1:
        sizes = np.array([len(c.faces) for c in components])
        keep_mask = sizes / sizes.sum() >= 0.05
        kept = [c for c, kk in zip(components, keep_mask) if kk]
        if len(kept) == 0:
            kept = [components[int(np.argmax(sizes))]]
        m = trimesh.util.concatenate(kept)
        m.merge_vertices()
        m.update_faces(m.unique_faces())
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
    # Scale + center
    m.apply_scale(SHARED_SCALE)
    m.apply_translation(-root_ctr)
    color = pal[k % len(pal)]
    meshes[acro] = (sid, color, m)

print(f'\nloaded {len(meshes)} cortical-area meshes')

# Save each as its own STL (for multi-color per-area printing)
for acro, (sid, color, m) in meshes.items():
    stl = OUT_DIR / 'stl' / 'cortical_areas' / f'allen_ccf_area_{acro}_{int(TARGET_MM)}mm.stl'
    m.export(stl)

print('saved per-area STLs')

# Render combined + exploded grid
pv.OFF_SCREEN = True
pv.set_plot_theme('document')

def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

all_v = np.vstack([m.vertices for _, _, m in meshes.values()])
d = (all_v.max(0) - all_v.min(0)).max() * 1.8

# Combined render (all areas overlaid, different colors)
views = [
    ('dorsal',     (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
    ('lateral R',  (0,  0, d), (0, 0, 0), (0, -1, 0)),
    ('lateral L',  (0,  0, -d),(0, 0, 0), (0, -1, 0)),
    ('ventral',    (0,  d, 0), (0, 0, 0), (1, 0, 0)),
]
pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for k, (vname, cam, focal, up) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    for acro, (sid, color, m) in meshes.items():
        pl.add_mesh(make_poly(m), color=color,
                     smooth_shading=True, specular=0.20,
                     ambient=0.25, diffuse=0.90)
    pl.add_text(vname, font_size=12)
    pl.camera_position = (cam, focal, up)
    pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'per_cortical_area_combined.png'))
pl.close()
print('wrote per_cortical_area_combined.png')

# Legend image
legend_fig_w = max(8, int(np.ceil(len(meshes) / 10) * 4))
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(legend_fig_w, 6), dpi=140, facecolor='white')
ax = fig.add_subplot(111); ax.axis('off')
ncol = 5; nrow = int(np.ceil(len(meshes) / ncol))
for i, (acro, (sid, color, m)) in enumerate(meshes.items()):
    col = i % ncol
    row = i // ncol
    x = col / ncol
    y = 1.0 - (row + 0.5) / nrow
    ax.add_patch(plt.Rectangle((x, y - 0.04), 0.015, 0.04,
                                 facecolor=color, edgecolor='k', lw=0.4,
                                 transform=ax.transAxes))
    ax.text(x + 0.02, y - 0.02, acro,
             transform=ax.transAxes, va='center', fontsize=8)
fig.suptitle(f'Cortical areas — {len(meshes)} regions', fontsize=12)
fig.savefig(OUT_DIR / 'png' / 'per_cortical_area_legend.png', dpi=140,
             bbox_inches='tight')
plt.close(fig)
print('wrote per_cortical_area_legend.png')

print('done')
