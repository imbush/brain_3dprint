"""Standalone test of smooth-core construction + smooth-groove etching."""
import sys, time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import measure
import pyvista as pv
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt

REPO_ROOT = Path('/Users/inlebush/lab/ila/microns')
OUT_DIR = REPO_ROOT / 'data' / 'ccf_stl'

root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
TARGET_MM = 100.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE

RES_UM = 25
rspc = ReferenceSpaceCache(resolution=RES_UM,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
ann, _ = rspc.get_annotation_volume()
tree = rspc.get_structure_tree()

ISO_ID = 688
MAJOR_DIVISIONS = {
    623:  ('CNU',  'Cerebral nuclei'),
    1089: ('HPF',  'Hippocampal formation'),
    703:  ('CTXsp','Cortical subplate'),
    698:  ('OLF',  'Olfactory areas'),
    549:  ('TH',   'Thalamus'),
    1097: ('HY',   'Hypothalamus'),
    313:  ('MB',   'Midbrain'),
    1065: ('HB',   'Hindbrain'),
    512:  ('CB',   'Cerebellum'),
}
MAJDIV_IDS = set(MAJOR_DIVISIONS.keys())
DIV_COLORS = {
    623: '#5dd39e', 1089: '#ff9f1c', 703: '#bdb2ff',
    698: '#ffd166', 549: '#cf6cf6', 1097: '#f08080',
    313: '#5a9bff', 1065: '#118ab2', 512: '#fa8072',
}

# Core mask: brain envelope ∩ not-cortex.
print('building core mask...')
ctx_desc = list(tree.descendant_ids([ISO_ID])[0])
ctx_labels_arr = np.array(ctx_desc, dtype=ann.dtype)
brain_mask = binary_fill_holes(ann > 0)
ctx_mask = np.isin(ann, ctx_labels_arr)
core_mask = brain_mask & ~ctx_mask

core_float = gaussian_filter(core_mask.astype(np.float32), sigma=1.5)
verts, faces, _, _ = measure.marching_cubes(
    core_float, level=0.5, spacing=(RES_UM, RES_UM, RES_UM))
core = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
trimesh.smoothing.filter_taubin(core, iterations=20)

pv_in = pv.wrap(core)
pv_dec = pv_in.decimate(target_reduction=0.85)
core = trimesh.Trimesh(vertices=np.asarray(pv_dec.points),
                        faces=pv_dec.regular_faces, process=True)
print(f'  smoothed core: {len(core.faces):,} faces')

# Label vertices by major division.
verts_um = core.vertices.copy()
vox = np.clip(np.round(verts_um / RES_UM).astype(int), 0,
               np.array(ann.shape) - 1)
leaf_labels = ann[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.int64)
inward = verts_um - 200.0 * core.vertex_normals
vox2 = np.clip(np.round(inward / RES_UM).astype(int), 0,
                np.array(ann.shape) - 1)
leaf2 = ann[vox2[:, 0], vox2[:, 1], vox2[:, 2]].astype(np.int64)
leaf_labels = np.where(leaf_labels == 0, leaf2, leaf_labels)

leaf_to_majdiv = {0: 0}
for L in np.unique(leaf_labels):
    if L == 0: continue
    try:
        path = tree.get_structures_by_id([int(L)])[0]['structure_id_path']
    except Exception:
        path = []
    match = [s for s in path if int(s) in MAJDIV_IDS]
    leaf_to_majdiv[int(L)] = int(match[0]) if match else 0
core_va = np.array([leaf_to_majdiv[int(l)] for l in leaf_labels],
                    dtype=np.int64)
labeled = core_va != 0
if (~labeled).any() and labeled.any():
    kd = cKDTree(verts_um[labeled])
    _, ii = kd.query(verts_um[~labeled])
    src = np.where(labeled)[0]
    core_va[~labeled] = core_va[src[ii]]

# Smooth labels.
N = len(core.vertices)
edges = core.edges_unique
rows = np.concatenate([edges[:, 0], edges[:, 1]])
cols = np.concatenate([edges[:, 1], edges[:, 0]])
adj = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                  shape=(N, N))
for it in range(8):
    ua = np.array(sorted(set(core_va.tolist())))
    a2c = {int(a): i for i, a in enumerate(ua)}
    own = np.array([a2c[int(L)] for L in core_va])
    counts = np.zeros((N, len(ua)), dtype=np.int32)
    for j, A in enumerate(ua):
        counts[:, j] = adj @ (core_va == int(A)).astype(np.int32)
    counts[np.arange(N), own] += 1
    core_va = ua[np.argmax(counts, axis=1)]

# Scale + center
core_print = core.copy()
core_print.apply_scale(SHARED_SCALE)
core_print.apply_translation(-root_ctr)

# Smooth-groove etching (deeper than cortex so it shows through color).
fv = core_va[core_print.faces]
bf = (fv[:, 0] != fv[:, 1]) | (fv[:, 0] != fv[:, 2]) | \
     (fv[:, 1] != fv[:, 2])
bv_idx = np.unique(core_print.faces[bf].ravel())

ETCH_DEPTH = 1.5
HALFW = 0.8
kd_bd = cKDTree(core_print.vertices[bv_idx])
dist, _ = kd_bd.query(core_print.vertices, k=1)
prof = np.cos(np.pi/2 * np.clip(dist / HALFW, 0, 1)) ** 2
core_print.vertices = (core_print.vertices
                        - core_print.vertex_normals
                        * (ETCH_DEPTH * prof)[:, None])

core_print.export(OUT_DIR / 'stl' / 'other' / 'allen_ccf_core_etched_100mm.stl')

core_face_div = fv[:, 0]
core_face_div = np.where((fv[:, 1] == fv[:, 2]) & (fv[:, 0] != fv[:, 1]),
                          fv[:, 1], core_face_div)
core_face_rgb_color = np.array([to_rgb(DIV_COLORS.get(int(d), '#cccccc'))
                                  for d in core_face_div])
core_face_rgb_solid = np.full_like(core_face_rgb_color, 0.85)

# OPTION B: sagittal-cut half mesh + fine-rasterized cut-face slab.
midline_z_um = ann.shape[2] / 2 * RES_UM
half = core.copy().slice_plane(plane_origin=[0, 0, midline_z_um],
                                plane_normal=[0, 0, -1], cap=True)
half_print = half.copy()
half_print.apply_scale(SHARED_SCALE)
half_print.apply_translation(-root_ctr)
half_print.export(OUT_DIR / 'stl' / 'other' / 'allen_ccf_core_sagittal_half_100mm.stl')

# Identify which faces of the (un-scaled) half-mesh are on the cut plane,
# and DROP them so the cut face renders as a clean cream colour from
# behind; we replace them with a fine rasterized slab.
half_tri_z = half.triangles[..., 2]
on_cut = np.abs(half_tri_z.mean(axis=1) - midline_z_um) < (RES_UM * 0.6)
keep_idx = np.where(~on_cut)[0]
half_outer = trimesh.Trimesh(vertices=half.vertices.copy(),
                              faces=half.faces[keep_idx].copy(),
                              process=False)
half_outer.apply_scale(SHARED_SCALE)
half_outer.apply_translation(-root_ctr)

# Build a fine 2D label slab at z = midline_z. For each voxel of the
# annotation slice at midline that's INSIDE the core mask, emit a small
# quad (two triangles) colored by major division.
print('building fine cut-face slab...')
slice_z_idx = int(round(midline_z_um / RES_UM))
slice_ann = ann[..., slice_z_idx]
# Lookup table: leaf → major-division id (or 0).
unique_slice_leaves = np.unique(slice_ann)
slice_l2d = np.zeros(int(unique_slice_leaves.max()) + 1, dtype=np.int64)
for L in unique_slice_leaves:
    if L == 0: continue
    if int(L) in leaf_to_majdiv:
        slice_l2d[int(L)] = leaf_to_majdiv[int(L)]
    else:
        try:
            path = tree.get_structures_by_id([int(L)])[0]['structure_id_path']
        except Exception:
            path = []
        match = [s for s in path if int(s) in MAJDIV_IDS]
        slice_l2d[int(L)] = int(match[0]) if match else 0
# For the cut-face slab, ALSO include cortex (so the whole brain
# cross-section is colored, not just the core). Add CTX as a 10th
# "division" for slab coloring only.
SLICE_DIV_COLORS = dict(DIV_COLORS)
SLICE_DIV_COLORS[688] = '#d8b4fe'  # pale purple for isocortex
ctx_leaf_set = set(ctx_desc)
for L in unique_slice_leaves:
    if L == 0: continue
    if int(L) in ctx_leaf_set and slice_l2d[int(L)] == 0:
        slice_l2d[int(L)] = 688
slice_div = slice_l2d[slice_ann]
# Now keep every in-brain midline voxel (whether core or cortex).
brain_slice = brain_mask[..., slice_z_idx]
slab_voxels = np.argwhere(brain_slice & (slice_div > 0))   # (M, 2)
print(f'  slab voxels: {len(slab_voxels):,}')

H = RES_UM
verts_slab = []
faces_slab = []
slab_rgb = []
for idx, (i, j) in enumerate(slab_voxels):
    x0, x1 = i * H, (i + 1) * H
    y0, y1 = j * H, (j + 1) * H
    z = midline_z_um
    base = len(verts_slab)
    verts_slab.extend([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])
    faces_slab.append([base, base + 1, base + 2])
    faces_slab.append([base, base + 2, base + 3])
    color = to_rgb(SLICE_DIV_COLORS[int(slice_div[i, j])])
    slab_rgb.append(color)
    slab_rgb.append(color)
verts_slab = np.array(verts_slab, dtype=float)
faces_slab = np.array(faces_slab, dtype=np.int64)
slab_rgb = np.array(slab_rgb)
slab = trimesh.Trimesh(vertices=verts_slab, faces=faces_slab, process=False)
slab.apply_scale(SHARED_SCALE)
slab.apply_translation(-root_ctr)
# Nudge the slab a tiny bit out of the cut plane so it doesn't z-fight.
slab.vertices[:, 2] -= 0.05
print(f'  slab: {len(slab.faces):,} colored triangles')

# RENDER
pv.OFF_SCREEN = True
pv.set_plot_theme('document')

def make_poly(mesh, face_rgb):
    poly = pv.PolyData(
        mesh.vertices,
        np.hstack([np.full((len(mesh.faces), 1), 3),
                     mesh.faces]).astype(np.int64))
    poly.cell_data['rgb'] = (face_rgb * 255).astype(np.uint8)
    return poly

d = core_print.bounding_box.extents.max() * 2.2

# Option A: colored + solid (grooves visible) in 2x4 grid.
views_a = [
    ('dorsal',  [(0, -d, 0), (0, 0, 0), (-1, 0,  0)]),
    ('right',   [(0,  0,  d), (0, 0, 0), ( 0, -1, 0)]),
    ('caudal',  [(d,  0,  0), (0, 0, 0), ( 0, -1, 0)]),
    ('oblique', [(0.7*d, -0.5*d, 0.7*d), (0, 0, 0), (0, -1, 0)]),
]
pl = pv.Plotter(shape=(2, 4), window_size=(2200, 1200),
                 off_screen=True)
for k, (vname, cam) in enumerate(views_a):
    pl.subplot(0, k)
    pl.add_mesh(make_poly(core_print, core_face_rgb_color),
                 scalars='rgb', rgb=True,
                 smooth_shading=True, specular=0.20,
                 ambient=0.22, diffuse=0.95)
    pl.add_text(f'colored — {vname}', font_size=9)
    pl.camera_position = cam; pl.reset_camera()
    pl.subplot(1, k)
    pl.add_mesh(make_poly(core_print, core_face_rgb_solid),
                 scalars='rgb', rgb=True,
                 smooth_shading=True, specular=0.20,
                 ambient=0.20, diffuse=0.95)
    pl.add_text(f'solid (grooves) — {vname}', font_size=9)
    pl.camera_position = cam; pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'core_optA_surface_engraved.png'))
pl.close()
print('wrote core_optA_surface_engraved.png')

# Option B: half-mesh + colored slab.
#   Cut face is at z = midline_z_um*SHARED_SCALE - root_ctr[2] ≈ 0.
#   The kept half has z < that value; the cut face NORMAL points to +Z.
views_b = [
    ('cut face',     [(0, 0,  d), (0, 0, 0), (0, -1, 0)]),
    ('oblique cut',  [(0.3*d, -0.3*d,  0.9*d), (0, 0, 0), (0, -1, 0)]),
    ('outer (left)', [(0, 0, -d), (0, 0, 0), (0, -1, 0)]),
    ('dorsal',       [(0, -d, 0), (0, 0, 0), (-1, 0, 0)]),
]
pl = pv.Plotter(shape=(1, 4), window_size=(2200, 700), off_screen=True)
cream_rgb = np.full((len(half_outer.faces), 3), (0.91, 0.88, 0.83))
for k, (vname, cam) in enumerate(views_b):
    pl.subplot(0, k)
    pl.add_mesh(make_poly(half_outer, cream_rgb),
                 scalars='rgb', rgb=True,
                 smooth_shading=True, specular=0.15,
                 ambient=0.30, diffuse=0.85)
    pl.add_mesh(make_poly(slab, slab_rgb),
                 scalars='rgb', rgb=True,
                 smooth_shading=False, specular=0.05,
                 ambient=0.55, diffuse=0.55)
    pl.add_text(vname, font_size=10)
    pl.camera_position = cam; pl.reset_camera()
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'core_optB_sagittal_half.png'))
pl.close()
print('wrote core_optB_sagittal_half.png')

print('done')
