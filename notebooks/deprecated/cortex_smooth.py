"""Smoother cortex boundaries via diffusion of label indicator fields."""
import sys, time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
import pyvista as pv
import matplotlib as mpl
from matplotlib.colors import to_rgb

REPO_ROOT = Path('/Users/inlebush/lab/ila/microns')
OUT_DIR = REPO_ROOT / 'data' / 'ccf_stl'
CCF_VERSION = ReferenceSpaceApi.CCF_2017
rsa = ReferenceSpaceApi()

root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
# Print scale: 5x real-life mouse brain (real AP ≈ 13.2 mm → ≈ 66 mm print).
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE
print(f'TARGET_MM = {TARGET_MM:.2f} mm  ({SCALE_FACTOR}x real life)')

RES_UM = 25
rspc = ReferenceSpaceCache(resolution=RES_UM,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
ann, _ = rspc.get_annotation_volume()
tree = rspc.get_structure_tree()

ISO_ID = 688
def is_layer_acro(a):
    return (a.endswith('2/3') or a.endswith('6a') or a.endswith('6b')
             or (len(a) >= 2 and a[-1].isdigit()))
iso_kid_set = set()
for sid in tree.descendant_ids([ISO_ID])[0]:
    if sid == ISO_ID: continue
    kids = tree.child_ids([int(sid)])[0]
    if not kids: continue
    kid_acros = [tree.get_structures_by_id([int(k)])[0]['acronym']
                  for k in kids]
    if all(is_layer_acro(a) for a in kid_acros):
        iso_kid_set.add(int(sid))
print(f'{len(iso_kid_set)} cortical areas')

cortex = trimesh.load(OUT_DIR / 'obj' / '688_isocortex.obj', force='mesh')
cortex.merge_vertices()
cortex.update_faces(cortex.unique_faces())
cortex.update_faces(cortex.nondegenerate_faces())
cortex.remove_unreferenced_vertices()
nv, nf = trimesh.remesh.subdivide(cortex.vertices, cortex.faces)
cortex = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
print(f'cortex: {len(cortex.vertices):,} verts, {len(cortex.faces):,} faces')

INWARD_UM = 400.0
v_um = cortex.vertices.copy()
v_in = v_um - INWARD_UM * cortex.vertex_normals
vox = np.clip(np.round(v_in / RES_UM).astype(int), 0,
               np.array(ann.shape) - 1)
leaf = ann[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.int64)
leaf_to_area = {0: 0}
for L in np.unique(leaf):
    if L == 0: continue
    try:
        path = tree.get_structures_by_id([int(L)])[0]['structure_id_path']
    except Exception:
        path = []
    match = [s for s in path if int(s) in iso_kid_set]
    leaf_to_area[int(L)] = int(match[0]) if match else 0
vert_area = np.array([leaf_to_area[int(l)] for l in leaf], dtype=np.int64)
labeled = vert_area != 0
if (~labeled).any() and labeled.any():
    kd = cKDTree(v_um[labeled])
    _, ii = kd.query(v_um[~labeled])
    vert_area[~labeled] = vert_area[np.where(labeled)[0][ii]]

# Build mesh adjacency.
N = len(cortex.vertices)
edges = cortex.edges_unique
rows = np.concatenate([edges[:, 0], edges[:, 1]])
cols = np.concatenate([edges[:, 1], edges[:, 0]])
adj = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                  shape=(N, N))
deg = np.asarray(adj.sum(axis=1)).ravel()
inv_deg = 1.0 / np.maximum(deg, 1.0)
# Normalized adjacency: W[i,j] = 1/deg[i] if (i,j) in adj.
W = diags(inv_deg).dot(adj).tocsr()

# Diffuse the label indicator fields. After K iterations with alpha,
# each indicator becomes a smoothed scalar over the mesh; argmax gives
# the final label. This produces boundaries that approximate the
# Cheeger curves of the original labels — clean and smooth.
unique_areas = np.array(sorted(set(int(a) for a in vert_area if a > 0)))
n_areas = len(unique_areas)
print(f'{n_areas} unique areas to diffuse')

S = np.zeros((N, n_areas), dtype=np.float32)
for j, A in enumerate(unique_areas):
    S[:, j] = (vert_area == int(A)).astype(np.float32)

DIFFUSE_ITERS = 200
ALPHA = 0.5
print(f'diffusing {DIFFUSE_ITERS} iters at alpha={ALPHA}...')
tic = time.time()
for it in range(DIFFUSE_ITERS):
    S = (1 - ALPHA) * S + ALPHA * (W @ S)
print(f'  diffusion: {time.time()-tic:.1f}s')

new_idx = np.argmax(S, axis=1)
vert_area_new = unique_areas[new_idx]
n_changed = int((vert_area_new != vert_area).sum())
vert_area = vert_area_new
print(f'  diffusion changed {n_changed} labels')

# Boundary vertices = endpoints of mesh edges spanning different labels.
fv = vert_area[cortex.faces]
bf = (fv[:, 0] != fv[:, 1]) | (fv[:, 0] != fv[:, 2]) | \
     (fv[:, 1] != fv[:, 2])
bv_idx = np.unique(cortex.faces[bf].ravel())
print(f'boundary faces: {bf.sum():,}  boundary verts: {len(bv_idx):,}')

# Smooth the *positions* of boundary vertices toward their boundary-only
# neighbors. This produces a smoothed phantom curve that we'll use as
# the reference for the etch distance — the mesh itself is NOT moved
# here, only the reference curve. The result: a clean groove along a
# smooth curve, regardless of how jagged the original boundary vertex
# set is.
b_adj = adj[bv_idx][:, bv_idx]
b_deg = np.asarray(b_adj.sum(axis=1)).ravel()
b_inv_deg = 1.0 / np.maximum(b_deg, 1.0)
b_W = diags(b_inv_deg).dot(b_adj).tocsr()

b_pos = cortex.vertices[bv_idx].copy().astype(np.float32)
CURVE_SMOOTH_ITERS = 80
CURVE_ALPHA = 0.5
for _ in range(CURVE_SMOOTH_ITERS):
    b_pos = (1 - CURVE_ALPHA) * b_pos + CURVE_ALPHA * (b_W @ b_pos)

# Scale + center the (phantom) boundary curve to print coords.
b_pos_print = b_pos * SHARED_SCALE - root_ctr

# Scale the cortex mesh.
cp = cortex.copy()
cp.apply_scale(SHARED_SCALE)
cp.apply_translation(-root_ctr)

# Distance from each mesh vertex to the SMOOTHED phantom curve.
ETCH_DEPTH = 1.0
HALFW = 1.0
kd_bd = cKDTree(b_pos_print)
dist, _ = kd_bd.query(cp.vertices, k=1)
prof = np.cos(np.pi/2 * np.clip(dist / HALFW, 0, 1)) ** 2
vn = cp.vertex_normals.copy()
cp.vertices = cp.vertices - vn * (ETCH_DEPTH * prof)[:, None]

# Final polish: light Taubin smoothing on the cortex mesh — averages out
# residual high-frequency wiggle in the etched grooves without
# collapsing them (Taubin is non-shrinking).
trimesh.smoothing.filter_taubin(cp, iterations=10)

cp.export(OUT_DIR / 'stl' / 'cortex_hat' / 'allen_ccf_cortex_hat_etched_100mm.stl')
print(f'wrote cortex STL: {len(cp.faces):,} faces')

# Face color
face_area = fv[:, 0]
face_area = np.where((fv[:, 1] == fv[:, 2]) & (fv[:, 0] != fv[:, 1]),
                      fv[:, 1], face_area)
areas_present = sorted({int(a) for a in face_area if a > 0})
pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors) +
       list(mpl.colormaps['tab20c'].colors))
area_color = {sid: pal[i % len(pal)] for i, sid in enumerate(areas_present)}
face_rgb = np.array([area_color.get(int(a), (0.6, 0.6, 0.6))
                      for a in face_area])
face_rgb_solid = np.full_like(face_rgb, fill_value=0.83)

pv.OFF_SCREEN = True
pv.set_plot_theme('document')
d = cp.bounding_box.extents.max() * 2.2
views = [
    ('dorsal',  [(0, -d, 0), (0, 0, 0), (-1, 0,  0)]),
    ('right',   [(0,  0,  d), (0, 0, 0), ( 0, -1, 0)]),
    ('caudal',  [(d,  0,  0), (0, 0, 0), ( 0, -1, 0)]),
    ('oblique', [(0.7*d, -0.5*d, 0.7*d), (0, 0, 0), (0, -1, 0)]),
]
def render(face_colors, fname):
    pl = pv.Plotter(shape=(2, 2), window_size=(1600, 1400),
                     off_screen=True)
    for k, (vname, cam) in enumerate(views):
        pl.subplot(k // 2, k % 2)
        poly = pv.PolyData(
            cp.vertices,
            np.hstack([np.full((len(cp.faces), 1), 3),
                         cp.faces]).astype(np.int64))
        poly.cell_data['rgb'] = (face_colors * 255).astype(np.uint8)
        pl.add_mesh(poly, scalars='rgb', rgb=True,
                     smooth_shading=False, specular=0.20,
                     ambient=0.20, diffuse=0.95, show_edges=False)
        pl.add_text(vname, font_size=10)
        pl.camera_position = cam
        pl.reset_camera()
    pl.background_color = 'white'
    pl.screenshot(str(fname))
    pl.close()
    print(f'wrote {fname.name}')

render(face_rgb, OUT_DIR / 'png' / 'cortex_hat_etched_color.png')
render(face_rgb_solid, OUT_DIR / 'png' / 'cortex_hat_etched_solid.png')
print('done')
