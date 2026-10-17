"""Etch cortical-area labels into the watertight cortex via
boolean subtraction. Each label is a vector-perfect text mesh from
matplotlib's TextPath, extruded to a thin plate, positioned + oriented
at the area's outside-surface centroid, then concatenated and
subtracted in one boolean op.
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
import pyvista as pv

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
WATERTIGHT_STL = OUT_DIR / 'stl' / 'other' / '688_isocortex_watertight.stl'

# Load + scale to 5× print
root = trimesh.load(OUT_DIR / 'obj' / '997_root.obj', force='mesh')
SCALE_FACTOR = 5.0
TARGET_MM = SCALE_FACTOR * root.bounding_box.extents.max() / 1000.0
SHARED_SCALE = TARGET_MM / root.bounding_box.extents.max()
root_ctr = root.bounding_box.centroid * SHARED_SCALE
print(f'TARGET_MM = {TARGET_MM:.2f} mm  ({SCALE_FACTOR}× real life)')

cortex = trimesh.load(str(WATERTIGHT_STL), force='mesh')
print(f'watertight cortex (input units): {len(cortex.vertices):,} verts, '
      f'{len(cortex.faces):,} faces  wt={cortex.is_watertight}')
cortex.apply_scale(SHARED_SCALE)
cortex.apply_translation(-root_ctr)
print(f'scaled cortex: extents = '
      f'{[round(x, 2) for x in cortex.bounding_box.extents.tolist()]} mm')
print(f'  wt={cortex.is_watertight} vol={cortex.is_volume}')

# Annotation volume + cortical-area set
RES_UM = 25
rspc = ReferenceSpaceCache(resolution=RES_UM,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
ann, _ = rspc.get_annotation_volume()
tree = rspc.get_structure_tree()

ISO_ID = 315  # 315 = true Isocortex (688 was Cerebral cortex CTX, broader)
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
print(f'{len(iso_kid_set)} cortical areas in ontology')

# Sample annotation inward to label each cortex vertex by area.
# The watertight cortex has been scaled by SHARED_SCALE and centered;
# to get back to ANNOTATION ORIGINAL coords (Allen µm), we undo that.
verts_print = cortex.vertices                  # in print mm
verts_um = (verts_print + root_ctr) / SHARED_SCALE   # back to Allen µm
print(f'vert range in µm: '
      f'{verts_um.min(0).round(0).tolist()} → '
      f'{verts_um.max(0).round(0).tolist()}')

INWARD_STEP_UM = 400.0
verts_inward = verts_um - INWARD_STEP_UM * cortex.vertex_normals
vox = np.clip(np.round(verts_inward / RES_UM).astype(int), 0,
               np.array(ann.shape) - 1)
leaf_labels = ann[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.int64)

leaf_to_area = {0: 0}
for leaf in np.unique(leaf_labels):
    if leaf == 0: continue
    try:
        path = tree.get_structures_by_id([int(leaf)])[0]['structure_id_path']
    except Exception:
        path = []
    match = [s for s in path if int(s) in iso_kid_set]
    leaf_to_area[int(leaf)] = int(match[0]) if match else 0
vert_area = np.array([leaf_to_area[int(l)] for l in leaf_labels],
                       dtype=np.int64)

# Fill unlabeled
labeled = vert_area != 0
if (~labeled).any() and labeled.any():
    kd = cKDTree(verts_um[labeled])
    _, ii = kd.query(verts_um[~labeled])
    vert_area[~labeled] = vert_area[np.where(labeled)[0][ii]]
print(f'unique areas hit: {len(set(vert_area)) - (1 if 0 in vert_area else 0)}')

# Curated dorsal set
DORSAL_AREAS = {
    'FRP', 'MOp', 'MOs',
    'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul', 'SSp-un',
    'SSs',
    'VISp', 'VISal', 'VISam', 'VISpm', 'VISpl', 'VISl', 'VISrl', 'VISa',
    'AUDp',
    'ACAd', 'ACAv',
    'RSPd', 'RSPv',
    'ORBm',
}
area_acro = {int(sid): tree.get_structures_by_id([int(sid)])[0]['acronym']
              for sid in iso_kid_set}
dorsal_ids = {sid for sid, a in area_acro.items() if a in DORSAL_AREAS}
print(f'dorsal areas to label: {len(dorsal_ids)}')


def text_to_extruded_mesh(text, height_mm=2.0, depth_mm=2.0,
                           font_family='Helvetica'):
    """Convert text → 3D mesh via matplotlib TextPath + extrude.

    Output: trimesh, centered at origin in (x, y), with z spanning
    [-depth_mm/2, +depth_mm/2]."""
    fp = FontProperties(family=font_family, weight='bold')
    tp = TextPath((0, 0), text, size=height_mm, prop=fp)
    polys = tp.to_polygons()
    # Build polygons and union to handle holes correctly
    parts = []
    for ring in polys:
        if len(ring) < 3:
            continue
        try:
            p = Polygon(ring)
            if not p.is_valid:
                p = make_valid(p)
            if p.is_valid and p.area > 1e-9:
                parts.append(p)
        except Exception:
            pass
    if not parts:
        return None
    # symmetric_difference for even-odd fill (so 'O' has a hole)
    result = parts[0]
    for p in parts[1:]:
        result = result.symmetric_difference(p)
    if isinstance(result, Polygon):
        polylist = [result]
    elif isinstance(result, MultiPolygon):
        polylist = list(result.geoms)
    else:
        return None

    # Extrude each polygon and concatenate
    meshes = []
    for p in polylist:
        if not p.is_valid or p.area < 1e-9:
            continue
        try:
            ext = trimesh.creation.extrude_polygon(p, height=depth_mm)
            meshes.append(ext)
        except Exception:
            pass
    if not meshes:
        return None
    m = trimesh.util.concatenate(meshes)
    # Center on origin (x, y); z in [0, depth_mm] → shift to [-d/2, +d/2]
    bb = m.bounding_box
    ctr = bb.centroid.copy()
    m.apply_translation(-ctr)
    return m


# Build a label site + frame for each dorsal area
WORLD_RIGHT = np.array([0.0, 0.0, 1.0])   # +Z = right (dorsal view)
WORLD_DOWN  = np.array([1.0, 0.0, 0.0])   # +X = down (= posterior)

def tangent_frame(normal):
    u = WORLD_RIGHT - normal * (WORLD_RIGHT @ normal)
    if np.linalg.norm(u) < 1e-3:
        u = np.cross(normal, WORLD_DOWN)
    u = u / np.linalg.norm(u)
    v = WORLD_DOWN - normal * (WORLD_DOWN @ normal) - u * (WORLD_DOWN @ u)
    if np.linalg.norm(v) < 1e-3:
        v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    if v[0] < 0:
        v = -v
        u = -u
    return u, v


# Per-area: site (in scaled cortex coords) + tangent frame.
vn = cortex.vertex_normals.copy()
DORSAL_NORM_TH = -0.55
text_meshes = []
labels_placed = []
LABEL_HEIGHT_MM = 4.0
LABEL_DEPTH_MM  = 2.0
FIT_FRAC_W = 0.55
FIT_FRAC_H = 0.55
MIN_LETTER_MM = 1.8
MAX_LETTER_MM = 5.0

for area_id in sorted(dorsal_ids):
    text = area_acro.get(int(area_id), '')
    if not text:
        continue
    area_mask_full = vert_area == int(area_id)
    if area_mask_full.sum() < 50: continue
    for hemi_name, hemi_sign in (('L', -1), ('R', +1)):
        hemi_mask = (area_mask_full
                      & (cortex.vertices[:, 2] * hemi_sign > 0)
                      & (vn[:, 1] < DORSAL_NORM_TH))
        if hemi_mask.sum() < 80: continue
        hpts = cortex.vertices[hemi_mask]
        hnorms = vn[hemi_mask]
        xz = hpts[:, [0, 2]]
        xz_ext = xz.max(axis=0) - xz.min(axis=0)
        # PCA for rotation
        xz_ctr = xz.mean(axis=0); xz_c = xz - xz_ctr
        cov = np.cov(xz_c.T) if len(xz_c) > 1 else np.eye(2)
        evals, evecs = np.linalg.eigh(cov)
        pc1 = evecs[:, 1].astype(float)
        ext1 = float((xz_c @ pc1).ptp())
        ext2 = float((xz_c @ evecs[:, 0]).ptp())
        # Auto-size letter to fit
        # Letters width = text_width = (#chars * avg_char_w) ~ height * 4
        # For Helvetica-Bold, ratio is ~0.6 per character. Estimate.
        # Easier: render once at unit size, measure, then scale.
        ref_m = text_to_extruded_mesh(text, height_mm=1.0,
                                        depth_mm=LABEL_DEPTH_MM)
        if ref_m is None:
            continue
        ref_w, ref_h = ref_m.bounding_box.extents[:2]
        # Want: ref_w * letter_mm ≤ FIT_FRAC_W * ext1 (along PC1 = u)
        #       ref_h * letter_mm ≤ FIT_FRAC_H * ext2 (along PC2 = v)
        letter_mm = min(FIT_FRAC_W * ext1 / ref_w,
                         FIT_FRAC_H * ext2 / ref_h,
                         MAX_LETTER_MM)
        if letter_mm < MIN_LETTER_MM:
            continue

        # Build real text mesh at the chosen size
        text_m = text_to_extruded_mesh(text, height_mm=letter_mm,
                                         depth_mm=LABEL_DEPTH_MM)
        if text_m is None:
            continue

        # Site placement: dorsal-most vertex near XZ centroid
        d_xz = np.linalg.norm(xz_c, axis=1)
        closest = np.argsort(d_xz)[:min(80, len(hpts))]
        best = closest[np.argmin(hpts[closest, 1])]
        site = hpts[best]
        site_n = hnorms[best]

        # Tangent frame from PCA-rotated u
        pc1_3d = np.array([pc1[0], 0.0, pc1[1]])
        u_ax = pc1_3d - site_n * (pc1_3d @ site_n)
        nu = np.linalg.norm(u_ax)
        if nu < 1e-6:
            u_ax, _ = tangent_frame(site_n)
        else:
            u_ax = u_ax / nu
        v_ax = np.cross(site_n, u_ax)
        v_ax = v_ax / np.linalg.norm(v_ax)
        if v_ax[0] < 0:
            v_ax = -v_ax; u_ax = -u_ax

        # The text mesh has its (x, y) plane in the bitmap plane and
        # extrudes along +z. Build a transform: text-local (x, y, z) →
        # world (u, v, n) at site, with the text plate straddling the
        # cortex surface (z=0 sits AT the surface, so half the depth
        # is inside the cortex → that's the part that boolean-subtracts).
        # Flip y of text so it reads right-side-up in the (u, v) frame.
        text_m.vertices[:, 1] *= -1.0
        text_m.invert()
        R = np.column_stack([u_ax, v_ax, site_n])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = site
        text_m.apply_transform(T)
        text_meshes.append(text_m)
        labels_placed.append((text, hemi_name, letter_mm))

print(f'\nplaced {len(labels_placed)} label meshes')
for name, h, lm in labels_placed:
    print(f'  {name:>10s} {h}  letter={lm:.2f}mm')

# Concatenate all text meshes (each watertight by construction).
text_union = trimesh.util.concatenate(text_meshes)
print(f'\nconcatenated text mesh: {len(text_union.vertices):,} verts, '
      f'{len(text_union.faces):,} faces, wt={text_union.is_watertight}')

# Boolean subtract
print('\n=== BOOLEAN SUBTRACT (manifold3d) ===')
tic = time.time()
labeled = cortex.difference(text_union, engine='manifold')
print(f'done in {time.time()-tic:.1f}s')
print(f'result: {len(labeled.vertices):,} verts, '
      f'{len(labeled.faces):,} faces, wt={labeled.is_watertight}')

out_stl = OUT_DIR / 'stl' / 'cortex_hat' / f'allen_ccf_cortex_hat_boollabeled_{int(TARGET_MM)}mm.stl'
labeled.export(str(out_stl))
print(f'wrote {out_stl.name} ({out_stl.stat().st_size/1024/1024:.1f} MB)')

# Render
pv.OFF_SCREEN = True
pv.set_plot_theme('document')
d = labeled.bounding_box.extents.max() * 1.8

def make_poly(m):
    return pv.PolyData(
        m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

views = [
    ('dorsal',                    (0, -d, 0), (0, 0, 0), (-1, 0, 0), 1.0),
    ('dorsal (zoom posterior)',   (0, -d, 0), (20, 0, 0), (-1, 0, 0), 3.5),
    ('dorsal (zoom anterior)',    (0, -d, 0), (-25, 0, 0), (-1, 0, 0), 3.5),
    ('dorsal (zoom mid)',         (0, -d, 0), (0, 0, 0), (-1, 0, 0), 3.5),
]

pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
for k, (vname, cam, focal, up, zoom) in enumerate(views):
    pl.subplot(k // 2, k % 2)
    pl.add_mesh(make_poly(labeled), color=(0.85, 0.85, 0.85),
                 smooth_shading=True, specular=0.20, ambient=0.20, diffuse=0.95)
    pl.add_text(vname, font_size=10)
    pl.camera_position = (cam, focal, up)
    pl.reset_camera()
    pl.camera.zoom(zoom)
pl.background_color = 'white'
pl.screenshot(str(OUT_DIR / 'png' / 'cortex_hat_boollabeled.png'))
pl.close()
print('wrote cortex_hat_boollabeled.png')
