"""Watertight cortex v2:
- Voxelize the Allen cortex at a moderate pitch (so the remesh is ~1M
  faces, not 7.7M).
- Verify watertight + manifold.
- Optionally simplify via manifold3d.Manifold.simplify() to bring face
  count down while preserving the manifold property.
- Boolean PoC.
"""
import time
from pathlib import Path
import numpy as np
import trimesh
import manifold3d

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
SRC = OUT_DIR / 'obj' / '688_isocortex.obj'
DST = OUT_DIR / 'stl' / 'other' / '688_isocortex_watertight.stl'

m = trimesh.load(str(SRC), force='mesh')
print(f'allen cortex: {len(m.vertices):,} verts, {len(m.faces):,} faces')
print(f'bbox extents (input µm): '
      f'{[round(x) for x in m.bounding_box.extents.tolist()]}')

# Pitch 30 µm in input space → about 350 voxels along longest axis →
# remesh ~ 500-700k faces. Watertight by construction.
PITCH = 30.0
print(f'\n=== voxelize @ pitch={PITCH} µm + marching cubes ===')
tic = time.time()
v = m.voxelized(pitch=PITCH).fill()
clean = v.marching_cubes
# marching_cubes returns vertices in VOXEL INDEX coords. Scale back
# to original input units (µm) using the voxelization pitch.
clean.apply_scale(PITCH)
clean.merge_vertices()
clean.update_faces(clean.unique_faces())
clean.update_faces(clean.nondegenerate_faces())
clean.remove_unreferenced_vertices()
trimesh.smoothing.filter_taubin(clean, iterations=20)
print(f'  remesh in {time.time()-tic:.1f}s: '
      f'{len(clean.vertices):,} verts, {len(clean.faces):,} faces')
print(f'  wt={clean.is_watertight}  vol={clean.is_volume}')

# Verify manifold3d acceptance
mf = manifold3d.Mesh(clean.vertices.astype(np.float32),
                      clean.faces.astype(np.uint32))
M = manifold3d.Manifold(mf)
print(f'  manifold3d: status={M.status()}, volume={M.volume():.1f} mm³')

# Simplify via manifold3d (tolerance-based, preserves manifold)
# Tolerance in input units (µm). 50 µm tolerance = ~250 µm in print
# scale (5×), still well below 0.4 mm nozzle resolution.
print('\n=== manifold3d.simplify ===')
tic = time.time()
SIMPLIFY_TOL = 50.0
M_s = M.simplify(SIMPLIFY_TOL)
ms_data = M_s.to_mesh()
v_s = np.asarray(ms_data.vert_properties)[:, :3]
f_s = np.asarray(ms_data.tri_verts)
simplified = trimesh.Trimesh(vertices=v_s, faces=f_s, process=True)
print(f'  simplify in {time.time()-tic:.1f}s: '
      f'{len(simplified.vertices):,} verts, '
      f'{len(simplified.faces):,} faces')
print(f'  wt={simplified.is_watertight}  vol={simplified.is_volume}')

# Verify it survives a boolean
sph = trimesh.creation.icosphere(subdivisions=3, radius=300.0)
sph.apply_translation(simplified.centroid +
                       np.array([0, -simplified.bounding_box.extents[1]/2, 0]))
print(f'  sphere: {len(sph.vertices):,} verts, wt={sph.is_watertight}')
try:
    tic = time.time()
    diff = simplified.difference(sph, engine='manifold')
    print(f'  ✅ boolean.difference: {time.time()-tic:.1f}s, '
          f'{len(diff.vertices):,} verts, {len(diff.faces):,} faces, '
          f'wt={diff.is_watertight}, vol={diff.is_volume}')
except Exception as e:
    print(f'  ✗ {e}')

# Save the simplified watertight cortex
# Simplify broke watertight (trimesh round-trip issue). Use the
# unsimplified 1.47M-face mesh — it IS watertight and acceptable.
clean.export(str(DST))
print(f'\nwrote {DST.name} '
      f'({DST.stat().st_size/1024/1024:.1f} MB) — '
      f'{len(clean.vertices):,} verts, {len(clean.faces):,} faces')
print(f'   ↳ use this STL as input to boolean label etching.')

# Verify the unsimplified version still boolean-subtracts
sph = trimesh.creation.icosphere(subdivisions=3, radius=300.0)
sph.apply_translation(clean.centroid +
                       np.array([0, -clean.bounding_box.extents[1]/2, 0]))
try:
    tic = time.time()
    diff = clean.difference(sph, engine='manifold')
    print(f'\n  ✅ boolean test on unsimplified: {time.time()-tic:.1f}s, '
          f'{len(diff.vertices):,} verts, '
          f'wt={diff.is_watertight}')
except Exception as e:
    print(f'  ✗ {e}')
