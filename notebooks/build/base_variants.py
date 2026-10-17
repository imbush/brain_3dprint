"""Explore base-fill strategies for the mouse body STL. Produces three
variants:

  body_base_hull.stl    — base = convex hull of all brain pieces at the
                          flat-cut. Aggressive smoothing of concavities,
                          ONE piece guaranteed.
  body_base_mst.stl     — base = original silhouette + Kruskal MST strut
                          bars connecting all substantive pieces. Preserves
                          silhouette, definitively one connected base.
  body_base_compare.png — 3-panel render of underside: current / hull / mst

Run:
    python notebooks/build/base_variants.py
"""
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
import pyvista as pv
import shapely.geometry as sg
import shapely.ops as so
from shapely.geometry import LineString, MultiPoint

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
ASM  = DATA / 'mouse_brain_assembly'

print('=== loading current body ===')
body = trimesh.load(str(ASM/'body.stl'), force='mesh')
print(f'body: vol {body.volume:.0f}mm³  bbox {[round(x,1) for x in body.bounding_box.extents]}  '
      f'is_volume={body.is_volume}')
flat_y = body.bounds[1, 1]                 # ventral-most
SLAB_THICK = 2.5

# ---- helpers ------------------------------------------------------------
def ev(m):
    if m is None or len(m.faces) == 0: return m
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
    cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

def base_section_polys():
    """Get the body's base cross-section polygons (each component, exteriors only)."""
    section = body.section(plane_origin=[0, flat_y - 0.05, 0],
                            plane_normal=[0, 1, 0])
    paths, T = section.to_2D()
    polys = [sg.Polygon(p.exterior.coords) for p in paths.polygons_full
             if sg.Polygon(p.exterior.coords).area > 5]
    return polys, T

def slab_from_poly(poly2d, T):
    """Extrude a 2D polygon into a 3D slab. T's z-axis points along -y in
    world coords, so the natural extrude_polygon + transform places the
    slab at y range [flat_y - SLAB_THICK, flat_y] — exactly where we want
    it (inside the body, hugging the flat-cut bottom). No extra translation."""
    slab = trimesh.creation.extrude_polygon(poly2d, SLAB_THICK)
    slab.apply_transform(T)
    return slab

# ---- Variant A: convex hull ---------------------------------------------
print('\n=== Variant A: CONVEX HULL ===')
polys, T = base_section_polys()
print(f'  raw polys: {len(polys)}, total area {sum(p.area for p in polys):.0f}mm²')
all_pts = [pt for p in polys for pt in p.exterior.coords]
hull = MultiPoint(all_pts).convex_hull
print(f'  convex hull area: {hull.area:.0f}mm²')
hull_slab = ev(slab_from_poly(hull, T))
body_ev = ev(body)
print(f'  hull_slab is_volume={hull_slab.is_volume}  body_ev is_volume={body_ev.is_volume}')
# Try boolean union (with ev'd inputs), fall back to concatenate if it fails
try:
    body_hull = ev(trimesh.boolean.union([body_ev, hull_slab], engine='manifold'))
    print(f'  boolean union OK')
except Exception as e:
    print(f'  boolean union failed ({e}) -> falling back to concatenate')
    body_hull = trimesh.util.concatenate([body, hull_slab])
body_hull.export(str(ASM/'body_base_hull.stl'))
print(f'  -> body_base_hull.stl  vol {body_hull.volume:.0f}mm³ (slab added {hull_slab.volume:.0f}mm³)')

# ---- Variant B: Kruskal MST struts --------------------------------------
print('\n=== Variant B: KRUSKAL MST STRUTS ===')
big = sorted([p for p in polys if p.area > 100], key=lambda q: -q.area)
print(f'  substantive pieces (>100mm²): {len(big)}')
# pairwise nearest distance between every pair
edges = []  # (dist, i, j, pt_i, pt_j)
for i in range(len(big)):
    for j in range(i+1, len(big)):
        p1, p2 = so.nearest_points(big[i], big[j])
        d = p1.distance(p2)
        edges.append((d, i, j, p1, p2))
edges.sort()
print(f'  computed {len(edges)} candidate edges')

# Union-Find
parent = list(range(len(big)))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def unite(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb; return True
    return False

# Build MST
mst_strips = []
STRUT_W = 4.0
for d, i, j, p1, p2 in edges:
    if unite(i, j):
        strip = LineString([p1, p2]).buffer(STRUT_W/2, cap_style=2)
        mst_strips.append(strip)
        print(f'  MST edge: piece{i}<->piece{j} (d={d:.1f}mm)')
print(f'  MST has {len(mst_strips)} edges -> all pieces now connected')

# Combine: filled polys (exteriors) + MST strips
filled = [sg.Polygon(p.exterior.coords) for p in big]
merged = so.unary_union(filled + mst_strips)
pieces = list(merged.geoms) if hasattr(merged, 'geoms') else [merged]
print(f'  merged silhouette: {len(pieces)} piece(s)')
slabs_mst = [ev(slab_from_poly(p, T)) for p in pieces]
mst_union = trimesh.util.concatenate(slabs_mst)
body_mst = trimesh.util.concatenate([body, mst_union])
body_mst.export(str(ASM/'body_base_mst.stl'))
print(f'  -> body_base_mst.stl  vol {body_mst.volume:.0f}mm³ (slab added {mst_union.volume:.0f}mm³)')

# ---- 3-panel comparison render: underside view --------------------------
print('\n=== rendering 3-panel comparison ===')
def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

def section_stats(m):
    sec = m.section([0, m.bounds[1,1] - 0.5, 0], [0, 1, 0])
    if sec is None: return 0, 0, 0
    paths = sec.to_2D()[0]
    pp = paths.polygons_full
    main = max((q.area for q in pp), default=0)
    return len(pp), main, paths.area

stats_cur  = section_stats(body)
stats_hull = section_stats(body_hull)
stats_mst  = section_stats(body_mst)

views = [
    ('current\n(closing + bridges)', body,      stats_cur),
    ('hull\n(convex hull)',          body_hull, stats_hull),
    ('mst\n(Kruskal struts)',        body_mst,  stats_mst),
]

pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2400, 1100))
for i, (title, mesh, stats) in enumerate(views):
    pl.subplot(0, i)
    # render the BASE polygon as a flat slab on top (visualization layer)
    sec = mesh.section([0, mesh.bounds[1,1]-0.5, 0], [0, 1, 0])
    if sec is not None:
        paths_2d, _ = sec.to_2D()
        # build a 2D polygon "footprint" as a thin slab to render
        footprint_polys = paths_2d.polygons_full
        for j, p in enumerate(footprint_polys):
            try:
                slab = trimesh.creation.extrude_polygon(p, 0.5)
                slab.apply_translation([0, 0, 0])
                col = ((0.30,0.55,0.85) if j==0 else (0.95,0.45,0.45)) if j<2 else (0.55,0.85,0.55)
                pl.add_mesh(poly(slab), color=col, smooth_shading=True, opacity=0.9)
            except Exception:
                pass
    pl.view_vector((0, 0, 1), (0, 1, 0))   # 2D top-down on the base footprint
    pl.reset_camera()
    pl.add_text(f'{title}\npieces: {stats[0]}\nmain: {stats[1]:.0f}mm²\ntotal: {stats[2]:.0f}mm²',
                font_size=11, color='black', position='upper_left')
    pl.background_color = 'white'
png = ASM / 'body_base_compare.png'
pl.screenshot(str(png))
print(f'  -> {png.relative_to(DATA.parent)}')
