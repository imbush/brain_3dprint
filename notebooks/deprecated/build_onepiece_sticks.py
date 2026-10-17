"""Cortex ONE-piece held together by short little STICKS.

Keep the area pieces at TRUE size (no inflation, no shell connector). For
every pair of nearby pieces (adjacent within a hemisphere, and the medial
areas across the midline) drop a short cylinder that pokes into both, then
union. Cylinder + clean-volume unions stay watertight (no intersection
booleans). Letters engraved LAST.

Output: stl/FINAL_onepiece_sticks_arial.stl + png/FINAL_onepiece_sticks_arial.png
"""
import json
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
from scipy.spatial import cKDTree
import pyvista as pv
import label_common as lc

O = lc.OUT_DIR; TARGET_MM = 66
lc.set_font('/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf', 0)
GAP = 6.0          # mm: connect pieces whose L4/5 points are within this
R_STICK = 0.9      # mm cylinder radius (thin, hidden)
POKE = 1.2         # mm the stick pokes into each piece
L45_DEPTH = 3.0    # mm under the pia -> layer 4/5 (mid-deep cortex)

def ev(m):
    if m.is_volume: return m
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces(); ms.meshing_remove_unreferenced_vertices()
    for _ in range(4):
        try: ms.meshing_repair_non_manifold_edges()
        except Exception: pass
        try: ms.meshing_close_holes(maxholesize=30000)
        except Exception: pass
    ms.meshing_re_orient_faces_coherently(); cm = ms.current_mesh()
    r = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
    if r.is_volume: return r
    d = float(np.linalg.norm(m.bounding_box.extents)); ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.generate_alpha_wrap(alpha=ml.PureValue(d*0.01), offset=ml.PureValue(d*0.003)); cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)

# nodes = each hemisphere component of each area. Connection points are at
# L4/5 depth: take the pia (outward) vertices and step ~3 mm INWARD along
# the normal. Struts placed there are buried mid-cortex -> hidden from the
# pia AND from the hollow underside (they don't touch the inner surface).
files = sorted((O/'stl'/'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl'))
nodes = []   # (mesh, l45pts, kdtree, centroid)
for f in files:
    m = ev(trimesh.load(str(f), force='mesh'))
    for c in m.split(only_watertight=False):
        if len(c.faces) < 50: continue
        vn = c.vertex_normals; od = c.vertices - c.centroid
        od /= np.linalg.norm(od, axis=1, keepdims=True) + 1e-12
        pia = (vn*od).sum(1) > 0.2
        pts = (c.vertices[pia] - L45_DEPTH * vn[pia]) if pia.sum() > 10 else c.vertices
        nodes.append((c, pts, cKDTree(pts), c.centroid))
print(f'{len(nodes)} hemisphere-area nodes (L4/5 connection points)')

# candidate edges between nearby nodes (closest deep point pair)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
N = len(nodes); rows=[]; cols=[]; ws=[]; ept={}
for i in range(N):
    ci, di, ti, pi = nodes[i]
    for j in range(i+1, N):
        cj, dj, tj, pj = nodes[j]
        if np.linalg.norm(pi - pj) > 22: continue
        dist, jj = tj.query(di); k = int(np.argmin(dist))
        if dist[k] > GAP: continue
        rows.append(i); cols.append(j); ws.append(float(dist[k]))
        ept[(i, j)] = (di[k], dj[jj[k]])
# minimum spanning tree -> fewest sticks that still connect everything
BOW = 2.5          # mm the strut dips INWARD at its midpoint -> never pokes out
# brain centre (areas are centred near origin); used as the "inward" direction
_allc = np.vstack([n[3] for n in nodes])
BRAIN_CTR = _allc.mean(axis=0)

def make_stick(A, B):
    """Return a list of clean cylinders forming a strut that bows inward
    (toward the brain interior) so it stays buried and never pokes the pia."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    M = (A + B) / 2.0
    inw = BRAIN_CTR - M; n = np.linalg.norm(inw)
    M2 = M + (inw/n)*BOW if n > 1e-6 else M       # midpoint pulled inward
    segs = []
    for p, q, poke_p, poke_q in [(A, M2, POKE, 0.0), (M2, B, 0.0, POKE)]:
        u = q - p; L = np.linalg.norm(u)
        if L < 1e-6: continue
        u /= L
        segs.append(trimesh.creation.cylinder(
            radius=R_STICK, segment=np.array([p - poke_p*u, q + poke_q*u]), sections=12))
    return segs

# strut set lives in struts.json so it can be pruned in the viewer. If it
# doesn't exist yet, compute the minimal spanning set and save it.
import json as _j
SJ = O/'struts.json'
if SJ.exists():
    strut_pairs = [(np.array(e['A'], float), np.array(e['B'], float))
                   for e in _j.loads(SJ.read_text())]
    print(f'loaded {len(strut_pairs)} struts from struts.json')
else:
    mst = minimum_spanning_tree(csr_matrix((ws, (rows, cols)), shape=(N, N))).tocoo()
    strut_pairs = [ept[(min(i, j), max(i, j))] for i, j in zip(mst.row, mst.col)]
    SJ.write_text(_j.dumps([{'A': a.tolist(), 'B': b.tolist()} for a, b in strut_pairs], indent=2))
    print(f'{len(strut_pairs)} auto spanning struts (saved struts.json)')

sticks = [c for a, b in strut_pairs for c in make_stick(a, b)]
print(f'{len(sticks)} cylinder segments (inward-bowed struts)')

parts = [n[0] for n in nodes] + sticks
print('unioning pieces + sticks...')
fused = trimesh.boolean.union(parts, engine='manifold')
comps = sorted(fused.split(only_watertight=False), key=lambda c: -len(c.faces))
print(f'fused components={len(comps)} (top {[len(c.faces) for c in comps[:3]]})')
fused = comps[0]

# engrave labels last
pos = json.loads((O/'label_positions.json').read_text())
kd = cKDTree(fused.vertices); fn = fused.vertex_normals
solids = []
for acro, lst in pos.items():
    for e in lst:
        _, jj = kd.query(np.asarray(e['point'], float))
        s = lc.build_text_world(str(e.get('text', acro)), fused.vertices[jj], fn[jj],
                                float(e.get('letter_mm', 2.5)), float(e.get('angle', 0.0)))
        if s is not None: solids.append(s)
print(f'subtracting {len(solids)} labels...')
labeled = fused.difference(trimesh.util.concatenate(solids), engine='manifold')
labeled = max(labeled.split(only_watertight=False), key=lambda c: len(c.faces))
labeled.merge_vertices()
print(f'final (reload-equiv): faces={len(labeled.faces):,}, '
      f'components={len(labeled.split(only_watertight=False))}, watertight={labeled.is_watertight}')
out = O/'stl'/'FINAL_onepiece_sticks_arial.stl'; labeled.export(str(out)); print('wrote', out.name)

pv.OFF_SCREEN = True; pv.set_plot_theme('document')
def poly(m): return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64))
from PIL import Image
TMP = O/'png'/'_opk'; TMP.mkdir(exist_ok=True, parents=True)
dd = labeled.bounding_box.extents.max()
views=[('dorsal',(0,-1.7*dd,0),(-1,0,0)),('ventral',(0,1.7*dd,0),(1,0,0)),
       ('lateral R',(0,0,1.7*dd),(0,-1,0)),('front',(-1.7*dd,0,0),(0,-1,0))]
panels=[]
for nm,cam,up in views:
    pl=pv.Plotter(window_size=(1200,1050),off_screen=True)
    pl.add_mesh(poly(labeled),color=(0.80,0.80,0.83),smooth_shading=True,specular=0.35,specular_power=18,ambient=0.15,diffuse=0.9)
    pl.camera_position=(cam,(0,0,0),up); pl.reset_camera(); pl.enable_anti_aliasing('ssaa')
    pl.background_color='white'; pl.add_text(nm,font_size=13)
    f=TMP/f'{nm}.png'.replace(' ',''); pl.screenshot(str(f)); pl.close(); panels.append(str(f))
imgs=[Image.open(p) for p in panels]; w,h=imgs[0].size
grid=Image.new('RGB',(w*2,h*2),'white')
for i,im in enumerate(imgs): grid.paste(im,((i%2)*w,(i//2)*h))
grid.save(str(O/'png'/'FINAL_onepiece_sticks_arial.png')); print('wrote FINAL_onepiece_sticks_arial.png')
print('done')
