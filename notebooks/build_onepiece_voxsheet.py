"""Cortex ONE-piece: a THIN voxelized sheet at LAYER 4 fuses the areas
WITHIN each hemisphere (hidden in the ribbon, doesn't block the hollow
inside, watertight), and a few inward-bowed STICKS join the two
hemispheres across the midline. Letters engraved last.

Output: stl/FINAL_onepiece_voxsheet_arial.stl + png/FINAL_onepiece_voxsheet_arial.png
"""
import sys, json
from pathlib import Path
import numpy as np
import trimesh
import pymeshlab as ml
from scipy.spatial import cKDTree
import pyvista as pv
import label_common as lc

O = lc.SET_DIR; CCF = lc.CCF_DIR; TM = 66
MODE = sys.argv[1] if len(sys.argv) > 1 else 'text'   # 'text' or 'notext'
lc.set_font('/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf', 0)
lc.TEXT_CHORD_PT = 0.08    # finer flattening -> crisper letter curves
L45_DEPTH = 3.0; R_STICK = 1.0; POKE = 1.2; BOW = 2.5
SHEET_OUT, SHEET_IN = 1.5, 3.0   # mm depths bounding the L4 band

def ev(m):
    if m.is_volume: return m
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices.astype(np.float64), m.faces.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_duplicate_faces(); ms.meshing_remove_unreferenced_vertices()
    for _ in range(3):
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

# thin L4 voxel sheet (clean manifold, hidden in the ribbon)
root = trimesh.load(CCF/'obj'/'997_root.obj', force='mesh')
SH = (5*root.bounding_box.extents.max()/1000)/root.bounding_box.extents.max(); CTR = root.bounding_box.centroid*SH
iso = trimesh.load(CCF/'obj'/'315_Isocortex.obj', force='mesh'); iso.merge_vertices(); iso.apply_scale(SH); iso.apply_translation(-CTR)
ic = iso.centroid; R = np.linalg.norm(iso.vertices-ic, axis=1).max()
def shr(fr): c = iso.copy(); c.vertices = ic+(c.vertices-ic)*fr; return c
band = ev(shr(1-SHEET_OUT/R)).difference(ev(shr(1-SHEET_IN/R)), engine='manifold')
sheet = ev(band.voxelized(pitch=0.5).fill().marching_cubes)
print(f'L4 voxel sheet: {len(sheet.faces):,} faces, watertight={sheet.is_watertight}')

# pieces + nodes (L4/5 points) for the midline sticks
files = sorted((O/'stl').glob(f'wt_*_{TM}mm.stl'))
pieces = []; nodes = []
for f in files:
    m = ev(trimesh.load(str(f), force='mesh'))
    for c in m.split(only_watertight=False):
        if len(c.faces) < 50: continue
        pieces.append(c)
        vn = c.vertex_normals; od = c.vertices-c.centroid; od /= np.linalg.norm(od, axis=1, keepdims=True)+1e-12
        pia = (vn*od).sum(1) > 0.2
        pts = c.vertices[pia]-L45_DEPTH*vn[pia] if pia.sum() > 10 else c.vertices
        nodes.append((c.centroid, pts))

BRAIN_CTR = np.vstack([n[0] for n in nodes]).mean(0)
R_MID, BOW_MID = 0.5, 5.0   # midline sticks: thinner
MID_EXTEND = 2.0            # extra length so sticks reach deeper into both walls
def bowed(A, B, r=R_STICK, bow=BOW):
    A = np.asarray(A, float); B = np.asarray(B, float); M = (A+B)/2
    inw = BRAIN_CTR-M; n = np.linalg.norm(inw); M2 = M+(inw/n)*bow if n > 1e-6 else M
    segs = []
    for p, q, pp, qq in [(A, M2, POKE, 0.0), (M2, B, 0.0, POKE)]:
        u = q-p; L = np.linalg.norm(u)
        if L < 1e-6: continue
        u /= L; segs.append(trimesh.creation.cylinder(radius=r, segment=np.array([p-pp*u, q+qq*u]), sections=12))
    return segs

# midline sticks: closest L4/5 pairs between L and R nodes (a few, redundant)
L = [i for i, n in enumerate(nodes) if n[0][2] < 0]
Rr = [i for i, n in enumerate(nodes) if n[0][2] >= 0]
cands = []
for i in L:
    ti = cKDTree(nodes[i][1])
    for j in Rr:
        if abs(nodes[i][0][2]-nodes[j][0][2]) > 30: continue
        d, jj = ti.query(nodes[j][1]); k = int(np.argmin(d))
        cands.append((d[k], nodes[j][1][k], nodes[i][1][jj[k]]))
cands.sort(key=lambda x: x[0])
# midline sticks live in midline_sticks.json (center C, unit dir D, half-len H,
# radius r) so they can be positioned by hand in place_midline_sticks.py.
MSJ = O/'midline_sticks.json'
if MSJ.exists():
    defs = json.loads(MSJ.read_text())
    print(f'loaded {len(defs)} midline sticks from json')
else:
    defs = []
    for d, B, A in cands[:4]:
        A = np.asarray(A, float); B = np.asarray(B, float)
        Dv = A - B; Dv = Dv/np.linalg.norm(Dv)
        defs.append({'C': ((A+B)/2).tolist(), 'D': Dv.tolist(),
                     'H': float(np.linalg.norm(A-B)/2 + POKE + MID_EXTEND), 'r': R_MID})
    MSJ.write_text(json.dumps(defs, indent=2))
    print(f'{len(defs)} auto midline sticks (saved midline_sticks.json)')
midsticks = []
for s in defs:
    C = np.asarray(s['C'], float); D = np.asarray(s['D'], float)
    midsticks.append(trimesh.creation.cylinder(
        radius=s['r'], segment=np.array([C - s['H']*D, C + s['H']*D]), sections=12))
print(f'{len(midsticks)} midline sticks')

parts = pieces + [sheet] + midsticks
fused = trimesh.boolean.union(parts, engine='manifold')
fused = max(fused.split(only_watertight=False), key=lambda c: len(c.faces))

if MODE == 'notext':
    labeled = fused
    out = O/'stl'/'FINAL_onepiece_voxsheet_notext_arial.stl'
    png = 'FINAL_onepiece_voxsheet_notext_arial.png'
else:
    # subdivide the fused mesh once so the engraved groove RIMS are fine
    # (the cause of the rough-looking text), then boolean-subtract labels.
    nv, nf = trimesh.remesh.subdivide(fused.vertices, fused.faces)
    feng = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
    pos = json.loads((O/'label_positions.json').read_text())
    kd = cKDTree(feng.vertices); fn = feng.vertex_normals
    solids = []
    for acro, lst in pos.items():
        for e in lst:
            _, jj = kd.query(np.asarray(e['point'], float))
            s = lc.build_text_world(str(e.get('text', acro)), feng.vertices[jj], fn[jj],
                                    float(e.get('letter_mm', 2.5)), float(e.get('angle', 0.0)))
            if s is not None: solids.append(s)
    labeled = feng.difference(trimesh.util.concatenate(solids), engine='manifold')
    out = O/'stl'/'FINAL_onepiece_voxsheet_arial.stl'
    png = 'FINAL_onepiece_voxsheet_arial.png'
labeled = max(labeled.split(only_watertight=False), key=lambda c: len(c.faces)); labeled.merge_vertices()
labeled.export(str(out))
rl = trimesh.load(str(out), force='mesh'); rl.merge_vertices()
print(f'RELOAD: faces={len(rl.faces):,}, components={len(rl.split(only_watertight=False))}, watertight={rl.is_watertight}')

pv.OFF_SCREEN = True; pv.set_plot_theme('document')
def poly(m): return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64))
from PIL import Image
TMP = O/'png'/'_vxs'; TMP.mkdir(exist_ok=True, parents=True)
dd = labeled.bounding_box.extents.max()
views=[('dorsal',(0,-1.7*dd,0),(-1,0,0)),('ventral',(0,1.7*dd,0),(1,0,0)),
       ('lateral R',(0,0,1.7*dd),(0,-1,0)),('front',(-1.7*dd,0,0),(0,-1,0))]
panels=[]
for nm,cam,up in views:
    pl=pv.Plotter(window_size=(1200,1050),off_screen=True)
    pl.add_mesh(poly(labeled),color=(0.80,0.80,0.83),smooth_shading=True,specular=0.35,specular_power=18,ambient=0.15,diffuse=0.9)
    pl.camera_position=(cam,(0,0,0),up); pl.reset_camera(); pl.enable_anti_aliasing('ssaa'); pl.background_color='white'; pl.add_text(nm,font_size=13)
    f=TMP/f'{nm}.png'.replace(' ',''); pl.screenshot(str(f)); pl.close(); panels.append(str(f))
ims=[Image.open(p) for p in panels]; w,h=ims[0].size; g=Image.new('RGB',(w*2,h*2),'white')
for i,im in enumerate(ims): g.paste(im,((i%2)*w,(i//2)*h))
g.save(str(O/'png'/png)); print('wrote', png)
print('done')
