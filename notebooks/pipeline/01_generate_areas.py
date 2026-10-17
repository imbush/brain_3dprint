"""STEP 1 — for each structure set, download the per-area Allen meshes and
render a preview image.

Leaf substructures are taken from Allen's "Mouse Connectivity - Summary
Structures" set (167587189), intersected with descendants of the major
division. The isocortex and cerebellum reuse the curated sets we already
have on disk; everything else is downloaded fresh.

Outputs per set, under data/structure_sets/<set>/:
  obj/area_<sid>_<acro>.obj    one mesh per leaf substructure
  png/preview.png              4-up render of all areas, colored, for QA
  areas.json                   {acronym: {sid, name, ancestry}} for reference

Run:
  conda activate microns-m2
  python pipeline/01_generate_areas.py [setname [setname ...]]    # default: all
"""
import sys, json, traceback
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv
import matplotlib as mpl
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sets import (DATA, STRUCTURE_SETS_DIR, MAJOR_DIVISIONS, SUMMARY_SET_ID,
                  set_dir, obj_dir, png_dir)

CCF_DIR = DATA / 'ccf'
rspc = ReferenceSpaceCache(resolution=25, reference_space_key='annotation/ccf_2017',
                            manifest=str(CCF_DIR / 'reference_space_manifest.json'))
tree = rspc.get_structure_tree()
rsa = ReferenceSpaceApi()

# all summary-set leaves (curated by Allen for the connectivity atlas — a
# good "named-area" granularity across the whole brain)
summary = tree.get_structures_by_set_id([SUMMARY_SET_ID])
summary_ids = set(int(s['id']) for s in summary)
print(f'{len(summary_ids)} summary structures available')

def leaves_for(parent_sid):
    """Summary-set structures that lie under this major division. Accepts a
    single sid or a list (for combined sets like cnu_ctxsp)."""
    if isinstance(parent_sid, (list, tuple)):
        all_leaves = set()
        for p in parent_sid:
            all_leaves.update(leaves_for(p))
        return sorted(all_leaves)
    desc = set(int(x) for x in tree.descendant_ids([parent_sid])[0])
    return sorted(desc & summary_ids)

def download_one(sid, acro, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return 'cached'
    try:
        rsa.download_structure_mesh(structure_id=sid,
                                    ccf_version=ReferenceSpaceApi.CCF_2017,
                                    file_name=str(dst))
        return 'ok' if dst.exists() and dst.stat().st_size > 0 else 'empty'
    except Exception as e:
        if dst.exists(): dst.unlink()
        return f'fail: {type(e).__name__}'

def render_preview(name):
    files = sorted(obj_dir(name).glob('area_*.obj'))
    if not files:
        print(f'  no .obj files to render for {name}'); return
    pv.OFF_SCREEN = True; pv.set_plot_theme('document')
    pal = (list(mpl.colormaps['tab20'].colors) + list(mpl.colormaps['tab20b'].colors)
           + list(mpl.colormaps['tab20c'].colors))
    meshes = []
    for f in files:
        try:
            m = trimesh.load(str(f), force='mesh')
            if len(m.faces) > 0: meshes.append(m)
        except Exception: pass
    if not meshes: return
    allv = np.vstack([m.vertices for m in meshes])
    ctr = allv.mean(0); dd = float((allv.max(0)-allv.min(0)).max())
    views = [('dorsal', (0,-1.7*dd,0), (-1,0,0)),
             ('ventral', (0,1.7*dd,0), (1,0,0)),
             ('lateral R', (0,0,1.7*dd), (0,-1,0)),
             ('front', (-1.7*dd,0,0), (0,-1,0))]
    pl = pv.Plotter(shape=(2,2), window_size=(1800,1500), off_screen=True)
    for k,(vn,cam,up) in enumerate(views):
        pl.subplot(k//2, k%2)
        for i,m in enumerate(meshes):
            faces = np.hstack([np.full((len(m.faces),1),3), m.faces]).astype(np.int64)
            pl.add_mesh(pv.PolyData(m.vertices, faces), color=pal[i%len(pal)],
                        smooth_shading=True, specular=0.2, ambient=0.25, diffuse=0.9)
        pl.add_text(f'{name} — {vn}  ({len(meshes)} areas)', font_size=11)
        pl.camera_position = (tuple(np.asarray(cam)+ctr), tuple(ctr), up); pl.reset_camera()
    pl.background_color = 'white'
    png_dir(name).mkdir(parents=True, exist_ok=True)
    out = png_dir(name) / 'preview.png'
    pl.screenshot(str(out)); pl.close()
    print(f'  wrote {out.relative_to(DATA.parent)}')

def process(name, parent_sid):
    out = obj_dir(name); out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob('area_*.obj'))
    if existing:
        print(f'{name}: {len(existing)} obj files already present, skipping download')
    else:
        leaves = leaves_for(parent_sid)
        print(f'{name} (parent {parent_sid}): {len(leaves)} summary substructures to download')
        log = {'ok': 0, 'cached': 0, 'fail': 0, 'empty': 0}
        meta = []
        for sid in leaves:
            s = tree.get_structures_by_id([sid])[0]
            acro = s['acronym'].replace('/', '_')
            dst = out / f'area_{sid}_{acro}.obj'
            status = download_one(sid, acro, dst)
            if status.startswith('fail'):
                log['fail'] += 1
            elif status == 'empty':
                log['empty'] += 1
            else:
                log[status if status in log else 'ok'] += 1
                meta.append({'sid': sid, 'acronym': s['acronym'], 'name': s['name'],
                             'ancestry': [tree.get_structures_by_id([int(a)])[0]['acronym']
                                          for a in s['structure_id_path']]})
        (set_dir(name) / 'areas.json').write_text(json.dumps(meta, indent=2))
        print(f'  ok={log["ok"]} cached={log["cached"]} '
              f'empty={log["empty"]} fail={log["fail"]}  -> areas.json')
    render_preview(name)

if __name__ == '__main__':
    wanted = sys.argv[1:] if len(sys.argv) > 1 else list(MAJOR_DIVISIONS)
    for name in wanted:
        if name not in MAJOR_DIVISIONS:
            print(f'unknown set: {name}'); continue
        try:
            process(name, MAJOR_DIVISIONS[name])
        except Exception:
            traceback.print_exc()
