"""STEP 4 — build the one-piece printable assembly for a structure set.

The cortex uses `build_onepiece_voxsheet.py` (thin L4 voxel sheet + midline
sticks). Other sets need a connector strategy that fits their geometry —
they're not all thin sheets like cortex:

  isocortex      -> voxel L4 sheet + midline sticks  (DONE — build_onepiece_voxsheet.py)
  cerebellum     -> TODO: lobules tile densely; likely L4-sheet-equivalent
                   in the molecular layer + vermis bridge for hemisphere connect
  olfactory      -> TODO: a few blobs; minimal-spanning bowed struts likely enough
  hippocampal    -> TODO: curved C-shape; struts at the deep face
  subplate       -> TODO: few small nuclei; struts
  cerebral_nuclei-> TODO: 3D blobs (striatum, pallidum); area-carve solid body
  thalamus       -> TODO: many 3D nuclei; area-carve from TH parent solid
  hypothalamus   -> TODO: many 3D nuclei; area-carve from HY parent solid
  midbrain       -> TODO: mix; area-carve + struts
  hindbrain      -> TODO: mix; area-carve + struts

See `notebooks/strategy.md` for which fusion methods work / fail with what
inputs. Each set will end up with its own build script under pipeline/build/.

Run:
  conda activate microns-m2
  python pipeline/04_build_set.py <setname>
"""
import sys, runpy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sets import MAJOR_DIVISIONS

BUILDERS = {
    'isocortex':  '../build_onepiece_voxsheet.py',   # existing voxsheet builder
    'cerebellum': '../build/cerebellum_engrave.py',  # per-area boolean-subtract
    'cnu_ctxsp':  '../build/cnu_ctxsp_engrave.py',   # per-area boolean-subtract
    'olfactory':  '../build/olfactory_engrave.py',   # per-area boolean-subtract
    # add per-set builders as they're written:
    # 'thalamus':   'build/thalamus_carve.py',
}

if len(sys.argv) < 2 or sys.argv[1] not in MAJOR_DIVISIONS:
    print('usage: 04_build_set.py <setname> [extra args ...]')
    print('       sets:', ', '.join(MAJOR_DIVISIONS)); sys.exit(1)
name = sys.argv[1]
extra = sys.argv[2:]
if name not in BUILDERS:
    print(f'no builder yet for {name}. Strategies sketched in this script\'s docstring.')
    sys.exit(1)

nbdir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(nbdir))
import label_common as lc
lc.use_set(name)
builder = (Path(__file__).resolve().parent / BUILDERS[name]).resolve()
print(f'building {name} via {builder.relative_to(nbdir)}')
sys.argv = [str(builder)] + extra
runpy.run_path(str(builder), run_name='__main__')
