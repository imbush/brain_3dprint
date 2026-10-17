"""STEP 3 — launch the interactive label-placement app pointed at a
specific structure set.

Reuses notebooks/place_labels_interactive.py (which was originally cortex-
only) by overriding `label_common.SET_DIR` for the chosen set before
launching the app.

Run:
  conda activate microns-m2
  python pipeline/03_label_set.py <setname>      # e.g. 'thalamus'
"""
import sys, runpy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sets import MAJOR_DIVISIONS

if len(sys.argv) != 2 or sys.argv[1] not in MAJOR_DIVISIONS:
    print('usage: 03_label_set.py <setname>')
    print('       sets:', ', '.join(MAJOR_DIVISIONS)); sys.exit(1)
name = sys.argv[1]

# point label_common at this set BEFORE the app imports anything from it
nbdir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(nbdir))
import label_common as lc
lc.use_set(name)
print(f'launching labeling app on structure set: {name}')
# run the app; its `OUT_DIR = lc.OUT_DIR` picks up the new SET_DIR
runpy.run_path(str(nbdir / 'place_labels_interactive.py'), run_name='__main__')
