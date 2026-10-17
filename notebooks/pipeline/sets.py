"""Definitions for the 10 Allen CCF structure sets we print.

Each "structure set" here is one of the 9 major brain divisions (isocortex,
olfactory areas, hippocampal formation, cortical subplate, cerebral nuclei,
thalamus, hypothalamus, midbrain, hindbrain) plus the cerebellum. Each one
prints as a single assembled piece composed of its named sub-structures.

Leaf substructures (the things that get individual STLs + labels) are taken
from Allen's curated "Mouse Connectivity - Summary Structures" set
(set_id 167587189), intersected with the descendants of each major division.
The cortex and cerebellum override that with the sets we already curated.
"""
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / 'data'
STRUCTURE_SETS_DIR = DATA / 'structure_sets'
SUMMARY_SET_ID = 167587189

# Set acronym -> parent structure id(s). A list means the set merges all
# substructures of every parent into one folder + one printable piece.
MAJOR_DIVISIONS = {
    'isocortex':       315,
    'olfactory':       698,    # OLF
    'hippocampal':     1089,   # HPF
    # CNU (striatum + pallidum + extended amygdala) and CTXsp (claustrum +
    # BLA + BMA + EP + LA + PA) abut each other and print as one piece.
    'cnu_ctxsp':       [623, 703],
    'thalamus':        549,    # TH
    'hypothalamus':    1097,   # HY
    'midbrain':        313,    # MB
    'hindbrain':       1065,   # HB
    'cerebellum':      512,    # CB
}

def set_dir(name):
    """All artifacts for a structure set live under data/structure_sets/<name>/."""
    return STRUCTURE_SETS_DIR / name

def obj_dir(name):  return set_dir(name) / 'obj'
def stl_dir(name):  return set_dir(name) / 'stl'
def png_dir(name):  return set_dir(name) / 'png'
