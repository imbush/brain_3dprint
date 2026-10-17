"""STEP 5 — for each pair of neighboring printed pieces (different
structure sets that touch), add matching circular magnet inlay sockets so
the assembled brain holds together with magnets.

NOT YET IMPLEMENTED — sketched here for the next agent. Inputs:
  - magnet diameter D and thickness T (TBD by user; common 6x3 or 8x3 mm)
  - the per-set FINAL STLs (data/structure_sets/<set>/stl/FINAL_*.stl)
  - the parent-region meshes in data/ccf/obj/ for adjacency reasoning

Outline of the algorithm to implement:
  1. For each pair of sets A, B whose parent meshes touch (e.g. isocortex
     <-> CTXsp, isocortex <-> CB at the cerebellar tentorium, etc.), find
     the contact patch: vertices of A's outer surface within ~0.5 mm of B.
  2. On that patch, place K magnet-socket centres at well-spaced spots
     (k-means / poisson disk) -- typically 2-3 magnets per interface so the
     pieces snap together with the right orientation.
  3. At each centre, subtract a cylinder (radius = D/2 + 0.1 mm tolerance,
     length T + 0.3 mm) recessed perpendicular to the surface, on BOTH A
     and B (mirrored so magnets align). Save the socket positions to
     `data/magnet_sockets.json` so the user can review / move them.
  4. Allow interactive review (a viewer similar to place_midline_sticks.py
     where the user clicks a socket and translates / removes it).

Useful starting points already in the repo:
  - `label_common.py`           -> `ev()` for guaranteed-volume booleans
  - `place_midline_sticks.py`   -> click-to-edit cylinder template
  - `build_onepiece_voxsheet.py`-> manifold-difference pattern for sockets

Once the user provides D, T, and approves an inter-set adjacency policy,
this script + an interactive viewer should take a couple of hours to wire.
"""
raise SystemExit(__doc__)
