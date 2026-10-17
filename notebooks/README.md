# Allen CCF 3D-printable brain — multi-set pipeline

Builds 3D-printable mouse-brain pieces from the Allen Common Coordinate
Framework (CCF). The whole brain is split into **10 structure sets** (one
per major division), each assembled as one watertight printable piece;
neighboring pieces will eventually snap together with magnet inlays.

**Read [`strategy.md`](strategy.md) first** — it explains what worked and
what failed making the cortex, so the same wheels don't get reinvented for
the other sets.

## Structure sets (one printable piece each)

| set name           | parent acronym | parent sid |
|--------------------|----------------|-----------:|
| `isocortex`        | Isocortex      | 315 |
| `cerebellum`       | CB             | 512 |
| `olfactory`        | OLF            | 698 |
| `hippocampal`      | HPF            | 1089 |
| `subplate`         | CTXsp          | 703 |
| `cerebral_nuclei`  | CNU            | 623 |
| `thalamus`         | TH             | 549 |
| `hypothalamus`     | HY             | 1097 |
| `midbrain`         | MB             | 313 |
| `hindbrain`        | HB             | 1065 |

## Data layout

```
data/
├── ccf/                                 # raw Allen downloads (parent meshes, annotation, manifest)
│   ├── obj/{315_Isocortex,997_root,...}
│   ├── annotation/
│   ├── reference_space_manifest.json
│   └── structures.json
├── structure_sets/<setname>/            # one folder per printable piece
│   ├── obj/area_<sid>_<acro>.obj        # per-area Allen meshes (step 1)
│   ├── stl/wt_<acro>_66mm.stl           # watertight cleaned (step 2)
│   ├── png/preview.png                  # QA render
│   ├── areas.json                       # acronym -> {sid, name, ancestry}
│   ├── label_positions.json             # (step 3 output)
│   ├── midline_sticks.json              # (set-specific connector state)
│   └── FINAL_*.stl                      # one-piece printable (step 4)
└── deprecated/                          # earlier intermediates, kept for reference
```

## Pipeline

```bash
conda activate microns-m2

# Step 1 — download per-area meshes + preview render (for one or all sets)
python notebooks/pipeline/01_generate_areas.py [setname [setname ...]]

# Step 2 — clean each area to a watertight printable solid (wt_<acro>_66mm.stl)
python notebooks/pipeline/02_watertight.py [setname [setname ...]]

# Step 3 — interactive: place / spin / resize labels per area on the pial surface
python notebooks/pipeline/03_label_set.py <setname>

# Step 4 — fuse all areas in a set into one watertight printable piece
#   (set-specific connector strategy — see 04_build_set.py docstring)
python notebooks/pipeline/04_build_set.py <setname>

# Step 5 — place magnet inlays at neighboring-set interfaces (NOT YET IMPLEMENTED)
python notebooks/pipeline/05_magnet_inlays.py
```

## What's kept in this folder

| file | role |
|---|---|
| `strategy.md` | **read first** — what worked, what failed, how to extend |
| `label_common.py` | shared engraving helpers (vector text, reading frame, manifold boolean). `lc.use_set(name)` switches the active structure set. |
| `place_labels_interactive.py` | the interactive labeling app (cortex was first user, now driven by `pipeline/03_label_set.py` for any set) |
| `place_midline_sticks.py` | interactive midline-stick positioning (cortex-specific midline; analogous tool can be written per set) |
| `build_onepiece_voxsheet.py` | the **cortex** one-piece builder (thin L4 voxel sheet + midline sticks); other sets will get their own builders |
| `pipeline/sets.py` | configuration: the 10 structure sets, helper paths, summary-set id |
| `pipeline/01_generate_areas.py` → `05_magnet_inlays.py` | the multi-set pipeline steps |
| `deprecated/` | earlier exploration scripts |

## Outputs already produced (isocortex)

`data/structure_sets/isocortex/`:
- `stl/FINAL_onepiece_voxsheet_arial.stl` — labeled one-piece cortex (5×, ~52 mm)
- `stl/FINAL_onepiece_voxsheet_arial_10x.stl` — same scaled 2× → ~103 mm (Bambu X1C printable)
- `stl/FINAL_onepiece_voxsheet_notext_arial*.stl` — no-label variants
- `label_positions.json`, `midline_sticks.json`, `number_legend.json` — human-placed state

## Print scale

The whole-brain root is scaled so its max extent = **66 mm (= 5× real life)**.
All sets share that frame, so individual pieces fit together at the
correct relative size. The 10× deliverables are just the 5× STLs scaled ×2.

## Environment

`conda activate microns-m2` (Python 3.11). Key deps: `allensdk`, `trimesh`,
`manifold3d`, `pymeshlab`, `pyvista==0.47.3`, `shapely`, `scipy`, `PIL`,
`matplotlib`, `numpy`.
