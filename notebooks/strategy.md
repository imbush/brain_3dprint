# Strategy — how the final cortex was built, and how to apply this to other CCF regions

This document captures everything a future agent (human or Claude) needs to
know to (a) understand why the current pipeline is shaped the way it is and
(b) extend it to other CCF regions (cerebellum, thalamus, etc.). The "why"
matters more than the "what" because most of the work was learning what
*doesn't* work.

---

## TL;DR — what produces the final cortex

```
per_cortical_area.py        # downloads the 43 Allen area meshes
   ↓
watertight_areas_final.py   # cleans each into a closed solid (wt_<acro>_66mm.stl)
   ↓
place_labels_interactive.py # human picks the label spot + spin + size per area
   ↓
place_midline_sticks.py     # human positions the few midline connecting struts
   ↓
build_onepiece_voxsheet.py  # unions everything into one watertight printable solid,
                            # engraves labels last on a SUBDIVIDED mesh for clean rims
```

The pipeline is **scripts + small JSON state** (`label_positions.json`,
`midline_sticks.json`, `number_legend.json`). The JSONs are the "human
input"; the scripts are deterministic given those.

---

## The hard problems and what actually worked

### 1. Allen per-area meshes are already smooth + closed (mostly)

The 43 per-cortical-area .obj meshes from Allen are watertight closed
volumes already. The earlier mistake was voxelizing + unioning them (the
"wavy" cortex), which destroyed Allen's smooth shape. **Use the area meshes
directly** with minimal cleanup (dedupe, drop floaters, keep components
≥ 5%, close holes only if needed) — see `watertight_areas_final.py`. Result
is 43 clean, smooth, watertight solids per the original Allen geometry.

`RSPagl` is the one persistent troublemaker; it sometimes needs alpha-wrap.
The `ev()` helper in `build_onepiece_voxsheet.py` (and similar in
`label_common`) is the robust repair sequence: dedupe → repair-nonmanifold-edges
(×3) → close-holes → re-orient → if still not `is_volume`, alpha-wrap with
`alpha=diag*0.01`, `offset=diag*0.003`. Use that anywhere you need to
guarantee a manifold input to `manifold3d`.

### 2. Engraved text via boolean must be vector + chordal-flattened

The text is a vector outline (matplotlib `TextPath`) flattened to a polygon
with **adaptive De Casteljau Bézier subdivision** at chord tolerance
~0.01 mm (1/20 of the layer height — the standard print-fidelity rule).
matplotlib's default `to_polygons()` flattening is too coarse and produces
faceted curved letters. The flattening lives in `label_common.glyph_contours`
+ `_flatten_quad` / `_flatten_cubic`. Then extrude → `manifold3d.difference`
on the cortex.

**Engrave LAST** (not on individual area pieces). Engraving on the
already-fused mesh gives one boolean over a clean volume → crisp grooves.
Engraving on the area pieces first and then unioning leads to coincident-
face slivers.

**Subdivide the engrave target** once before the boolean. Without
subdivision the groove rim follows the coarse area tessellation and looks
rough. `build_onepiece_voxsheet.py` does
`trimesh.remesh.subdivide(...)` on the fused mesh before subtracting text.
File grows from ~190 k → ~950 k faces, which is fine for slicing.

The screen-aligned reading frame in `label_common.reading_frame()` keeps
text forward-readable (`det[u, up, n] = +1` → no mirror) when projected
to the surface from a clicked normal.

### 3. Fusing 43 disjoint pieces into one watertight solid — what works and what doesn't

This is where most of the effort went. The pieces don't touch each other
(86 components total — 43 areas × 2 hemispheres). Any one-piece deliverable
needs a connector that fuses them, **AND** the result has to stay watertight
on STL **reload** (not just in-memory — in-memory checks lied repeatedly).

Approaches that DON'T work:
- **Thin shell connectors via boolean difference** (e.g., shrink(iso, A) − shrink(iso, B) at L4 depth) — shatters the final mesh into thousands of components in the union. Thin shells are degenerate for boolean unions.
- **Clipping the connector to the cortex footprint via intersection** — the intersection introduces near-degenerate geometry and shatters.
- **Naive normal-offset inflation** of areas to make them overlap — self-intersects on concave regions.
- **`root − iso_315` for the rest-of-brain body** — the iso mesh is misaligned with root, so the subtraction does almost nothing visible.

Approaches that DO work:
- **Per-area uniform-scale inflation toward each area centroid** — affine, no self-intersect. At ~1.1 mm inflation the medial areas overlap across the midline and the whole cortex fuses into one watertight piece conformally. Trade-off: areas grow ~1 mm visually.
- **Voxelize a thin sheet, then union** — voxel-remesh of the L4 band (shrink(iso, 1−1.5/R) − shrink(iso, 1−3/R), then `voxelized(0.5).fill().marching_cubes`) gives a **clean manifold thin shell** that unions cleanly with the area pieces. This is the trick that made the L4 sheet viable. Slight voxel smoothing is invisible because the sheet is buried in the ribbon.
- **Carving the full area-union footprint from `root_wt`** for a rest-of-brain solid base — gives a watertight body that areas inlay into.
- **Cylinder struts** unioned with clean volumes — never shatter (no intersection involved). MST over an adjacency graph gives a minimal hidden connecting set.

### 4. Connecting the hemispheres invisibly is genuinely hard

The cortex cap is two thin shells separated by the open longitudinal
fissure with a hollow underside. There is **no hidden interior** to route a
bridge through. Any cortex-only connector either shows in surface gaps or
sits in the open fissure. Pragmatic resolutions:

- **A few thin hand-positioned sticks deep in the fissure** — the current
  approach. `place_midline_sticks.py` lets the user translate (in the
  sagittal plane), tilt, and resize each stick individually until it sits
  inside the fissure walls. Default 4 sticks, r = 0.5 mm, picked from the
  closest L-R medial L4/5 point pairs.
- The voxel L4 sheet fuses each hemisphere internally but does NOT cross
  the midline (the iso mesh has separate L/R components, and at typical
  shrink the halves don't touch). So the architecture is **L4 sheet
  intra-hemi + midline sticks inter-hemi**.

For a fully-hidden one-piece, switch to the **whole-brain rest-of-brain
solid** (`FINAL_onepiece_arial.stl` in deprecated/) — the solid body
bridges hemispheres internally and there's no fissure to hide a bridge in.
Lose the hollow-cap look, gain a robust truly-hidden join.

### 5. In-memory mesh checks lie. Always reload-verify

`trimesh.is_watertight` on a freshly-unioned mesh reported `True` many
times when the exported STL, on reload, was 16,000+ shattered components
non-watertight. **Always export → reload → `merge_vertices()` → check
`is_watertight` and component count**. The `RELOAD:` print line in
`build_onepiece_voxsheet.py` does this. Trust nothing else.

### 6. Boolean inputs must each be `is_volume`. Concatenations are not

`manifold3d` raises `"Not all meshes are volumes!"` if any input fails
`is_volume`. Run `ev()` (the repair helper) on EVERY mesh you pass to
`trimesh.boolean.union/difference`. For text subtraction, **union the per-
letter solids first** rather than concatenating them — concatenated solids
are technically a single non-manifold mesh that degrades boolean output.

### 7. Interactive apps and the `pyvista 0.47` pickpoint bug

`pyvista==0.47.3`'s `enable_surface_point_picking(left_clicking=True)`
tries to set `self.pickpoint` on the Plotter, which the new attribute-
guard rejects → every click errors. Fix at the top of every app:
```python
pv.set_new_attribute(pl, 'pickpoint', None)
```
Without that, clicks silently fail and the user thinks the app is broken.

### 8. Interactive apps need visual feedback, not console prints

The Claude Code UI hides script stdout from the user. If clicking a
button only prints to console, the user sees nothing happen and assumes
it's broken (we hit this twice). **Every interactive action should
produce a visible change** — a sphere marker, a colour flip, the actor
itself updating. Print is for the agent reading logs, not the user.

---

## Architecture of `build_onepiece_voxsheet.py`

1. Load `997_root.obj` (whole brain) and `315_Isocortex.obj` — only used to
   define the print frame (`SHARED_SCALE`, `ROOT_CTR`) and to build the L4 sheet.
2. Build the **L4 voxel sheet**: `band = shrink(iso, 1-1.5/R) − shrink(iso, 1-3/R)`,
   then `band.voxelized(0.5).fill().marching_cubes` → guaranteed manifold
   thin shell at ~1.5–3.0 mm under the pia. 2 components (L/R hemispheres).
3. Load all 43 `wt_<acro>_66mm.stl` area pieces, split into hemisphere
   components.
4. Load **midline sticks** from `midline_sticks.json` (`C, D, H, r` per
   stick — created/edited by `place_midline_sticks.py`), build each as a
   cylinder of length 2H along axis D centred at C.
5. `manifold` union of `pieces + sheet + sticks` → keep the largest
   component (drops tiny boolean slivers).
6. If `text` mode: subdivide the fused mesh once (finer label rims), then
   `manifold` difference of the per-label extruded text solids (built via
   `label_common.build_text_world` using `label_positions.json` and
   `number_legend.json`).
7. Export, reload, verify watertight + components, render.

---

## Applying this pipeline to other CCF regions

The same architecture transfers cleanly to other Allen structures (the
cerebellum, thalamus, hippocampal formation, basal ganglia, etc.):

1. **Generate per-area meshes** by adapting `per_cortical_area.py`:
   - Replace the root structure ID (`ISO_ID = 315` for isocortex) with the
     parent of your target. e.g. `CB = 512` for cerebellum, `TH = 549` for
     thalamus, `HPF = 1089`, `CNU = 623`, etc. (see
     `cortical_areas_table.md` for the full hierarchy, or
     `tree.descendant_ids([sid])` via `allensdk`).
   - The `is_layer_acro` filter is cortex-specific; for non-cortex regions
     either drop it (take all descendants with non-layer children) or use
     the appropriate granularity for the region. For the cerebellum,
     `deprecated/per_cerebellum_area.py` and `deprecated/watertight_cb_regions.py`
     are starting points — copy and adapt.
2. **Watertight cleanup** is identical — copy `watertight_areas_final.py`
   and just change the input glob.
3. **Labels** — `place_labels_interactive.py` and `label_common.py` are
   region-agnostic. They load `wt_*_66mm.stl` from `stl/cortical_areas/` —
   either point them at a new folder or load from your region's folder.
4. **Connector** — the L4 sheet trick is specific to a layered cortex.
   For other regions:
   - **Cerebellum** has its own folial structure; a similar mid-depth voxel
     shell from `512_Cerebellum.obj` would work, plus midline sticks for the
     vermis vs hemispheres.
   - **Subcortical nuclei** (thalamus, basal ganglia) are 3D blobs, not
     thin shells. You probably want one of these instead:
     - **Whole-region solid base** (analog of `FINAL_onepiece_arial.stl`):
       carve the area-union footprint from the parent region solid; nuclei
       inlay into it. See `deprecated/build_onepiece.py` and
       `deprecated/fix_v1_restofbrain.py` for the area-carve pattern.
     - **MST struts** (see `deprecated/build_onepiece_sticks.py`) — works for
       any geometry; sticks bow inward (toward region centroid) to hide.
5. **Midline / cross-region bridges** — copy `place_midline_sticks.py`. The
   `to_true()` un-spread logic assumes the dividing axis is z; for other
   pairs of regions to bridge, change the spread axis.

Common pitfalls when extending:
- **Path scope.** All kept scripts now use `Path(__file__).resolve().parent.parent / 'data'`. Stick with that pattern in new scripts so the project stays portable.
- **`ISO_ID = 315` is the true isocortex.** `688` ("Cerebral cortex" / CTX) is broader and includes olfactory bulb / HPF / OLF — picking it was the source of the early "weird olfactory bulb" bug. Always verify the parent structure ID with `tree.get_structures_by_id([sid])[0]['acronym']`.
- **Per-area children with layer suffixes.** The `is_layer_acro` helper
  identifies cortical leaf areas by checking if all children end in a layer
  suffix (`'2/3'`, `'6a'`, `'6b'`, or a digit). Three areas (`329 SSp-bfd`,
  `1011 AUDd`, `894 RSPagl`) have a stray mis-parented child that breaks
  the `all()` check; the script manually `update({329, 1011, 894})`.
  Watch for analogous edge cases in other regions.

---

## Files / data layout the pipeline expects

```
data/
├── obj/                           # Allen .obj meshes (downloaded by per_cortical_area.py)
│   ├── 997_root.obj
│   ├── 315_Isocortex.obj
│   └── area_<sid>_<acro>.obj × 43
├── stl/
│   ├── cortical_areas/
│   │   └── wt_<acro>_66mm.stl × 43  # watertight cleaned (watertight_areas_final.py)
│   ├── FINAL_onepiece_voxsheet_arial.stl       # the labeled final
│   ├── FINAL_onepiece_voxsheet_notext_arial.stl
│   └── FINAL_onepiece_voxsheet_*_10x.stl       # 2× scaled
├── label_positions.json           # interactive labels: point, normal, hemi, angle, letter_mm, text-override
├── midline_sticks.json            # interactive sticks: C, D, H, r
├── number_legend.json             # number → acronym key for tiny areas (1=ORBm, 2=PERI, 3=SSp-un, 4=VISam)
├── cortical_areas_table.md        # generated by area_table.py
└── png/                           # render outputs
```

---

## Print-ready notes (Bambu Lab X1 Carbon)

- **Build volume**: 256 × 256 × 256 mm. The 10× cortex is 103 × 86 × 54 mm
  — fits with huge margin.
- **Orientation**: dorsal/pia side **up** keeps labels on top surfaces
  (crispest groove rendering).
- **Supports**: the cap is a hollow dome → the underside / medial walls
  need supports. Tree/organic supports peel cleanly.
- **Layer**: 0.12–0.16 mm for clean label grooves (chord tolerance was set
  for ~0.01 mm at print scale).
- **Material**: ~116 cm³ at 5×, ~29 g at 15% infill. Scales with the cube
  of the scale factor.
- **Single colour** is fine — labels are recessed grooves, read by shadow.

---

## Things still on the table (if you continue this)

- The numbered tiny areas (`1=ORBm, 2=PERI, 3=SSp-un, 4=VISam`) want a
  printed **legend card / plaque** (engrave or print as a separate flat
  tile). Acronym → full name is in `cortical_areas_table.md`.
- The midline-stick app saves `C, D, H, r` per stick but currently does NOT
  reload a saved spread display position — pre-existing sticks are
  rendered in the un-spread cortex frame. Minor.
- `RSPagl`'s plain `wt_*.stl` still fails `is_volume` (the labeled copy was
  alpha-wrap-repaired). `watertight_areas_final.py` could be hardened to
  always alpha-wrap RSPagl on production.
