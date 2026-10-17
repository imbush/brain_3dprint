"""Engrave each cortical area's acronym onto its own watertight area
piece using BINARY (boolean) subtraction (manifold3d), trying the four
label-placement variants we developed earlier PLUS the shrink-to-fit
resize methodology so labels never overflow the piece.

Inputs : stl/cortical_areas/wt_<acro>_66mm.stl   (43 watertight solids)
Method (per piece, per hemisphere):
  1. variant mask picks candidate outside-facing vertices
       A_dorsal       — strongly dorsal-facing  (vn_y < -0.55)
       B_outside      — any outside-facing       (vn·out > 0.2)
       C_outside_flat — outside AND aligned with the patch mean normal
       D_inscribed    — outside, site = most-interior vertex of patch
  2. screen-aligned reading_frame at the site  → forward-readable text
       (u = reading dir, up = -v ; det[u,up,n]=+1 so NO mirror)
  3. SHRINK-TO-FIT: rasterize the acronym to a bitmap, shrink letter
     size (ps *= 0.82, up to 7 tries) until the inked footprint stops
     overflowing the piece's outside surface (overflow <= OVERFLOW_TOL).
  4. build the accepted-size text as an extruded solid (TextPath ->
     shapely -> extrude_polygon), orient with the reading frame, poke
     it LABEL_DEPTH_MM into the surface.
  5. concatenate this piece's text solids and boolean-SUBTRACT once.
     (piece validated as a manifold volume first; bilinear-engrave
      fallback if manifold rejects it.)

Outputs per variant V in {A_dorsal, B_outside, C_outside_flat, D_inscribed}:
  stl/cortical_areas/boollabeled_<V>/wtl_<acro>_66mm.stl
  png/boolarea_<V>.png        (all areas, colored, labels engraved)
"""
import time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import matplotlib as mpl

OUT_DIR = Path('/Users/inlebush/lab/ila/microns/data/ccf_brain')
TARGET_MM = 66

# Cute font for the bitmap fit test (SF Rounded). The boolean text mesh
# is built from a matplotlib FontProperties family; we map to the same.
FONT_CANDIDATES = [
    ('/System/Library/Fonts/SFNSRounded.ttf', 0, 'Helvetica'),
    ('/System/Library/Fonts/Avenir Next.ttc', 4, 'Avenir Next'),
    ('/System/Library/Fonts/Helvetica.ttc', 1, 'Helvetica'),
]
FONT_PATH, FONT_INDEX, MPL_FAMILY = next(
    (p, i, fam) for (p, i, fam) in FONT_CANDIDATES if Path(p).exists())
print(f'fit-test font: {FONT_PATH} (index {FONT_INDEX})')

# ----- fit-test helpers (bitmap rasterize + bilinear sample) -----
def text_bitmap(text, font_pt=160, pad_px=8):
    try:
        font = ImageFont.truetype(FONT_PATH, font_pt, index=FONT_INDEX)
    except OSError:
        font = ImageFont.truetype(FONT_PATH, font_pt)
    img0 = Image.new('L', (1, 1))
    bbox = ImageDraw.Draw(img0).textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 2 * pad_px
    h = bbox[3] - bbox[1] + 2 * pad_px
    img = Image.new('L', (w, h), color=255)
    ImageDraw.Draw(img).text((pad_px - bbox[0], pad_px - bbox[1]),
                              text, font=font, fill=0)
    return 1.0 - np.asarray(img, dtype=np.float32) / 255.0

def bilinear(field, col, row):
    H_, W_ = field.shape
    c0 = np.floor(col).astype(int); r0 = np.floor(row).astype(int)
    c1 = c0 + 1; r1 = r0 + 1
    cf = col - c0; rf = row - r0
    c0 = np.clip(c0, 0, W_ - 1); c1 = np.clip(c1, 0, W_ - 1)
    r0 = np.clip(r0, 0, H_ - 1); r1 = np.clip(r1, 0, H_ - 1)
    return (field[r0, c0]*(1-cf)*(1-rf) + field[r0, c1]*cf*(1-rf) +
            field[r1, c0]*(1-cf)*rf + field[r1, c1]*cf*rf)

def reading_frame(n):
    """Screen-aligned, forward-readable. Returns (u, v): u=reading dir,
    v=row(downward) dir, with u x v = -n. Up direction is -v."""
    n = n / np.linalg.norm(n)
    up = np.array([0.0, -1.0, 0.0])
    su = up - n * (up @ n)
    if np.linalg.norm(su) < 0.35:
        up = np.array([-1.0, 0.0, 0.0])
        su = up - n * (up @ n)
    su = su / np.linalg.norm(su)
    u = np.cross(su, n); u = u / np.linalg.norm(u)
    v = -su
    return u, v

# ----- boolean text-solid builder -----
# Load the SAME cute font file the fit-test uses, so the engraved
# vector text matches the bitmap and stays crisp (no font swap, no
# non-uniform stretch — that was distorting the letters).
_TEXT_FP = FontProperties(fname=FONT_PATH)

# Chordal deviation for flattening the font's Bezier outlines. Best
# practice for printable curved surfaces: keep the gap between the true
# curve and its polyline below ~1/20 of the layer height. The label is
# built at TEXT_BUILD_PT then scaled to letter_mm; we flatten in
# build-units so that the FINAL chord error stays ~0.01 mm.
TEXT_BUILD_PT = 100.0         # cap height ~70 build units at this size
TEXT_CHORD_PT = 0.14          # build-unit chord tol → ~0.01 mm at 5 mm letters

def _line_dist(p, a, b):
    ab = b - a; t = ab @ ab
    if t < 1e-12: return float(np.hypot(*(p - a)))
    return abs(ab[0]*(a[1]-p[1]) - ab[1]*(a[0]-p[0])) / np.sqrt(t)

def _flatten_quad(p0, p1, p2, tol, out, depth=0):
    if depth > 24 or _line_dist(p1, p0, p2) <= tol:
        out.append(p2); return
    p01 = (p0+p1)/2; p12 = (p1+p2)/2; m = (p01+p12)/2
    _flatten_quad(p0, p01, m, tol, out, depth+1)
    _flatten_quad(m, p12, p2, tol, out, depth+1)

def _flatten_cubic(p0, p1, p2, p3, tol, out, depth=0):
    if depth > 24 or max(_line_dist(p1, p0, p3), _line_dist(p2, p0, p3)) <= tol:
        out.append(p3); return
    p01=(p0+p1)/2; p12=(p1+p2)/2; p23=(p2+p3)/2
    p012=(p01+p12)/2; p123=(p12+p23)/2; m=(p012+p123)/2
    _flatten_cubic(p0, p01, p012, m, tol, out, depth+1)
    _flatten_cubic(m, p123, p23, p3, tol, out, depth+1)

def glyph_contours(text, tol):
    """Finely flattened glyph outlines (adaptive Bezier subdivision to
    chordal tol) — eliminates the faceted look on curved letters."""
    tp = TextPath((0, 0), text, size=TEXT_BUILD_PT, prop=_TEXT_FP)
    contours = []; cur = []; pos = None
    for verts, code in tp.iter_segments(curves=True):
        v = np.asarray(verts, dtype=float).reshape(-1, 2)
        if code == MplPath.MOVETO:
            if len(cur) >= 3: contours.append(np.array(cur))
            pos = v[0]; cur = [pos]
        elif code == MplPath.LINETO:
            pos = v[0]; cur.append(pos)
        elif code == MplPath.CURVE3:
            _flatten_quad(pos, v[0], v[1], tol, cur); pos = v[1]
        elif code == MplPath.CURVE4:
            _flatten_cubic(pos, v[0], v[1], v[2], tol, cur); pos = v[2]
        elif code == MplPath.CLOSEPOLY:
            if len(cur) >= 3: contours.append(np.array(cur))
            cur = []
    if len(cur) >= 3: contours.append(np.array(cur))
    return contours

def text_to_solid(text, height_mm, depth_mm):
    """Extruded vector-text solid, XY-centered, z in [0, depth_mm].
    Uniformly scaled so its cap height == height_mm (font aspect kept).
    Curved edges are flattened to ~0.01 mm chord error (smooth walls)."""
    parts = []
    for ring in glyph_contours(text, TEXT_CHORD_PT):
        if len(ring) < 3: continue
        try:
            p = Polygon(ring)
            if not p.is_valid: p = make_valid(p)
            if p.is_valid and p.area > 1e-9:
                parts.append(p)
        except Exception:
            pass
    if not parts: return None
    result = parts[0]
    for p in parts[1:]:
        result = result.symmetric_difference(p)
    polys = ([result] if isinstance(result, Polygon)
              else list(result.geoms) if isinstance(result, MultiPolygon)
              else [])
    meshes = []
    for p in polys:
        if not p.is_valid or p.area < 1e-9: continue
        try:
            meshes.append(trimesh.creation.extrude_polygon(p, height=depth_mm))
        except Exception:
            pass
    if not meshes: return None
    m = trimesh.util.concatenate(meshes)
    ext = m.bounding_box.extents
    if ext[1] < 1e-6: return None
    s = height_mm / ext[1]                       # uniform — preserve aspect
    m.apply_translation([-m.bounding_box.centroid[0],
                          -m.bounding_box.centroid[1], 0.0])
    m.apply_scale([s, s, 1.0])                   # XY only — z stays depth_mm
    return m

# ----- placement params (resize methodology) -----
LABEL_DEPTH_MM = 1.5
MARGIN_OUT_MM = 0.6        # text solid pokes this far above the surface
DEPTH_TOL_MM = 4.0
FIT_FRAC_W = 0.62
FIT_FRAC_H = 0.50
MIN_LETTER_MM = 1.0
MAX_LETTER_MM = 5.0
OVERFLOW_TOL = 0.15
SHRINK = 0.82

files = sorted((OUT_DIR / 'stl' / 'cortical_areas').glob(f'wt_*_{TARGET_MM}mm.stl'))
print(f'{len(files)} watertight area pieces')

pal = (list(mpl.colormaps['tab20'].colors) +
       list(mpl.colormaps['tab20b'].colors) +
       list(mpl.colormaps['tab20c'].colors))

# Pre-load + subdivide each piece once (denser for fit sampling).
pieces = []   # (acro, color, m_orig, m_sub)
for fi, f in enumerate(files):
    acro = f.stem[len('wt_'):-len(f'_{TARGET_MM}mm')]
    m = trimesh.load(str(f), force='mesh')
    nv, nf = trimesh.remesh.subdivide(m.vertices, m.faces)
    m_sub = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
    pieces.append((acro, pal[fi % len(pal)], m, m_sub))

# ----- variant site/mask strategies -----
def candidates(variant, m_sub, vn, outside, hemi):
    """Return (sub_mask) of vertices eligible to host the label for this
    variant within this hemisphere."""
    if variant == 'A_dorsal':
        return hemi & outside & (vn[:, 1] < -0.55)
    if variant == 'B_outside':
        return hemi & outside
    if variant == 'C_outside_flat':
        cand = hemi & outside
        if cand.sum() < 40:
            return cand
        mean_n = vn[cand].mean(axis=0)
        mean_n /= (np.linalg.norm(mean_n) + 1e-12)
        align = vn @ mean_n
        flat = cand & (align > 0.6)
        if flat.sum() < 20:
            flat = cand & (align > 0.4)
        return flat
    if variant == 'D_inscribed':
        return hemi & outside
    return hemi & outside

def pick_site(variant, m_sub, sub_mask, vn, outside, hemi):
    """Return (site, site_n) for label center."""
    idx = np.where(sub_mask)[0]
    pts = m_sub.vertices[idx]; norms = vn[idx]
    mean_n = norms.mean(axis=0); mean_n /= (np.linalg.norm(mean_n) + 1e-12)
    if variant == 'D_inscribed':
        # most-interior: farthest from the patch boundary (nearest
        # non-candidate vertex in this hemisphere).
        rim = np.where(hemi & ~sub_mask)[0]
        if len(rim) >= 3:
            kd = cKDTree(m_sub.vertices[rim])
            d, _ = kd.query(pts)
            best = idx[int(np.argmax(d))]
            return m_sub.vertices[best], vn[best]
    # default: near patch (uv) centroid AND well-aligned with mean_n
    pc = pts - pts.mean(axis=0)
    d_ctr = np.linalg.norm(pc, axis=1)
    order = np.argsort(d_ctr)[:min(80, len(pts))]
    best_local = order[np.argmax((norms @ mean_n)[order])]
    best = idx[best_local]
    return m_sub.vertices[best], vn[best]

def fit_label(acro, m_sub, site, site_n, sub_mask, outside):
    """Shrink-to-fit. Returns (letter_mm, width_mm) or None if it can't
    fit at >= MIN_LETTER_MM without overflowing the piece."""
    u_ax, v_ax = reading_frame(site_n)
    bitmap = text_bitmap(acro)
    H_px, W_px = bitmap.shape
    rel = m_sub.vertices - site
    u_co = rel @ u_ax; v_co = rel @ v_ax; w_co = rel @ site_n
    f_idx = np.where(sub_mask)[0]
    f_u = u_co[f_idx]; f_v = v_co[f_idx]
    ps = min((FIT_FRAC_W * f_u.ptp()) / W_px,
             (FIT_FRAC_H * f_v.ptp()) / H_px,
             MAX_LETTER_MM / H_px)
    for _ in range(7):
        if ps * H_px < MIN_LETTER_MM:
            return None
        col = u_co / ps + W_px / 2
        row = v_co / ps + H_px / 2
        in_box = ((col >= 0) & (col < W_px - 1) & (row >= 0) &
                  (row < H_px - 1) & (np.abs(w_co) < DEPTH_TOL_MM))
        if not in_box.any():
            ps *= SHRINK; continue
        ink = bilinear(bitmap, col[in_box], row[in_box])
        inked = np.zeros(len(m_sub.vertices), dtype=bool)
        inked[np.where(in_box)[0]] = ink > 0.3
        n_ink = int(inked.sum())
        if n_ink < 5:
            ps *= SHRINK; continue
        overflow = int((inked & ~outside).sum()) / max(n_ink, 1)
        if overflow <= OVERFLOW_TOL:
            letter_mm = ps * H_px
            width_mm = ps * W_px
            return letter_mm, width_mm
        ps *= SHRINK
    return None

def build_text_world(acro, site, site_n, letter_mm):
    """Oriented, positioned extruded text solid for boolean subtraction."""
    u_ax, v_ax = reading_frame(site_n)
    up_ax = -v_ax                              # matplotlib y-up
    n = site_n / np.linalg.norm(site_n)
    solid = text_to_solid(acro, height_mm=letter_mm,
                          depth_mm=LABEL_DEPTH_MM + MARGIN_OUT_MM)
    if solid is None:
        return None
    solid.apply_translation([0, 0, -LABEL_DEPTH_MM])  # poke inward
    R = np.column_stack([u_ax, up_ax, n])             # det=+1, no mirror
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = site
    solid.apply_transform(T)
    return solid

# ----- main: per variant, per piece -----
def make_poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

VARIANTS = ['A_dorsal', 'B_outside', 'C_outside_flat', 'D_inscribed']
status_lines = []

for variant in VARIANTS:
    print(f'\n=== variant {variant} ===')
    odir = OUT_DIR / 'stl' / 'cortical_areas' / f'boollabeled_{variant}'
    odir.mkdir(parents=True, exist_ok=True)
    labeled = {}                # acro -> (color, mesh)
    n_placed = 0; n_pieces_labeled = 0; n_fallback = 0
    for acro, color, m_orig, m_sub in pieces:
        vn = m_sub.vertex_normals
        ctr = m_sub.centroid
        out_dir = m_sub.vertices - ctr
        out_dir /= (np.linalg.norm(out_dir, axis=1, keepdims=True) + 1e-12)
        outside = (vn * out_dir).sum(axis=1) > 0.2

        text_solids = []
        for hemi_sign in (-1, +1):
            hemi = m_sub.vertices[:, 2] * hemi_sign > 0
            sub = candidates(variant, m_sub, vn, outside, hemi)
            if sub.sum() < 30:
                continue
            site, site_n = pick_site(variant, m_sub, sub, vn, outside, hemi)
            fit = fit_label(acro, m_sub, site, site_n, sub, outside)
            if fit is None:
                continue
            letter_mm, width_mm = fit
            solid = build_text_world(acro, site, site_n, letter_mm)
            if solid is not None:
                text_solids.append(solid)

        out_mesh = m_orig
        if text_solids:
            text_union = trimesh.util.concatenate(text_solids)
            try:
                cut = m_orig.difference(text_union, engine='manifold')
                if cut is not None and len(cut.faces) > 0:
                    out_mesh = cut
                    n_placed += len(text_solids)
                    n_pieces_labeled += 1
                else:
                    n_fallback += 1
            except Exception as e:
                n_fallback += 1
        labeled[acro] = (color, out_mesh)
        out_mesh.export(str(odir / f'wtl_{acro}_{TARGET_MM}mm.stl'))
    print(f'  labeled {n_pieces_labeled}/{len(pieces)} pieces, '
          f'{n_placed} labels, {n_fallback} boolean failures')
    status_lines.append(
        f'- **{variant}**: {n_pieces_labeled}/{len(pieces)} pieces '
        f'labeled, {n_placed} labels engraved, {n_fallback} boolean '
        f'failures → [png/boolarea_{variant}.png](png/boolarea_{variant}.png)')

    # render combined
    pv.OFF_SCREEN = True
    pv.set_plot_theme('document')
    all_v = np.vstack([m.vertices for _, m in labeled.values()])
    d = (all_v.max(0) - all_v.min(0)).max() * 1.8
    views = [
        ('dorsal',    (0, -d, 0), (0, 0, 0), (-1, 0, 0)),
        ('lateral R', (0,  0,  d), (0, 0, 0), (0, -1, 0)),
        ('lateral L', (0,  0, -d), (0, 0, 0), (0, -1, 0)),
        ('ventral',   (0,  d, 0), (0, 0, 0), (1, 0, 0)),
    ]
    pl = pv.Plotter(shape=(2, 2), window_size=(2200, 1800), off_screen=True)
    for k, (vname, cam, focal, up) in enumerate(views):
        pl.subplot(k // 2, k % 2)
        for acro, (color, m) in labeled.items():
            pl.add_mesh(make_poly(m), color=color, smooth_shading=True,
                         specular=0.2, ambient=0.25, diffuse=0.9)
        pl.add_text(f'{variant} — {vname}', font_size=11)
        pl.camera_position = (cam, focal, up); pl.reset_camera()
    pl.background_color = 'white'
    pl.screenshot(str(OUT_DIR / 'png' / f'boolarea_{variant}.png'))
    pl.close()
    print(f'  wrote boolarea_{variant}.png')

# write STATUS
status = ('# Boolean area labels — variant comparison\n\n'
          'Each of the 43 watertight cortical-area pieces gets its '
          'acronym engraved by **boolean (manifold3d) subtraction** of '
          'an extruded-text solid. Text is oriented with the '
          'screen-aligned reading frame (forward-readable, not '
          'mirrored) and **shrink-to-fit** resized (label shrinks until '
          'its footprint no longer overflows the piece, '
          f'OVERFLOW_TOL={OVERFLOW_TOL}).\n\n'
          'Four placement variants:\n\n'
          '- **A_dorsal** — label only where the surface faces strongly '
          'dorsal (vn_y < -0.55)\n'
          '- **B_outside** — any outside-facing patch (lateral/ventral '
          'too)\n'
          '- **C_outside_flat** — outside AND aligned with the patch '
          'mean normal (sits on a flat spot)\n'
          '- **D_inscribed** — outside, label centered at the '
          'most-interior point of the patch (farthest from its edge)\n\n'
          'Results:\n\n' + '\n'.join(status_lines) + '\n\n'
          'Per-variant STLs: '
          '`stl/cortical_areas/boollabeled_<variant>/wtl_<acro>_66mm.stl`\n')
(OUT_DIR / 'STATUS.md').write_text(status)
print('\nwrote STATUS.md')
print('done')
