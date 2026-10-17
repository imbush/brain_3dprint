"""Shared helpers for boolean cortical-area labeling: cute-font vector
text flattened to a chordal tolerance (smooth, non-faceted grooves),
screen-aligned forward-readable reading frame, shrink-to-fit sizing, and
manifold boolean engraving. Imported by the variant labeler, the manual
(picked-position) labeler, and the interactive placer.
"""
from pathlib import Path
import numpy as np
import trimesh
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from PIL import Image, ImageDraw, ImageFont

_DATA = Path(__file__).resolve().parent.parent / 'data'
CCF_DIR = _DATA / 'ccf'                              # raw Allen meshes / annotation
SET_DIR = _DATA / 'structure_sets' / 'isocortex'     # ACTIVE structure set (override per script)
OUT_DIR = SET_DIR                                    # back-compat for older scripts
TARGET_MM = 66

def use_set(name):
    """Point label_common at structure_set <name>. Call this before load_pieces()
    in scripts that work on a non-cortex set."""
    global SET_DIR, OUT_DIR
    SET_DIR = _DATA / 'structure_sets' / name
    OUT_DIR = SET_DIR

# ---- cute font (SF Rounded), used for BOTH the fit bitmap and vector text ----
FONT_CANDIDATES = [
    ('/System/Library/Fonts/SFNSRounded.ttf', 0),
    ('/System/Library/Fonts/Avenir Next.ttc', 4),
    ('/System/Library/Fonts/Helvetica.ttc', 1),
]
FONT_PATH, FONT_INDEX = next((p, i) for (p, i) in FONT_CANDIDATES
                              if Path(p).exists())
_TEXT_FP = FontProperties(fname=FONT_PATH)

def set_font(path, index=0):
    """Switch the engraving font (affects both the fit bitmap and the
    vector text)."""
    global FONT_PATH, FONT_INDEX, _TEXT_FP
    FONT_PATH = path
    FONT_INDEX = index
    _TEXT_FP = FontProperties(fname=path)

# ---- placement / engraving params ----
LABEL_DEPTH_MM = 1.5
MARGIN_OUT_MM = 0.6
DEPTH_TOL_MM = 4.0
FIT_FRAC_W = 0.62
FIT_FRAC_H = 0.50
MIN_LETTER_MM = 1.0
MAX_LETTER_MM = 5.0
OVERFLOW_TOL = 0.15
SHRINK = 0.82
TEXT_BUILD_PT = 100.0          # cap height ~70 build units at this size
TEXT_CHORD_PT = 0.14           # build-unit chord tol -> ~0.01 mm at 5 mm letters


# ---- fit-test bitmap helpers ----
def text_bitmap(text, font_pt=160, pad_px=8):
    try:
        font = ImageFont.truetype(FONT_PATH, font_pt, index=FONT_INDEX)
    except OSError:
        font = ImageFont.truetype(FONT_PATH, font_pt)
    bbox = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox((0, 0), text, font=font)
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
    v=row(downward) dir, u x v = -n. Label up direction is -v."""
    n = n / np.linalg.norm(n)
    up = np.array([0.0, -1.0, 0.0])
    su = up - n * (up @ n)
    if np.linalg.norm(su) < 0.35:
        up = np.array([-1.0, 0.0, 0.0])
        su = up - n * (up @ n)
    su = su / np.linalg.norm(su)
    u = np.cross(su, n); u = u / np.linalg.norm(u)
    return u, -su


# ---- vector text flattened to a chordal tolerance (smooth curves) ----
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


def glyph_contours(text, tol=TEXT_CHORD_PT):
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
    """Extruded vector-text solid, XY-centered, z in [0, depth_mm],
    uniformly scaled to cap height == height_mm, smooth curved edges."""
    parts = []
    for ring in glyph_contours(text):
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
    s = height_mm / ext[1]
    m.apply_translation([-m.bounding_box.centroid[0],
                          -m.bounding_box.centroid[1], 0.0])
    m.apply_scale([s, s, 1.0])
    return m


def rotate_in_plane(u, up, deg):
    """Spin the reading frame about the surface normal by `deg`."""
    th = np.radians(deg); c, s = np.cos(th), np.sin(th)
    return c * u + s * up, -s * u + c * up


def oriented_text(acro, site, site_n, letter_mm, depth_mm, z_base,
                  angle_deg=0.0):
    """Extruded text solid placed on the surface at `site`, spun by
    angle_deg in the tangent plane, base offset z_base along the normal.
    det[u,up,n]=+1 (no mirror)."""
    u_ax, v_ax = reading_frame(site_n)
    up_ax = -v_ax
    u_ax, up_ax = rotate_in_plane(u_ax, up_ax, angle_deg)
    n = site_n / np.linalg.norm(site_n)
    solid = text_to_solid(acro, height_mm=letter_mm, depth_mm=depth_mm)
    if solid is None:
        return None
    solid.apply_translation([0, 0, z_base])
    R = np.column_stack([u_ax, up_ax, n])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = site
    solid.apply_transform(T)
    return solid


def build_text_world(acro, site, site_n, letter_mm, angle_deg=0.0):
    """Text solid that pokes LABEL_DEPTH_MM into the surface (boolean cut)."""
    return oriented_text(acro, site, site_n, letter_mm,
                         LABEL_DEPTH_MM + MARGIN_OUT_MM, -LABEL_DEPTH_MM,
                         angle_deg)


def preview_text(acro, site, site_n, letter_mm, angle_deg=0.0, proud_mm=0.12):
    """Thin text solid sitting just proud of the surface, for live preview."""
    return oriented_text(acro, site, site_n, letter_mm, 0.5, proud_mm,
                         angle_deg)


def fit_label(acro, m_sub, site, site_n, sub_mask, outside):
    """Shrink-to-fit. Returns letter_mm or None if it can't fit at
    >= MIN_LETTER_MM without overflowing the piece's outside surface."""
    u_ax, v_ax = reading_frame(site_n)
    bitmap = text_bitmap(acro)
    H_px, W_px = bitmap.shape
    rel = m_sub.vertices - site
    u_co = rel @ u_ax; v_co = rel @ v_ax; w_co = rel @ site_n
    f_idx = np.where(sub_mask)[0]
    if len(f_idx) < 5:
        return None
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
            return ps * H_px
        ps *= SHRINK
    return None


# ---- mesh loading / boolean engraving ----
def outside_mask(m):
    vn = m.vertex_normals
    od = m.vertices - m.centroid
    od /= (np.linalg.norm(od, axis=1, keepdims=True) + 1e-12)
    return (vn * od).sum(axis=1) > 0.2


def load_pieces(out_dir=None, target_mm=TARGET_MM):
    # default is looked up at CALL time (not def time) so use_set() works
    if out_dir is None: out_dir = OUT_DIR
    """Returns list of (acro, m_orig, m_sub). m_sub is subdivided once
    for denser site/normal sampling; boolean is applied to m_orig."""
    import matplotlib as mpl
    pal = (list(mpl.colormaps['tab20'].colors) +
           list(mpl.colormaps['tab20b'].colors) +
           list(mpl.colormaps['tab20c'].colors))
    files = sorted((out_dir / 'stl').glob(
        f'wt_*_{target_mm}mm.stl'))
    pieces = []
    for fi, f in enumerate(files):
        acro = f.stem[len('wt_'):-len(f'_{target_mm}mm')]
        m = trimesh.load(str(f), force='mesh')
        nv, nf = trimesh.remesh.subdivide(m.vertices, m.faces)
        m_sub = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
        pieces.append((acro, pal[fi % len(pal)], m, m_sub))
    return pieces


def engrave(m_orig, text_solids):
    """Boolean-subtract concatenated text solids. Returns (mesh, ok)."""
    if not text_solids:
        return m_orig, False
    union = trimesh.util.concatenate(text_solids)
    try:
        cut = m_orig.difference(union, engine='manifold')
        if cut is not None and len(cut.faces) > 0:
            return cut, True
    except Exception:
        pass
    return m_orig, False
