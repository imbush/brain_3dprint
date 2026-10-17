"""Interactive label placement tool with live preview, rotation, and
resize.

Opens a 3D window with all 43 watertight cortical-area pieces (colored).
LEFT-CLICK the pial (outer) surface to place a label: the actual acronym
appears engraved-style on the surface (red = active). Click an existing
label to re-select it. Each click also mirrors to the same area's other
hemisphere (toggle with 'b').

Edit the ACTIVE label (red):
  Left / Right  or  a / d   : spin the label in the surface plane (-/+)
  Up / Down     or  w / x   : resize bigger / smaller
  n / p                     : select next / previous label
  f                         : delete active label
  b                         : toggle bilateral mirroring
  u                         : undo last placement
  r                         : reset all
  s                         : save -> data/ccf_brain/label_positions.json
  q                         : quit

The saved JSON stores point, normal, hemi, angle, and letter_mm for each
label; boolean_area_labels_manual.py engraves exactly what you see.

Run from a terminal WITH a display:
    conda activate microns-m2
    python notebooks/ccf_brain/place_labels_interactive.py
"""
import json
import numpy as np
import pyvista as pv

import label_common as lc

OUT_DIR = lc.OUT_DIR
TARGET_MM = lc.TARGET_MM
POS_JSON = OUT_DIR / 'label_positions.json'

ROT_STEP = 10.0          # degrees per key press
SIZE_STEP = 1.12         # multiplier per key press
SELECT_RADIUS = 2.5      # mm: click within this of a label selects it
DEFAULT_SIZE = 2.5       # mm fallback if auto-fit fails
NUMBER_SIZE = 2.2        # mm for areas shown as a number
NUDGE_MM = 1.0           # step for medial/lateral label nudging

# areas engraved as a number (legend) -> keep the number when re-placed
NUM_FOR = {}
_nlp = OUT_DIR / 'number_legend.json'
if _nlp.exists():
    NUM_FOR = {a: n for n, a in json.loads(_nlp.read_text()).items()}

pieces = lc.load_pieces()
print(f'loaded {len(pieces)} pieces')
msub = {}; out = {}; is_bilateral = {}
m_orig_by_acro = {}
snap_pts = []; snap_nrm = []; snap_acro = []
for acro, color, m_orig, m_sub in pieces:
    o = lc.outside_mask(m_sub)
    msub[acro] = m_sub; out[acro] = o
    m_orig_by_acro[acro] = m_orig
    # detect bilateral structures: components on BOTH sides of the midline.
    # midline / unilateral structures (e.g. thalamic PVT, CM, MD) get a SINGLE
    # label that does not auto-mirror.
    zs = [float(c.centroid[2]) for c in m_orig.split(only_watertight=False)]
    is_bilateral[acro] = any(z > 0.5 for z in zs) and any(z < -0.5 for z in zs)
    idx = np.where(o)[0]
    snap_pts.append(m_sub.vertices[idx])
    snap_nrm.append(m_sub.vertex_normals[idx])
    snap_acro += [acro] * len(idx)
print(f'  {sum(is_bilateral.values())}/{len(is_bilateral)} areas bilateral; '
      f'midline (no-mirror): {sorted(a for a,b in is_bilateral.items() if not b)[:10]}'
      f'{"..." if sum(not b for b in is_bilateral.values()) > 10 else ""}')
snap_pts = np.vstack(snap_pts)
snap_nrm = np.vstack(snap_nrm)
snap_acro = np.array(snap_acro)

from scipy.spatial import cKDTree
kd = cKDTree(snap_pts)

pal = {acro: color for acro, color, _, _ in pieces}

# ---- exploded view: per-piece radial offset from the set centroid ----
# At factor f, piece is translated by (centroid - set_centroid) * f. Picking and
# labels stay anchored to the original coordinates — the visual translation is
# undone before snapping. 'e' cycles through ramp levels (0 = normal).
SET_CENTROID = np.mean([m_orig_by_acro[a].centroid
                        for a in m_orig_by_acro], axis=0)
piece_offset_dir = {acro: (m_orig_by_acro[acro].centroid - SET_CENTROID)
                    for acro in m_orig_by_acro}
EXPLODE_LEVELS = [0.0, 0.4, 0.8, 1.2, 1.8, 2.5]
EXPLODE_IDX = [0]
def explode_factor():
    return EXPLODE_LEVELS[EXPLODE_IDX[0]]
def piece_offset(acro):
    return piece_offset_dir[acro] * explode_factor()

# ---- hide mode: click a piece to make it disappear (reveal interior nuclei) ----
HIDE_MODE = [False]
hidden = set()

def poly(m):
    return pv.PolyData(m.vertices,
        np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64))

pl = pv.Plotter(window_size=(1700, 1200))
# pyvista 0.47's left_clicking surface picker assigns self.pickpoint, which
# the new attribute-guard otherwise rejects — register it up front.
pv.set_new_attribute(pl, 'pickpoint', None)
piece_actors = []   # (actor, acro) so a click resolves to the piece under it
def _build_piece_actors():
    """Rebuild piece actors at the current explosion factor. Called once at
    startup and on every 'e' press."""
    for act, _ in piece_actors:
        pl.remove_actor(act)
    piece_actors.clear()
    have = {lbl['acro'] for g in groups for lbl in g} if 'groups' in globals() else set()
    for acro, color, m_orig, _ in pieces:
        if acro in hidden:
            continue
        m = m_orig.copy()
        m.vertices = m.vertices + piece_offset(acro)
        c = color if (not have or acro in have) else UNLABELED_C
        a = pl.add_mesh(poly(m), color=c, smooth_shading=True,
                        specular=0.2, ambient=0.25, diffuse=0.9, pickable=True,
                        reset_camera=False)
        piece_actors.append((a, acro))

# ---- label state ----
# Each label: dict(acro, site, normal, hemi, angle, size, actor).
# groups: list of [label] or [label, mirror]; active = index into groups.
BILATERAL = [True]
groups = []
active = [-1]

UNLABELED_C = (1.0, 0.0, 0.0)   # bright red = no label yet -> easy to find
_base_color = {p[0]: p[1] for p in pieces}

def _recolor_pieces():
    have = {lbl['acro'] for g in groups for lbl in g}
    for act, acro in piece_actors:
        act.prop.color = (_base_color[acro] if acro in have else UNLABELED_C)

def _fit_size(acro, site, normal):
    sign = 1 if site[2] > 0 else -1
    sm = out[acro] & (msub[acro].vertices[:, 2] * sign > 0)
    lm = lc.fit_label(acro, msub[acro], site, normal, sm, out[acro])
    return float(lm) if lm else DEFAULT_SIZE

def _draw(lbl, is_active):
    if lbl.get('actor') is not None:
        pl.remove_actor(lbl['actor']); lbl['actor'] = None
    site = lbl['site'] + piece_offset(lbl['acro'])
    m = lc.preview_text(lbl.get('text') or lbl['acro'], site,
                        lbl['normal'], lbl['size'], lbl['angle'])
    color = 'red' if is_active else (0.08, 0.08, 0.08)
    if m is not None:
        lbl['actor'] = pl.add_mesh(poly(m), color=color, pickable=False,
                                   reset_camera=False)

def _redraw_group(gi):
    for lbl in groups[gi]:
        _draw(lbl, gi == active[0])

def _set_active(gi):
    prev = active[0]
    active[0] = gi
    if 0 <= prev < len(groups) and prev != gi:
        _redraw_group(prev)
    if 0 <= gi < len(groups):
        _redraw_group(gi)

def _make(acro, site, normal, angle, size, text=None, midline=False):
    s = np.asarray(site, float)
    hemi = 'M' if midline else ('R' if s[2] > 0 else 'L')
    return dict(acro=acro, site=s, normal=np.asarray(normal, float),
                hemi=hemi, angle=float(angle), size=float(size),
                text=text, midline=bool(midline), actor=None)

def _snap_midline(acro, target_xy):
    """Return (site, normal) on area acro's pial surface, snapped onto the
    midline (z=0). target_xy hints which spot to pick when the area's
    midline ridge has multiple candidate vertices."""
    V = msub[acro].vertices; N = msub[acro].vertex_normals; o = out[acro]
    for tol in (1.0, 2.5, 5.0, 1e9):
        mask = o & (np.abs(V[:, 2]) < tol)
        if mask.any(): break
    idx = np.where(mask)[0]
    j = idx[int(np.argmin(np.linalg.norm(V[idx, :2] - target_xy, axis=1)))]
    site = V[j].copy(); site[2] = 0.0
    return site, N[j].copy()

def _reflect_fields(prim):
    """Exact mirror of a label across the midline (z=0): position and
    normal reflect in z, angle negates. Guarantees bilateral symmetry."""
    s = prim['site']; n = prim['normal']
    site = np.array([s[0], s[1], -s[2]])
    return dict(site=site, normal=np.array([n[0], n[1], -n[2]]),
                hemi='R' if site[2] > 0 else 'L',
                angle=-prim['angle'], size=prim['size'], text=prim.get('text'))

def _sync_mirror(group):
    if len(group) == 2:
        group[1].update(_reflect_fields(group[0]))

def _all_labels():
    return [(gi, li) for gi, g in enumerate(groups) for li in range(len(g))]

def click(point, *args):
    point = np.asarray(point, float)
    # In exploded mode the picked point is in the *visible* (offset) frame;
    # undo the offset of the piece under the cursor to get original-frame
    # coordinates that match snap_pts and stored label sites.
    pa = getattr(pl, 'picked_actor', None)
    acro_under = next((ac for act, ac in piece_actors if act is pa), None)
    if HIDE_MODE[0]:
        if acro_under is None:
            print('hide mode: no piece under cursor'); return
        hidden.add(acro_under)
        _build_piece_actors()
        for gi in range(len(groups)):
            _redraw_group(gi)
        print(f'hidden {acro_under}  ({len(hidden)} hidden; v to show all)')
        return
    off_under = piece_offset(acro_under) if acro_under is not None else np.zeros(3)
    orig_point = point - off_under
    # select existing if close
    best = None; bestd = SELECT_RADIUS
    for gi, g in enumerate(groups):
        for lbl in g:
            d = np.linalg.norm(lbl['site'] - orig_point)
            if d < bestd:
                bestd = d; best = gi
    if best is not None:
        _set_active(best)
        print(f'selected {groups[best][0]["acro"]} '
              f'(angle {groups[best][0]["angle"]:.0f}, '
              f'size {groups[best][0]["size"]:.1f}mm)')
        return
    # else create — resolve the area from the PIECE under the cursor (not
    # nearest vertex across all pieces), so tiny areas like PERI next to
    # ECT get the right label. Then snap within that area's pial surface.
    acro = acro_under
    if acro is not None:
        sel = np.where(snap_acro == acro)[0]
        i = sel[int(np.argmin(np.linalg.norm(snap_pts[sel] - orig_point, axis=1)))]
    else:
        _, i = kd.query(orig_point)
        acro = str(snap_acro[i])
    site = snap_pts[i]; normal = snap_nrm[i]
    num = NUM_FOR.get(acro)                  # numbered area -> engrave its number
    # if this area already has a label, MOVE it here (don't duplicate)
    for gi, g in enumerate(groups):
        if g[0]['acro'] == acro:
            if g[0].get('midline'):           # locked to z=0: re-snap to midline at click xy
                site, normal = _snap_midline(acro, site[:2])
            g[0]['site'] = site; g[0]['normal'] = normal
            g[0]['hemi'] = 'M' if g[0].get('midline') else ('R' if site[2] > 0 else 'L')
            _sync_mirror(g); _set_active(gi); _redraw_group(gi)
            print(f'moved {acro}'); return
    size = NUMBER_SIZE if num else _fit_size(acro, site, normal)
    group = [_make(acro, site, normal, 0.0, size, text=num)]
    # only mirror across midline if the area actually exists on both sides
    if BILATERAL[0] and is_bilateral.get(acro, True):
        group.append(_make(acro, site, normal, 0.0, size, text=num))
        _sync_mirror(group)
    groups.append(group)
    _set_active(len(groups) - 1)
    _recolor_pieces()
    print(f'+ {acro} ({"bilateral" if len(group) == 2 else "one-sided"}) '
          f'size {size:.1f}mm  (total {len(_all_labels())})')

def _edit(d_angle=0.0, scale=1.0):
    gi = active[0]
    if not (0 <= gi < len(groups)):
        print('no active label — click to place one'); return
    prim = groups[gi][0]
    prim['angle'] += d_angle
    prim['size'] = float(np.clip(prim['size'] * scale, 0.6, 9.0))
    _sync_mirror(groups[gi])             # keep the other side an exact mirror
    _redraw_group(gi)
    print(f'  {prim["acro"]}: angle {prim["angle"]:.0f}  '
          f'size {prim["size"]:.2f}mm')

def rot_neg(): _edit(d_angle=-ROT_STEP)
def rot_pos(): _edit(d_angle=+ROT_STEP)
def big():     _edit(scale=SIZE_STEP)
def small():   _edit(scale=1.0 / SIZE_STEP)

def _nudge(direction):
    """Slide the active label along its area's surface toward the midline
    ('medial') or away ('lateral'). Snaps to the area's own mesh, so it can
    reach the medial wall (not just the strongly-outward patch)."""
    gi = active[0]
    if not (0 <= gi < len(groups)):
        print('no active label'); return
    prim = groups[gi][0]; acro = prim['acro']; site = prim['site']
    toward_mid = -np.sign(site[2]) or 1.0          # medial = toward z=0
    step = toward_mid * NUDGE_MM * (1.0 if direction == 'medial' else -1.0)
    target = site + np.array([0.0, 0.0, step])
    V = msub[acro].vertices
    j = int(np.argmin(np.linalg.norm(V - target, axis=1)))
    prim['site'] = V[j]; prim['normal'] = msub[acro].vertex_normals[j]
    prim['hemi'] = 'R' if V[j][2] > 0 else 'L'
    _sync_mirror(groups[gi]); _redraw_group(gi)
    print(f'  {acro}: nudged {direction} -> z={V[j][2]:.1f}')

def nudge_medial():  _nudge('medial')
def nudge_lateral(): _nudge('lateral')

def toggle_hide_mode():
    """'h': enter/exit hide-mode. While ON, left-click hides the picked piece
    (its label stays visible). 'v' brings everything back."""
    HIDE_MODE[0] = not HIDE_MODE[0]
    print(f'HIDE MODE {"ON — click pieces to hide" if HIDE_MODE[0] else "OFF"}')

def show_all():
    if not hidden:
        print('nothing hidden'); return
    n = len(hidden); hidden.clear()
    _build_piece_actors()
    for gi in range(len(groups)):
        _redraw_group(gi)
    print(f'restored {n} hidden piece(s)')

def toggle_number():
    """'j' on the active label: replace its text with the next free integer
    (so very small areas engrave a digit + go in the legend). Press again to
    revert to the acronym. The legend is written next to label_positions.json
    on 's'."""
    gi = active[0]
    if not (0 <= gi < len(groups)):
        print('no active label'); return
    prim = groups[gi][0]
    if prim.get('text'):                       # revert
        for lbl in groups[gi]:
            lbl['text'] = None
            lbl['size'] = _fit_size(prim['acro'], lbl['site'], lbl['normal'])
        _redraw_group(gi)
        print(f'  {prim["acro"]}: number -> acronym'); return
    used = {int(lbl['text']) for g in groups for lbl in g
            if (lbl.get('text') or '').isdigit()}
    n = 1
    while n in used: n += 1
    for lbl in groups[gi]:
        lbl['text'] = str(n)
        lbl['size'] = NUMBER_SIZE
    _redraw_group(gi)
    print(f'  {prim["acro"]} -> {n}')

def toggle_midline_lock():
    """'k' on the active label: snap it onto the midline (z=0) and drop any
    mirror copy. Press again to release the lock (label stays where it is)."""
    gi = active[0]
    if not (0 <= gi < len(groups)):
        print('no active label'); return
    prim = groups[gi][0]
    if prim.get('midline'):
        prim['midline'] = False
        prim['hemi'] = 'R' if prim['site'][2] > 0 else 'L'
        _redraw_group(gi)
        print(f'  {prim["acro"]}: midline lock OFF'); return
    site, normal = _snap_midline(prim['acro'], prim['site'][:2])
    prim['site'] = site; prim['normal'] = normal
    prim['midline'] = True; prim['hemi'] = 'M'
    if len(groups[gi]) > 1:                  # drop the mirror copy
        for lbl in groups[gi][1:]:
            if lbl.get('actor') is not None:
                pl.remove_actor(lbl['actor'])
        groups[gi] = [prim]
    _redraw_group(gi)
    print(f'  {prim["acro"]}: locked to midline (z=0)')

def nxt():
    if groups: _set_active((active[0] + 1) % len(groups))
def prv():
    if groups: _set_active((active[0] - 1) % len(groups))

def delete():
    gi = active[0]
    if not (0 <= gi < len(groups)):
        print('no active label to delete'); return
    for lbl in groups[gi]:
        if lbl.get('actor') is not None:
            pl.remove_actor(lbl['actor'])
    acro = groups[gi][0]['acro']
    groups.pop(gi)
    active[0] = -1                       # nothing active; click to add a new one
    _recolor_pieces()
    print(f'deleted {acro} — click to add a new label')

def undo():
    if not groups: print('nothing to undo'); return
    active[0] = len(groups) - 1
    delete()

def reset():
    for g in groups:
        for lbl in g:
            if lbl.get('actor') is not None: pl.remove_actor(lbl['actor'])
    groups.clear(); active[0] = -1
    _recolor_pieces()
    print('reset all')

def save():
    data = {}
    for g in groups:
        for lbl in g:
            entry = dict(point=lbl['site'].tolist(), normal=lbl['normal'].tolist(),
                         hemi=lbl['hemi'], angle=lbl['angle'], letter_mm=lbl['size'])
            if lbl.get('text') is not None:
                entry['text'] = lbl['text']
            if lbl.get('midline'):
                entry['midline'] = True
            data.setdefault(lbl['acro'], []).append(entry)
    POS_JSON.write_text(json.dumps(data, indent=2))
    # legend: number -> acronym, derived from labels whose text is a digit
    legend = {}
    for g in groups:
        prim = g[0]
        t = prim.get('text')
        if t and t.isdigit():
            legend[t] = prim['acro']
    legend = {k: legend[k] for k in sorted(legend, key=int)}
    legend_path = OUT_DIR / 'number_legend.json'
    if legend:
        legend_path.write_text(json.dumps(legend, indent=2))
        print(f'>> saved {POS_JSON.name} ({len(_all_labels())} labels), '
              f'{legend_path.name} ({len(legend)} entries)')
    else:
        if legend_path.exists(): legend_path.unlink()
        print(f'>> saved {POS_JSON.name} ({len(_all_labels())} labels)')

def toggle_bilateral():
    BILATERAL[0] = not BILATERAL[0]
    print(f'bilateral mirroring: {"ON" if BILATERAL[0] else "OFF"}')

def _load_existing():
    if not POS_JSON.exists():
        return
    data = json.loads(POS_JSON.read_text())
    for acro, lst in data.items():
        R = [e for e in lst if e['point'][2] > 0]
        L = [e for e in lst if e['point'][2] <= 0]
        used = set()
        def mk(e):
            return _make(acro, e['point'], e['normal'],
                         e.get('angle', 0.0), e.get('letter_mm', DEFAULT_SIZE),
                         text=e.get('text'), midline=e.get('midline', False))
        for er in R:                       # pair each R with its mirror L
            group = [mk(er)]
            refl = np.array([er['point'][0], er['point'][1], -er['point'][2]])
            best = None; bestd = 5.0
            for j, el in enumerate(L):
                if j in used: continue
                dd = np.linalg.norm(np.array(el['point']) - refl)
                if dd < bestd: bestd = dd; best = j
            if best is not None:
                used.add(best); group.append(mk(L[best]))
            groups.append(group)
        for j, el in enumerate(L):         # any unmatched L
            if j not in used:
                groups.append([mk(el)])
    for gi in range(len(groups)):
        _redraw_group(gi)
    print(f'loaded {len(_all_labels())} existing labels ({len(groups)} groups)')

def cycle_explode():
    """'e' cycles outward through EXPLODE_LEVELS, wrapping back to 0. Labels
    stay anchored in original coordinates; only the visible mesh + label
    actors shift."""
    EXPLODE_IDX[0] = (EXPLODE_IDX[0] + 1) % len(EXPLODE_LEVELS)
    _build_piece_actors()
    for gi in range(len(groups)):
        _redraw_group(gi)
    print(f'explode x{explode_factor():.1f}'
          + ('  (normal)' if explode_factor() == 0 else ''))

_build_piece_actors()
_load_existing()
_recolor_pieces()   # unlabeled areas turn red

pl.enable_surface_point_picking(callback=click, show_point=False,
                                left_clicking=True, show_message=False,
                                tolerance=0.01)
for keys, fn in [(('Left', 'a'), rot_neg), (('Right', 'd'), rot_pos),
                 (('Up', 'w'), big), (('Down', 'x'), small),
                 (('m',), nudge_medial), (('l',), nudge_lateral),
                 (('n',), nxt), (('p',), prv), (('f',), delete),
                 (('b',), toggle_bilateral), (('e',), cycle_explode),
                 (('k',), toggle_midline_lock), (('j',), toggle_number),
                 (('h',), toggle_hide_mode), (('v',), show_all),
                 (('u',), undo), (('r',), reset), (('s',), save)]:
    for k in keys:
        pl.add_key_event(k, fn)

pl.add_text('RED pieces = not labeled yet\n'
            'LEFT-CLICK pial surface to place/select a label\n'
            'arrows or a/d spin   up/down or w/x resize   m/l nudge medial/lateral\n'
            'h hide-mode (click to hide piece)   v show-all\n'
            'e explode (cycle)   k lock to midline   j toggle number (legend)\n'
            'n/p select   f delete   b bilateral\n'
            'u undo   r reset   s save   q quit',
            font_size=10, position='upper_left')
pl.background_color = 'white'
print('window opening — place, spin, resize; press s to save, q to quit')
pl.show()
