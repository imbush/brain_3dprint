from allensdk.core.reference_space_cache import ReferenceSpaceCache
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / 'data'
rspc = ReferenceSpaceCache(resolution=25,
                            reference_space_key='annotation/ccf_2017',
                            manifest=str(OUT_DIR / 'reference_space_manifest.json'))
tree = rspc.get_structure_tree()

# What is structure 688?
s = tree.get_structures_by_id([688])[0]
print(f'688 = {s["acronym"]!r}: {s["name"]!r}\n')

have = {}
for f in OUT_DIR.glob('area_*.obj'):
    parts = f.stem.split('_', 2)
    if len(parts) >= 3:
        try:
            have[parts[2]] = int(parts[1])
        except ValueError:
            pass
print(f'areas with mesh files: {len(have)}\n')

# Build markdown table
print(f'{"sid":>10s}  {"acronym":<12s}  {"name":<55s} ancestry')
print('-' * 140)

rows = []
for acro in sorted(have.keys()):
    sid = have[acro]
    s = tree.get_structures_by_id([sid])[0]
    chain = [tree.get_structures_by_id([int(asid)])[0]['acronym']
              for asid in s['structure_id_path']]
    ancestry = ' > '.join(chain[1:-1])
    rows.append((acro, sid, s['name'], ancestry))
    print(f'{sid:>10d}  {acro:<12s}  {s["name"][:53]:<55s} {ancestry}')

# Identify suspect non-cortex
print('\n=== suspected non-cortex (ancestors include non-CTX divisions) ===')
for acro, sid, name, ancestry in rows:
    if any(x in ancestry for x in ('HPF', 'OLF', 'CTXsp', 'sAMY')):
        print(f'  {acro:<12s}  {name:<50s}  via {ancestry}')

# Save as markdown table for the user
md = ['# Cortical areas included in the per-area meshes',
       '',
       f'(structure_id 688 in Allen is {s["acronym"]!r} = '
       f'{s["name"]!r}; descendants include some allocortex/archicortex)',
       '',
       '| sid | acronym | full name | ancestry |',
       '|---:|---|---|---|']
for acro, sid, name, ancestry in rows:
    md.append(f'| {sid} | **{acro}** | {name} | {ancestry} |')
(OUT_DIR / 'cortical_areas_table.md').write_text('\n'.join(md))
print(f'\nwrote cortical_areas_table.md ({len(rows)} rows)')
