import json, sys

with open('notebooks/06_complete_project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
# Print key output cells: final summary (preproc), model results, evaluation
TARGET_CELLS = [34, 35, 36, 48, 49, 50, 51, 52, 53, 56, 57, 58, 59, 60, 77, 85, 86, 87, 88, 89, 96, 97, 98, 104, 105, 106]

for i in TARGET_CELLS:
    if i >= len(cells):
        continue
    c = cells[i]
    ctype = c['cell_type']
    src = ''.join(c['source'])
    print(f"\n{'='*60}")
    print(f"CELL {i} [{ctype.upper()}]")
    print('='*60)
    print(src[:800])
    if ctype == 'code' and c.get('outputs'):
        for out in c['outputs']:
            otype = out.get('output_type','')
            if otype == 'stream':
                text = ''.join(out.get('text', []))
                print(f"\n--- OUTPUT (stream) ---\n{text[:1000]}")
            elif otype in ('execute_result', 'display_data'):
                plain = ''.join(out.get('data', {}).get('text/plain', []))
                html = ''.join(out.get('data', {}).get('text/html', []))
                if plain:
                    print(f"\n--- OUTPUT (result) ---\n{plain[:800]}")
