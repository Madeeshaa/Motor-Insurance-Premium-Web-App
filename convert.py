import json

with open('premium_app.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('premium_app_script.py', 'w', encoding='utf-8') as out:
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                out.write("".join(source))
            else:
                out.write(source)
            out.write("\n\n")
