import ast
import json

with open('premium_app.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('"execution_count": None', '"execution_count": null')
text = text.replace(': True', ': true').replace(': False', ': false')

data = json.loads(text)

with open('premium_app_script.py', 'w', encoding='utf-8') as out:
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                out.write("".join(source))
            else:
                out.write(source)
            out.write("\n\n")
