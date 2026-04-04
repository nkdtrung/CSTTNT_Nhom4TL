import json
with open('modeling_experiments.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('scale_columns_path', 'selected_features_path')
text = text.replace('scale_columns.npz', 'selected_features.npz')
text = text.replace('scale_columns', 'selected_features')

with open('modeling_experiments.ipynb', 'w', encoding='utf-8') as f:
    f.write(text)
