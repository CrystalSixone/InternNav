import gzip, json
import os,sys

raw_data_path = 'data/vln_pe/raw_data/gruvln10/val_seen/val_seen.json.gz'

with gzip.open(raw_data_path, 'rt', encoding='utf-8') as f:
    data = json.load(f)

for item in data['episodes']:
    print(item['instruction_tokens'])
    break