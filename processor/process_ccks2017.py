import json
from pathlib import Path

from configuration.config import data_dir

mapping = {
    '症状和体征': 'symptom',
    '检查和检验': 'diagnosis',
    '疾病和诊断': 'disease',
    '治疗': 'diagnosis'
}
data = []

for fn in (Path(data_dir)/'ccks2017').iterdir():
    if 'original' not in fn.name:
        continue

    text = ''.join([l.strip() for l in fn.open().readlines()])
    mention = []
    labeled_fn = fn.name.replace('original.txt','')
    labeled_path = (Path(data_dir)/'ccks2017'/labeled_fn).open()

    for l in labeled_path:
        m, s, _, t = l.strip().split('\t')
        if t not in mapping.keys():
            continue
        if t == '治疗' and '术' not in m:
            continue

        mention.append((m, int(s), mapping[t]))

    data.append({
        'text':text,
        'mention':mention
    })


json.dump(data, (Path(data_dir)/'ccks2017.json').open('w'), ensure_ascii=False, indent=4)

print('Done')
