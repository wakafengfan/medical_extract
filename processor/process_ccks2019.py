import json
from pathlib import Path

from configuration.config import data_dir

p1 = (Path(data_dir) / 'ccks2019' / 'ccks2019_subtask1_training_part1.txt').open(encoding='utf-8-sig')
p2 = (Path(data_dir) / 'ccks2019' / 'ccks2019_subtask1_training_part2.txt').open(encoding='utf-8-sig')

data = []

mapping = {
    '药物': 'drug',
    '实验室检验': 'diagnosis',
    '影像检查': 'diagnosis',
    '手术': 'diagnosis',
    '疾病和诊断': 'disease',
}

keys_set = []
for p in [p1, p2]:
    for line in p:
        l = json.loads(line.strip())

        text = l['originalText']
        mention = []

        for ent in l['entities']:
            t = ent['label_type']
            if t not in ['药物', '实验室检验', '影像检查', '手术', '疾病和诊断']:
                continue
            s = ent['start_pos']
            e = ent['end_pos']
            m = text[s:e]

            mention.append((m, s, mapping[t]))
        data.append({
            'text': text,
            'mention': mention
        })

json.dump(data, (Path(data_dir)/'ccks2019.json').open('w'), ensure_ascii=False, indent=4)

print('Done')
