import json
from pathlib import Path

import requests
from tqdm import tqdm

rule_url = 'http://17510-tech-other-insurance-kbqa.test.za-tech.net/v1/kbqa/get_kbqa_answer/'

model_url = 'http://127.0.0.1:21628/medical_extractor_v2'

o_path = Path('res.csv').open('w')

for line in tqdm(Path('log_disease_corpus_0821.txt').open()):
    line = line.strip().replace(',', '，')
    print(line)
    answer_list = json.loads(requests.post(rule_url, json={'question': line}).content)['details']['result']['answer_list']
    if isinstance(answer_list[0], str) or isinstance(answer_list[0]['right_info'][0], str):
        rule_res = ''
    else:
        rule_res = [r['entity_name'] for r in answer_list[0]['right_info'] if r['entity_type'] == 'disease']
        rule_res = '__'.join(rule_res)

    model_res = json.loads(requests.post(model_url, json={'text': line}).content)
    model_res = [r[0] for r in model_res['tags'] if r[2] in ['disease','symptom','diagnosis']]
    model_res = '__'.join(model_res)
    print(f'== res == : {line},{rule_res},{model_res}')

    o_path.write(f'{line},{rule_res},{model_res}\n')



