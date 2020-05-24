import json
from copy import deepcopy
fname = '42-ensemble.json'
f = open(fname, 'r', encoding='utf8')
js = json.load(f)

res = js['submit_result']

new_js = {"team_name": "ddl自动机",}
news = []
pn = None
i = None

for item in res:
    # if len(item['text']) > 137:
        # print(item['text']) 
    if item['text_id'] == i:
        pn['mention_result'] += item['mention_result']
    else:
        i = item['text_id']
        pn = deepcopy(item)
        news.append(pn)

new_js['submit_result'] = news

json.dump(new_js, open('merged1-'+fname, 'w', encoding='utf8'), ensure_ascii=False, indent=True)