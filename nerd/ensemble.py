import numpy as np
import math
import datetime
import json
import pickle
import copy
import os

ensemble_dir = '../results/ensemble/'

dirs = os.listdir(ensemble_dir)
n = len(dirs)
que = None
conf = None
preds = None
for filename in dirs:
    pred_pkl = pickle.load(open(ensemble_dir + filename, 'rb'))
    if preds is not None:
        for ix, pred in enumerate(pred_pkl):
            # assert pred['question_id'] == pred_list[ix]['question_id']
            preds[ix]['confidence'] += pred['confidence']
    else:
        preds = copy.deepcopy(pred_pkl)

result_json = {}
result_json["team_name"] = "ddl自动机"
result_json["submit_result"] = []
for pred in preds:
    result_item = {}
    text_id, text, mention, offset, kb_id = pred['question']
    result_item["text_id"] = text_id
    result_item["text"] = text
    result_item["mention_result"] = []
    mention_item = {}
    mention_item["mention"] = mention
    mention_item["offset"] = offset
    # pred /= n 
    p = pred['confidence'] / n
    p = 1 / (1 + math.exp(-p))
    mention_item["kb_id"] = kb_id if p >= 0.5 else -1
    mention_item["confidence"] = max(p, 1 - p)
    result_item["mention_result"].append(mention_item)
    result_json["submit_result"].append(result_item)


filename = '../results/'+str(n)+'-ensemble.json'
json.dump(
    result_json,
    open(filename, 'w', encoding='utf8'), 
    ensure_ascii=False, 
    indent=True)
