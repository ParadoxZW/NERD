import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import os
import math
import datetime
import warnings 
warnings.filterwarnings('ignore')

from model.nerd import NerdModel
from dataset.nerd_loader import InferData
from config import args, config
from utils.desolve import eval_file
from utils.engine import infer

from flask import Flask, Response, request

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.backends.cudnn.benchmark = True
if args.seed != -1:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

version = args.version
result_dir = '../results/' + version
os.makedirs(result_dir, exist_ok=True)

print('Load datasets!')
name_to_com = eval_file('../datasets/name_2_company.json')
id_to_com = eval_file('../datasets/id_2_company.json')
print('datasets successfully loaded!\n')


print('config model...')
model = NerdModel(config)
model = model.cuda()
model = torch.nn.DataParallel(model)
state_dict_ = torch.load('../' + args.ckpt_path, map_location='cpu')
model.load_state_dict(state_dict_)

desolve = lambda item: item.strip().split(split_token)

@app.route('/query/', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        req = request.data
        queries = [desolve(item) for item in req.split('\n')]
        infer_data = InferData(queries, name_to_com, id_to_com, threshold=args.threshold, result_dir=result_dir)
        infer_loader = DataLoader(infer_data, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
        result_json = infer(model, test_data, test_loader)
        return result_json
    else:
        return 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002)