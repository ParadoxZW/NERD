import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import os
import math
from tqdm import tqdm
import datetime
# import warnings 
# warnings.filterwarnings('ignore')

from model.nerd import NerdModel
from dataset.nerd_loader import TestData
from config import args, config
from utils.desolve import eval_file
from utils.engine import evaluate, train, test
from utils.optim import get_optim, adjust_lr

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.backends.cudnn.benchmark = True
if args.seed != -1:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

curr_time = datetime.datetime.now()
result_dir = '../results/' + curr_time.strftime("%m-%d-%H-%M")
os.mkdir(result_dir)
os.mkdir(result_dir+'/ckpts')

print('Load datasets!')
name_to_com = eval_file('../datasets/name_2_company.json')
id_to_com = eval_file('../datasets/id_2_company.json')
test_data = TestData('../datasets/test.txt', name_to_com, id_to_com, threshold=args.threshold, result_dir=result_dir)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
# val_data = AnnoData('../datasets/dev.json', name_to_com, id_to_com, threshold=args.threshold, result_dir=result_dir)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
print('datasets successfully loaded!\n')

print('config model...')

model = NerdModel(config)
# torch.cuda.empty_cache()
model = model.cuda()
model = torch.nn.DataParallel(model)
state_dict_ = torch.load('../' + args.ckpt_path, map_location='cpu')
model.load_state_dict(state_dict_)

# optim = get_optim(args, model) #, momentum=0.98, weight_decay=2e-5)
# criterion = nn.BCEWithLogitsLoss()

print('start testing!')

test(model, test_data, test_loader)

# torch.save(model.state_dict(), result_dir + '/ckpts/last.pkl')
