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
from dataset.nerd_loader import AnnoData, TestData
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
# version = curr_time.strftime("%m-%d-%H-%M")
version = args.version + '-' + str(args.seed)
result_dir = '../results/' + version
os.makedirs(result_dir, exist_ok=True)
os.makedirs(result_dir+'/ckpts', exist_ok=True)

print('Load datasets!')
name_to_com = eval_file('../datasets/name_2_company.json')
id_to_com = eval_file('../datasets/id_2_company.json')
train_data = AnnoData('../datasets/train_all.json', name_to_com, id_to_com)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

val_data = AnnoData('../datasets/dev.json', name_to_com, id_to_com, threshold=args.threshold, result_dir=result_dir)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, drop_last=False)

test_data = TestData('../datasets/test.txt', name_to_com, id_to_com, threshold=args.threshold, result_dir=result_dir, version=version)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
print('datasets successfully loaded!\n')

print('config model and optim...')

model = NerdModel(config)
# torch.cuda.empty_cache()
model = model.cuda()
model = torch.nn.DataParallel(model)

optim = get_optim(args, model) #, momentum=0.98, weight_decay=2e-5)
criterion = nn.BCEWithLogitsLoss()

print('start training!')

for epoch in range(args.epochs):
    if epoch == args.decay_epoch:
        adjust_lr(optim, args.lr_decay)
    print('\nEpoch: %d, LR: %e' % (epoch, optim.param_groups[0]['lr']))
    train(model, optim, criterion, train_loader)
    f1 = evaluate(model, val_data, val_loader, epoch)
    # if f1 > 0.97:
        # torch.save(model.state_dict(), result_dir + '/ckpts/epoch%d_%5f.pkl'%(epoch, f1))

torch.save(model.state_dict(), result_dir + '/ckpts/last.pkl')

print('start testing!')

test(model, test_data, test_loader)
