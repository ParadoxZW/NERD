import argparse
class NertConfig:
    def __init__(self):
        self.first_ent_emb = True
        self.sum_ent_emb = False
        self.sum_sent_emb = False
        self.cls_pooled = True
        self.hidden_size = 768
        self.activation = 'GeLU'
        self.vocab_size = 327

config = NertConfig()

parser = argparse.ArgumentParser(
    description='nerd training args')
parser.add_argument('--gpu', type=str, required=True,
                    metavar='G', help='gpu')
parser.add_argument('--seed', default=-1, type=int,
                    metavar='S', help='random seed(default: 888)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrr',, default=0.1, type=float,
                    metavar='LRR', help='rate of learning rate, for bert')
parser.add_argument('--lr_decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay')
parser.add_argument('--b1', default=0.9, type=float, metavar='B1',
                    help='beta1')
parser.add_argument('--b2', default=0.98, type=float, metavar='B2',
                    help='beta2')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--decay_epoch', default=7, type=int, metavar='N',
                    help='epochs to decay')
args = parser.parse_args()
print('args:\n' + args.__str__())
