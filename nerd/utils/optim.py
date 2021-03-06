from torch import optim

def param_split(named_params, lr, lrr):
    p1 = []
    p2 = []
    for name, p in named_params:
        if 'bert' in name or 'Bert' in name:
            p2.append(p)
        else:
            p1.append(p)
    p1 = {"params": p1, "lr": lr, "lr_base": lr}
    p2 = {"params": p2, "lr": lr * lrr, "lr_base": lr * lrr}
    return [p1, p2]

def get_optim(args, model):
    params = param_split(model.named_parameters(), args.lr, args.lrr)

    optimization = optim.Adam(params, betas=(args.b1, args.b2))

    return optimization


def adjust_lr(optim, decay_r):
    for p in optim.param_groups:
        p['lr'] *= decay_r
