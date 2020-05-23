from tqdm import tqdm
import torch
def train(model, optim, criterion, train_loader):
    model.train()
    # optim = torch.optim.Adam(model.parameters(), lr=lr) #, momentum=0.98, weight_decay=2e-5)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    # criterion = nn.BCEWithLogitsLoss()
    for i, sample in pbar:
        sent = sample['ids'].cuda()
        sent_mask = sample['mask_mat'].cuda()
        ent_mask = sample['ent_mask'].float().cuda()
        entity_id = sample['kb_ids'].cuda()
        label = sample['labels'].float().cuda().view(-1, 1)
        offset = sample['offset'].cuda()
        pred = model(sent, sent_mask, entity_id, offset, ent_mask)
        loss = criterion(pred, label)
        loss_ = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description('loss %f' % (loss_))

def evaluate(model, dataset, dev_loader, epoch):
    with torch.no_grad():
        model.eval()
        pbar = tqdm(enumerate(dev_loader), total=len(dev_loader))
        for i, sample in pbar:
            sent = sample['ids'].cuda()
            sent_mask = sample['mask_mat'].cuda()
            ent_mask = sample['ent_mask'].float().cuda()
            entity_id = sample['kb_ids'].cuda()
            label = sample['labels'].cuda().view(-1, 1)
            offset = sample['offset'].cuda()
            pred = model(sent, sent_mask, entity_id, offset, ent_mask)
            dataset.get_batch_result(sample, pred)
            ta = ((pred>0.5) == label).sum().item()
            pbar.set_description('acc %f' % (ta / label.shape[0]))
        dataset.write_json(epoch)
        print('evaluating!')
        R, P, F1 = dataset.evaluate(epoch)
        print('R: %f, P: %f, F1, %f \n' % (R, P, F1))
        return F1

def test(model, dataset, loader):
    with torch.no_grad():
        model.eval()
        pbar = tqdm(enumerate(loader), total=len(loader))
        for i, sample in pbar:
            sent = sample['ids'].cuda()
            sent_mask = sample['mask_mat'].cuda()
            ent_mask = sample['ent_mask'].float().cuda()
            entity_id = sample['kb_ids'].cuda()
            label = sample['labels'].cuda().view(-1, 1)
            offset = sample['offset'].cuda()
            pred = model(sent, sent_mask, entity_id, offset, ent_mask)
            dataset.get_batch_result(sample, pred)
            # ta = ((pred>0.5) == label).sum().item()
            # pbar.set_description('acc %f' % (ta / label.shape[0]))
        dataset.write_json(-1)
        print('result saved')
