def train(model, optim, criterion, train_loader):
    model.train()
    # optim = torch.optim.Adam(model.parameters(), lr=lr) #, momentum=0.98, weight_decay=2e-5)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    # criterion = nn.BCEWithLogitsLoss()
    for i, sample in pbar:
        sent = torch.tensor(sample['ids'], dtype=torch.long).cuda()
        sent_mask = torch.tensor(sample['mask_mat'], dtype=torch.long).cuda()
        ent_mask = torch.tensor(sample['ent_mask'], dtype=torch.float).cuda()
        entity_id = torch.tensor(sample['kb_ids'], dtype=torch.long).cuda()
        label = torch.tensor(sample['labels'], dtype=torch.float).cuda().view(-1, 1)
        offset = torch.tensor(sample['offset'], dtype=torch.long).cuda()
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
            sent = torch.tensor(sample['ids'], dtype=torch.long).cuda()
            sent_mask = torch.tensor(sample['mask_mat'], dtype=torch.long).cuda()
            ent_mask = torch.tensor(sample['ent_mask'], dtype=torch.float).cuda()
            entity_id = torch.tensor(sample['kb_ids'], dtype=torch.long).cuda()
            label = torch.tensor(sample['labels'], dtype=torch.uint8).cuda().view(-1, 1)
            offset = torch.tensor(sample['offset'], dtype=torch.long).cuda()
            pred = model(sent, sent_mask, entity_id, offset, ent_mask)
            dataset.get_batch_result(sample, pred)
            ta = ((pred>0.5) == label).sum().item()
            sa += ta
            pbar.set_description('acc %f' % (ta / label.shape[0]))
        dataset.write_json(epoch)
        print('evaluating!')
        R, P, F1 = dataset.evaluate(epoch)
        print('R: %f, P: %f, F1, %f \n' % (R, P, F1))
        return F1
