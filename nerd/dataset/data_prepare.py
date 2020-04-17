import numpy as np

from utils.desolve import eval_file


def get_pos(tokens, pos):
    s = 0
    cnt = 0
    for token in tokens:
        if s == pos:
            break
        s += len(token)
        cnt += 1
    return cnt + 1


def train_data_prepare(json_data, name_to_com, tokenizer, max_length):
    # json_data = eval_file(filename)
    # name_to_com = eval_file('../datasets/name_2_company.json')
    ids_mat = []
    offsets = []
    mask_mat = []
    ent_mask_mat = []
    kb_ids = []
    labels = []
    raw_data = []
    for item in json_data:
        sent = item["text"]
        text_id = item["text_id"]
        # because every sample in traindata is annotated only one mention 
        lab_result = item["lab_result"][0]
        mention = lab_result['mention']
        offset_ = lab_result['offset']
        label = lab_result['kb_id']
        kb_id = int(name_to_com[mention][0])
        tokens = tokenizer.tokenize(sent)
        offset = get_pos(tokens, offset_) # count ['CLS']
        tokens = ['[CLS]'] + tokens + ['SEP']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(ids)
        ent_mask = [0] * len(ids)
        ent_mask[offset:] = [1] * len(mention)
        # padding
        length = max_length
        ids += [0] * (length - len(ids))
        mask += [0] * (length - len(mask))
        ent_mask += [0] * (length - len(ent_mask))
        # append
        ids_mat.append(ids)
        # print(len(mask))
        raw_data.append((text_id, sent, mention, offset_, kb_id))
        mask_mat.append(mask)
        ent_mask_mat.append(ent_mask)
        kb_ids.append(kb_id)
        labels.append(0 if label == -1 else 1)
        offsets.append(offset)
    ids_mat = np.array(ids_mat)
    mask_mat = np.array(mask_mat)
    ent_mask_mat = np.array(ent_mask_mat)
    kb_ids = np.array(kb_ids)
    labels = np.array(labels)
    offsets = np.array(offsets)
    # print(ids_mat.shape)
    # print(labels.shape)
    data = {
        'ids': ids_mat,
        'mask_mat': mask_mat,
        'ent_mask': ent_mask_mat,
        'offsets': offsets,
        'kb_ids': kb_ids,
        'labels': labels
    }
    return raw_data, data
