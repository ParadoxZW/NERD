import numpy as np

from utils.desolve import eval_file


def get_pos(tokens, pos):
    s = 0
    cnt = 0
    for token in tokens:
        s += len(token)
        cnt += 1
        if s == pos:
            break
    return cnt + 1


def train_data_prepare(json_data, name_to_com, tokenizer, max_length):
    # json_data = eval_file(filename)
    # name_to_com = eval_file('../datasets/name_2_company.json')
    ids_mat = []
    # offsets = []
    mask_mat = []
    ent_mask_mat = []
    kb_ids = []
    labels = []
    for item in json_data:
        sent = item["text"]
        # because every sample in traindata is annotated only one mention 
        lab_result = item["lab_result"][0]
        mention = lab_result['mention']
        offset = lab_result['offset']
        kb_id = lab_result['kb_id']
        tokens = tokenizer.tokenize(sent)
        offset = get_pos(tokens, offset) # count ['CLS']
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
        mask_mat.append(mask)
        ent_mask_mat.append(ent_mask)
        kb_ids.append(int(name_to_com[mention][0]))
        labels.append(0 if kb_id == -1 else 1)
        # offsets.append(offset)
    ids_mat = np.array(ids_mat)
    mask_mat = np.array(mask_mat)
    ent_mask_mat = np.array(ent_mask_mat)
    kb_ids = np.array(kb_ids)
    labels = np.array(labels)
    # print(ids_mat.shape)
    # print(labels.shape)
    return {
        'ids': ids_mat,
        'mask_mat': mask_mat,
        'ent_mask': ent_mask_mat,
        'kb_ids': kb_ids,
        'labels': labels
    }