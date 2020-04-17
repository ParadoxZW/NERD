eval_file = lambda file: eval(open(file, 'r', encoding='utf8').read())

def desolve_str_to_list(file, split_token='\t'):
    desolve = lambda item: item.strip().strip('\n').split(split_token)
    with open(file,'r', encoding='utf8') as file:
        content = [desolve(item) for item in list(file)]
        return content