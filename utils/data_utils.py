import os

def load_raw_data(data_path):
    # read from the jsonl file
    print('Reading from the jsonl file ...')
    with open(data_path,'r') as f:
        raw = f.readlines()
        docs = [eval(item)["doc"] for item in raw]
        queries = [eval(item)["query"] for item in raw]
        
    return (docs, queries)

