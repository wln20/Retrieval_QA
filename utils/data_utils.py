import os

def load_raw_data(doc_path, query_path):
    # read from the jsonl file
    print('Reading from the jsonl file ...')
    with open(doc_path,'r') as f:
        docs_raw = f.readlines()
        docs = [eval(item)["doc"] for item in docs_raw]
    with open(query_path,'r') as f:
        queries_raw = f.readlines()
        queries = [eval(item)["query"] for item in queries_raw]
    return (docs, queries)

