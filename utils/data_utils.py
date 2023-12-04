import os

def load_raw_data(data_path):
    # read from the jsonl file
    print('Reading from the jsonl file ...')
    with open(data_path,'r') as f:
        raw = f.readlines()
        regions = [eval(item)["region"] for item in raw]
        docs = [eval(item)["doc"] for item in raw]
        queries = [eval(item)["query"] for item in raw]
        choices = [eval(eval(item)["choice"]) for item in raw]  # become a real list, no longer a str
        answers = [eval(item)["choice"] for item in raw]
        
    return regions, docs, queries, choices, answers

