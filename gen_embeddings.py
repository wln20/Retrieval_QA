# @Author: Luning Wang

from utils.encoder_modeling import BaseEncoder, BertEncoder, BaichuanEncoder
from utils.data_utils import load_raw_data
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import torch
import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-uncased', help='The retrieval encoder model.')
    parser.add_argument('--load_raw', action='store_true', help='Load the local .jsonl dataset file if specified.')
    parser.add_argument('--save_name', default='bert', help='The generated embeddings would be saved to `./embeddings/[save_name]`')
    parser.add_argument('--additional_info', default='', help='You could add some additional information, and let it be saved to `info.json`')
    args = parser.parse_args()

    save_path = os.path.join('embeddings', args.save_name)
    os.makedirs(save_path, exist_ok=True)
    # three files would be saved to `save_path`
    doc_save_path = os.path.join(save_path, 'doc_embeddings.npy')
    query_save_path = os.path.join(save_path, 'query_embeddings.npy')
    info_save_path = os.path.join(save_path, 'info.json')
    
    # construct encoder model
    print('constructing encoder model ...')
    if 'bert' in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
        encoder = BertEncoder(model, tokenizer)
    elif 'baichuan' in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
        encoder = BaichuanEncoder(model, tokenizer)

    
    # load dataset
    print('loading dataset ...')
    if args.load_raw:
        raw_doc_path = './raw_data/docs_countries.jsonl'
        raw_query_path = './raw_data/queries_countries.jsonl'
        docs, queries = load_raw_data(raw_doc_path, raw_query_path)
    else:
        dataset = load_dataset('lnwang/retrieval_qa')
        docs, queries = dataset['test']['doc'], dataset['test']['query']

    # encode and save
    print('encoding ...')
    encoded_docs, num_items, embed_dim = encoder.encode(docs)
    encoded_queries, num_items_q, _ = encoder.encode(queries)
    assert num_items == num_items_q

    np.save(doc_save_path, encoded_docs)
    np.save(query_save_path, encoded_queries)

    info = {
        'model': args.model.split('/')[-1],
        'embed_dim': embed_dim,
        'num_items': num_items,
        'additional_info': args.additional_info
    }
    with open(info_save_path, 'w') as f:
        json.dump(info, f)
    
    print('embeddings saved.')














