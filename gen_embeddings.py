# @Author: Luning Wang

from utils.encoder_modeling import encoder_list
from utils.data_utils import load_raw_data
from transformers import AutoConfig
from datasets import load_dataset
import numpy as np
import torch
import os
import json
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-uncased', help='Name of path to the model that acts as the retrieval encoder.')
    parser.add_argument('--load_raw', action='store_true', help='Load the local .jsonl dataset file if specified.')
    parser.add_argument('--subset', default='en', choices=['en', 'zh_cn', 'zh_tw', 'ja', 'es', 'de', 'ru'])
    parser.add_argument('--save_name', default='bert', help='The generated embeddings would be saved to `./embeddings/[save_name]_[subset]`')
    parser.add_argument('--additional_info', default='', help='You could add some additional information, and let it be saved to `info.json`')
    args = parser.parse_args()
    print(args)

    save_path = os.path.join('./embeddings', args.save_name + f'_{args.subset}')
    print(f'The generated embeddings would be saved to {save_path}')

    os.makedirs(save_path, exist_ok=True)
    # 2 binary files would be saved to `save_path`, containing the embeddings of docs and queries
    doc_save_path = os.path.join(save_path, 'doc_embeddings.npy')
    query_save_path = os.path.join(save_path, 'query_embeddings.npy')
    info_save_path = os.path.join(save_path, 'info.json')
    
    # construct encoder model
    print('Constructing encoder model ...')
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    arch = model_config.architectures 
    if arch:
        encoder = encoder_list.get(arch[0])(args.model)
        if not encoder:
            logging.error(f'The encoder class for `{arch}` has not been implemented yet, please manually add it in `utils/encoder_modeling.py` according to the instruction.')
            raise NotImplementedError
    elif 'sentence-transformers' in args.model:   # Sbert
        encoder = encoder_list.get('SBert')(args.model)
    else:
        logging.error(f"Could not identify the architecture of this model, for it doesn't have `architectures` in its config file.")
        raise NotImplementedError


    # load dataset
    print('loading dataset ...')
    if args.load_raw:
        raw_path = f'./raw_data/{args.subset}/data_{args.subset}.jsonl'
        _, docs, queries, _, _ = load_raw_data(raw_path)
    else:
        dataset = load_dataset('lnwang/retrieval_qa', name=args.subset)
        docs, queries = dataset['test']['doc'], dataset['test']['query']

    # encode and save
    print('encoding ...')
    encoded_docs, num_items, embed_dim = encoder.encode(docs)
    encoded_queries, num_items_q, _ = encoder.encode(queries)
    assert num_items == num_items_q

    np.save(doc_save_path, encoded_docs)
    np.save(query_save_path, encoded_queries)

    info = {
        'model': args.model,
        'embed_dim': embed_dim,
        'subset': args.subset,
        'num_items': num_items,
        'additional_info': args.additional_info
    }
    with open(info_save_path, 'w') as f:
        json.dump(info, f)
    
    print('embeddings saved.')

