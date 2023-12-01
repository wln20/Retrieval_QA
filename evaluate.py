from utils.prompt_utils import format_query_prompt
from utils.retrieval_utils import encoder_evaluator, qa_retriever
from utils.plot_utils import draw
import numpy as np
import matplotlib.pyplot as plt
import faiss
import math
import os
from tqdm import tqdm
import json
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bert_en', help='Name of the directory containing saved embeddings')
    parser.add_argument('--task', default='retrieve_only', choices=['retrieve_only', 'qa'])
    parser.add_argument('--num_samples', default=30, help='The number of sample points of top-k')
    parser.add_argument('--verbose', action='store_true', help='Whether to mark some key points in the result graph')
    parser.add_argument('--checkpoints', default='[0.5,0.8,0.9]', help='The key points to mark at verbose mode')
    args = parser.parse_args()

    
    docs_cache_path = os.path.join('./embeddings/retrieval_embeddings', args.name, 'doc_embeddings.npy')
    queries_cache_path = os.path.join('./embeddings/retrieval_embeddings', args.name, 'query_embeddings.npy')
    result_save_path = os.path.join('./results/retrieval_results', args.name)
    os.makedirs(result_save_path, exist_ok=True)
    print(f'Results would be saved to: {result_save_path}.')
    
    with open(os.path.join('./embeddings/retrieval_embeddings', args.name, 'info.json'), 'r') as f:
        info = json.load(f)

    accs = []
    baselines = []
    top_ks = []
    print('Evaluating ...')
    for top_k in tqdm(range(1, info['num_items'], info['num_items']//args.num_samples )):
        top_ks.append(top_k)
        baseline = math.comb(info['num_items']-1, top_k-1) / math.comb(info['num_items'], top_k)
        baselines.append(baseline)
        acc = encoder_evaluator(embed_dim=info['embed_dim'], top_k=top_k,\
             docs_cache_path=docs_cache_path, queries_cache_path=queries_cache_path)
        accs.append(acc)
    
    draw(top_ks, accs, baselines, num_samples=args.num_samples, subset=info['subset'], model_name=info['model'], \
        result_save_path=result_save_path, verbose=args.verbose, checkpoints=eval(args.checkpoints))

    print('Results saved.')



