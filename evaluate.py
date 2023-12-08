from utils.prompt_utils import format_query_prompt
from utils.retrieval_utils import encoder_evaluator, qa_evaluator
from utils.plot_utils import draw
from utils.data_utils import load_raw_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
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
    parser.add_argument('--encoder_name', default='bert_en', help='Name of the directory containing saved embeddings')
    parser.add_argument('--task', default='retrieve_only', choices=['retrieve_only', 'qa'])
    
    # for retrieval_only
    parser.add_argument('--num_samples', default=30, help='The number of sample points of top-k')
    parser.add_argument('--verbose', action='store_true', help='Whether to mark some key points in the result graph')
    parser.add_argument('--checkpoints', default='[0.5,0.8,0.9]', help='The key points to mark at verbose mode')
    # for qa
    parser.add_argument('--lm_model', default='lmsys/vicuna-7b-v1.3', help='The LM model to be tested with QA task')
    parser.add_argument('--load_raw', action='store_true', help='Load the local .jsonl dataset file if specified.')
    parser.add_argument('--mode', default='with_correct_doc', choices=['without_doc', 'with_correct_doc', 'with_retrieved_docs'], help='The mode of constructing prompt')
    # specially for retrieval-augmented qa
    parser.add_argument('--subset', default='en', choices=['en', 'zh_cn', 'zh_tw', 'ja', 'es', 'de', 'ru'])
    parser.add_argument('--top_k', default=5)
    args = parser.parse_args()

    print(f"================ TASK: {args.task} ================")

    
    docs_cache_path = os.path.join('./embeddings', args.encoder_name, 'doc_embeddings.npy')
    queries_cache_path = os.path.join('./embeddings', args.encoder_name, 'query_embeddings.npy')
    if args.task == 'retrieve_only':
        result_save_path = os.path.join('./results/retrieval_results', args.encoder_name)
    elif args.task == 'qa':
        result_save_path = os.path.join('./results/qa_results', args.lm_model.split('/')[-1])
    # elif args.mode == 'without_doc':
    #     result_save_path = os.path.join('./results/qa_results', 'without_doc', args.lm_model.split('/')[-1])
    # elif args.mode == 'with_correct_doc':
    #     result_save_path = os.path.join('./results/qa_results', 'with_correct_doc', args.lm_model.split('/')[-1])
    # elif args.mode == 'with_retrieved_doc':
    #     result_save_path = os.path.join('./results/qa_results', 'with_retrieved_docs', args.encoder_name, args.lm_model.split('/')[-1])

    os.makedirs(result_save_path, exist_ok=True)
    print(f'Results would be saved to: {result_save_path}')
    
    if args.task == 'retrieve_only':
        with open(os.path.join('./embeddings', args.encoder_name, 'info.json'), 'r') as f:
            info = json.load(f)
        accs = []
        baselines = []
        top_ks = []

        print('Evaluating ...')
        for top_k in tqdm(range(1, info['num_items'], info['num_items']//args.num_samples)):
            top_ks.append(top_k)
            baseline = math.comb(info['num_items']-1, top_k-1) / math.comb(info['num_items'], top_k)
            baselines.append(baseline)
            acc = encoder_evaluator(embed_dim=info['embed_dim'], top_k=top_k,\
                docs_cache_path=docs_cache_path, queries_cache_path=queries_cache_path)
            accs.append(acc)
        
        draw(top_ks, accs, baselines, num_samples=args.num_samples, subset=info['subset'], model_name=info['model'], \
            result_save_path=result_save_path, verbose=args.verbose, checkpoints=eval(args.checkpoints))

        print('Results saved.')
    
    elif args.task == 'qa':
        embed_dim = -1  # a placeholder for non-retrieval QA
        if args.mode == 'with_retrieved_docs':
            with open(os.path.join('./embeddings', args.encoder_name, 'info.json'), 'r') as f:
                info = json.load(f)
            embed_dim = info['embed_dim']
            assert args.subset == info['subset']    # make sure the language of the embeddings is the same as the QA-prompt

        subset = args.subset
        lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.lm_model, use_fast=False, trust_remote_code=True)

        # load dataset
        print('loading dataset ...')    # column_based
        if args.load_raw:
            raw_path = f'./raw_data/{subset}/data_{subset}.jsonl'
            regions, docs, queries, choices, answers = load_raw_data(raw_path)   
        else:
            dataset = load_dataset('lnwang/retrieval_qa', name=subset)
            regions, docs, queries, choices, answers = dataset['test']['region'], dataset['test']['doc'], dataset['test']['query'], \
                dataset['test']['choice'], dataset['test']['answer']
        
        examples = [{'region':i[0], 'doc': i[1], 'query': i[2], 'choice': i[3], 'answer': i[4]} \
            for i in zip(regions, docs, queries, choices, answers)]   # format the data to adapt to the `qa_evaluator`, being row_based

        acc = qa_evaluator(lm_model=lm_model, tokenizer=tokenizer, examples=examples, mode=args.mode, embed_dim=embed_dim, \
            top_k=args.top_k, subset=subset, docs_cache_path=docs_cache_path, queries_cache_path=queries_cache_path)
        print("========================")
        print(f"Accuracy: {acc}")

        result_save_name = f"result_{args.mode}.md"
        with open(os.path.join(result_save_path, 'result.txt'), 'w') as f:
            f.write(f"# QA-Task Result For {args.lm_model.split('/')[-1]}\n")
            f.write(f"+ Subset: {subset}\n")
            f.write(f"+ Accuracy: {acc}\n")
            if args.mode == 'with_etrieved_docs':
                f.write(f"+ Encoder_name: {args.encoder_name}\n")
                f.write(f"+ Top-k: {args.top_k}")

        print('Results saved.')
        

        





