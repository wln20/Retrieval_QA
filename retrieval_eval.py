import numpy as np
import matplotlib.pyplot as plt
import faiss
import math
import os
from tqdm import tqdm
import json
import argparse



def encoder_evaluator(embed_dim=768, top_k=5, docs_cache_path=None, queries_cache_path=None):
    """
    embed_dim: the dimension of the representation vectors
    top_k: if the top_k results include the correct result, then judge it as correct
    docs_cache_path: the cached representation vectors of docs, should be a path of a .npy file if specified, and the cached ndarray should be a 2-d one with shape (docs_num, embed_dim)
    queries_cache_path: the cached representation vectors of queries, should be a path of a .npy file if specified, and the cached ndarray should be a 2-d one with shape (queries_num, embed_dim)
    """
    # print('Reading from cache ...')
    xb, xq = np.load(docs_cache_path), np.load(queries_cache_path)

    # print('Building index ...')
    index = faiss.IndexFlatL2(embed_dim)   # build the index
    index.add(xb)                  # add vectors to the index 

    # print('Searching ...')             
    D, I = index.search(xq, top_k)         # I.size = (queries_num, top_k)     
    I = I.tolist()

    # calculate acc
    # print('Evaluating ...')
    acc = 0
    for i in range(len(I)): # traverse the search result of all the queries
        if i in I[i]:
            acc += 1
    acc = acc / len(I)
    return acc


def draw(top_ks, accs, baselines, num_samples, model_name, subset, result_save_path, verbose=False, checkpoints = [0.2,0.5,0.8]):
    """
    verbose: whether to draw the details on the graph
    checkpoints: the accs that need to be checked
    """
    plt.figure(figsize=(num_samples//2, 4))
    plt.title(f'acc vs. top_k, subset={subset}')
    plt.xlabel('top_k')
    plt.ylabel('acc')
    plt.xlim(0, len(top_ks))
    plt.ylim(0, 1)
    plt.xticks(top_ks)
    plt.yticks([i / 10 for i in range(11)])
    plt.plot(top_ks, baselines)
    plt.plot(top_ks, accs)
    plt.legend(['baseline', model_name])

    verbose_result = []
    if verbose:
        for y0 in checkpoints:
            index = np.argmin(np.abs(np.array(accs) - y0))
            verbose_result.append((index, top_ks[index], accs[index]))
            # print(index, top_ks[index], accs[index])
            plt.scatter(top_ks[index], accs[index], s=50, color='b')
            plt.text(top_ks[index]-10, accs[index]+0.05, rf'$acc \approx {y0}$', fontdict={'size':15, 'color':'b'})
            plt.plot([top_ks[index], top_ks[index]], [0, accs[index]], 'r--')
        # save a report
        eval_report_path = os.path.join(result_save_path, 'eval_report.md')
        with open(eval_report_path, 'w') as f:
            f.write('# Evaluation report\n')
            f.write('This report shows the top-k levels which are nearest to the checkpoints.\n')
            for i in range(len(checkpoints)):
                f.write(f'+ top_k={verbose_result[i][1]} ~ acc={checkpoints[i]}, true_acc={verbose_result[i][2]:.3f}\n')
            
    save_name = 'eval_results.jpg' if not verbose else 'eval_results_verbose.jpg'
    plt.savefig(os.path.join(result_save_path, save_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bert_en', help='Name of the directory containing saved embeddings')
    parser.add_argument('--num_samples', default=30, help='The number of sample points of top-k')
    parser.add_argument('--verbose', action='store_true', help='Whether to mark some key points in the result graph')
    parser.add_argument('--checkpoints', default='[0.5,0.8,0.9]', help='The key points to mark at verbose mode')

    args = parser.parse_args()

    
    docs_cache_path = os.path.join('embeddings', args.name, 'doc_embeddings.npy')
    queries_cache_path = os.path.join('embeddings', args.name, 'query_embeddings.npy')
    result_save_path = os.path.join('results', args.name)
    os.makedirs(result_save_path, exist_ok=True)
    print(f'Results would be saved to: {result_save_path}.')
    
    with open(os.path.join('embeddings', args.name, 'info.json'), 'r') as f:
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





