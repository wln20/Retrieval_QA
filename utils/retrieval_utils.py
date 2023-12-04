from utils.prompt_utils import format_query_prompt
import numpy as np
import faiss
import random
import re

def faiss_retrieve(embed_dim, top_k, docs_cache_path, queries_cache_path):
    """
    The basic function for faiss retrieval
    Params:
        embed_dim: the dimension of the representation vectors
        top_k: if the top_k results include the correct result, then judge it as correct
        docs_cache_path: the cached representation vectors of docs, should be a path of a .npy file if specified, and the cached ndarray should be a 2-d one with shape (docs_num, embed_dim)
        queries_cache_path: the cached representation vectors of queries, should be a path of a .npy file if specified, and the cached ndarray should be a 2-d one with shape (queries_num, embed_dim)
    """
    xb, xq = np.load(docs_cache_path), np.load(queries_cache_path)
    index = faiss.IndexFlatL2(embed_dim)   # build the index
    index.add(xb)                  # add vectors to the index           
    D, I = index.search(xq, top_k)         # I.size = (queries_num, top_k)     
    I = I.tolist()
    return I

def get_response(lm_model, tokenizer, txt):
    in_tokens = tokenizer(txt, return_tensors="pt")['input_ids']
    out_ids = lm_model.generate(in_tokens, max_new_tokens=3, temperature=0.1) # expect to only generate a letter: A/B/C/D
    out_txt = tokenizer.batch_decode(out_ids)[0]
    response = out_txt[len(txt)+3:] # truncate the input
    match = re.findall("[AaBbCcDd]", response)
    return match[0] if match else 'F'   # random.choice(['A', 'B', 'C', 'D'])  # if not a regular response, select a random answer for prediction

# for the evaluation of the recall rate of a encoder
def encoder_evaluator(embed_dim, top_k, docs_cache_path, queries_cache_path):
    # retrieve
    I = faiss_retrieve(embed_dim, top_k, docs_cache_path, queries_cache_path)   # I.size = (queries_num, top_k)
    # calculate acc
    acc = 0
    for i in range(len(I)): # traverse the search result of all the queries
        if i in I[i]:
            acc += 1
    acc = acc / len(I)
    return acc


# for the evaluation of QA task
def qa_evaluator(lm_model, tokenizer, examples, mode, embed_dim, top_k, subset, docs_cache_path, queries_cache_path):
    """
    `lm_model`, `tokenizer`: both should be instantialized objects
    `examples`: contain doc, query, choice, answer, 
        a data item would be like: {"region": "XXX", "doc": "XXX", "query": "XXX?", "choice": [('A', 'XXX'), ('B', 'XXX'), ('C', 'XXX'), ('D', 'XXX')], "answer": "X"}
    `mode`: choose from 'without_doc', 'with_correct_doc', 'with_retrieved_docs'
    if mode == 'with_retrieved_docs', then `retrieved_docs` should be a list of retrieved docs.
    """
    acc = 0
    num = 0
    if mode == 'without_doc' or mode == 'with_correct_doc':
        for example in examples:
            query, answer_gt = format_query_prompt(example, subset=subset, mode=mode)
            answer_predict = get_response(lm_model, tokenizer, query)
            print('----------------------------')
            print(f"{example['region']}\nresponse:{answer_predict}\nanswer_gt:{answer_gt}")
            if answer_predict.lower() == answer_gt.lower():
                acc += 1
            num += 1
        
        
    elif mode == 'with_retrieved_docs':
        # retrieve
        I = faiss_retrieve(embed_dim, top_k, docs_cache_path, queries_cache_path)   # I.size = (queries_num, top_k)
        for i in range(len(examples)):
            example = examples[i]
            retrieved_docs = [examples[j]['doc'] for j in I[i]]
            query, answer_gt = format_query_prompt(example, subset=subset, mode=mode, retrieved_docs=retrieved_docs)
            answer_predict = get_response(lm_model, tokenizer, query)
            print('----------------------------')
            print(f"{example['region']}\nresponse:{answer_predict}\nanswer_gt:{answer_gt}")
            if answer_predict.lower() == answer_gt.lower():
                acc += 1
            num += 1

    acc = acc / num
    return acc

