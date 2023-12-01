import numpy as np
import faiss

# for the evaluation of the recall rate of a encoder: retrieve and evaluate
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


# to be utilized to retrieve relevant docs, for QA task
def qa_retriever():
    pass
