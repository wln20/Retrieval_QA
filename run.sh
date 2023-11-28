python gen_embeddings.py --model all-MiniLM-L6-v2 --is_sbert --subset en --save_name sbert
python gen_embeddings.py --model all-MiniLM-L6-v2 --is_sbert --subset zh_cn --save_name sbert
python gen_embeddings.py --model all-MiniLM-L6-v2 --is_sbert --subset zh_tw --save_name sbert
python retrieval_eval.py --name sbert_en --verbose
python retrieval_eval.py --name sbert_zh_cn --verbose
python retrieval_eval.py --name sbert_zh_tw --verbose

