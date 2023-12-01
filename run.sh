# python gen_embeddings.py --model bert-base-multilingual-cased --subset en --save_name bert-multilingual
# python gen_embeddings.py --model bert-base-multilingual-cased --subset zh_cn --save_name bert-multilingual
# python gen_embeddings.py --model bert-base-multilingual-cased --subset zh_tw --save_name bert-multilingual
python gen_embeddings.py --model bert-base-uncased --load_raw --subset ja --save_name bert
# python retrieval_eval.py --name bert-multilingual_en --verbose
# python retrieval_eval.py --name bert-multilingual_zh_cn --verbose
# python retrieval_eval.py --name bert-multilingual_zh_tw --verbose
python retrieval_eval.py --name bert_ja --verbose

