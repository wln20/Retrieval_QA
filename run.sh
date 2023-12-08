# python gen_embeddings.py --model lmsys/vicuna-7b-v1.3 --subset en --save_name vicuna-7b-v1.3
# python gen_embeddings.py --model lmsys/vicuna-7b-v1.3 --subset zh_cn --save_name vicuna-7b-v1.3
# python gen_embeddings.py --model lmsys/vicuna-7b-v1.3 --subset zh_tw --save_name vicuna-7b-v1.3
# python gen_embeddings.py --model lmsys/vicuna-7b-v1.3 --subset ja --save_name vicuna-7b-v1.3
# python gen_embeddings.py --model lmsys/vicuna-7b-v1.3 --subset es --save_name vicuna-7b-v1.3
# python gen_embeddings.py --model lmsys/vicuna-7b-v1.3 --subset de --save_name vicuna-7b-v1.3
# python gen_embeddings.py --model lmsys/vicuna-7b-v1.3 --subset ru --save_name vicuna-7b-v1.3

# python evaluate.py --encoder_name vicuna-7b-v1.3_en --task retrieve_only --verbose
# python evaluate.py --encoder_name vicuna-7b-v1.3_zh_cn --task retrieve_only --verbose
# python evaluate.py --encoder_name vicuna-7b-v1.3_zh_tw --task retrieve_only --verbose
# python evaluate.py --encoder_name vicuna-7b-v1.3_ja --task retrieve_only --verbose
# python evaluate.py --encoder_name vicuna-7b-v1.3_es --task retrieve_only --verbose
# python evaluate.py --encoder_name vicuna-7b-v1.3_de --task retrieve_only --verbose
# python evaluate.py --encoder_name vicuna-7b-v1.3_ru --task retrieve_only --verbose

# python gen_embeddings.py --model /mnt/bn/yukunfeng-nasdrive/wln/models/baichuan2-7B-chat --subset en --save_name baichuan2-7b
# python gen_embeddings.py --model /mnt/bn/yukunfeng-nasdrive/wln/models/baichuan2-7B-chat --subset zh_cn --save_name baichuan2-7b
# python gen_embeddings.py --model /mnt/bn/yukunfeng-nasdrive/wln/models/baichuan2-7B-chat --subset zh_tw --save_name baichuan2-7b
python gen_embeddings.py --model /mnt/bn/yukunfeng-nasdrive/wln/models/baichuan2-7B-chat --subset ja --save_name baichuan2-7b
python gen_embeddings.py --model /mnt/bn/yukunfeng-nasdrive/wln/models/baichuan2-7B-chat --subset es --save_name baichuan2-7b
python gen_embeddings.py --model /mnt/bn/yukunfeng-nasdrive/wln/models/baichuan2-7B-chat --subset de --save_name baichuan2-7b
python gen_embeddings.py --model /mnt/bn/yukunfeng-nasdrive/wln/models/baichuan2-7B-chat --subset ru --save_name baichuan2-7b

python evaluate.py --encoder_name baichuan2-7b_en --task retrieve_only --verbose
python evaluate.py --encoder_name baichuan2-7b_zh_cn --task retrieve_only --verbose
python evaluate.py --encoder_name baichuan2-7b_zh_tw --task retrieve_only --verbose
python evaluate.py --encoder_name baichuan2-7b_ja --task retrieve_only --verbose
python evaluate.py --encoder_name baichuan2-7b_es --task retrieve_only --verbose
python evaluate.py --encoder_name baichuan2-7b_de --task retrieve_only --verbose
python evaluate.py --encoder_name baichuan2-7b_ru --task retrieve_only --verbose