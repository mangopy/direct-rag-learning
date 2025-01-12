import argparse
from collections import defaultdict
from src.utilize.utilize import *
from src.instruction import *
import random
from typing import Dict, Union, List
import json
import pdb
import os
import requests

SEARCH_URL = "http://0.0.0.0:8893" # you should use your local retrieval ip and port

def document_retrieval(query, k=20):
    url = f'{SEARCH_URL}/api/search?query={query}&k={k}'
    response = requests.get(url)
    res = response.json()
    knowledge = []
    for doc in res['topk']:
        text = doc['text'][doc['text'].index('|') + 1:].replace('"','').strip()
        title = doc['text'][:doc['text'].index('|')].replace('"','').strip()
        knowledge.append(f"Title: {title}. Content: {text}")
    return knowledge

class DataPrepare:
    """
    _transform -> {"question": "", "positive": ["Title: Content"], "retrieval": ["Title: Content"]}
    """
    @staticmethod
    def transform_with_label_assignment(data: List[Dict]) -> List[Dict]:
        dataset = data[0]['dataset']
        for i, example in enumerate(data):
            prefix = ''
            if 'id' in example:
                prefix = example['id']
            elif '_id' in example:
                prefix = example['_id']

            example['query_id'] = f"loop_{i}###{dataset}###{prefix}"
        return [example for example in data]

    @staticmethod
    def post_retrieval_padding(example: Dict):
        for i, doc in enumerate(example['positive']):
            assert 'Content:' in example['positive'][i]
            if '. Content: ' not in example['positive'][i]:
                example['positive'][i] = example['positive'][i].replace('Content:', '. Content: ')

        candidates = document_retrieval(example['question'], 100)
        retrieval_ctx = []
        for doc in candidates[10:]:
            if len(retrieval_ctx)>25:
                break
            # if not has_answer(example['answer'], doc):
            retrieval_ctx.append(doc)
        example['retrieval'] = retrieval_ctx
        return example

    @staticmethod
    def transform_asqa(example: Dict, dataset: str):
        gt = []
        for e in example['annotations']:
            gt += e['knowledge']
        gt = ['Title: ' + e['wikipage'] + '. Content: ' + e['content'] for e in gt]

        return {
            "question": example['ambiguous_question'],
            "positive": gt,
            "retrieval": [],
            "answer": '. '.join([e['long_answer'] for e in example['annotations']]),
            "dataset": dataset,
            "ctx": example['padding_ctxs'],
        }

    @staticmethod
    def transform_wikimultihop_hoppotQA(example: Dict, dataset: str):
        evidence = [e[0] for e in example['supporting_facts']]
        gt, noise = [], []
        for e in example['context']:
            doc = f"Title: {e[0]}. Content: " + ' '.join(e[1])
            gt.append(doc) if e[0] in evidence else noise.append(doc)
        return {
            "question": example['question'],
            "positive": gt,
            "retrieval": noise,
            "answer": example['answer'],
            "dataset": dataset,
            "ctx": example['padding_ctxs'],
        }

    @staticmethod
    def transform_msmacro(example, dataset):
        positive, retrieval = [], []
        answer = sorted(example['answers'], key=lambda x: len(x), reverse=True)[0]
        for i, flag in range(example['passages']['is_selected']):
            if flag:
                positive.append(example['passage']['passage_text'][i])
            else:
                retrieval.append(example['passage']['passage_text'][i])

        return {
            "question": example['query'],
            "answer": answer,
            "positive": positive,
            "retrieval": retrieval,
            "dataset": "msmacro",
            "ctx": [],
        }
    
    @staticmethod
    def transform_musique(example: Dict, dataset: str):
        gt, noise = [], []
        for e in example['paragraphs']:
            doc = f'Title: ' + e['title'] + ". Content: " + e['paragraph_text']
            gt.append(doc) if e['is_supporting'] else noise.append(doc)

        return {
            "question": example['question'],
            "positive": gt,
            "retrieval": noise,
            'answer': example['answer'],
            'dataset': dataset,
            "ctx": example['padding_ctxs'],
        }

    @staticmethod
    def transform_nq(example: Dict, dataset: str):
        gt = [f'Title: ' + e['title'] + ". Content: " + e['text'] for e in example['positive_passages']]
        noise = [f'Title: ' + e['title'] + ". Content: " + e['text'] for e in example['retrieval_passages']]
        return {
            "question": example['query'],
            "positive": gt,
            "retrieval": noise,
            'answer': example['answers'][0],
            'dataset': dataset,
            "ctx": example['padding_ctxs'],
        }

    @staticmethod
    def filter(example: Dict):
        """
        :param example:
        :return:
        """
        # if len(line['output'].split())<5:
        # return False
        if example['answer'] == None or example['answer'] == '':
            return False
        if example['positive'] == [] or example['retrieval'] == []:
            return False

        return True


def prepare_warmup_data_for_generation(
        data: List[Dict],
        k_positive: int,
        k_retrieval: int
) -> List[Dict]:
    def _transform_generation_data(example: Dict):
        n = min(k_positive, len(example['positive']))
        m = k_positive + k_retrieval - n
        docs = [doc for i, doc in enumerate(example['positive'][:n] + example['retrieval'][:m], 1)]
        # random.shuffle(docs)
        return {
            "prompt": generation_instruction(question=example['question'], docs=docs),
            "response": example['answer'],
            "dataset": example['dataset'] if 'dataset' in example else 'msmarco',
            "query_id": example['query_id']
        }

    results = [_transform_generation_data(line) for i, line in enumerate(data)]
    random.shuffle(results)
    return results


def gather_data(
        configs: Dict,
        save_cache: bool = True,
        output_folder: Union[str, None] = None
) -> [List[Dict], List[Dict]]:
    train, test = defaultdict(list), defaultdict(list)
    for k, config in configs.items():
        print(k)
        for split in ['train', 'test', 'dev']:
            if split not in config:
                continue
            tmp = load_data(config[split]['in']) if os.path.exists(config[split]['in']) else []
            tmp = tmp[:config[split]['trunc']]
            tmp = [config['func'](example=line, dataset=k) for line in tqdm(tmp)]
            if tmp != []:
                print(json.dumps(tmp[0], indent=4))
            else:
                print(f"empty file...{config[split]['in']}")
                continue
            tmp = DataPrepare.transform_with_label_assignment(tmp)
            # filter the data using pre-defined criteria
            tmp = [v for v in tmp if (DataPrepare.filter(v) if config['filter'] is None else config['filter'](v))]
            # retrieval for padding
            # tmp = [DataPrepare.post_retrieval_padding(example) for example in tqdm(tmp)]

            if split == 'train':
                train[k] = tmp
            else:
                test[k] = tmp

            key = ['question', 'answer', 'positive', 'retrieval', 'query_id', 'dataset']
            if save_cache and output_folder is not None:
                cache_file = os.path.join(output_folder, f"{k}_{split}.json")
                write_file([{k: line[k] for k in key} for line in tmp], cache_file)

    train = sum(list(train.values()), [])
    test = sum(list(test.values()), [])
    return train, test


root = "DATA_FOLDER"
CONFIG = {
    "hotpotqa": {
        "func": DataPrepare.transform_wikimultihop_hoppotQA,
        "train": {
            # "in": f"{root}/raw/hotpotqa/hotpot_train_v1.1.json",
            "in": f"{root}/hotpot_train_v1.1.jsonl",
            "out": "",
            "trunc": 100000000,
            "sampling_type": "kmeans"
        },
        "dev": {
            # "in": f"{root}/raw/hotpotqa/hotpot_dev_distractor_v1.json",
            "in": f"{root}/hotpot_dev_distractor_v1.jsonl",
            "out": "",
            "trunc": 100000000,
            "sampling_type": "trunc"
        },
        "filter": None,
        "post_retrieval": True
    },
    "wikimultihop": {
        "func": DataPrepare.transform_wikimultihop_hoppotQA,
        "train": {
            # "in": f'{root}/raw/2wikimultiqa/data_ids_april7/train.json',
            "in": f'{root}/2wikitrain.jsonl',
            "out": "",
            "trunc": 100000000,
            "sampling_type": "kmeans"
        },
        "dev": {
            # "in": f'{root}/raw/2wikimultiqa/data_ids_april7/dev.json',
            "in": f'{root}/2wikidev.jsonl',
            "out": "",
            "trunc": 100000000,
            "sampling_type": "trunc"
        },
        "filter": None,
        "post_retrieval": True
    },
    'nq': {
        "func": DataPrepare.transform_nq,
        "train": {
            # "in": f'{root}/raw/nq/train.jsonl',
            "in": f'{root}/nq_train.jsonl',
            "out": "",
            "trunc": 100000000,
            "sampling_type": "kmeans"
        },
        "dev": {
            # "in": f'{root}/raw/nq/dev.jsonl',
            "in": f'{root}/nq_dev.jsonl',
            "out": "",
            "trunc": 100000000,
            "sampling_type": "trunc"
        },
        "filter": None,
        "post_retrieval": True
    },
    "musiqueqa": {
        "func": DataPrepare.transform_musique,
        "train": {
            # "in": f'{root}/raw/musique/musique_ans_v1.0_train.jsonl',
            "in": f'{root}/musique_ans_v1.0_train.jsonl',
            "out": "",
            "trunc": 10000000,
            "sampling_type": "kmeans"
        },
        "dev": {
            # "in": f'{root}/raw/musique/musique_ans_v1.0_dev.jsonl',
            "in": f'{root}/musique_ans_v1.0_dev.jsonl',
            "out": "",
            "trunc": 10000000,
            "sampling_type": "trunc"
        },
        "filter": None,
        "post_retrieval": True
    },
}

def run_data_prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=False)
    parser.add_argument("--warmup_data_file", type=str)
    parser.add_argument("--type", default='short', type=str, )
    parser.add_argument("--dataset", type=lambda x: x.split(','), default='all')
    parser.add_argument("--save_cache", type=bool, default=True)
    parser.add_argument("--shuffle", type=bool, default=True)

    args = parser.parse_args()
    assert args.output_folder is not None

    if args.dataset != ['all']:
        configs = {k: v for k, v in CONFIG.items() if k in args.dataset}
    else:
        configs = CONFIG

    train, test = gather_data(configs=configs, output_folder=args.output_folder, save_cache=args.save_cache)

    train = prepare_warmup_data_for_generation(data=train, k_positive=2, k_retrieval=3)
    write_file(train, os.path.join(args.output_folder, args.warmup_data_file))
    for line in train[:5]:
        print(json.dumps(line, indent=4))
