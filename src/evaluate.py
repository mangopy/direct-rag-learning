import math
import os.path
from src.utilize.utilize import load_data, write_file, extract_numbers_from_ordered_brackets
from tqdm import tqdm
import argparse
import json
from src.utilize.metrics import *
from typing import Dict
import pytrec_eval
import random
import logging
import subprocess
import time
import multiprocessing
from instruction import *

logger = logging.getLogger('evaluation')

def collect_docs(example: Dict, k_positive: int = 3, k_retrieval: int = 17):
    positive, retrieval = example['positive'], example['retrieval']
    n = min(k_positive, len(positive))
    m = k_positive + k_retrieval - n
    # docs = [doc for i, doc in enumerate(positive[:n] + retrieval[:m], 1)]
    docs = [doc for i, doc in enumerate(retrieval[:m], 1)]
    return docs

class Evaluator:
    @staticmethod
    def gen_eval(predicts, answers, metrics=None):
        if metrics is None:
            metrics = ['f1', 'em']
        assert len(predicts) == len(answers)
        score = {m: 0. for m in metrics}
        if len(predicts) == 0:
            return score

        if 'f1' in metrics:
            score['f1'] = eval_f1(predicts, [[e] if type(e) == str else e for e in answers])
        if 'acc' in metrics:
            score['acc'] = eval_acc(predicts, [[e] if type(e) == str else e for e in answers])
        if 'em' in metrics:
            score['em'] = eval_em(predicts, [[e] if type(e) == str else e for e in answers])

        return score

    @staticmethod
    def trec_eval(qrels: Dict[str, Dict[str, int]],
                  results: Dict[str, Dict[str, float]],
                  k_values: List[int]) -> Dict[str, float]:
        ndcg, _map, recall = {}, {}, {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
        scores = evaluator.evaluate(results)

        for query_id in scores:
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

        def _normalize(m: dict) -> dict:
            return {k: round(v / len(scores), 5) for k, v in m.items()}

        ndcg = _normalize(ndcg)
        _map = _normalize(_map)
        recall = _normalize(recall)

        all_metrics = {}
        for mt in [ndcg, _map, recall]:
            all_metrics.update(mt)

        return all_metrics

def shell_command(command, directory='./'):
    process = subprocess.Popen(
        command, cwd=directory, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True
    )
    return process  # return `process` class to manage the process

def launch(data, devices, model_name, base_port, func):
    pids = []  # record the sub-processes
    # start the command
    for device in devices:
        port = base_port + int(device)
        print(port)
        command = f"""CUDA_VISIBLE_DEVICES={device} vllm serve {model_name} --port {port} --dtype auto --api-key token-abc123"""
        print(f"Launching model on GPU {device}, listening on port {port}")
        process = shell_command(command)
        if process is not None:
            pids.append(process)  # Store process objects into a list
        else:
            print(f"Failed to launch model on device {device}")

    # wait for command starting
    time.sleep(120)

    # split the data
    length = len(data) // len(pids) + 1
    pool = multiprocessing.Pool(processes=len(pids))
    collects = []
    print('---------- Starting Evaluation ----------')
    for ids, base_url in enumerate(pids):
        collect = data[ids * length: (ids + 1) * length]
        print(f"Processing batch {ids} on {base_url} with {len(collect)} items")
        collects.append(pool.apply_async(func, (ids, model_name, base_url, collect)))
    pool.close()
    pool.join()

    print('---------- Finished Evaluation ----------')
    # collect the data
    results = []
    for i, result in enumerate(collects):
        ids, res = result.get()
        assert ids == i
        results.extend(res)

    # kill of the process
    for process in pids:
        if process is not None:
            process.terminate()  # send SIGTERM signal to kill the process
            process.wait()  # wait for the process finishing

    return results

global_log = './global_log'
def launch_command(model_name, data, devices):
    processes = []
    length = math.floor(len(data) // len(devices)) + 1
    output_files = []
    for ids, device in enumerate(devices):
        model = model_name

        left, right = ids * length, (ids + 1) * length

        input_file = os.path.join(global_log, f'raw.{left}.{right}.json')
        write_file(data[left: right], input_file)

        output_file = os.path.join(global_log, f'cache.{left}.{right}.json')

        log_file = os.path.join(global_log, f'log.{left}.{right}.txt')

        port = 8000 + ids
        # torchrun --master_addr {port} --nproc_per_node=1
        # if os.path.exists(output_file):
            # output_files.append(output_file)
            # continue
        command = f"""CUDA_VISIBLE_DEVICES={device} python ./src/_evaluate.py \
--model_name {model} \
--input_file {input_file} \
--output_file {output_file} > {log_file} 2>&1"""
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
        output_files.append(output_file)
        time.sleep(10)

    print(f'launch {len(processes)} processes...')
    for process in processes:
        process.wait()
    print(f'end {len(processes)} processes...')

    cache = []
    for file in output_files:
        cache.extend(load_data(file))

    assert len(data) == len(cache)
    # for line, response in zip(data, cache):
        # assert line['query_id'] == response['query_id']
    cache = [response['response'] for response in cache]
    return cache

def commend_eval():
    parser = argparse.ArgumentParser(description="Parse configuration.")
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--rank_model_name", type=str, required=False)
    parser.add_argument("--generation_model_name", type=str, required=False)
    parser.add_argument("--k_positive", type=int, required=False, default=0)
    parser.add_argument("--k_retrieval", type=int, required=False, default=20)
    parser.add_argument("--k", type=int, required=False, default=5)
    parser.add_argument("--devices", type=str, required=False, default='0,1,2')
    parser.add_argument("--output_dir", type=str, required=False, default='./log')
    parser.add_argument("--left", type=int, required=False, default=0)
    parser.add_argument("--right", type=int, required=False, default=10000)
    parser.add_argument("--shuffle", type=bool, required=False, default=False)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    devices = args.devices.split(',')
    data = load_data(args.data_path)[args.left:args.right]
    file = ''.join(args.data_path.split('/')[-2:])+ '+' + '/'.join(args.rank_model_name.split('/')[-3:]).replace('/', '_') + '+' + '/'.join(args.generation_model_name.split('/')[-3:]).replace('/', '_') + str(args.shuffle) + '.json'
    output_file = os.path.join(args.output_dir, file)

    instructions, candidates = [], []
    for i, line in enumerate(tqdm(data)):
        candidate = collect_docs(line, args.k_positive, args.k_retrieval)
        if args.shuffle:
            random.shuffle(candidate) # shuffle  47.526.   44.044
        instructions.append({
            "query_id": line['query_id'] if 'query_id' in line else '',
            "instruction": rank_instruction(question=line['question'], docs=candidate)
        })
        print(len(candidate))
        candidates.append(candidate)

    rank_strs = launch_command(args.rank_model_name, instructions, devices)
    instructions = []

    for line, rank_str, candidate in zip(data, rank_strs, candidates):
        docids = extract_numbers_from_ordered_brackets(rank_str)
        print(rank_str, docids)
        docs = [candidate[docid - 1] for docid in docids[:args.k] if 1 <= int(docid) <= len(candidate)]
        instructions.append({
            "query_id": line['query_id'] if 'query_id' in line else '',
            "instruction": generation_instruction(line['question'], docs=docs)
        })
    predicts = launch_command(args.generation_model_name, instructions, devices)

    answers = [line['answer'] for line in data]
    score = Evaluator.gen_eval(predicts, answers)
    print(score)
    print(json.dumps(score, indent=4))
    results = {"score": score,  "config": vars(args), "data": predicts}
    write_file(results, output_file)


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    commend_eval()