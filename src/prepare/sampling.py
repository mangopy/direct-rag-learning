import os.path
import sys
import time
import random
from src.instruction import generation_instruction, rank_instruction
from typing import Dict, List, Sequence, Tuple, Any
import argparse
from src.utilize.utilize import extract_numbers_from_ordered_brackets, load_data, write_file
import math
import subprocess
import logging

local_cache = './cache'


def get_command(execute_file: str, command_args: Dict, device: List[int], log_file: str):
    args_str = []
    for k, v in command_args.items():
        args_str.append(f'--{k} {v}')
    args_str = ' '.join(args_str)
    command = f"""CUDA_VISIBLE_DEVICES={device} python -u {execute_file} \
{args_str} > {log_file} 2>&1"""
    logging.info(command)
    return command


def launch_command(execute_file, command_args, data, devices, cache_folder) -> List[Dict]:
    processes = []
    length = math.ceil(len(data) / len(devices))
    output_files = []
    for ids, device in enumerate(devices):
        left, right = ids * length, (ids + 1) * length

        input_file = os.path.join(cache_folder, f'raw.{left}.{right}.json')
        write_file(data[left: right], input_file)

        output_file = os.path.join(cache_folder, f'cache.{left}.{right}.json')

        command_args['input_file'] = input_file
        command_args['output_file'] = output_file
        log_file = os.path.join(cache_folder, f'log.{left}.{right}.txt')
        command = get_command(execute_file, command_args, device, log_file)
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
        output_files.append(output_file)
        time.sleep(5)

    for process in processes:
        process.wait()

    cache = []
    for file in output_files:
        cache.extend(load_data(file))

    return cache


def _get_rank_input(example, k_positive, k_retrieval, shuffle=False):
    positive, retrieval = example['positive'], example['retrieval']
    a = min(k_positive, len(positive))
    b = k_positive + k_retrieval - a
    candidate = positive[:a] + retrieval[:b]
    if shuffle:
        random.shuffle(candidate)
    instruction = rank_instruction(example['question'], docs=candidate)
    return candidate, instruction


def compute_posterior(args, permutation_reward_file, devices):
    # prepare the rank data
    data = load_data(args.data_path)[args.left:args.right]
    samples = []
    if args.permutation_file is not None and os.path.exists(args.permutation_file):
        samples = load_data(args.permutation)
    else:
        prompts, extras = [], []
        for example in data:
            candidate, prompt = _get_rank_input(example, args.k_positive, args.k_retrieval, False)
            prompts.append({"query_id": example['query_id'], "prompt": prompt})
            extras.append({"query_id": example['query_id'], "answer": example['answer'], "candidate": candidate, "question": example['question'], "dataset": example['dataset'] if 'dataset' in example else 'htopot'})
        # rank the doc
        command_args = {"model_name": args.ranker_name, "n": args.sampling_num}
        outputs = launch_command(execute_file='./src/prepare/rank.py', command_args=command_args, data=prompts, devices=devices, cache_folder=local_cache)
        assert len(data) == len(outputs)
        for response, extra in zip(outputs, extras):
            assert response['query_id'] == extra['query_id']
            for i, sample in enumerate(response['outputs']):
                tmp = {"logprobs": sample['cumulative_logprob'], "sample_id": i, "text": sample['text']}
                tmp.update(extra)
                samples.append(tmp)
        write_file(samples, args.permutation)

    key = ['query_id', "sample_id", 'dataset']

    def _extract_doc_id(_sample):
        _candidate = _sample['candidate']
        _doc_ids = extract_numbers_from_ordered_brackets(_sample['text'])
        _doc_ids = [_doc_id for _doc_id in _doc_ids if 1 <= _doc_id <= len(_candidate)]
        if len(_doc_ids) < len(_candidate):
            _doc_ids += [i for i in range(1, len(_candidate) + 1) if i not in _doc_ids]
        _docs = [_candidate[doc_id - 1] for doc_id in _doc_ids]
        return _doc_ids, _docs

    def construct_training_generation(_samples):
        _data = []
        for _sample in _samples:
            _doc_ids, _docs = _extract_doc_id(_sample)
            _tmp = {
                "prompt": generation_instruction(question=_sample['question'], docs=_docs[:args.top_k]),
                "response": _sample['answer'],
            }
            _tmp.update({k: _sample[k] for k in key if k in _sample})
            _data.append(_tmp)
        return _data

    def construct_training_rank(_samples):
        _data = []
        for _sample in _samples:
            _doc_ids, _docs = _extract_doc_id(_sample)
            _tmp = {
                "prompt": rank_instruction(question=_sample['question'], docs=_docs),
                "response": ' > '.join([f"[{_doc_id}]" for _doc_id in _doc_ids])
            }
            _tmp.update({k: _sample[k] for k in key if k in _sample})
            _data.append(_tmp)
        return _data

    examples = construct_training_generation(_samples=samples)
    command_args = {"model_name": args.generator_name, "cutoff_len": args.cutoff_len, "batch_size": args.batch_size}
    results = launch_command(execute_file='./src/prepare/entropy.py', command_args=command_args, data=examples, devices=devices, cache_folder=local_cache)
    for line, example in zip(results, examples):
        assert example['query_id'] == line['query_id'] and example['sample_id'] == line['sample_id']
        example['log_probs'] = line['log_probs']

    write_file(examples, permutation_reward_file)


def offpolicy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--left", default=0, type=int)
    parser.add_argument("--right", default=100000000, type=int)

    parser.add_argument("--ranker_name", type=str)
    parser.add_argument("--generator_name", type=str)

    parser.add_argument("--k_positive", type=int, default=3)
    parser.add_argument("--k_retrieval", type=int, default=17)
    parser.add_argument("--sampling_num", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--devices", type=str, default='0,1')
    parser.add_argument("--permutation", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        logging.info(f'create the folder: {args.output_folder}')
        os.makedirs(args.output_folder, exist_ok=True)

    devices = args.devices.split(',')
    args.permutation = os.path.join(args.output_folder, 'permutation.json')
    permutation_reward = os.path.join(args.output_folder, f'permutation_reward.device-num={len(devices)}.{args.left}.{args.right}.json')
    compute_posterior(args=args, permutation_reward_file=permutation_reward, devices=devices)
