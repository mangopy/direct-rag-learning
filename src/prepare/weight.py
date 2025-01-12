import json
from collections import defaultdict
import numpy as np
from src.utilize.utilize import extract_numbers_from_ordered_brackets, load_data, write_file
import argparse

user_rank_prompt = """You are RankLLM, an intelligent assistant that can rank passages based on their relevance and usefulness to the user's query.

I will provide you with {n_docs} passages. Please rank these passages based on their usefulness in answering the user's search query: "{question}". 

A passage's usefulness is defined as:
1. Relevance to the question.
2. Contains necessary information to help the user.

The passages are listed below, each with a numerical identifier [].
{docs}

Rank the {n_docs} passages based on their usefulness in descending order. Use the format [] > [], e.g., [2] > [1]. Only respond with the ranking results; do not provide explanations.

Search Query: {question}
Your output: """

def _rank_input(example):
    assert len(example['candidate']) >= 20
    return user_rank_prompt.format(n_docs=len(example['candidate']), docs='\n'.join(example['candidate']), question=example['question'])

def _re_weighted(args):
    gen = json.load(open(args.permutation_reward_file))
    rank = json.load(open(args.permutation_file))
    data = defaultdict(list)
    for line1, line2 in zip(gen, rank):
        assert line1['query_id'] == line2['query_id']
        assert line1['sample_id'] == line2['sample_id']
        data[line1['query_id']].append((line1, line2))
    gen_data = []
    rank_data = []
    sample_num = 1
    keys = list(data.keys())

    def unique(examples):
        _s = set()
        _results = []
        for example in examples:
            if example[1]['text'] not in _s:
                _s.add(example[1]['text'])
                _results.append(example)
        return _results

    cnt = 0
    for k in keys:
        samples = data[k][:sample_num]
        # Deduplication
        samples = unique(samples)
        posterior = [e[0]['log_probs'] for e in samples]
        posterior = -np.array(posterior)
        posterior_probs = np.exp(posterior - np.max(posterior))  # Avoid overflow by subtracting the maximum value
        posterior_probs /= np.sum(posterior_probs)  # Normalization

        for i, sample in enumerate(samples):
            rank = extract_numbers_from_ordered_brackets(sample[1]['text'])
            if len(sample[1]['candidate']) < 20 or len(rank) != 5:
                cnt += 1
                continue

            gen_data.append({
                "prompt": sample[0]['prompt'],
                "response": sample[0]['response'],  # generation model
                "sample_id": i,
                "query_id": k,
                "probs": posterior_probs[i]
            })
            rank_data.append({
                "prompt": _rank_input(sample[1]),
                "response": ' > '.join([f"[{e}]" for e in rank]),  # rank model
                "sample_id": i,
                "query_id": k,
                "probs": posterior_probs[i]
            })
    print(len(gen_data))
    print(cnt)

    file1 = f'{args.output_folder}/train.gen.{len(gen_data)}.sample={sample_num}.json'
    json.dump(gen_data, open(file1, 'w'), indent=4)
    print(file1)

    file2 = f'{args.output_folder}/train.rank.{len(rank_data)}.sample={sample_num}.json'
    json.dump(rank_data, open(file2, 'w'), indent=4)
    print(file2)

def re_weighted():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--permutation_reward_file", type=str)
    parser.add_argument("--permutation_file", type=str)
    args = parser.parse_args()
    _re_weighted(args)