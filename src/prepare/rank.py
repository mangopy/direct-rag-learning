import argparse
import json
import os
from vllm import LLM, SamplingParams


def launch():
    parser = argparse.ArgumentParser(description="Parse configuration.")
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--n", type=int, required=False, default=1)
    parser.add_argument("--input_file", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=False)

    args = parser.parse_args()
    device = os.environ['CUDA_VISIBLE_DEVICES']
    print(f'---------- Begin rank {device}----------')
    llm = LLM(model=args.model_name, tensor_parallel_size=1)
    instructions = json.load(open(args.input_file))
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128, n=args.n)
    outputs = llm.generate(prompts=instructions, sampling_params=sampling_params, use_tqdm=True)
    samples = []
    for line, response in zip(instructions, outputs):
        sample = []
        for output in response.outputs:
            sample.append({"cumulative_logprob": output.cumulative_logprob, "text": output.text})
        samples.append({"query_id": line['query_id'], "outputs": sample})
    json.dump(samples, open(args.output_file, 'w'), indent=4)
    print('---------- Finished rank ----------')

if __name__ == '__main__':
    launch()