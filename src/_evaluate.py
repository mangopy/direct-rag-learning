import argparse
import json
import os
from vllm import LLM, SamplingParams


def launch():
    parser = argparse.ArgumentParser(description="Parse configuration.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    device = os.environ['CUDA_VISIBLE_DEVICES']
    print(f'---------- Begin Evaluation {device}----------')
    llm = LLM(model=args.model_name, tensor_parallel_size=1)
    instruction = json.load(open(args.input_file))
    generation_kwargs = {
        "top_p": 0.95, # no nucleus sampling
        # "do_sample": True, # yes, we want to sample
        # "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_tokens": 128, # specify how many tokens you want to generate at most
        "temperature": 0.8,
    }
    sampling_params = SamplingParams(**generation_kwargs)
    outputs = llm.generate(
        prompts=[line['instruction'] for line in instruction],
        sampling_params=sampling_params,
        use_tqdm=True
    )
    results = [{"query_id": line['query_id'], "response": response.outputs[0].text} for line, response in zip(instruction, outputs)]
    json.dump(results, open(args.output_file, 'w'), indent=4)
    print('---------- Finished Evaluation ----------')

if __name__ == '__main__':
    launch()