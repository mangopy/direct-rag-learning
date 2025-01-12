from datasets import DatasetDict, load_dataset
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple
from src.tuning.data.processors.supervised import (
    preprocess_supervised_dataset,
)
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from src.tuning.extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments
    from ..hparams import DataArguments


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))


def get_preprocess_and_print_func(
    cutoff_len: int,
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
) -> Tuple[Callable, Callable]:
    if stage == "sft":
        preprocess_func = partial(
            preprocess_supervised_dataset,
            tokenizer=tokenizer,
            cutoff_len=cutoff_len,
        )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    elif stage=='parallel sft':
        preprocess_func = partial(
            preprocess_supervised_dataset,
            tokenizer=tokenizer,
            cutoff_len=cutoff_len,
        )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    else:
       raise NotImplementedError

    return preprocess_func, print_function



def get_dataset(
        stage: Literal["pt", "sft", "rm", "ppo", "kto"],
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        tokenizer: "PreTrainedTokenizer",
        remove_column=True,
):
    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = load_dataset('json', data_files=data_args.dataset_name_or_path)['train']
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    with training_args.main_process_first(desc="pre-process dataset"):

        preprocess_func, print_function = get_preprocess_and_print_func(data_args.cutoff_len, stage, tokenizer)

        column_names = list(next(iter(dataset)).keys())
        kwargs = dict(
            num_proc=8 if data_args.preprocessing_num_workers is None else data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache),
            desc="Running tokenizer on dataset",
            remove_columns=column_names if remove_column else [],
            batched=True
        )
        # preprocess_func(dataset)
        dataset = dataset.map(preprocess_func, **kwargs)

        print("training example:")
        print_function(next(iter(dataset)))
        
        dataset_dict = {"train_dataset": dataset}
        return dataset_dict

# def func(line):
#     return {"a": 1}
#
# if __name__ == '__main__':
#     # dataset = load_dataset('json', data_files='/Users/shizhl/Documents/GitHub/FactReward/dataset/small.jsonl')['train']
#     dataset = load_dataset('json', data_files='/Users/shizhl/Documents/GitHub/FactReward/dataset/alpaca_data_en_52k.json')['train']
#     list(next(iter(dataset)).keys())
#     dataset = dataset.map(func,remove_columns=[])
#     for line in dataset:
#         print(line)