from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from transformers import PreTrainedTokenizer

logger = get_logger(__name__)

def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def _encode_parallel_supervised_example(
        tokenizer: "PreTrainedTokenizer",
        prompt: Union[Dict[str, str], str],
        response: Union[Dict[str, str], str],
        system: Optional[str],
        cutoff_len: int,
) -> Tuple[List[int], List[int]]:
    input_ids = tokenizer.apply_chat_template(conversation=[prompt, response])
    mask_len = len(tokenizer.apply_chat_template(conversation=[prompt]))
    labels = [IGNORE_INDEX] * mask_len + input_ids[mask_len:]
    return input_ids, labels


def preprocess_supervised_dataset(
        examples: Dict[str, List[Any]],
        tokenizer: "PreTrainedTokenizer",
        cutoff_len: int,
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        input_ids, labels = _encode_parallel_supervised_example(
            tokenizer=tokenizer,
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            cutoff_len=cutoff_len,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs

