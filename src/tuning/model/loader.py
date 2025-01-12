from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_ms
from .adapter import init_adapter
from .model_utils.misc import register_autoclass
from .patcher import patch_config, patch_model, patch_tokenizer

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

    from ..hparams import FinetuningArguments, ModelArguments

logger = get_logger(__name__)


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def fix_tokenizer(
    tokenizer: "PreTrainedTokenizer",
):
    def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            logger.info("Add eos token: {}".format(tokenizer.eos_token))
        else:
            logger.info("Replace eos token: {}".format(tokenizer.eos_token))

        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))
    return tokenizer


def load_tokenizer(model_args: "ModelArguments") -> "PreTrainedTokenizer":
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)
    tokenizer=fix_tokenizer(tokenizer)
    return tokenizer


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
        tokenizer: "PreTrainedTokenizer",
        model_args: "ModelArguments",
        finetuning_args: "FinetuningArguments",
        is_trainable: bool = False,
        add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(**init_kwargs)
    patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
    register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model
