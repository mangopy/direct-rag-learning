import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional,Literal,Sequence
from src.tuning.extras import ploting
from src.tuning.data import SFTDataCollatorWith4DAttentionMask, get_dataset
from src.tuning.model import load_model, load_tokenizer
from src.tuning.hparams import get_train_args
from configuration import preparation
from transformers import Seq2SeqTrainingArguments, TrainerCallback
from src.tuning.trainer import SFTSeq2SeqTrainer, WeightedTrainer
from dataclasses import dataclass
import torch


@dataclass
class SFTDataCollator(SFTDataCollatorWith4DAttentionMask):
    r"""
    Data collator for 4d attention mask.
    """  

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        # keys = features[0].keys()
        # print(keys)
        keys = list([k for k in features[0].keys() if k in ['input_ids', 'labels', 'probs', 'sample_id']])
        pad_features = ['input_ids', 'labels']
        extra = {k: torch.tensor([example[k] for example in features]) for k in keys if k not in pad_features}
        features = super().__call__([{k: example[k] for k in pad_features} for example in features])
        features.update(extra)
        return features

def train(callbacks :List['TrainerCallback']= [])-> None:

    args = preparation()
    print(args)
    model_args, data_args, training_args, finetuning_args = get_train_args(vars(args))

    tokenizer = load_tokenizer(model_args)
    dataset_module = get_dataset(stage="sft", data_args=data_args, training_args=training_args, tokenizer=tokenizer)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    print(getattr(model, "is_quantized", False))
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    print("2 -----------------")
    data_collator = SFTDataCollator( # 
        tokenizer=tokenizer,
        pad_to_multiple_of=8,  # for shift short attention
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
    )

    training_args.remove_unused_columns = False
    print('remove columns: ', training_args.remove_unused_columns)

    print("3 -----------------")
    # Initialize our Trainer
    kwargs = dict(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module, 
    )
    _type = os.environ.get('TRAINER', 'SFT')
    print(_type)
    trainer = SFTSeq2SeqTrainer(**kwargs) if  _type == 'sft' else  WeightedTrainer(**kwargs)

    print("starting training")
    # Training
    # if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        ploting.plot_loss(training_args.output_dir, keys=["loss"])
