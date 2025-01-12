from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override
from src.tuning.extras.packages import is_transformers_version_equal_to_4_46
from src.tuning.utils import create_custom_optimizer, create_custom_scheduler
from src.tuning.hparams import FinetuningArguments
from torch.nn import CrossEntropyLoss


class SFTSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            print('using create_custom_optimizer... ')
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        # print(inputs.keys())
        loss = super().compute_loss(model, {"labels": inputs['labels'], "input_ids": inputs['input_ids'], "attention_mask": inputs['attention_mask']}, return_outputs, **kwargs)
        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
            # other model should not scale the loss
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps
        return loss


class WeightedTrainer(SFTSeq2SeqTrainer):

    @override
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        probs = inputs.pop('probs')
        sample_id = inputs.pop('sample_id')
        logits = model(**inputs).get("logits")
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_length-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_length-1]
        # # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        
        loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=-100)
        loss = torch.stack([loss_fct(slogit, slabel) for slabel, slogit in zip(shift_labels, shift_logits)])
        # print('-----', loss, probs)
        loss = loss * probs
        loss = loss.sum() / 4# Final loss scalar
        return (loss, outputs) if return_outputs else loss


    # def get_train_dataloader(self) -> DataLoader:
    #     """
    #     Returns the training [`~torch.utils.data.DataLoader`].

    #     Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    #     training if necessary) otherwise.

    #     Subclass and override this method if you want to inject some custom behavior.
    #     """
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")

    #     train_dataset = self.train_dataset
    #     # print(train_dataset['query_id'][:100])
    #     print(train_dataset['sample_id'][:100])
    #     # train_dataset = train_dataset.remove_columns(["query_id"])
    #     # train_dataset = train_dataset.remove_columns(["sample_id"])
    #     data_collator = self.data_collator
    #     if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
    #         train_dataset = self._remove_unused_columns(train_dataset, description="training")
    #     else:
    #         data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    #     dataloader_params = {
    #         "batch_size": self._train_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": 0, #self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #         "persistent_workers": self.args.dataloader_persistent_workers,
    #     }

    #     if not isinstance(train_dataset, torch.utils.data.IterableDataset):
    #         dataloader_params["sampler"] = self._get_train_sampler()
    #         # dataloader_params["sampler"] = SequentialSampler(train_dataset)
    #         dataloader_params["drop_last"] = self.args.dataloader_drop_last
    #         dataloader_params["worker_init_fn"] = seed_worker
    #         dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
    #         print('.............')
    #     dataloader = DataLoader(train_dataset, shuffle=False, **dataloader_params)
    #     cnt=0
    #     for batch in dataloader:
    #         print(batch['probs'])
    #         print(batch['sample_id'])
    #         cnt+=1
    #         if cnt>10:
    #             break
    #     #     print(batch['query_id'])
    #     return self.accelerator.prepare(dataloader)
