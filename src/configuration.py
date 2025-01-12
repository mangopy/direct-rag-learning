import argparse


def preparation():
    parser = argparse.ArgumentParser(description="Training configuration")

    # Model arguments
    parser.add_argument('--model_name_or_path', type=str)

    # Method arguments
    parser.add_argument('--stage', type=str, default='sft')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--finetuning_type', type=str, default='full')
    parser.add_argument('--deepspeed', type=str, default='./script/ds_z3_config.json')

    # Dataset arguments
    parser.add_argument('--dataset_name_or_path', type=str)
    parser.add_argument('--cutoff_len', type=int, default=1024)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--overwrite_cache', type=bool, default=True)
    parser.add_argument('--preprocessing_num_workers', type=int, default=16)
    parser.add_argument('--remove_unused_columns', type=bool, default=False)

    # wandb
    parser.add_argument('--report_to', type=str, default=None)
    parser.add_argument('--run_name', type=str, default='default_train')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='../output')
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--plot_loss', type=bool, default=True)
    parser.add_argument('--overwrite_output_dir', type=bool, default=True)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    # Training arguments
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1.0e-4)
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--bf16', type=bool, default=False)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--ddp_timeout', type=int, default=180000000)

    args = parser.parse_args()
    return args