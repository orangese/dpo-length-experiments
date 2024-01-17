#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --time=240:00:00
#SBATCH --job-name=tldr-pythia28-mod-0
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python -u train.py model.archive=/iris/u/rypark/cache/rypark/tldr-pythia28-sft_2024-01-16_15-36-36_165241/LATEST/policy.pt model=pythia28 batch_size=128 eval_batch_size=4 trainer=FSDPTrainer model.fsdp_policy_mp=bfloat16 gradient_accumulation_steps=32 loss=dpo loss.beta=0.1 loss.alpha=0.0 datasets=[tldr] exp_name=tldr-pythia28-mod-0
