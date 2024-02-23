#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:a40:4
#SBATCH --time=240:00:00
#SBATCH --job-name=tldr-b5-a02
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python -u train.py model.archive=/iris/u/rypark/cache/rypark/tldr-pythia28-sft_2024-02-19_18-52-36_556575/step-59904/policy.pt model=pythia28 batch_size=128 eval_batch_size=4 trainer=FSDPTrainer model.fsdp_policy_mp=bfloat16 gradient_accumulation_steps=8 loss=dpo loss.beta=0.5 loss.alpha=0.02 datasets=[tldr] exp_name=pythia28-tldr-b5-a02 n_epochs=1
