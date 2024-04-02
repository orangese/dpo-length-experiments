#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:4
#SBATCH --time=240:00:00
#SBATCH --job-name=hh-b1-a0-no-sft
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python -u train.py model=pythia28 batch_size=128 eval_batch_size=8 trainer=FSDPTrainer model.fsdp_policy_mp=bfloat16 gradient_accumulation_steps=8 loss=dpo loss.beta=0.1 loss.alpha=0.0 datasets=[hh] exp_name=pythia28-hh-b1-a0-no-sft n_epochs=1
