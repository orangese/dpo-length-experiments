#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --time=240:00:00
#SBATCH --job-name=hh-b05-a01
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python -u train.py model=pythia28 batch_size=128 eval_batch_size=4 trainer=FSDPTrainer model.archive=/iris/u/rafailov/DPOExperiments/models/rafailov/pythia2.8b_sft_hh/LATEST/policy.pt model.fsdp_policy_mp=bfloat16 gradient_accumulation_steps=32 loss=dpo loss.beta=0.05 loss.alpha=0.01 datasets=[hh] exp_name=pythia28-hh-b05-a01
