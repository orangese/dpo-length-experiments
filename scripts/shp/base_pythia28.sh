#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=240:00:00
#SBATCH --job-name=pythia28-hh-length
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python -u train.py model=pythia28 batch_size=64 eval_batch_size=4 trainer=FSDPTrainer model.archive=/iris/u/rafailov/DPOExperiments/models/rafailov/pythia2.8b_sft_shp/LATEST/policy.pt model.fsdp_policy_mp=bfloat16 gradient_accumulation_steps=16 loss=dpo loss.beta=0.1 loss.alpha=0.0 datasets=[hh] exp_name=hh-pythia28-base
