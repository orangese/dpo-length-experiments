#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --time=240:00:00
#SBATCH --job-name=tldr-pythia28-sft
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python -u train.py model=pythia28 batch_size=64 eval_batch_size=8 gradient_accumulation_steps=8 loss=sft exp_name=tldr-pythia28-sft trainer=FSDPTrainer sample_during_eval=false datasets=[tldr]
