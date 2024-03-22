#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:4
#SBATCH --time=240:00:00
#SBATCH --job-name=hh-b1-a01
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python -u train.py model=pythia28 batch_size=64 eval_batch_size=4 trainer=FSDPTrainer model.fsdp_policy_mp=bfloat16 gradient_accumulation_steps=16 loss=dpo loss.beta=0.1 loss.alpha=0.01 datasets=[hh] exp_name=pythia28-hh-b1-a01-continuation optimizer_archive=/iris/u/rypark/cache/rypark/pythia28-hh-length-small-64batch-flipped_2023-11-16_09-59-44_843430/step-199680/optimizer.pt scheduler_archive=/iris/u/rypark/cache/rypark/pythia28-hh-length-small-64batch-flipped_2023-11-16_09-59-44_843430/step-199680/scheduler.pt model.archive=/iris/u/rypark/cache/rypark/pythia28-hh-length-small-64batch-flipped_2023-11-16_09-59-44_843430/step-199680/policy.pt n_epochs=1 sft_archive=/iris/u/rafailov/DPOExperiments/models/rafailov/pythia2.8b_sft_hh/LATEST/policy.pt
