#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=240:00:00
#SBATCH --job-name=pythia28-hh-base
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000

python -u train.py model.archive=/iris/u/rypark/cache/rypark/pythia28-hh-length-small-64batch-flipped_2023-11-25_14-36-34_071782/step-59904/policy.pt eval_batch_size=4 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_hh_alpha005.json exp_name=hh-sample n_eval_examples=256 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[hh]
