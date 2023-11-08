#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:titanrtx:3
#SBATCH --time=240:00:00
#SBATCH --job-name=pythia28-hh-base
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python -u train.py model.archive=/iris/u/rafailov/DPOExperiments/models/rafailov/pythia2.8b_sft_hh/LATEST/policy.pt eval_batch_size=12 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_hh_sft.json exp_name=pythia28-hh-sft-sample n_eval_examples=1000 n_eval_model_samples=100 debug=true trainer=BasicTrainer datasets=[hh]
python -u train.py model.archive=/iris/u/rypark/cache/rypark/pythia28-hh-base_2023-11-03_15-41-09_113519/step-180000/policy.pt eval_batch_size=12 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_hh_base_dpo.json exp_name=pythia28-hh-base-dpo-sample n_eval_examples=1000 n_eval_model_samples=100 debug=true trainer=BasicTrainer datasets=[hh]
