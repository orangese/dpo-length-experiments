#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=240:00:00
#SBATCH --job-name=shp-sample
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000

python -u train.py model.archive={model_archive} eval_batch_size=16 sample_only=true samples_per_prompt=1 sample_path={sample_path} exp_name={dataset}-sample n_eval_examples=256 debug=true trainer=BasicTrainer datasets=[{dataset}] max_length=1024
