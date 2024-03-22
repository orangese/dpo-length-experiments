#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=240:00:00
#SBATCH --job-name={dataset_id}-rewards
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000

python -u train.py model.archive={sft_archive} policy_archive={model_archive} eval_batch_size={batch_size} reward_only=true rewards_save_path={sample_path} trainer=BasicTrainer datasets=[{dataset}] n_eval_model_samples=256 exp_name={dataset_id}-rewards debug=true model={model}
