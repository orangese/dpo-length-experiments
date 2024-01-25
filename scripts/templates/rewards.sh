#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=240:00:00
#SBATCH --job-name={dataset}-rewards
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000

python -u train.py model.archive={sft_archive} policy_archive={model_archive} eval_batch_size=2 reward_only=true rewards_save_path={sample_path} trainer=BasicTrainer datasets=[{dataset}] n_eval_model_samples=256 exp_name={dataset}-rewards debug=true
