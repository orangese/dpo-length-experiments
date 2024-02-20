#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:a40:4
#SBATCH --time=480:00:00
#SBATCH --job-name=phi2
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python train.py model=microsoft/phi-2 batch_size=128 eval_batch_size=4 trainer=FSDPTrainer model.archive=/iris/u/rafailov/cache/rafailov/ultrafeedback-phi-sft_2024-02-06_19-37-03_807659/LATEST/policy.pt model.fsdp_policy_mp=bfloat16 gradient_accumulation_steps=16 loss=dpo loss.beta=0.01 loss.alpha=0.0 datasets=[ultrafeedback] exp_name=phi-ultrafeedback-b01-a0 n_epochs=10
