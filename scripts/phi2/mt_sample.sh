#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --time=240:00:00
#SBATCH --job-name=phi-save
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000

cd ../FastChat/fastchat/llm_judge/
python gen_model_answer.py --model-path /iris/u/rypark/cache/rypark/phi2-ultrafeedback-b0-a0 --model-id phi2-b0-a0 --num-gpus-total 4
python gen_model_answer.py --model-path /iris/u/rypark/cache/rypark/phi2-ultrafeedback-b01-a0 --model-id phi2-b01-a0 --num-gpus-total 4
python gen_model_answer.py --model-path /iris/u/rypark/cache/rypark/phi2-ultrafeedback-b01-a0025 --model-id phi2-b01-a0025 --num-gpus-total 4
python gen_model_answer.py --model-path /iris/u/rypark/cache/rypark/phi2-ultrafeedback-b01-a005 --model-id phi2-b01-a005 --num-gpus-total 4
