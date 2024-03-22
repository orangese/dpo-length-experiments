#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:4
#SBATCH --time=240:00:00
#SBATCH --job-name=uf-sample
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000

cd ../FastChat/fastchat/llm_judge/
python gen_model_answer.py --model-path alignment-handbook/zephyr-7b-sft-full --model-id zephyr7b-sft-beam50-ngram4 --num-gpus-per-model 4 --num_beams 50 --no_repeat_ngram_size 4 --num-gpus-total 4

python gen_model_answer.py --model-path HuggingFaceH4/pref_models_mistral-7b-dpo --revision "v37.0" --model-id zephyr7b-beta0.01-beam50-ngram4 \
	--num-gpus-per-model 4 --num_beams 50 --num-gpus-total 4 --no_repeat_ngram_size 4
python gen_model_answer.py --model-path HuggingFaceH4/pref_models_mistral-7b-dpo --revision "v37.1" --model-id zephyr7b-beta0.1-beam50-ngram4 \
	--num-gpus-per-model 4 --num_beams 50 --num-gpus-total 4 --no_repeat_ngram_size 4
