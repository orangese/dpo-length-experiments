#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=240:00:00
#SBATCH --job-name=save
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000

python -u train.py datasets=[tldr] model=gptj model.name_or_path=CarperAI/openai_summarize_tldr_sft model.tokenizer_name_or_path=EleutherAI/gpt-j-6b save_dpo_format=/iris/u/rypark/cache/gptj-tldr/gptj-tldr-b0-a0 trainer=BasicTrainer debug=true exp_name=gptj-save-sft
