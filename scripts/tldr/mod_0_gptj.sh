#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:a40:4
#SBATCH --time=240:00:00
#SBATCH --job-name=tldr-b5-a0
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000
python -u train.py model=gptj model.name_or_path=CarperAI/openai_summarize_tldr_sft model.tokenizer_name_or_path=EleutherAI/gpt-j-6b batch_size=64 eval_batch_size=4 trainer=FSDPTrainer model.fsdp_policy_mp=bfloat16 gradient_accumulation_steps=16 loss=dpo loss.beta=0.5 loss.alpha=0.0 datasets=[tldr] exp_name=gptj-tldr-b5-a0 n_epochs=1 lr=1e-6
