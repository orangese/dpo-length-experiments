#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=240:00:00
#SBATCH --job-name=shp-sample
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate
ulimit -n 64000

#### SFT SAMPLING
#python -u train.py model.archive=/iris/u/rafailov/DPOExperiments/models/rafailov/pythia2.8b_sft_shp/LATEST/policy.pt eval_batch_size=16 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_sft_full.json exp_name=pythia28-shp-sft-sample n_eval_examples=10000 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]

#### LENGTH DPO SAMPLING
# alpha = 0.005
python -u train.py model.archive=/iris/u/rypark/cache/rypark/shp-pythia28-mod-005_2023-12-28_19-22-43_109845/step-199680/policy.pt eval_batch_size=16 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_alpha005.json exp_name=pythia28-shp-alpha005-sample n_eval_examples=10000 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]

# alpha = 0.01
#python -u train.py model.archive=/iris/u/rypark/cache/rypark/shp-pythia28-mod-01_2023-12-27_22-12-47_942163/step-199680/policy.pt eval_batch_size=16 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_alpha01.json exp_name=pythia28-shp-alpha01-sample n_eval_examples=10000 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]

# alpha = 0.02
#python -u train.py model.archive= eval_batch_size=16 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_alpha02.json exp_name=pythia28-shp-alpha02-sample n_eval_examples=10000 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]

#### BASE DPO SAMPLING
#python -u train.py model.archive=/iris/u/rypark/cache/rypark/shp-pythia28-mod-0_2023-12-27_12-09-55_579462/step-179712/policy.pt eval_batch_size=16 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_base_dpo_full.json exp_name=pythia28-shp-base-dpo-sample n_eval_examples=10000 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]
