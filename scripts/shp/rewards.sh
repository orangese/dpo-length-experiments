#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=240:00:00
#SBATCH --job-name=pythia28-shp-rewards
#SBATCH --output slurm/%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rypark@stanford.edu

source env/bin/activate

##### BASE DPO
python -u train.py model.archive=/iris/u/rafailov/DPOExperiments/models/rafailov/pythia2.8b_sft_shp/LATEST/policy.pt policy_archive=/iris/u/rypark/cache/rypark/shp-pythia28-mod-0_2023-12-27_12-09-55_579462/step-159744/policy.pt eval_batch_size=4 reward_only=true rewards_save_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia2.8b_shp_base_dpo_rewards.csv trainer=BasicTrainer datasets=[shp] n_eval_model_samples=1000 n_eval_examples=1000

#### ALPHA 0.005 OLD
python -u train.py model.archive=/iris/u/rafailov/DPOExperiments/models/rafailov/pythia2.8b_sft_shp/LATEST/policy.pt policy_archive=/iris/u/rypark/cache/rypark/shp-pythia28-mod-005_2023-12-28_19-22-43_109845/step-199680/policy.pt eval_batch_size=4 reward_only=true rewards_save_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia2.8b_shp_alpha005_rewards.csv trainer=BasicTrainer datasets=[shp] n_eval_model_samples=1000 n_eval_examples=1000

#### ALPHA 0.01 OLD
python -u train.py model.archive=/iris/u/rafailov/DPOExperiments/models/rafailov/pythia2.8b_sft_shp/LATEST/policy.pt policy_archive=/iris/u/rypark/cache/rypark/shp-pythia28-mod-01_2023-12-27_22-12-47_942163/step-199680/policy.pt eval_batch_size=4 reward_only=true rewards_save_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia2.8b_shp_alpha01_rewards.csv trainer=BasicTrainer datasets=[shp] n_eval_model_samples=1000 n_eval_examples=1000

#### SFT SAMPLING
#python -u train.py model.archive=/iris/u/rafailov/DPOExperiments/models/rafailov/pythia2.8b_sft_shp/LATEST/policy.pt eval_batch_size=6 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_sft_full.json exp_name=pythia28-shp-sft-sample n_eval_examples=10000 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]

#### LENGTH DPO SAMPLING
# alpha = 0.005
#python -u train.py model.archive=/iris/u/rypark/cache/rypark/pythia28-shp-length-small_2023-11-09_01-04-32_208183/step-239976/policy.pt eval_batch_size=4 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_alpha005.json exp_name=pythia28-shp-sft-sample n_eval_examples=100 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]

# alpha = 0.005, inverse sign
#python -u train.py model.archive=/iris/u/rypark/cache/rypark/pythia28-shp-length-small-64batch-flipped_2023-11-25_14-36-34_071782/step-59904/policy.pt eval_batch_size=4 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_alpha005_invsign.json exp_name=pythia28-shp-sft-sample n_eval_examples=100 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]

# alpha = 0.01
#python -u train.py model.archive=/iris/u/rypark/cache/rypark/pythia28-shp-length-small_2023-11-12_23-57-37_544017/step-99990/policy.pt eval_batch_size=4 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_alpha01.json exp_name=pythia28-shp-sft-sample n_eval_examples=100 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]

# alpha = 0.01, inverse sign
#python -u train.py model.archive=/iris/u/rypark/cache/rypark/pythia28-shp-length-small-64batch-flipped_2023-11-16_09-59-44_843430/step-179712/policy.pt eval_batch_size=4 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_alpha01_invsign_full.json exp_name=pythia28-shp-sft-sample n_eval_examples=10000 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]

#### BASE DPO SAMPLING
#python -u train.py model.archive=/iris/u/rypark/cache/rypark/pythia28-shp-base_2023-11-03_15-41-09_113519/step-180000/policy.pt eval_batch_size=12 sample_only=true samples_per_prompt=1 sample_path=/sailhome/rypark/dpo-length-experiments/sampled/pythia28_shp_base_dpo_full.json exp_name=pythia28-shp-base-dpo-sample n_eval_examples=10000 n_eval_model_samples=10000 debug=true trainer=BasicTrainer datasets=[shp]
