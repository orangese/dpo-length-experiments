import os
import argparse
import random
import torch
import json
import transformers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from all_datasets import get_lens, DATASETS

import sys
sys.path.insert(1, "../")
from trainers import _get_batch_logps


def load_model(name_or_path, cache_dir, archive=None):
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        name_or_path,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    print(f"loaded model {name_or_path}")
    if archive is not None:
        state_dict = torch.load(archive, map_location='cuda:0')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {archive} '
              f'with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
    policy.eval()
    return policy


def logps(model, tokens):
    logits = model(tokens).logits.to(torch.float32)
    logps_ = _get_batch_logps(logits, tokens)
    return logps_


def get_rewards(ds, sft_model, base_model, mod_model, tok, beta, local_cache_dir, ds_name):
    for entry in tqdm(ds):
        chosen_tokens = tok(entry["Dispreferred completion"], add_special_tokens=False)["input_ids"]
        chosen_tokens = torch.LongTensor([chosen_tokens + [tok.eos_token_id]])

        rejected_tokens = tok(entry["Preferred completion"], add_special_tokens=False)["input_ids"]
        rejected_tokens = torch.LongTensor([rejected_tokens + [tok.eos_token_id]])

        # Get sft logps for reward normalization
        sft_chosen_logps = logps(sft_model, chosen_tokens)
        sft_rejected_logps = logps(sft_model, rejected_tokens)

        # Get logps for base and modified models
        base_chosen_logps = logps(base_model, chosen_tokens)
        base_rejected_logps = logps(base_model, rejected_tokens)

        mod_chosen_logps = logps(mod_model, chosen_tokens)
        mod_rejected_logps = logps(mod_model, rejected_tokens)
    
        # Compute the implicit rewards for both models
        entry["base_chosen_rewards"] = beta * (base_chosen_logps - sft_chosen_logps).item()
        entry["base_rejected_rewards"] = beta * (base_rejected_logps - sft_rejected_logps).item()
 
        entry["mod_chosen_rewards"] = beta * (mod_chosen_logps - sft_chosen_logps).item()
        entry["mod_rejected_rewards"] = beta * (mod_rejected_logps - sft_rejected_logps).item()

    # Save everything
    dataset = pd.DataFrame()
    dataset.to_csv(os.path.join(local_cache_dir, f"{ds_name}_{beta}_rewards.csv"))


def plot_rewards(dataset, name, plot="concat"):
    dataset = pd.DataFrame(dataset)

    # Fix types
    for seq_type in ["chosen", "rejected"]:
        for model in ["base", "mod"]:
            s = f"{model}_{seq_type}_rewards"
            try:
                dataset[s] = dataset[s].apply(lambda x: x.item())
            except:
                dataset[s] = dataset[s].apply(lambda x: float(x[x.find("[") + 1:x.find("]")]))
    print(name)
    print(dataset)

    # Create the dataset
    min_length = min(dataset["Preferred length"].min(), dataset["Dispreferred length"].min())
    max_length = max(dataset["Preferred length"].max(), dataset["Dispreferred length"].max())
    length_bins = np.linspace(min_length, max_length, 10)

    preferred_length_binned = np.digitize(dataset["Preferred length"], length_bins)
    dispreferred_length_binned = np.digitize(dataset["Dispreferred length"], length_bins)
    
    if plot == "concat":
        df = pd.DataFrame({
            'all_len_binned': np.concatenate([preferred_length_binned, dispreferred_length_binned]),
            'all_len': np.concatenate([dataset["Preferred length"], dataset["Dispreferred length"]]),
            'reward_delta_all': np.concatenate(
                [dataset["mod_chosen_rewards"] - dataset["base_chosen_rewards"],
                dataset["mod_rejected_rewards"] - dataset["base_rejected_rewards"]]
            ),
        })
        categories = {"all": "grey"}
    
    elif plot == "split":
        df = pd.DataFrame({
            'chosen_len_binned': preferred_length_binned,
            'rejected_len_binned': dispreferred_length_binned,
            'chosen_len': dataset["Preferred length"],
            'rejected_len': dataset["Dispreferred length"],
            'reward_delta_chosen': dataset["mod_chosen_rewards"] - dataset["base_chosen_rewards"],
            'reward_delta_rejected': dataset["mod_rejected_rewards"] - dataset["base_rejected_rewards"],
        })
        categories = {"chosen": "red", "rejected": "blue"}

    elif plot == "margins":
        length_bins = np.linspace(dataset["Length difference"].min(), dataset["Length difference"].max(), 10)
        df = pd.DataFrame({
            'margins_len_binned': np.digitize(dataset["Length difference"], length_bins),
            'margins_len': dataset["Length difference"],
            'reward_delta_margins': (
                ((dataset["mod_chosen_rewards"]) - dataset["mod_chosen_rewards"])
                - (dataset["base_rejected_rewards"] - dataset["base_rejected_rewards"])
            )
        })
        categories = {"margins": "grey"}
    
    else:
        raise Exception("Invalid plot type")

    # Plot the categories based on the figure in the rlhf length correlation paper
    plt.figure(figsize=(12, 9))

    for seq_type, color in categories.items():
        for bin_number, group in df.groupby(f'{seq_type}_len_binned'):
            bin_size = group[f'{seq_type}_len'].count()
            plt.scatter(
                group[f'{seq_type}_len'].mean(),
                group[f'reward_delta_{seq_type}'].mean(), 
                s=bin_size * 10, color=color,
                label=seq_type.title() if bin_number == 1 else ""
            )
            plt.arrow(
                group[f'{seq_type}_len'].mean(),
                group[f'reward_delta_{seq_type}'].mean(), 
                0, group[f'reward_delta_{seq_type}'].mean(),
                head_width=2.0, head_length=0.005, fc=color, ec=color
            )

    plt.title(f'{name} Length-Reward Exploitation')
    plt.xlabel('Length')
    plt.ylabel('Implicit DPO Reward')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hh')
    parser.add_argument('--name_or_path', type=str, default='EleutherAI/pythia-2.8b')
    parser.add_argument('--sft_archive', type=str, default=None)
    parser.add_argument('--base_archive', type=str, default=None)
    parser.add_argument('--mod_archive', type=str, default=None)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--cache_dir', type=str, default='~/.cache')
    parser.add_argument('--local_cache_dir', type=str, default='cache')
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()

    print(f"loading tokenizer {args.name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.name_or_path,
        cache_dir=args.cache_dir
    )
    dataset, dataset_lens = get_lens(args.dataset, tokenizer, 'test', include_completions=True)
    random.shuffle(dataset_lens)
    dataset_lens = dataset_lens[:args.n]

    print(f"loading sft model weights from {args.sft_archive}")
    sft_model = load_model(args.name_or_path, args.cache_dir, args.sft_archive)

    print(f"loading base dpo model weights from {args.base_archive}")
    base_model = load_model(args.name_or_path, args.cache_dir, args.base_archive)
    
    print(f"loading regularized dpo model weights from {args.mod_archive}")
    mod_model = load_model(args.name_or_path, args.cache_dir, args.mod_archive)

    with torch.no_grad():
        get_rewards(dataset_lens, sft_model, base_model, mod_model, tokenizer, args.beta, args.local_cache_dir, args.dataset)

