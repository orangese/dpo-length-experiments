import argparse
import torch
import json
import transformers
import matplotlib.pyplot as plt
import numpy as np
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
        device_map='balanced',
    )
    state_dict = torch.load(archive, map_location='cpu')
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


def get_rewards(ds, sft_model, base_model, mod_model, tok, beta):
    for entry in tqdm(ds):
        chosen_tokens = tok(entry["Dispreferred completion"])["input_ids"]
        rejected_tokens = tok(entry["Preferred completion"])["input_ids"]

        # Get sft logps for reward normalization
        sft_chosen_logps = logps(sft_model, chosen_tokens)
        sft_rejected_logps = logps(sft_model, rejected_tokens)

        # Get logps for base and modified models
        base_chosen_logps = logps(base_model, chosen_tokens)
        base_rejected_logps = logps(base_model, rejected_tokens)

        mod_chosen_logps = logps(mod_model, chosen_tokens)
        mod_rejected_logps = logps(mod_model, rejected_tokens)
    
        # Compute the implicit rewards for both models
        entry["base_chosen_rewards"] = beta * (base_chosen_logps - sft_chosen_logps)
        entry["base_rejected_rewards"] = beta * (base_rejected_logps - sft_rejected_logps)
 
        entry["mod_chosen_rewards"] = beta * (mod_chosen_logps - sft_chosen_logps)
        entry["mod_rejected_rewards"] = beta * (mod_rejected_logps - sft_rejected_logps)


def plot_rewards(dataset):
    dataset = pd.DataFrame(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hh')
    parser.add_argument('--name_or_path', type=str, default='EleutherAI/pythia-2.8b')
    parser.add_argument('--cache_dir', type=str, default='~/.cache')
    parser.add_argument('--sft_archive', type=str, default=None)
    parser.add_argument('--base_archive', type=str, default=None)
    parser.add_argument('--mod_archive', type=str, default=None)
    parser.add_argument('--beta', type=float, default=0.01)
    args = parser.parse_args()

    print(f"loading tokenizer {args.name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.name_or_path,
        cache_dir=args.cache_dir
    )
    dataset, dataset_lens = get_lens(args.dataset, tokenizer, 'train', include_completions=True)

    print(f"loading sft model weights from {args.sft_archive}")
    sft_model = load_model(args.name_or_path, args.cache_dir, args.sft_archive)

    print(f"loading base dpo model weights from {args.base_archive}")
    base_model = load_model(args.name_or_path, args.cache_dir, args.base_archive)
 p   
    print(f"loading regularized dpo model weights from {args.mod_archive}")
    mod_model = load_model(args.name_or_path, args.cache_dir, args.mod_archive)

    with torch.no_grad():
        from IPython import embed ; embed()
        get_rewards(dataset_lens, sft_model, base_model, mod_model, args.beta)

    print("plotting now...")
    plot_rewards(dataset_lens, args.dataset)
