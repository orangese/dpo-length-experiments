import pickle
import json
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from collections import defaultdict
from functools import partial

client = OpenAI()#timeout=10)


def load_prompt(pfile, pdir):
    with open(os.path.join(pdir, pfile)) as t:
        prompt = "\n".join(map(str.strip, t.readlines()))
    
    print(f"loaded {pfile}")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print()

    return prompt


def load_hh():
    """
    Loads HH dataset test split.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    reformatted = {}
    i = 0
    kword = "Assistant:"
    for entry in tqdm(dataset):
        s = entry["chosen"].rfind(kword) + len(kword)
        reformatted[entry["chosen"][:s]] = entry["chosen"][s + 1:]
        i += 1
    print(f"loaded {i} examples from HH test split")
    return reformatted


def load_samples(sample_dir, to_process=None):
    """
    Get samples from directory and list of json files.
    Returns a dict with keys corresponding to model names, and values
    corresponding to dict of prompt: response pairs.
    """
    if to_process is None:
        to_process = os.listdir(sample_dir)
    
    sampled = defaultdict(dict)
    kword = "Assistant:"
    for f in to_process:
        if f.endswith(".json"):
            with open(os.path.join(sample_dir, f), "r") as fi:
                tmp = json.load(fi)
                for prompt, v in tmp.items():
                    v = v[0]
                    response = v[v.rfind(kword) + len(kword) + 1:]
                    sampled[f.replace(".json", "")][prompt] = response
   
    print(f"loaded samples from {len(to_process)} models")
    return sampled


def cache_key(a, b, query, prompt, model):
    """
    Formulates cache request for GPT critic.
    """
    return f"{a}::{b}::{query}::{prompt}::{model}"


def judge(a, b, query, prompt, system, model="gpt-4-0613", seed=None, cache_file=None):
    """
    Judges responses a and b based on quality or brevity prompt
    from GPT critic. Randomly flips a and b as well.
    """
    key = cache_key(a, b, query, prompt, model)
    try:
        with open(cache_file, "rb") as c:
            cache = pickle.load(c)
    except (EOFError, FileNotFoundError):
        with open(cache_file, "wb+") as d:
            pass
        cache = {}

    try:
        return cache[key]
    except KeyError:
        pass

    flipped = False
    if random.random() < 0.5:
        a, b = b, a
        flipped = True
    
    prompt = prompt.replace("{{QUERY}}", query.replace("\n", "\\n"))
    prompt = prompt.replace("{{A}}", a.replace("\n", "\\n"))
    prompt = prompt.replace("{{B}}", b.replace("\n", "\\n"))

    ans = None
    while True:
        try:
            compl = client.chat.completions.create(
                messages=[
                    dict(role="system", content=system),
                    dict(role="user", content=prompt)
                ],
                model=model,
                seed=seed
            )
            ans = compl.choices[0].message.content
        except:
            continue
        if ans[-1] in ("A", "B"):
            break

    judgement = ans[-1] == "A" if not flipped else ans[-1] == "B"
    cache[key] = judgement
    with open(cache_file, "wb+") as d:
        pickle.dump(cache, d)

    return judgement


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gpt-4-0613",
        help="gpt critic model to use"
    )
    parser.add_argument(
        "--sample_dir",
        default="sampled",
        help="directory with sampled completions"
    )
    parser.add_argument(
        "--sample_files",
        nargs="*",
        help="list of files in sample_dir to load"
    )
    parser.add_argument(
        "--prompt_dir",
        default="./",
        help="prompt directory"
    )
    parser.add_argument(
        "--seed",
        default=1234,
        help="seed for gpt critic"
    )
    parser.add_argument(
        "--cache",
        default=".gptcache",
        help="cache file for gpt responses"
    )
    args = parser.parse_args()

    quality = load_prompt("quality.prompt", args.prompt_dir)
    brevity = load_prompt("brevity.prompt", args.prompt_dir)
    system = load_prompt("system.prompt", args.prompt_dir)

    sampled = load_samples(args.sample_dir, args.sample_files)
    truth = load_hh()

    model_wins = {}
    judger = partial(
        judge,
        system=system,
        model=args.model,
        seed=args.seed,
        cache_file=args.cache
    )

    for prompt in tqdm(truth):
        try:
            for model in sampled:
                if model not in model_wins:
                    model_wins[model] = defaultdict(list)
                a, b = sampled[model][prompt], truth[prompt]
                model_wins[model]["quality"].append(judger(a, b, prompt, quality))
                model_wins[model]["brevity"].append(judger(a, b, prompt, brevity))

        except KeyError:
            pass

    for model in model_wins:
        print("model:", model)
        for key, scores in model_wins[model].items():
            print("-" * 20)
            print("metric:", key)
            print("len:   ", len(scores))
            print("mean:  ", np.mean(scores))
            print("std:   ", np.std(scores))
        print("=" * 60)

