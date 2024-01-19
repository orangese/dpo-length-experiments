import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import transformers

import sys
sys.path.insert(1, "../")
import preference_datasets
from preference_datasets import get_dataset

DATASETS = {
    # datasets from dpo paper
    #"hh": ("Anthropic RLHF HH", 500),
    "shp": ("Stanford Human Preferences", 1000),
    #"se": ("Stack Exchange Full", 10000),
    #"tldr": ("Webis TLDR 17", 150),

    # datasets from length paper
    #"rlcd": ("RLCD Synthetic", 400),
    #"webgpt": ("WebGPT", 400),
    #"stack": ("Stack Exchange Paired", 200),

    # other datasets
    #"ultrafeedback": ("Ultrafeedback Preferences", 10000),
    #"alpaca": ("AlpacaFarm", 500),
}


def get_len(text, tokenizer, cache=None):
    if cache is None:
        cache = {}
    try:
        toklen = cache[text]
    except KeyError:
        toklen = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        cache[text] = toklen
    return toklen


def get_lens(ds_code, tok, split, include_completions=False):
    ds = get_dataset(ds_code, split)
    print(f"{ds_code}: {split=}, n_examples={len(ds)}")

    # Get token lengths
    name, maxlen = DATASETS[ds_code]
    results = []
    if split == "train":
        cfile = f"cache/{name}_lens.pkl"
    else:
        cfile = f"cache/{name}_{split}_lens.pkl"

    try:
        with open(cfile, "rb") as f:
            cache = pickle.load(f)
        print(f"loaded {cfile} from cache")
        loaded = True
    except FileNotFoundError:
        cache = {}
        loaded = False

    for completions in tqdm(ds.values(), desc=f"Tokenizing {ds_code}"):
        for pair in completions["pairs"]:
            chosen = completions["responses"][pair[0]]
            rejected = completions["responses"][pair[1]]
            info = {
                f"Preferred length": min(get_len(chosen, tok, cache=cache), maxlen),
                f"Dispreferred length": min(get_len(rejected, tok, cache=cache), maxlen),
            }
            info["Length difference"] = info["Preferred length"] - info["Dispreferred length"]
            if include_completions:
                info["Preferred completion"] = chosen
                info["Dispreferred completion"] = rejected
            results.append(info)

    os.makedirs(os.path.dirname(cfile), exist_ok=True)
    if not loaded:
        with open(cfile, "wb+") as f:
            print(f"caching {cfile} for later...")
            pickle.dump(cache, f)

    return ds, results


def plot_length(dataset, code, results):
    name, _ = DATASETS[code]

    # Printing for the latex table in the paper
    results = pd.DataFrame(results)
    pavg = round(results["Preferred length"].mean(), 1)
    pmed = round(results["Preferred length"].median(), 1)
    plen = round(results["Preferred length"].std(), 1)

    davg = round(results["Dispreferred length"].mean(), 1)
    dmed = round(results["Dispreferred length"].median(), 1)
    dlen = round(results["Dispreferred length"].std(), 1)

    prev = "\\textbf{%.1f}"
    if pavg > davg:
        pavg = prev % pavg
    else:
        davg = prev % davg

    if pmed > dmed:
        pmed = prev % pmed
    else:
        dmed = prev % dmed

    if plen > dlen:
        plen = prev % plen
    else:
        dlen = prev % dlen
    text = f"{name} & {pavg} & {pmed} & {plen} & {davg} & {dmed} & {dlen} \\\\"

    # Length distribution comparison plot
    b, r = sns.color_palette("Paired")[1], sns.color_palette("Paired")[5]
    sns.kdeplot(
        data=results.drop(["Length difference"], axis=1),
        palette=[b, r],
        linewidth=3
    )
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel(f"Density (n={len(dataset)})")

    plt.title(name)
    plt.gca().axes.get_yaxis().set_ticks([])

    plt.axvline(
        results["Preferred length"].mean(),
        color=b, linestyle='--',
        label=f'Mean preferred'
    )
    plt.axvline(
        results["Dispreferred length"].mean(),
        color=r, linestyle='--',
        label=f'Mean dispreferred'
    )
    plt.savefig(f"figs/{code}", dpi=500, bbox_inches="tight")
    plt.clf()

    # Length difference comparison plot
    sns.kdeplot(
        data=results.drop(["Preferred length", "Dispreferred length"], axis=1),
        x="Length difference",
        linewidth=3
    )
    plt.xlabel("Difference in sequence length in tokens")
    plt.ylabel(f"Density (n={len(dataset)})")
    plt.axvline(
        results["Length difference"].mean(),
        color=b, linestyle='--',
    )
    plt.legend(
        [f'Mean length difference'],
        prop={"size": 15},
        loc='upper left'
    )

    plt.title(name)
    plt.gca().axes.get_yaxis().set_ticks([])

    plt.savefig(f"figs/{code}_diff", dpi=500, bbox_inches="tight")
    plt.clf()

    acc = (results["Length difference"] > 0).values
    return text, acc


def plot_accs(accs):
    names = []
    acc = []
    for name, arr in accs.items():
        names.append(name)
        acc.append(sum(arr) / len(arr))

    df = pd.DataFrame({"Dataset": names, "Accuracy": acc})
    sns.barplot(x="Dataset", y="Accuracy", data=df)
    plt.xticks(rotation=45)
    plt.savefig("figs/accs", dpi=500, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    preference_datasets.LOCAL_PATH = "../data/"
    split = "train"
    if not os.path.exists("figs"):
        os.makedirs("figs")

    plt.rcParams.update({"font.size": 15})
    table_texts = []
    accs = {}

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-2.8b"
    )
    for ds_code in DATASETS:
        dataset, results = get_lens(ds_code, tokenizer, split)
        table, acc = plot_length(dataset, ds_code, results)
        table_texts.append(table)
        accs[ds_code] = acc

    print("\n" + "=" * 80 + "\n")
    print("Table 1: Length statistics")
    print("\n".join(table_texts) + "\n")

    print("Classification accuracy based on length alone")
    print({k: round(sum(v) / len(v), 3) for k, v in accs.items()})

    plot_accs(accs)
