import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preference_datasets import get_dataset


def plot_length(dataset, name, maxlen):
    results = []
    for completions in dataset.values():
        for pair in completions["pairs"]:
            chosen = completions["responses"][pair[0]]
            rejected = completions["responses"][pair[1]]
            results.append({
                f"Preferred length": min(len(chosen), maxlen),
                f"Dispreferred length": min(len(rejected), maxlen)
            })

    results = pd.DataFrame(results)
    pavg = round(results["Preferred length"].mean(), 1)
    pmed = round(results["Preferred length"].median(), 1)
    plen = round(results["Preferred length"].std(), 1)

    davg = round(results["Dispreferred length"].mean(), 1)
    dmed = round(results["Dispreferred length"].median(), 1)
    dlen = round(results["Dispreferred length"].std(), 1)

    if pavg > davg:
        pavg = "\\multicolumn{1}{B{.}{.}{2,3}}{%.1f}" % pavg
    else:
        davg = "\\multicolumn{1}{B{.}{.}{2,3}}{%.1f}" % davg

    if pmed > dmed:
        pmed = "\\multicolumn{1}{B{.}{.}{2,3}}{%.1f}" % pmed
    else:
        dmed = "\\multicolumn{1}{B{.}{.}{2,3}}{%.1f}" % dmed

    if plen > dlen:
        plen = "\\multicolumn{1}{B{.}{.}{2,3}}{%.0f}" % plen
    else:
        dlen = "\\multicolumn{1}{B{.}{.}{2,3}}{%.0f}" % dlen
    text = f"{name} & {pavg} & {pmed} & {plen} & {davg} & {dmed} & {dlen} \\\\"

    b, r = sns.color_palette("Paired")[1], sns.color_palette("Paired")[5]

    sns.kdeplot(
        data=results,
        palette=[b, r],
        linewidth=3
    )
    plt.xlabel("Sequence length (characters)")
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

    return text


if __name__ == "__main__":
    split = "train"
    info = {
        # datasets from dpo paper
        "hh": ("Anthropic RLHF HH", 1500),
        "shp": ("Stanford Human Preferences", 4000),
        # "se": ("Stack Exchange Full", 10000),
        "tldr": ("Webis TLDR 17", 500),

        # datasets from length paper
        "rlcd": ("RLCD Synthetic", 1200),
        "webgpt": ("WebGPT", 1700),
        "stack": ("Stack Exchange Paired", 6000),

        # other datasets
        # TODO: https://huggingface.co/datasets/openbmb/UltraFeedback/viewer/default/train?row=4
        "alpaca": ("AlpacaFarm", 1500),
    }
    
    if not os.path.exists("figs"):
        os.makedirs("figs")
    
    plt.rcParams.update({'font.size': 15})
    ts = []
    for dataset in info:
        ts.append(plot_length(get_dataset(dataset, split), *info[dataset]))
        plt.savefig(f"figs/{dataset}", dpi=500, bbox_inches="tight")
        plt.clf()

    print("\n".join(ts))
