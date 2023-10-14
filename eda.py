import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preference_datasets import get_dataset

def plot_length(datasets):
    results = []
    for name, dataset in datasets.items():
        for prompt, completions in dataset.items():
            for i, pair in enumerate(completions["pairs"]):
                chosen = completions["responses"][pair[0]]
                rejected = completions["responses"][pair[1]]

                results.append({
#                     "Dataset": name,
#                     "Prompt length": len(prompt),
                    f"({name.upper()}) Chosen length": np.log2(max(len(chosen), 1)),
                    f"({name.upper()}) Rejected length": np.log2(max(len(rejected), 1))
                })

    paired = sns.color_palette("Paired")
    results = pd.DataFrame(results)
    sns.kdeplot(
        data=results,
        palette=[paired[1], paired[5]],
        linewidth=2
    )
    plt.xlabel("log2(sequence length)")

    names = {
        "hh": "Anthropic RLHF HH",
        "shp": "Stanford Human Preferences",
        "se": "Stack Exchange Preferences",
        "tldr": "Webis TLDR 17 Preferences"
    }
    plt.title(f"{', '.join(names[n] for n in datasets)} length distributions")


if __name__ == "__main__":
    datasets = {}
    todo = ["hh"]#, "se"]
    for dataset in todo:
        datasets[dataset] = get_dataset(dataset, split="train")
    plot_length(datasets)
    plt.savefig("-".join(todo) + ".png", dpi=500, bbox_inches="tight")
    plt.show()
