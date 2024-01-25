import argparse
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import transformers
from all_datasets import get_len


def parse(s, legacy=False):
    try:
        s = s.replace(".json", "")
        if legacy:
            model, ds, _mod, alpha, _beta, beta = s.split("_")[0].split("-")
            step = int(s.split("_")[-1].replace(".json", "").replace("step-", ""))
            assert _mod == "mod" and _beta == "b"
            alpha, beta = float(f"0.{alpha}"), float(f"0.{beta[1:]}")
            return model, ds, alpha, beta, step
        else:
            raise NotImplementedError
    except:
        return


def get_sample_lens(sample_dir, sample_path, tokenizer, legacy=False):
    cfile = f"cache/{sample_path}_lens.pkl"
    try:
        with open(cfile, "rb") as f:
            cache = pickle.load(f)
        print(f"loaded {cfile} from cache")
        loaded = True
    except FileNotFoundError:
        cache = {}
        loaded = False

    ds = []
    kword = "Assistant:"

    with open(os.path.join(sample_dir, sample_path), "r") as f:
        *_, alpha, beta, step = parse(os.path.basename(sample_path), legacy=legacy)
        name = f"DPO (α={alpha}, β={beta})" if beta != 0 else "SFT"
        for v in json.load(f).values():
            v = v[v.rfind(kword) + len(kword) + 1:]
            ds.append({
                "len": get_len(v, tokenizer, cache=cache),
                "model": name,
                "step": step,
                "beta": alpha,
                "alpha": beta
            })

    os.makedirs(os.path.dirname(cfile), exist_ok=True)
    if not loaded:
        with open(cfile, "wb+") as f:
            print(f"caching {cfile} for later...")
            pickle.dump(cache, f)

    return pd.DataFrame(ds)


def plot_intermediate_lengths_single(df, ds_name, model_name):
    df = df.drop(columns=["beta", "alpha", "model"])
    df.loc[:, "step"] = df["step"] / df["step"].max()

    df = df.groupby("step").agg(["mean", "std", "count"]).reset_index()
    df.columns = ["step", "mean_len", "std_len", "count"]
    df['ci_90'] = df.apply(
        lambda row: 1.645 * (row['std_len'] / np.sqrt(row['count'])),
        axis=1
    )

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        df['step'],
        df['mean_len'],
        yerr=df['ci_90'],
        fmt='o',
        capsize=5,
        capthick=1,
        label='90% CI'
    )
    plt.plot(
        np.unique(df['step']),
        np.poly1d(np.polyfit(df['step'], df['mean_len'], 1))(np.unique(df['step'])),
        linestyle=':',
        label=f"R²={np.power(np.corrcoef(df['step'], df['mean_len'])[0, 1], 2):.2f}"
    )
    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Response Length')
    plt.title(f'{model_name} on {ds_name}: Sample Length Evolution')

    plt.savefig(f"lengths/{ds_name.lower()}_{model_name.lower()}_length_evo.png", bbox_inches='tight', dpi=300)


def plot_length_bars(df, ds_name):
    df = df.sort_values(by=["alpha", "beta"])
    mean = df.groupby("model")['len'].mean()
    sorted_models = {m: i for i, m in enumerate(df['model'].unique())}

    fig = plt.figure(figsize=(10, 8))
    ax_main = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=4)
    for model in df['model'].unique():
        sns.kdeplot(
            data=df[df['model'] == model],
            x='len',
            ax=ax_main,
            label=model,
            color=sns.color_palette()[list(df['model'].unique()).index(model)]
        )

    gridspec = fig.add_gridspec(4, 4)
    subplotspec = gridspec.new_subplotspec((0, 0), rowspan=1, colspan=4)
    ax_top = plt.subplot(subplotspec, sharex=ax_main)

    for model in df['model'].unique():
        ax_top.axvline(
            mean[model],
            color=sns.color_palette()[list(df['model'].unique()).index(model)],
            linestyle='--'
        )

    ax_top.get_xaxis().set_visible(False)
    ax_top.get_yaxis().set_visible(False)

    ax_main.set_xlabel('Response Length')
    ax_main.set_ylabel('Density')
    ax_top.set_title(f'{ds_name}: Distribution of Response Lengths by Model')

    handles, labels = ax_main.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: sorted_models[t[0]]))
    ax_main.legend(handles, labels)

    plt.savefig(f"lengths/{ds_name.lower()}_length_dist.png", bbox_inches='tight', dpi=300)


def plot_alpha_vs_length(df, ds_name):
    # plots alpha vs length
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='alpha', y='len', hue='model')

    # add line of best fit and r2 to legend
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        plt.plot(
            np.unique(model_df['alpha']),
            np.poly1d(np.polyfit(model_df['alpha'], model_df['len'], 1))(np.unique(model_df['alpha'])),
            linestyle=':',
            label=f"{model} (R²={np.power(np.corrcoef(model_df['alpha'], model_df['len'])[0, 1], 2):.2f})"
        )
    plt.legend()

    plt.xlabel('α')
    plt.ylabel('Response Length')
    plt.title(f'{ds_name}: Response Length vs. α')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default="../sampled", help="directory of samples")
    parser.add_argument("--dataset", type=str, help="dataset to filter samples by")
    parser.add_argument("--legacy", action="store_true", help="use legacy sample naming scheme")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-2.8b", help="tokenizer to use")
    args = parser.parse_args()

    assert args.legacy, "only legacy naming scheme supported for now"
    assert os.path.isdir(args.sample_dir), f"{args.sample_dir} is not a directory"
    assert args.dataset, "must specify a dataset to filter by"

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    sample_paths = os.listdir(args.sample_dir)
    if args.dataset:
        sample_paths = list(
            filter(
                lambda p: args.dataset in p and p.endswith(".json") and parse(p, args.legacy),
                sample_paths
            )
        )
    print("analyzing samples:")
    for p in sample_paths:
        print(f" - {p}")
    print()

    dfs = pd.concat([
        get_sample_lens(args.sample_dir, p, tokenizer, legacy=args.legacy)
        for p in sample_paths
    ])

    # Plot by individual model names
    for model_name in dfs["model"].unique():
        df = dfs[dfs["model"] == model_name]
        for plotter in [plot_intermediate_lengths_single]:
            plotter(df, args.dataset.upper(), model_name)
    
    # Plot by all model names
    last_steps_idx = dfs.groupby("model")["step"].idxmax()
    dfs = dfs.loc[last_steps_idx]
    print(df["len"].min())

    for plotter in [plot_length_bars]:
        plotter(dfs, args.dataset.upper())
