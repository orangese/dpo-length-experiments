import sys
import seaborn as sns
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from all_datasets import get_len


if __name__ == "__main__":
    SAMPLE_DIR = "../sampled"
    try:
        to_process = sys.argv[1:]
    except IndexError:
        pass
    if not to_process:
        to_process = os.listdir(SAMPLE_DIR)
        print('defaulting to listdir')

    print(to_process)

    ds = []
    kword = "Assistant:"

    for f in to_process:
        if f.endswith(".json"):
            print(f"processing {f}")
            with open(os.path.join(SAMPLE_DIR, f), "r") as fi:
                tmp = json.load(fi)
                for k, v in tmp.items():
                    v = v[0]  # only take one sample for now
                    v = v[v.rfind(kword) + len(kword) + 1:]
                    ds.append({
                        "Response length (tokens)": get_len(v),
                        "Model name": f.replace(".json", "")
                    })

    df = pd.DataFrame(ds)
    mean = df.groupby("Model name").mean()
    median = df.groupby("Model name").median()
    std = df.groupby("Model name").std()

    todo = []
    for k, v in {"mean": mean, "median": median, "std": std}.items():
        todo.append(v.rename(columns={"Response length (tokens)": k}).reset_index())
    stats = pd.concat(todo).groupby("Model name").first()
    print(stats)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Response length (tokens)', hue='Model name')

    plt.xlabel('Response Length')
    plt.ylabel('Density')
    plt.title('Distribution of Response Lengths by Model')

    plt.show()
