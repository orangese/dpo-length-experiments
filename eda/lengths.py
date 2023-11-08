import seaborn as sns
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from all_datasets import get_len


if __name__ == "__main__":
    SAMPLE_DIR = "../sampled"

    ds = []
    for f in os.listdir(SAMPLE_DIR):
        if f.endswith(".json"):
            print(f"processing {f}")
            with open(os.path.join(SAMPLE_DIR, f), "r") as fi:
                tmp = json.load(fi)
                for k, v in tmp.items():
                    v = v[0]  # only take one sample for now
                    ds.append({
                        "Response length (tokens)": get_len(v),
                        "Model name": f.rstrip("_1.json")
                    })

    df = pd.DataFrame(ds)
    print(df)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Response length (tokens)', hue='Model name')

    plt.xlabel('Response Length')
    plt.ylabel('Density')
    plt.title('Distribution of Response Lengths by Model')

    plt.show()
