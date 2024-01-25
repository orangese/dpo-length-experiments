# NOTE: UNDER CONTRUCTION!!

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

MAXLEN = 300

data_shp1 = pd.read_csv('../sampled/pythia2.8b_hh_alpha01_rewards.csv')
data_shp2 = pd.read_csv('../sampled/pythia2.8b_hh_alpha005_rewards.csv')
data_shp3 = pd.read_csv('../sampled/pythia2.8b_hh_base_dpo_rewards.csv')

# Prepare for linear regression and plotting
datasets_shp = {
    'hh_alpha01': data_shp1[data_shp1['lengths'] <= MAXLEN],
    'hh_alpha005': data_shp2[data_shp2['lengths'] <= MAXLEN],
    'hh_base_dpo': data_shp3[data_shp3['lengths'] <= MAXLEN]
}

# Creating subplots with scatterplots for each new dataset
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Iterate over new datasets to create scatterplots in subplots
for ax, (name, data) in zip(axes, datasets_shp.items()):
    # Fit linear regression
    X = data[['lengths']]
    y = data['rewards']
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Calculate R^2 value
    r2 = r2_score(y, y_pred)

    # Create scatterplot in subplot
    ax.scatter(X, y, alpha=0.3, s=5)
    ax.plot(X, y_pred, color='red')  # Adding the regression line
    ax.set_title(f"{name}")
    ax.set_xlabel("Length")
    ax.set_ylabel("Reward")
    ax.legend([f"R²={r2:.2f}"])  # Only R² in legend, no red line

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

