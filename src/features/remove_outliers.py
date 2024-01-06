import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import warnings
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

outlier_columns = list(df.columns[:6])

# Plotting outliers

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

df[["gyr_y", "label"]].boxplot(by="label", figsize=(20, 10))

df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )

    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# IQR function
def mark_outliers_iqr(dataset, col):
    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

# Plot a single column
col = "acc_x"
dataset = mark_outliers_iqr(df, col)
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
)


# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# Check for normal distribution

df[outlier_columns[:3] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)
df[outlier_columns[3:] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)

# Chauvenet's function

def mark_outliers_chauvenet(dataset, col, C=2):
    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    dataset = dataset.copy()
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    deviation = abs(dataset[col] - mean) / std

    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    for i in range(0, len(dataset.index)):
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

# Loop over all columns

for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# LOF function
    
def mark_outliers_lof(dataset, columns, n=20):
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

# Loop over all columns

dataset, outliers, X_scores = mark_outliers_lof(df, outlier_columns)

for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

# Check outliers grouped by label

label = "squat"

for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

dataset, outliers, X_scores = mark_outliers_lof(
    df[df["label"] == label], outlier_columns
)

for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    outliers_removed_df = df.copy()

    for col in outlier_columns:
        for label in df["label"].unique():
            dataset = mark_outliers_chauvenet(df[df["label"] == label], col)

            # Replace values marked as outliers with NaN
            dataset.loc[dataset[col + "_outlier"], col] = np.nan

            # Update the column in the original dataframe
            outliers_removed_df.loc[
                (outliers_removed_df["label"] == label), col
            ] = dataset[col]

            n_outliers = len(dataset) - len(dataset[col].dropna())

            print(f"Removed {n_outliers} from {col} from {label}")

outliers_removed_df.info()

outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
