import pandas as pd
import os


def remove_missing_data(pd_csv):
    cols = list(pd_csv.columns)

    for c in cols:
        pd_csv = pd_csv[pd_csv[c] != '?']

    pd_csv.to_csv("adult_cleaned.csv", index=False, header=True)
    return


def extract_unique_categorical(categorical_cols, df):
    for cat in categorical_cols:
        path = os.path.join("generalization", cat + ".csv")
        pd.DataFrame(df[cat].unique()).to_csv(path, index=False, header=False)


if __name__ == "__main__":
    """Uncomment to use"""
    # remove_missing_data(adult_csv)
    adult_csv = pd.read_csv("adult_cleaned.csv")

    categorical_col = ["workclass", "education", "martial-status", "occupation", "relationship", "race", "sex",
                       "native-country"]
    """uncomment to use"""
    # extract_unique_categorical(categorical_col,adult_csv)
