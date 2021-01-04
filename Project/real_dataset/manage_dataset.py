import pandas as pd


def remove_missing_data(pd_csv):
    cols = list(pd_csv.columns)

    for c in cols:
        pd_csv = pd_csv[pd_csv[c] != '?']

    pd_csv.to_csv("adult_cleaned.csv", index=False, header=True)


if __name__ == "__main__":
    adult_csv = pd.read_csv("adult.csv")

    remove_missing_data(adult_csv)
