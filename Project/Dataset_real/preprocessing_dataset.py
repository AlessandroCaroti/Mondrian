import pandas as pd
import os


def remove_missing_data(pd_csv):
    cols = list(pd_csv.columns)

    for c in cols:
        pd_csv = pd_csv[pd_csv[c] != '?']

    return pd_csv


def extract_unique_categorical(categorical_cols, df):
    for cat in categorical_cols:
        path = os.path.join("Hierarchies", cat + ".csv")
        pd.DataFrame(df[cat].unique()).to_csv(path, index=False, header=False)


if __name__ == "__main__":
    dataset = pd.DataFrame("adult.csv")

    # removing records with missing values
    dataset = remove_missing_data(dataset)

    # drop redundant and negligible columns
    dataset.drop(["education-num", "final-weight"], axis=1)

    # save final dataset
    dataset.to_csv("adult_final.csv", index=False, header=True)

    # to create the hierarchies...
    # adult_csv = pd.read_csv("adult_final.csv")
    # categorical_col = ["workclass", "education", "martial-status", "occupation", "relationship", "race", "sex",
    #                                                                                                 "native-country"]
    # extract_unique_categorical(categorical_col,adult_csv)
    # print(list(adult_csv.columns))
