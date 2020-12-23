import numpy as np
import pandas as pd
import os
from dataset_generator.database_generator import *

"""
:param zipcode_col: colum of original dataset with all the zip code
"""


def zipcode_generalization(data_path):
    csv = pd.read_csv(data_path, converters={'Zipcode': lambda x: str(x)})
    zip_gen = csv["Zipcode"]
    zip_generalizations = pd.DataFrame()
    zip_generalizations[0] = zip_gen

    for i in range(1, 6):
        new_col = []
        for zip in zip_gen:
            ast = ''
            for _ in range(i):
                ast += "*"
            zip_anon = zip[:-i] + ast
            new_col.append(zip_anon)
        zip_generalizations[i] = new_col

    zip_generalizations.to_csv(os.path.join("Generalization", "zipcode_generalization.csv"), header=False, index=False)
    print(zip_generalizations.sort_values(by=1))


def blood_groups_generalization(data_path):
    csv = pd.read_csv(data_path, converters={'Zipcode': lambda x: str(x)})
    blood_groups = csv["Blood type"]
    blood_generalizations = pd.DataFrame()
    blood_generalizations[0] = blood_groups

    for i in range(1, 3):
        new_col = []
        for group in blood_groups:
            ast = ''
            for _ in range(i):
                ast += "*"
            if i == 2 and len(group) == 3:
                group_anon = "***"
            else:
                group_anon = group[:-i] + ast

            new_col.append(group_anon)
        blood_generalizations[i] = new_col

    blood_generalizations.to_csv(os.path.join("Generalization", "BloodGroups_generalization.csv"), header=False,
                                 index=False)
    print(blood_generalizations.sort_values(1))


def date_Generalization():
    # TODO:
    pass


if __name__ == "__main__":
    # df = pd.read_csv("C:\\Users\\simoc\\MondrianMultidimentional_K-Anonymity\\"
    #                "Project\\dataset_generator\\data\\mainDB_21.csv")
    # print(df)
    # df["Zipcode"].tolist()
    # list_str = [str(x) for x in df["Zipcode"].tolist()]
    path = "C:\\Users\\simoc\\MondrianMultidimentional_K-Anonymity\\Project\\dataset_generator\\data\\mainDB_21.csv"
    zipcode_generalization(path)
    blood_groups_generalization(path)
