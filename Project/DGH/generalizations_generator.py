import numpy as np
import pandas as pd
import os
from pathlib import Path
from dataset_generator.database_generator import *

"""
:param zipcode_col: colum of original dataset with all the zip code
"""


def zipcode_generalization(relative_csv_path):
    path = Path(__file__)
    cur_work = path.parent.parent
    csv_path = os.path.join(cur_work, relative_csv_path)

    csv = pd.read_csv(csv_path, converters={'Zipcode': lambda x: str(x)})
    zip_gen = csv["Zipcode"].sort_values(axis=0)
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
    print(zip_generalizations)


def blood_groups_generalization():
    blood_groups = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
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


if __name__ == "__main__":
    csv_relative_path = r"dataset_generator/data/mainDB_100000.csv"
    zipcode_generalization(csv_relative_path)
    # uncomment to generate blood  groups
    # blood_groups_generalization()
    s = "01234"
    print(s[:2])
