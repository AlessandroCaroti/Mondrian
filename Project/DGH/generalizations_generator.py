import numpy as np
import pandas as pd
import os
from dataset_generator.database_generator import *

"""
:param zipcode_col: colum of original dataset with all the zip code
"""


def zipcode_generalization(data_path):
    csv = pd.read_csv(data_path, converters={'Zipcode': lambda x: str(x)})
    zip_gen = data["Zipcode"]
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
    print(zip_generalizations.sort_values(1))




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
