import numpy as np
import pandas as pd
import random

name_path = "./alexandra-baby-names/data/babynames_clean.csv"
age_bound = [18, 105]
n_entry = 200



if __name__ == "__main__":
    #load the name
    df_name = pd.read_csv(name_path, header=0, names=['Name', 'Sex'])
    print(df_name.shape)

    for i in n_entry:
        new_entry = []
        k = random.randrange(0, df_name.shape[1])
        pp = df_name.loc[k,:]


