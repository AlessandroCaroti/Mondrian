import numpy as np
import pandas as pd
import random

name_path = "data/babynames_clean.csv"
disease_path = "data/disease.csv"
gender_map = {'boy': 'Male', 'girl': 'Female'}
age_bound = [18, 105]
n_entry = 5



def random_age():
    return random.randint(age_bound[0], age_bound[1])


def random_zipcode():
    zipcode = ''
    for _ in range(5):
        zipcode += str(random.randint(0, 9))
    return zipcode


if __name__ == "__main__":
    # load the name
    data = []

    # Name, Gender dataset
    df_name = pd.read_csv(name_path, header=0, names=['Name', 'Gender'])
    df_name.replace({'Gender': gender_map}, inplace=True)

    # list of disease
    df_disease = pd.read_csv(disease_path)
    print(df_disease.columns)

    for i in range(n_entry):
        new_entry = []

        k = random.randrange(0, df_name.shape[0])
        pp = df_name.loc[k, :]

        # Name
        new_entry.append(pp.Name)
        # Gender
        new_entry.append(pp.Gender)
        # Age
        new_entry.append(random_age())
        # ZipCode
        new_entry.append(random_zipcode())
        # B-day
        # City_birth

        data.append(new_entry)

    column_name = ['Name', 'Gender', 'Age', 'Zipcode']
    df = pd.DataFrame(data, columns=column_name)
    print(df)

