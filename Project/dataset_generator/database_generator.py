import datetime
import os
import random

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# parameters for the data generation
gender_map = {'boy': 'Male', 'girl': 'Female'}
age_bound = [18, 105]
n_entry = 10

# path & filename variable
dataset_folder = "data"
name_path = os.path.join(dataset_folder, "babynames.csv")
disease_path = os.path.join(dataset_folder, "disease.csv")

mainDB_filename = 'mainDB_' + str(n_entry) + '.csv'
externalDB_filename = 'externalDB_' + str(n_entry) + '.csv'

# variable that specify the column of the main dataset and an external one public that can be used for a join
mainTable_indices = []
externalTable_indices = []
identifiers = ['Name', 'Gender', 'Age', 'Zipcode', 'B-day']
sensible_data = ['Disease', 'Blood type', 'Weight (Kg)']


def random_age():
    return random.randint(age_bound[0], age_bound[1])


def random_zipcode():
    zipcode = ''
    for _ in range(5):
        zipcode += str(random.randint(0, 9))
    return zipcode


def random_Bday(age):
    day = np.random.randint(1, 28)
    mouth = np.random.randint(1, 12)
    year = datetime.datetime.now().year - age
    return "{:02d}-{:02d}-{}".format(day, mouth, year)


def random_blood_group():
    groups = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
    distribution = [0.38, 0.07, 0.34, 0.06, 0.09, 0.02, 0.03, 0.01]

    return np.random.choice(groups, p=distribution)


def random_weight():
    return round(np.random.normal(80, 15, 1)[0], 1)


if __name__ == "__main__":
    # array to store all the data
    data = []

    # Name, Gender dataset
    df_name = pd.read_csv(name_path, header=0, names=['Name', 'Gender'])
    df_name.replace({'Gender': gender_map}, inplace=True)

    # Dataset with a list of disease
    df_disease = pd.read_csv(disease_path)

    for i in range(n_entry):
        new_entry = []

        # Name & Gender
        k = random.randrange(0, df_name.shape[0])
        pp = df_name.loc[k, :]

        new_entry.append(pp.Name)
        new_entry.append(pp.Gender)

        # Age
        new_entry.append(random_age())

        # ZipCode
        new_entry.append(random_zipcode())

        # B-day
        new_entry.append(random_Bday(new_entry[2]))

        # City_birth TODO: to implement if we want it

        # Disease
        k = random.randrange(0, df_disease.shape[0])
        disease = df_disease.loc[k, :]
        new_entry.append(disease.Name)

        # Blood type
        new_entry.append(random_blood_group())

        # Therapy day (start - end) TODO: to implement if we want it

        # Weight
        new_entry.append(random_weight())

        data.append(new_entry)

    column_name = ['Name', 'Gender', 'Age', 'Zipcode', 'B-day', 'Disease', 'Blood type', 'Weight (Kg)']
    df = pd.DataFrame(data, columns=column_name)
    print(df)

    # TODO: split all data into 2 dataset:
    #       -one with the sensible data and some Quasi-Identifier attribute (the Main_DataBase)
    #       -one that contain external information that can be joined with the previous
    #        dataset to re-identify individual records (the external_DataBase)
