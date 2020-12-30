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
n_entry = 1000

# path & filename variable
dataset_folder = "data"
name_path = os.path.join(dataset_folder, "babynames.csv")
disease_path = os.path.join(dataset_folder, "disease_small.csv")
geography_path = os.path.join(dataset_folder, "geography_dataset.csv")

mainDB_filename = 'mainDB_' + str(n_entry) + '.csv'
externalDB_filename = 'externalDB_' + str(n_entry) + '.csv'

# variable that specify the column of the main dataset and an external one public that can be used for a join
quasi_identifiers = ['Gender', 'Age', 'Zipcode', 'B-City', 'B-day', 'Height (cm)', 'Weight (Kg)',
                     'Blood type', 'Start ', 'Therapy', 'End Therapy']
explicit_identifiers = ['Name']
sensitive_data = ['Disease']

mainTable_indices = quasi_identifiers + sensitive_data
externalTable_indices = explicit_identifiers + quasi_identifiers


def random_age():
    return random.randint(age_bound[0], age_bound[1])


def random_zipcode():
    zipcode = ''
    for i in range(5):
        if i == 2 and zipcode[:2] == "00":
            zipcode += str(random.randint(1, 9))
        else:
            zipcode += str(random.randint(0, 9))
    return zipcode


def rand_day(month):
    if month != 2:
        return random.randint(1, 30)
    else:
        return random.randint(1, 28)


def random_Bday(age):
    month = np.random.randint(1, 12)
    day = rand_day(month)
    year = datetime.datetime.now().year - age
    return "{:02d}-{:02d}-{}".format(day, month, year)


def random_therapy_day():
    this_year = datetime.datetime.now().year

    start_month = np.random.randint(1, 12)
    start_day = rand_day(start_month)
    start_year = np.random.randint(this_year - 5, this_year)

    end_month = start_month + np.random.randint(1, 12)
    end_year = start_year + np.random.randint(0, 3)
    if end_month > 12 and end_year == start_year:
        end_month = end_month - 12
        end_year += 1
    elif end_month > 12:
        end_month = end_month - 12
    end_day = rand_day(end_month)

    start_date = "{:02d}-{:02d}-{}".format(start_day, start_month, start_year)
    end_date = "{:02d}-{:02d}-{}".format(end_day, end_month, end_year)

    return start_date, end_date


def random_height(gender):
    if gender == "Male":
        return np.random.randint(160, 200)
    else:
        return np.random.randint(150, 185)


def random_blood_group():
    groups = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
    distribution = [0.38, 0.07, 0.34, 0.06, 0.09, 0.02, 0.03, 0.01]

    return np.random.choice(groups, p=distribution)


def random_weight():
    return round(np.random.normal(80, 15, 1)[0], 1)


def pick_geography_from_csv(path):
    dataframe = pd.read_csv(path, converters={'zip': lambda x: str(x)})
    new_dataset = pd.DataFrame()

    new_dataset['Zipcode'] = dataframe['zip']
    new_dataset['B_City'] = dataframe['city']
    new_dataset['County-FIPS'] = dataframe["county_name"]
    new_dataset['state_name'] = dataframe["state_name"]
    new_dataset.to_csv(os.path.join(dataset_folder, "geography_dataset.csv"), index=False)


if __name__ == "__main__":

    # pick_geography_from_csv("data/original_geography_dataset.csv")

    # array to store all the data
    data = []

    # Name, Gender dataset
    df_name = pd.read_csv(name_path, header=0, names=['Name', 'Gender'])
    df_name.replace({'Gender': gender_map}, inplace=True)
    random.seed(25)
    # Dataset with a list of disease
    df_disease = pd.read_csv(disease_path)
    df_geo = pd.read_csv(geography_path, converters={'Zipcode': lambda x: str(x)})

    for i in range(n_entry):
        new_entry = []

        # Name & Gender
        k = random.randrange(0, df_name.shape[0])
        pp = df_name.loc[k, :]

        new_entry.append(pp.Name)
        new_entry.append(pp.Gender)

        # Age
        new_entry.append(random_age())

        # ZipCode and cities
        # new_entry.append(random_zipcode())
        k = random.randrange(0, df_geo.shape[0])
        row = df_geo.loc[k, :]
        new_entry.append(row.Zipcode)
        new_entry.append(row.B_City)

        # B-day
        new_entry.append(random_Bday(new_entry[2]))

        # Disease
        k = random.randrange(0, df_disease.shape[0])
        disease = df_disease.loc[k, :]
        new_entry.append(disease.Name)

        # Therapy day (start - end)
        start_th, end_th = random_therapy_day()
        new_entry.append(start_th)
        new_entry.append(end_th)

        # Blood type
        new_entry.append(random_blood_group())

        # Weight
        new_entry.append(random_weight())

        # Height
        new_entry.append(random_height(new_entry[1]))

        data.append(new_entry)

    column_name = ['Name', 'Gender', 'Age', 'Zipcode', 'B-City', 'B-day', 'Disease', 'Start Therapy', 'End Therapy',
                   'Blood type', 'Weight (Kg)', 'Height (cm)']

    df = pd.DataFrame(data, columns=column_name)

    df = df.infer_objects()
    df = df.convert_dtypes()

    print(df)

    # main_df = df[mainTable_indices]
    df.to_csv(os.path.join(dataset_folder, mainDB_filename))
