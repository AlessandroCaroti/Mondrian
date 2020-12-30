import numpy as np
import pandas as pd
import os
from pathlib import Path
from dataset_generator.database_generator import *
# import pycountry_convert as pc
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import multiprocessing
from itertools import product
import reverse_geocoder
from datetime import datetime

"""
:param zipcode_col: colum of original dataset with all the zip code
"""


def zipcode_generalization(relative_csv_path):
    path = Path(__file__)
    cur_work = path.parent.parent
    csv_path = os.path.join(cur_work, relative_csv_path)

    csv = pd.read_csv(csv_path, converters={'zip': lambda x: str(x)})
    zip_gen = csv["zip"].sort_values(axis=0)
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

    zip_generalizations.to_csv(os.path.join("Generalization", "Zipcode.csv"), header=False, index=False)
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

    blood_generalizations.to_csv(os.path.join("Generalization", "Blood type.csv"), header=False,
                                 index=False)
    print(blood_generalizations.sort_values(1))


def op(tupl):
    return reverse_geocoder.search(tupl[0], tupl[1])[0]['cc']
    # return tupl[2].reverse(str(tupl[0]) + "," + str(tupl[1])).raw['address']['country']


# city, county, state, continent
def city_generalization(relative_csv_path):
    path = Path(__file__)
    cur_work = path.parent.parent
    csv_path = os.path.join(cur_work, relative_csv_path)
    csv = pd.read_csv(csv_path, converters={'zip': lambda x: str(x)})

    timezones = csv['timezone']
    continent = list(map(lambda x: x.split('/')[0], timezones))
    timezones = list(map(lambda x: x.split('/')[1], timezones))

    # Latitude = csv['lat'].tolist()
    # Longitude = csv['lng'].tolist()
    # zipcode = csv['zip'].tolist()

    geolocator = Nominatim(user_agent="application")
    # aux_val = [(lat, lng) for lat, lng in zip(Latitude, Longitude)]  # [geolocator for _ in range(len(csv))])]
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #    countries = list(pool.map(op, aux_val))
    """
    t1 = datetime.now()
    countries = []
    i = 0
    for lat, lng in zip(Latitude, Longitude):
        print(i, '/', len(csv))
        country = geolocator.reverse(str(lat) + "," + str(lng)).raw['address']['country']
        # country = geolocator.geocode(str(zip)).address.split(',')[-1]
        countries.append(country)
        if i % 100 == 0:
            backup = pd.DataFrame()
            backup['countries'] = countries
            backup.to_csv(os.path.join("Generalization/backup", "backup_countries.csv_it_{}".format(i)))

        i += 1
    print(countries)
    """
    new_dataframe = pd.DataFrame()
    cols = ['City', 'County', 'State', 'Timezone', 'Continent']
    for col, data_col in zip(cols, [csv['city'], csv['county_name'], csv['state_name'], timezones, continent]):
        new_dataframe[col] = data_col

    print(new_dataframe)
    new_dataframe.to_csv(os.path.join("Generalization", "B-City.csv"), header=True, index=False)


# REMOVED: RHODE ISLAND
if __name__ == "__main__":
    csv_relative_path = r"dataset_generator/data/original_geography_dataset.csv"

    #blood_groups_generalization()
    #zipcode_generalization(csv_relative_path)
    #city_generalization(r"dataset_generator/data/original_geography_dataset.csv")
