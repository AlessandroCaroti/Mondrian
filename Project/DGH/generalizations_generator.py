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


def op(tupl):
    print('AAA')
    return reverse_geocoder.search(tupl[0], tupl[1])[0]['cc']
    # return tupl[2].reverse(str(tupl[0]) + "," + str(tupl[1])).raw['address']['country']


# city, county, state, continent
def city_generalization(relative_csv_path):
    path = Path(__file__)
    cur_work = path.parent.parent
    csv_path = os.path.join(cur_work, relative_csv_path)
    csv = pd.read_csv(csv_path, converters={'Zipcode': lambda x: str(x)})

    timezones = csv['timezone']
    continent = list(map(lambda x: x.split('/')[0], timezones))

    Latitude = csv['lat'].tolist()
    Longitude = csv['lng'].tolist()

    geolocator = Nominatim(user_agent="application")
    # aux_val = [(lat, lng) for lat, lng in zip(Latitude, Longitude)]  # [geolocator for _ in range(len(csv))])]
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #    countries = list(pool.map(op, aux_val))

    backup = pd.DataFrame()
    backup['countries'] = [0,0]
    backup.to_csv(os.path.join("Generalization", "backup_countries.csv"))
    t1 = datetime.now()
    countries = []
    i = 0
    for lat, lng in zip(Latitude, Longitude):
        print(i, '/', len(csv))
        country = geolocator.reverse(str(lat) + "," + str(lng)).raw['address']['country']
        countries.append(country)
        if i % 20 == 0:
            backup.drop(labels=['countries'])
            backup['countries'] = countries
            backup.to_csv(os.path.join("Generalization", "backup_countries.csv"))

        i += 1

    print(countries)

    new_dataframe = pd.DataFrame()

    for idx, col in zip(range(4), [csv['city'], csv['county_name'], csv['state_name'], continent]):
        new_dataframe[idx] = col

    print(new_dataframe)
    new_dataframe.to_csv(os.path.join("Generalization", "city_generalization.csv"), header=False,
                         index=False)
    t2 = datetime.now()

    print("\n\nEXECUTION TIME: ", t2 - t1)


# REMOVED: RHODE ISLAND
if __name__ == "__main__":
    csv_relative_path = r"dataset_generator/data/original_geography_dataset.csv"
    # zipcode_generalization(csv_relative_path)
    # uncomment to generate blood  groups
    # blood_groups_generalization()
    #city_generalization(r"dataset_generator/data/original_geography_dataset.csv")
    # es = pc.country_alpha2_to_continent_code('RI')
    # es = pc.convert_continent_code_to_continent_name(es)
    # es2 = pc.country_name_to_country_alpha2("Rhode Island")
    # initialize Nominatim API
    import pgeocode

    #nomi = pgeocode.Nominatim('US')
    #print(nomi.query_postal_code("01074"))
    from pyzipcode import ZipCodeDatabase

   # geolocator = Nominatim(user_agent="geoapiExercises")
    #location = geolocator.geocode("00601")