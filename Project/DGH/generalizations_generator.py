from Project.Dataset_synthetic.dataset_generator.database_generator import *


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


# city, county, state, continent
def city_generalization(relative_csv_path):
    path = Path(__file__)
    cur_work = path.parent.parent
    csv_path = os.path.join(cur_work, relative_csv_path)
    csv = pd.read_csv(csv_path, converters={'zip': lambda x: str(x)})

    timezones = csv['timezone']
    continent = list(map(lambda x: x.split('/')[0], timezones))
    timezones = list(map(lambda x: x.split('/')[1], timezones))

    new_dataframe = pd.DataFrame()
    cols = ['City', 'County', 'State', 'Timezone', 'Continent']
    for col, data_col in zip(cols, [csv['city'], csv['county_name'], csv['state_name'], timezones, continent]):
        new_dataframe[col] = data_col

    print(new_dataframe)
    new_dataframe.to_csv(os.path.join("Generalization", "B-City.csv"), header=True, index=False)


# REMOVED: RHODE ISLAND
if __name__ == "__main__":
    csv_relative_path = r"dataset_generator/data/geography_small.csv"

    blood_groups_generalization()
    zipcode_generalization(csv_relative_path)
    city_generalization(csv_relative_path)
