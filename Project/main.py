import pandas as pd
import numpy as np

data_folder = r"dataset_generator/data/"
db_path = data_folder + r"FILE_NAME.csv"

if __name__ == "__main__":
    # Load the database to anonymize
    original_db = pd.DataFrame(db_path)


