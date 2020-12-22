
class Data():

    NUMERICAL, CATEGORICAL, DATE = list(range(3)) # type of data

    def __init__(self, data, columns_type): # data is a path or Dataframe??? da decidere

        self.dataFrame = data
        self.columns_type = columns_type # dictionary column_name : type_of_data

        self.initial_ranges = {col: compute_width(df[col], col) for col in cols_to_anonymize}