# Mondrian K-Anonymity

### Usage 

There must be a folder containing the .csv file with the data to anonymize and the .csv file which specifies the type for each attribute.
All the results will be saved in a folder named *Results*.

This is an example of how you can use the program.
```
 python3 anonymizer.py
```
In this case _Mondrian_ is executed with all the default settings: dataset provided by us and K=10.

#### Parameters

- -folder_name: name of the folder of the data (default: Dataset_synthetic)
- -dataset_name: name of the dataset file. (default: mainDB_10000.csv)
- -columns_type: name of the file containing the types (default: columns_type.csv)
- -K: the integer K (defualt: 10)
- -result_name: name of the file that will contains the data anonymized (defualt: anonymized.csv)
- -save_info: boolean value indicating either save in a txt file some info about the resulting data or not save any .txt file (default: True)

Here a further example using the parameters
```
python3 anonymizer.py -dataset_name=mainDB_1000000.csv -K=100 -result_name=anonymized_1000000.csv -save_statistics=False
```

### Requirements
```
pip3 install -r requirements.txt
```

- numpy==1.18.5
- pandas==1.1.2
- argparse==1.4.0
- datetime==4.3

### Data
We provide 2 kind of data as an example: one synthetic and one real.

#### Synthetic
Each record has the following attributes:
- Gender: male or female
- Age
- Zipcode: code of the place where the person lives
- B-City: name of the city where the person was born
- B-day: birthday date
- Start therapy: date starting the therapy
- End therapy: date end treatment
- Blood type
- Weight (Kg)
- Height (cm)
- Disease: kind of disease affecting the person, attribute considered as _Sensitive data_

#### Real
This is taken from [here](https://archive.ics.uci.edu/ml/datasets/adult). The records containing missing values are removed and we get rid of *education-num* and *final-weight* attributes.
The *annual-gain* is treated as Sensitive data.
