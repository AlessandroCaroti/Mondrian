# Mondrian K-Anonymity

### Usage 

The folder *Dataset* must contain the .csv file containing the data to anonymize and the .csv file which specifies the type for each attribute.
All the results will be saved in a folder named *Results*.

This is an example of how you can use the program.
```
 python3 anonymizer.py
```
In this case _Mondrian_ is executed with all the default settings: dataset provided by us and K=10.

#### Parameters

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
- datatime==4.3
