import importlib.util

cudf_found = importlib.util.find_spec("cudf")
if cudf_found is not None:
    import cudf as pd
    from cuml import RandomForestClassifier
    from cuml import train_test_split
    from cuml import accuracy_score

    print("cudf found")
else:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
import os
from loading import *
from transform import *
from extraction import *
# from machinelearning import *
import tensorflow as tf
import keras

print("GPU:")
print(tf.config.list_physical_devices('GPU'))
print("\n\n")

directory = '../CSE-CIC-IDS2018'
loadProcess = LoadingProcess(directory)
loadProcess.load_files()
# loadProcess.store_small_portions(100000)

extraction = Extraction(loadProcess.dataframes)
extraction.extract_common_columns()
extraction.extract_dtypes()

print(extraction.unusual_columns_per_df)
print("\n\n")
print(extraction.common_columns)
print("\n\n")
extraction.dtype_df.to_csv("dtypesCSECIC2018.csv")
quit()
transform = TransformCSECICIDS2018(loadProcess.dataframes)

transform.exploration_single()