import importlib.util

cudf_found = importlib.util.find_spec("cudf")
if cudf_found is not None:
    import cudf as pd
else:
    import pandas as pd


class Extraction():
    def __init__(self, dataframes: dict):
        self.dataframes = dataframes
        self.columns_per_df = {}
        self.common_columns = []
        self.unusual_columns_per_df = {}
        self.all_columns = []

        for filename, df in self.dataframes.items():
            columns = df.columns.tolist()
            self.columns_per_df[filename] = columns
            self.unusual_columns_per_df[filename] = []
            for c in columns:
                if c not in self.all_columns:
                    self.all_columns.append(c)

        self.dtype_df = None

    def extract_common_columns(self):
        # for every column go through all other columns and check if its present everywhere
        for filename, columns_test in self.columns_per_df.items():
            for test_column in columns_test:
                is_common = True
                # go through every other columns and test if test_column is present
                for columns_comparison in self.columns_per_df.values():
                    # no need to test ourselves
                    if columns_test is columns_comparison:
                        continue
                    if test_column not in columns_comparison:
                        is_common = False
                        break
                if is_common and test_column not in self.common_columns:
                    self.common_columns.append(test_column)
                if not is_common:
                    self.unusual_columns_per_df[filename].append(test_column)

    def extract_dtypes(self):
        dtype_data = {}

        for column in self.all_columns:
            dtype_data[column] = []
        dtype_data["file"] = []

        for filename, df in self.dataframes.items():
            df: pd.DataFrame = df
            dtype_data["file"].append(filename)
            for column in self.all_columns:
                if column in df:
                    dtype_data[column].append(str(df[column].dtype))
                else:
                    dtype_data[column].append("-")
        self.dtype_df = pd.DataFrame(dtype_data)