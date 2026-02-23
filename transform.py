import importlib.util

cudf_found = importlib.util.find_spec("cudf")
if cudf_found is not None:
    import cudf as pd
    import pandas as pandas
    from cudf.api.types import *
else:
    import pandas as pd
    from pandas.api.types import *
    import pandas as pandas
from extraction import *
from utils import *
import multiprocessing
import os
from scipy.stats.contingency import association
import scipy.stats as stats


# resample so majority_class and minority_class have both the same amount of rows
def resample(majority_class: pd.DataFrame, minority_class: pd.DataFrame, minority_frac=0.5) -> pd.DataFrame:
    n_major = int(majority_class.shape[0] * (
            (minority_class.shape[0] / minority_frac - minority_class.shape[0]) / majority_class.shape[0]))
    combined = pd.concat([majority_class.sample(n=n_major), minority_class], axis=0)
    # shuffle
    return combined.sample(frac=1).reset_index(drop=True)


def remove_poorly_corr_with(df: pd.DataFrame, target_col_name: str, corr_threshold=0.1, exclude_columns: tuple = None):
    if target_col_name is None:
        return

    corr_label = df.select_dtypes(exclude='object').corr()[target_col_name]
    for index, corr in corr_label.to_pandas().items():
        if index == target_col_name:
            continue

        skip = False
        if exclude_columns is not None:
            for c in exclude_columns:
                if c in index:
                    skip = True
                    break

        if skip:
            continue

        if abs(corr) < corr_threshold:
            df.drop(columns=[index], axis=1, inplace=True)


def get_statistics_mp(df, results: dict, index: int, target_class_column: str = None):
    statistics = pandas.DataFrame(columns=df.columns,
                              index=["missings", "min", "max", "mean", "median", "25%", "50%", "75%", "std", "var_coef",
                                     "cramers v", "unique", "top", "freq", "count"])

    for column in df:
        desc = df[column].describe(datetime_is_numeric=True)

        if pd.api.types.is_numeric_dtype(df[column].dtype):
            statistics.loc["mean", column] = desc["mean"]
            statistics.loc["std", column] = desc["std"]
            statistics.loc["min", column] = desc["min"]
            statistics.loc["max", column] = desc["max"]
            statistics.loc["25%", column] = desc["25%"]
            statistics.loc["50%", column] = desc["50%"]
            statistics.loc["75%", column] = desc["75%"]
            statistics.loc["median", column] = df[column].median()
            if df[column].mean() != 0:
                statistics.loc["var_coef", column] = df[column].std() / df[column].mean()
            #only calculate cramers v for binary data
            if target_class_column is not None and target_class_column != column:
                if is_integer_dtype(df[column].dtype) and is_integer_dtype(df[target_class_column].dtype) \
                        and df[target_class_column].nunique() == 2 and df[column].nunique() == 2:
                    statistics.loc["cramers v", column] = df[[column, target_class_column]].corr()[column][target_class_column]
        elif pd.api.types.is_object_dtype(df[column].dtype):
            statistics.loc["unique", column] = desc["unique"]
            statistics.loc["top", column] = desc["top"]
            statistics.loc["freq", column] = desc["freq"]

        statistics.loc["count", column] = desc["count"]
        statistics.loc["missings", column] = df[column].isna().sum()
        statistics.loc["unique", column] = df[column].nunique()

    results[index] = statistics


def get_statistics_cool(df: pd.DataFrame, target_class_column: str = None):
    return multi_process_per_column(df, multiprocessing.cpu_count(),
                                    lambda df_chunk, results, index:
                                    get_statistics_mp(df_chunk, results, index, target_class_column),
                                    [target_class_column])


def get_statistics(df: pd.DataFrame, target_class_column: str = None) -> pd.DataFrame:
    proxy = {}
    get_statistics_mp(df, proxy, 0, target_class_column)
    result: pd.DataFrame = proxy[0].fillna('nan').astype(str, errors="ignore")
    return pd.from_pandas(result)


def normalize_columns(df: pd.DataFrame, columns):
    normalize_columns_types = {}

    for column in columns:
        normalize_columns_types[column] = "float64"

    df.astype(normalize_columns_types)
    normalized_df = df[list(normalize_columns_types.keys())]
    normalized_df = ((normalized_df - normalized_df.min()) / (normalized_df.max() - normalized_df.min()))
    df.update(normalized_df)


class Transform():
    def __init__(self, dataframes: dict):
        self.extraction = Extraction(dataframes)
        self.extraction.extract_common_columns()
        self.extraction.extract_dtypes()
        self.combined_df: pd.DataFrame = None

    def exploration_combined(self, save_stats_folder: str = None):
        self._exploration(self.combined_df, "combined", save_stats_folder)

    def exploration_single(self, save_stats_folder: str = None):
        for file_name, df in self.extraction.dataframes.items():
            print(file_name)
            self._exploration(df, file_name, save_stats_folder)
            print("".join("-" for i in range(40)))
            print("\n\n")

    def _exploration(self, df: pd.DataFrame, df_name: str, save_stats_folder: str = None):
        pass


class TransformNFUNSWNB(Transform):
    def combine(self):
        self.combined_df = None
        for file_name, df in self.extraction.dataframes.items():
            df: pd.DataFrame = self.extraction.dataframes[file_name]
            df = df.drop(self.extraction.unusual_columns_per_df[file_name], axis=1)

            if self.combined_df is None:
                self.combined_df = df
            else:
                self.combined_df = pd.concat([self.combined_df, df], axis=0)

    # sets the result into the results array at the specified index, useful for multiprocessing
    def get_tcp_dummies(self, df: pandas.DataFrame, results, index: int):
        temporary_df = pandas.DataFrame()
        temporary_df["TCP_FLAGS_BINARY"] = df["TCP_FLAGS"].apply(lambda x: bin(x)[2:].zfill(8))
        tcp_binary_dummies: pd.DataFrame = temporary_df["TCP_FLAGS_BINARY"].astype(str).str.extractall('(.)')[0].unstack()
        tcp_binary_dummies = tcp_binary_dummies.add_prefix('TCP_FLAG_')
        results[index] = tcp_binary_dummies.astype(int)

    def transform_combined(self):
        print_line()
        print("Transforming combined dataframe:\n")
        if self.combined_df is None:
            self.combine()

        print("Info before Transformation:")
        self.combined_df.info()
        print("\n")

        print("Drop categorical...")
        self.combined_df = self.combined_df.drop(["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"],
                                                 axis=1)

        print("Convert TCP_FLAGS to binary dummies...")
        self.combined_df = multi_process(self.combined_df, multiprocessing.cpu_count(), self.get_tcp_dummies)
        self.combined_df = self.combined_df.drop(["TCP_FLAGS"], axis=1)

        print("Convert PROTOCOL to dummies...")
        self.combined_df = pd.concat([self.combined_df, pd.get_dummies(self.combined_df[["PROTOCOL"]].astype(str))],
                                     axis=1)
        self.combined_df = self.combined_df.drop(["PROTOCOL"], axis=1)

        print("Convert L7_PROTO to dummies...")
        print(f"Convert L7_PROTO to int...")
        print(f'Unique L7_PROTO floats {self.combined_df["L7_PROTO"].nunique()}')
        self.combined_df["L7_PROTO"] = self.combined_df["L7_PROTO"].astype("Int64")
        print(f'Unique L7_PROTO ints {self.combined_df["L7_PROTO"].nunique()}')
        self.combined_df = pd.concat([self.combined_df, pd.get_dummies(self.combined_df[["L7_PROTO"]].astype(str))],
                                     axis=1)
        self.combined_df = self.combined_df.drop(["L7_PROTO"], axis=1)

        print("Transformed info:")
        print(self.combined_df.info())
        print_line()

        """ verify whether multi processing worked in converting TCP_FLAG correctly
        self.combined_df["TCP_FLAG_BINARY"] = (self.combined_df["TCP_FLAG_0"].astype(str) + self.combined_df["TCP_FLAG_1"].astype(str)
                                               + self.combined_df["TCP_FLAG_2"].astype(str) + self.combined_df["TCP_FLAG_3"].astype(str)
                                               + self.combined_df["TCP_FLAG_4"].astype(str) + self.combined_df["TCP_FLAG_5"].astype(str)
                                               + self.combined_df["TCP_FLAG_6"].astype(str) + self.combined_df["TCP_FLAG_7"].astype(str))
        for i, row in self.combined_df.to_pandas().iterrows():
            if row["TCP_FLAG_BINARY"] != bin(row["TCP_FLAGS"])[2:].zfill(8):
                print("ERROR")
        """

    def get_ready_for_ml(self) -> pd.DataFrame:
        ml = self.combined_df.copy()
        normalize_columns(ml, ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS"])
        return ml

    def exploration_combined(self, save_stats_folder: str = None):
        if self.combined_df is None:
            self.combine()
        self._exploration(self.combined_df, "combined", save_stats_folder)

    def exploration_single(self, save_stats_folder: str = None):
        for file_name, df in self.extraction.dataframes.items():
            print(file_name)
            self._exploration(df, file_name, save_stats_folder)
            print_line()
            print("\n\n")

    def _exploration(self, df: pd.DataFrame, df_name: str, save_stats_folder: str = None):
        print_line()
        print("Exploration\n")
        print("info")
        df.info()
        print("\n")

        print("head")
        print(df.head().to_string())
        print("\n")

        categorical_columns_to_drop = list(
            set(df.columns) & {"IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT"})
        categorical_dropped_df = self.combined_df.drop(categorical_columns_to_drop, axis=1)
        stats = multi_process_per_column(categorical_dropped_df, multiprocessing.cpu_count(),
                                         lambda df_chunk, results, index:
                                         get_statistics_mp(df_chunk, results, index, "Label"), ["Label"])

        print("statistics:")
        print(stats.head(999))
        print("\n")

        if save_stats_folder is not None:
            stats.to_csv(os.path.join(save_stats_folder, df_name + "_statistics.csv"), index=True)

        stats_per_label_col = None
        for column in df.columns:
            label_0 = df[df["Label"] == 0][[column]].rename(columns={column: column + "_0"})
            label_1 = df[df["Label"] == 1][[column]].rename(columns={column: column + "_1"})
            if stats_per_label_col is None:
                stats_per_label_col = pd.concat([get_statistics(label_0), get_statistics(label_1)], axis=1)
            else:
                stats_per_label_col = pd.concat([stats_per_label_col, get_statistics(label_0), get_statistics(label_1)],
                                                axis=1)

        if save_stats_folder is not None and stats_per_label_col is not None:
            stats_per_label_col.to_csv(os.path.join(save_stats_folder, df_name + "_statistics_label.csv"), index=True)

        print("Attack types")
        print(df["Attack"].unique())
        print("\n")

        print("Attack types per Label")
        print(df.groupby("Label")["Attack"].unique().to_string())
        print("\n")

        print("Count of Attack types")
        print(df.groupby("Attack")["Label"].count())
        print("\n")

        print("Correlation with Label")
        print(df.select_dtypes(exclude='object').corr()["Label"].to_string())
        print("\n")


class TransformNFUNSWNBV2(TransformNFUNSWNB):
    def transform(self):
        self.combined_df = None
        for file_name, df in self.extraction.dataframes.items():
            df: pd.DataFrame = self.extraction.dataframes[file_name]
            df = df.drop(self.extraction.unusual_columns_per_df[file_name], axis=1)
            df = df.drop(["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"], axis=1)

            if self.combined_df is None:
                self.combined_df = df
            else:
                self.combined_df = pd.concat([self.combined_df, df], axis=0)

    def normalize_necessary_columns(self):
        normalize_columns(self.combined_df,
                          ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS"])
