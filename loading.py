import os
import threading
import random
import importlib.util

cudf_found = importlib.util.find_spec("cudf")
if cudf_found is not None:
    import cudf as pd
else:
    import pandas as pd


class LoadingProcess():
    def __init__(self, root_directory: str, random_sample=None):
        self.dataframes = {}
        self.all_files = [file for file in os.listdir(root_directory) if file.endswith('.csv')]
        self.root_directory = root_directory
        self.random_sample = random_sample

    def _load_dataframe(self, filename, sample_size=None):
        if sample_size is not None:
            # number of records in file (excludes header)
            n = sum(1 for line in open(os.path.join(self.root_directory, filename))) - 1
            s = sample_size
            skip = sorted(random.sample(range(1, n + 1), n - s))

            self.dataframes[filename] = pd.read_csv(os.path.join(self.root_directory, filename), skiprows=skip)
        else:
            self.dataframes[filename] = pd.read_csv(os.path.join(self.root_directory, filename))

    def load_only_columns(self):
        for filename in self.all_files:
            self.dataframes[filename] = pd.read_csv(os.path.join(self.root_directory, filename), nrows=0)

    def load_files(self):
        threads = []
        for filename in self.all_files:
            thread = threading.Thread(target=lambda: self._load_dataframe(filename, sample_size=self.random_sample))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    def store_small_portions(self, sample_size):
        for file_name in self.dataframes:
            sample = self.dataframes[file_name].sample(n=sample_size)
            sample.to_csv(os.path.join(self.root_directory, "small", file_name), index=False)