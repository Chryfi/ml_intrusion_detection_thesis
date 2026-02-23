import importlib.util

cudf_found = importlib.util.find_spec("cudf")
if cudf_found is not None:
    import cudf as pd
    import pandas as pandas
else:
    import pandas as pd
import multiprocessing
import numpy as np


def multi_process_per_column(df: pd.DataFrame, threads: int, target_func, columns_to_include_always: list[str]) -> pd.DataFrame:
    df_columns = None
    if columns_to_include_always is not None:
        df_columns = df[columns_to_include_always]
        df = df.drop(columns=columns_to_include_always, axis=1)

    columns_num_per_thread = len(df.columns) // threads
    columns_per_thread = []

    for i in range(0, len(df.columns), columns_num_per_thread):
        columns_per_thread.append(np.array(df.columns)[i:i + columns_num_per_thread])

    manager = multiprocessing.Manager()
    return_dict: dict = manager.dict()
    threads = []
    for i in range(len(columns_per_thread)):
        df_concat = df[columns_per_thread[i]]
        if columns_to_include_always is not None:
            df_concat = pd.concat([df_concat, df_columns], axis=1)
        thread = multiprocessing.Process(target=target_func,
                                         args=(df_concat.to_pandas(), return_dict, i))
        thread.start()
        threads.append(thread)
    # wait for threads to finish
    for thread in threads:
        thread.join()

    df = None
    for key, result in return_dict.items():
        result: pd.DataFrame = result.fillna('nan').astype(str, errors="ignore")
        if columns_to_include_always is not None:
            result.drop(columns=columns_to_include_always, axis=1, inplace=True)

        if df is None:
            df = pd.from_pandas(result)
        else:
            df = pd.concat([df, pd.from_pandas(result)], axis=1)

    return df


# target function needs to take following args (pandas.DataFrame, results: dict, index)
# and set the result into the dict with index as key
def multi_process(df: pd.DataFrame, threads: int, target_func) -> pd.DataFrame:
    chunk_size = len(df) // threads
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i + chunk_size])

    manager = multiprocessing.Manager()
    return_dict: dict = manager.dict()
    threads = []
    for i in range(len(chunks)):
        thread = multiprocessing.Process(target=target_func, args=(chunks[i].to_pandas(), return_dict, i))
        thread.start()
        threads.append(thread)
    # wait for threads to finish
    for thread in threads:
        thread.join()

    for key, result in return_dict.items():
        result_cudf = pd.from_pandas(result)
        chunk = chunks[int(key)]
        chunk.drop(columns=result_cudf.columns, axis=1, inplace=True, errors="ignore")
        chunks[int(key)] = pd.concat([chunk, result_cudf], axis=1)

    df = None
    for chunk in chunks:
        if df is None:
            df = chunk
        else:
            df = pd.concat([df, chunk], axis=0)

    return df


def print_line(num=100):
    print("".join("-" for x in range(num)))
