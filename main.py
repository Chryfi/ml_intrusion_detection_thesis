import importlib.util

cudf_found = importlib.util.find_spec("cudf")
if cudf_found is not None:
    import cudf as pd
    from cuml import RandomForestClassifier
    from cuml import train_test_split
    from cuml import accuracy_score
    from cuml.metrics import confusion_matrix

    print("cudf found")
else:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
from loading import *
from transform import *
from extraction import *
from machinelearning import *
from visualisation import *
import tensorflow as tf
import keras

print("GPU:")
print(tf.config.list_physical_devices('GPU'))
print("\n\n")

directory = '../NF-UNSW-CSE-CIC-IDS2018'


def load_transform():
    loadProcess = LoadingProcess(directory)
    loadProcess.load_files()
    # loadProcess.store_small_portions(100000)

    transform = TransformNFUNSWNB(loadProcess.dataframes)
    transform.combine()

    print_line()
    print("Combined data info")
    transform.combined_df.info()
    print_line()
    print("Combined data head")
    print(transform.combined_df.head(100).to_string())
    print_line()

    transform.transform_combined()
    transform.exploration_combined(os.path.join(directory, "statistics"))

    df: pd.DataFrame = transform.get_ready_for_ml()
    df.to_csv(os.path.join(directory, "processed", "ml_ready.csv"), index=False, chunksize=500000)
    transform.combined_df.to_csv(os.path.join(directory, "processed", "combined.csv"), index=False)


def visualize():
    stats = pandas.read_csv(os.path.join(directory, "statistics/combined_statistics.csv"))

    stats.set_index("Unnamed: 0", inplace=True)
    stats.info()
    stats = stats.filter(like='L7_PROTO')
    print(stats.head(100).to_string())

    cramersv: pd.Series = stats.loc["cramers v"]
    cramersv = cramersv.abs()
    print(len(cramersv))
    print(len(cramersv[cramersv <= 0.002663]) / len(cramersv))
    print(len(cramersv[cramersv <= 0.001061]) / len(cramersv))
    print(len(cramersv[cramersv <= 0.000629]) / len(cramersv))
    print(cramersv.describe())

    print(cramersv.sort_values().to_string())

    df = pd.read_csv(os.path.join(directory, "processed", "combined.csv"))
    # x, y = RandomUnderSampler(random_state=42).fit_resample(df.drop(["Label"], axis=1), df[["Label"]])
    # df = pd.concat([x, y], axis=1)#.sample(n=1000)
    visualisation = NFCSECICIDS2018_Visualizer(df)
    #visualisation.visualize()


def load_test_mlp_models(X_test: np.ndarray, y_test: np.ndarray,
                         X_zd: np.ndarray, y_zd: np.ndarray,
                         models_filenames: list[str], directory: str) -> pd.DataFrame:
    performance_df = None
    for model_name in models_filenames:
        model = keras.models.load_model(os.path.join(directory, model_name))

        start_testing = time.time()
        y_pred = model.predict(X_test)
        end_testing = time.time()

        zd_metrics = model.evaluate(X_zd, y_zd, return_dict=True)

        y_pred = (y_pred > 0.5).astype(int).flatten()

        perf = get_model_performance(y_pred, y_test, 0, end_testing - start_testing)
        perf["model_name"] = model_name
        perf["zero_day_detection_rate"] = zd_metrics["recall"]

        if performance_df is None:
            performance_df = perf
        else:
            performance_df = pd.concat([performance_df, perf], axis=0)
    return performance_df

def evaluate_best_mlp_models(df: pd.DataFrame, zero_day_type: str,  directory: str):
    datasets = get_datasets(df, zero_day_type)
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]
    X_zd = datasets["X_zd"]
    y_zd = datasets["y_zd"]

    models_accuracy = []
    models_loss = []
    models_recall = []

    for filename in os.listdir(directory):
        if "best_accuracy" in filename:
            models_accuracy.append(filename)
        elif "best_loss" in filename:
            models_loss.append(filename)
        elif "best_recall" in filename:
            models_recall.append(filename)

    accuracy_df = load_test_mlp_models(X_test, y_test, X_zd, y_zd, models_accuracy, directory)
    accuracy_df["type"] = "accuracy"
    performance_df = accuracy_df

    loss_df = load_test_mlp_models(X_test, y_test, X_zd, y_zd, models_loss, directory)
    loss_df["type"] = "loss"
    performance_df = pd.concat([performance_df, loss_df], axis=0)

    recall_df = load_test_mlp_models(X_test, y_test, X_zd, y_zd, models_recall, directory)
    recall_df["type"] = "recall"
    performance_df = pd.concat([performance_df, recall_df], axis=0)

    performance_df.to_csv(os.path.join(directory, "performance_best_models.csv"), index=False)

def train_test_models(data_path: str, test_zero_days: list[str], models_path: str):
    df = pd.read_csv(data_path)
    df.info()
    # feature selection
    """df = df[["IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAG_6", "PROTOCOL_17",
             "PROTOCOL_6", "L7_PROTO_131", "L7_PROTO_91", "L7_PROTO_0", "Label",
             "Attack"]]"""

    df = df[["IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAG_6", "PROTOCOL_17",
             "PROTOCOL_6", "L7_PROTO_7", "L7_PROTO_88", "L7_PROTO_5", "L7_PROTO_131", "L7_PROTO_91", "L7_PROTO_0",
             "Label", "Attack"]]

    mlp_stats_path = os.path.join(models_path, "mlp_stats.csv")
    rf_stats_path = os.path.join(models_path, "random_forest_stats.csv")
    svm_stats_path = os.path.join(models_path, "svm_stats.csv")
    rf_stats_df = None
    mlp_stats_df = None
    svm_stats_df = None

    if os.path.exists(rf_stats_path):
        rf_stats_df = pd.read_csv(rf_stats_path)

    if os.path.exists(mlp_stats_path):
        mlp_stats_df = pd.read_csv(mlp_stats_path)

    if os.path.exists(svm_stats_path):
        svm_stats_df = pd.read_csv(svm_stats_path)

    for attack in test_zero_days:
        print_line()
        print(f"Testing models for Zero-Day Attack {attack}")

        datasets = get_datasets_pandas(df, attack)
        datasets_numpy = {"X_train": datasets["X_train"].drop(columns=["Attack"], axis=1).to_numpy(), "y_train": datasets["y_train"].to_numpy(),
                          "X_test": datasets["X_test"].drop(columns=["Attack"], axis=1).to_numpy(), "y_test": datasets["y_test"].to_numpy(),
                          "X_zd": datasets["X_zd"].drop(columns=["Attack"], axis=1).to_numpy(), "y_zd": datasets["y_zd"].to_numpy()}

        datasets["X_train"].info()
        print(datasets_numpy["X_train"])
        print_line()
        print("Number of Attack samples in train vs test.")
        for attack in df["Attack"].unique().to_pandas():
            print(f'Number of samples for "{attack}" in train {datasets["X_train"][datasets["X_train"]["Attack"] == attack].shape[0]}')
            print(f'Number of samples for "{attack}" in test {datasets["X_test"][datasets["X_test"]["Attack"] == attack].shape[0]}')
            print_line(num=50)
        print_line()

        svm_stats = SVM("SVM_ZeroDay_" + attack, models_path, **datasets_numpy)
        if svm_stats_df is None:
            svm_stats_df = svm_stats
        else:
            svm_stats_df = pd.concat([svm_stats_df, svm_stats], axis=0)

        rf_stats = RF("RF_ZeroDay_" + attack, models_path, **datasets_numpy)
        if rf_stats_df is None:
            rf_stats_df = rf_stats
        else:
            rf_stats_df = pd.concat([rf_stats_df, rf_stats], axis=0)

        mlp_stats = MLP("MLP_ZeroDay_" + attack, os.path.join(models_path, "mlp/"), **datasets_numpy)
        if mlp_stats_df is None:
            mlp_stats_df = mlp_stats
        else:
            mlp_stats_df = pd.concat([mlp_stats_df, mlp_stats], axis=0)
        print_line()

    if rf_stats_df is not None:
        rf_stats_df.to_csv(rf_stats_path)
    if mlp_stats_df is not None:
        mlp_stats_df.to_csv(mlp_stats_path)
    if svm_stats_df is not None:
        svm_stats_df.to_csv(svm_stats_path)


# load_transform()
#visualize()
#quit()

train_test_models(os.path.join(directory, "processed", "ml_ready.csv"), ["Brute Force -XSS"], "modelsfinalv3")
quit()

df = pd.read_csv(os.path.join(directory, "processed", "ml_ready.csv"))
df.info()
# feature selection
df = df[["IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAG_6", "PROTOCOL_17",
         "PROTOCOL_6", "L7_PROTO_7", "L7_PROTO_131", "L7_PROTO_91", "L7_PROTO_0", "Label",
         "Attack"]]

datasets = get_datasets_pandas(df, "Brute Force -XSS")

X_train = datasets["X_train"]
X_test = datasets["X_test"]

for attack in df["Attack"].unique().to_pandas():
    print(f'Number of samples for "{attack}" in train {X_train[X_train["Attack"] == attack].shape[0]}')
    print(f'Number of samples for "{attack}" in test {X_test[X_test["Attack"] == attack].shape[0]}')
    print_line()

metric_features = ["IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS"]

print("Correlation with Label")
print(df[["IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "Label"]].corr()["Label"].to_string())

for m in metric_features:
    for m2 in metric_features:
        if m == m2:
            continue
        test_linear_separability(df, m, m2, "Label")

#evaluate_best_mlp_models(df, "Brute Force -XSS", "modelsfinal/mlp_test/")

performance_df = pd.read_csv("modelsfinal/mlp_test/performance_best_models.csv")
sorted = performance_df.sort_values(by=["false_alarm_rate", "recall", "zero_day_detection_rate"], ascending=True)
print(sorted[["model_name", "false_alarm_rate", "accuracy", "recall", "zero_day_detection_rate"]])