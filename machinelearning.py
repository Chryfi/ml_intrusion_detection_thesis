import importlib.util
import time
from datetime import datetime

import numpy as np
import os

# deactivate stupid warning messages of keras that we are not in a datacenter environment wtf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

cudf_found = importlib.util.find_spec("cudf")
if cudf_found is not None:
    import cudf as pd
    from cuml import RandomForestClassifier
    from cuml import accuracy_score
    from cuml import svm
    # from cuml import train_test_split
    from cuml.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score

    print("cudf found")
else:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import *
import tensorflow as tf
import keras
import numpy
from utils import *

def get_model_performance(predicted: [int], expected: [int], train_time: float = None, test_time: float = None, prefix: str = ""):
    cfm = confusion_matrix(expected, predicted)
    tn, fp, fn, tp = cfm.ravel()

    data = {
        prefix + "recall": tp / (tp + fn), prefix + "accuracy": (tp + tn) / (tp + tn + fp + fn),
        prefix + "precision": tp / (tp + fp), prefix + "false_alarm_rate": fp / (tn + fp),
        prefix + "tn": tn, prefix + "fp": fp, prefix + "fn": fn, prefix + "tp": tp
    }

    if train_time is not None:
        data[prefix + "train_time"] = train_time

    if test_time is not None:
        data[prefix + "test_time"] = test_time

    return pd.DataFrame(data=data)


def test_linear_separability(df: pd.DataFrame, col1: str, col2: str, class_col: str):
    print_line()
    print(f"Testing for linear separability of {class_col} for {col1} and {col2}")

    train = df[[col1, col2]].to_numpy()
    y = df[class_col].to_numpy()

    params = {
        "C": 2e64
    }
    svm_classifier = svm.LinearSVC(**params)
    svm_classifier.fit(train, y)
    predictions = svm_classifier.predict(train)
    print(f"Accuracy {accuracy_score(y, predictions)}")
    print(f"Recall {recall_score(y, predictions)}")
    print_line()


def get_datasets_pandas(df: pd.DataFrame, zero_day_type: str) -> dict[str, pd.DataFrame]:
    """
        Get the datasets used for this machine learning experiment.
        Returns:
            dict[str, np.ndarray]: keys are X_train, y_train, X_test, y_test, X_zd, y_zd
    """

    df_no_zd = df[df["Attack"] != zero_day_type]
    df_zd = df[df["Attack"] == zero_day_type]

    print_line()
    print("Generating datasets...")
    print(f'Zero-Day "{zero_day_type}" has {df_zd.shape[0]} samples')

    X = df_no_zd.drop(["Label"], axis=1)
    X_zd = df_zd.drop(["Label"], axis=1)
    y = df_no_zd["Label"]
    y_zd = df_zd["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X.to_pandas(), y.to_pandas(), test_size=0.2, shuffle=True,
                                                        random_state=42)
    print_line(num=50)
    print(f"Samples in X_test {X_test.shape[0]}")
    print(f'Samples of Attacks in y_test {np.sum(y_test)}')
    print(f'Samples of Normal activity y_test {len(y_test) - np.sum(y_test)}')
    print_line(num=50)
    print(f"Samples in X_train {X_train.shape[0]}")
    print(f'Samples of Attacks in y_train {np.sum(y_train)}')
    print(f'Samples of Normal activity y_train {len(y_train) - np.sum(y_train)}')
    print_line(num=50)
    print("Undersampling...")
    X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
    print_line(num=50)
    print(f"Samples in X_train {X_train.shape[0]}")
    print(f'Samples of Attacks {np.sum(y_train)}')
    print(f'Samples of Normal activity {len(y_train) - np.sum(y_train)}')
    print_line()

    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
            "X_zd": X_zd, "y_zd": y_zd}


def get_datasets(df: pd.DataFrame, zero_day_type: str) -> dict[str, np.ndarray]:
    """
        Get the datasets used for this machine learning experiment.
        Returns:
            dict[str, np.ndarray]: keys are X_train, y_train, X_test, y_test, X_zd, y_zd
    """

    df_no_zd = df[df["Attack"] != zero_day_type]
    df_zd = df[df["Attack"] == zero_day_type]

    print_line()
    print("Generating datasets...")
    print(f'Zero-Day "{zero_day_type}" has {df_zd.shape[0]} samples')

    X = df_no_zd.drop(["Attack", "Label"], axis=1)
    X_zd = df_zd.drop(["Attack", "Label"], axis=1)
    y = df_no_zd["Label"]
    y_zd = df_zd["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    print_line(num=50)
    print(f"Samples in X_test {X_test.shape[0]}")
    print(f'Samples of Attacks in y_test {np.sum(y_test)}')
    print(f'Samples of Normal activity y_test {len(y_test) - np.sum(y_test)}')
    print_line(num=50)
    print(f"Samples in X_train {X_train.shape[0]}")
    print(f'Samples of Attacks in y_train {np.sum(y_train)}')
    print(f'Samples of Normal activity y_train {len(y_train) - np.sum(y_train)}')
    print_line(num=50)
    print("Undersampling...")
    X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train.to_numpy(), y_train.to_numpy())
    print_line(num=50)
    print(f"Samples in X_train {X_train.shape[0]}")
    print(f'Samples of Attacks {np.sum(y_train)}')
    print(f'Samples of Normal activity {len(y_train) - np.sum(y_train)}')
    print_line()
    X_train = X_train  # .to_numpy()
    y_train = y_train  # .to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    X_zd = X_zd.to_numpy()
    y_zd = y_zd.to_numpy()

    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
            "X_zd": X_zd, "y_zd": y_zd}


def get_training_callbacks(save_path: str, model_name: str):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_path, 'best_recall_{epoch:02d}-{val_recall:.2f}' + model_name + '.keras'),
            monitor='val_recall', save_best_only=True),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_path, 'best_loss_{epoch:02d}-{val_loss:.2f}' + model_name + '.keras'),
            monitor='val_loss', save_best_only=True),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_path, 'best_accuracy_{epoch:02d}-{val_accuracy:.2f}' + model_name + '.keras'),
            monitor='val_accuracy', save_best_only=True)
    ]


def SVM(modelname: str, save_path: str,
        X_train: numpy.ndarray, y_train: numpy.ndarray,
        X_test: numpy.ndarray, y_test: numpy.ndarray,
        X_zd: numpy.ndarray, y_zd: numpy.ndarray) -> pd.DataFrame:
    print_line()
    print("SVM")
    params = {
        "kernel": "rbf",
        "C": 1,
        "degree": 3,
        "gamma": "scale",
        "coef0": 0.0,
        "probability": False,
    }
    svm_classifier = svm.SVC(**params)

    start_training = time.time()
    svm_classifier.fit(X_train, y_train)
    end_training = time.time()

    start_testing = time.time()
    predictions = svm_classifier.predict(X_test)
    end_testing = time.time()

    test_accuracy = accuracy_score(y_test, predictions)
    test_precision = precision_score(y_test, predictions, zero_division=np.nan)
    test_recall = recall_score(y_test, predictions, zero_division=np.nan)
    print(f"Test accuracy: {test_accuracy} - precision: {test_precision} - recall: {test_recall}")

    zd_predictions = svm_classifier.predict(X_zd)
    zd_accuracy = accuracy_score(y_zd, zd_predictions)
    zd_precision = precision_score(y_zd, zd_predictions, zero_division=np.nan)
    zd_recall = recall_score(y_zd, zd_predictions, zero_division=np.nan)
    print(f"Zero-Day accuracy: {zd_accuracy} - precision: {zd_precision} - recall: {zd_recall}")
    print_line()

    test_performance = get_model_performance(predictions, y_test, train_time=end_training - start_training, test_time=end_testing - start_testing)
    test_performance["zero_day_detection_rate"] = zd_recall

    y_train_pred = svm_classifier.predict(X_train)
    y_train_pred = (y_train_pred > 0.5).astype(int).flatten()

    train_performance = get_model_performance(y_train_pred, y_train, prefix="train_")

    stats = pd.DataFrame(data={"modelname": modelname, "training date": datetime.now().strftime("%H-%M_%d-%m-%Y"),
                               "kernel": params["kernel"], "C": params["C"],
                               "degree": params["degree"], "gamma": params["gamma"],
                               "coef0": params["coef0"], "probability": params["probability"]})

    return pd.concat([stats, train_performance, test_performance], axis=1)


def MLP(modelname: str, save_path: str,
        X_train: numpy.ndarray, y_train: numpy.ndarray,
        X_test: numpy.ndarray, y_test: numpy.ndarray,
        X_zd: numpy.ndarray, y_zd: numpy.ndarray) -> pd.DataFrame:
    print_line()
    print("Multi Layer Perceptron")
    print(f"MLP Input size {X_train.shape[1]}")
    NUM_NEURONS = X_train.shape[1]  # round((X_train.shape[1] + 1) / 2)# + 2
    EPOCHS = 256
    LEARNING_RATE = 0.001
    BATCH_SIZE = 82
    NUM_HIDDENLAYERS = 3
    ACTIVATION = "relu"
    OUTPUT_ACTIVATION = "sigmoid"

    layers = [keras.Input(shape=(X_train.shape[1],))]

    for h in range(NUM_HIDDENLAYERS):
        layers.append(keras.layers.Dense(NUM_NEURONS, activation=ACTIVATION))

    layers.append(keras.layers.Dense(1, activation=OUTPUT_ACTIVATION))

    model = keras.models.Sequential(layers)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.99, beta_2=0.999),
                  # , weight_decay=0.01),
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', "recall", "precision"])

    start_training = time.time()
    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[get_training_callbacks(save_path, modelname)],
              validation_data=(X_test, y_test),
              verbose=1)
    end_training = time.time()

    model.save(os.path.join(save_path, modelname + ".keras"))

    print("Test Accuracy")
    test_metrics = model.evaluate(X_test, y_test, return_dict=True)
    print(test_metrics)

    print("Zero-Day-Attacks")
    zd_metrics = model.evaluate(X_zd, y_zd, return_dict=True)
    print(zd_metrics)
    print_line()

    start_testing = time.time()
    y_pred = model.predict(X_test)
    end_testing = time.time()

    y_pred = (y_pred > 0.5).astype(int).flatten()

    test_performance = get_model_performance(y_pred, y_test, train_time=end_training - start_training, test_time=end_testing - start_testing)
    test_performance["zero_day_detection_rate"] = zd_metrics["recall"]

    y_train_pred = model.predict(X_train)
    y_train_pred = (y_train_pred > 0.5).astype(int).flatten()

    train_performance = get_model_performance(y_train_pred, y_train, prefix="train_")

    stats = pd.DataFrame(data={"modelname": modelname, "training date": datetime.now().strftime("%H-%M_%d-%m-%Y"),
                               "epochs": EPOCHS, "hidden layers": NUM_HIDDENLAYERS, "neurons": NUM_NEURONS,
                               "learning rate": LEARNING_RATE,
                               "hidden neurons activation": ACTIVATION, "output activation": OUTPUT_ACTIVATION,
                               "batch size": BATCH_SIZE})



    return pd.concat([stats, train_performance, test_performance], axis=1)


def RF(modelname: str, save_path: str,
       X_train: numpy.ndarray, y_train: numpy.ndarray,
       X_test: numpy.ndarray, y_test: numpy.ndarray,
       X_zd: numpy.ndarray, y_zd: numpy.ndarray) -> pd.DataFrame:
    print_line()
    print("Random Forest")
    rf_params = {
        'n_estimators': 150,
        "max_depth": 50,
        "n_bins": 128,
        "random_state": 2,
        "n_streams": 18,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True
    }

    n_estimators = [int(x) for x in numpy.linspace(start=100, stop=500, num=25)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in numpy.linspace(10, 100, num=10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_classifier = RandomForestClassifier(**rf_params)

    start_training = time.time()
    rf_classifier.fit(X_train, y_train)
    end_training = time.time()

    """rf_classifier = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf_classifier, param_distributions=random_grid, n_iter=100, cv=3,
                                   verbose=2, random_state=rf_params["random_state"], n_jobs=-1)
    rf_random.fit(X_train, y_train)

    print_line()
    print("Best Random Forest Paramters according to RandomizedSearchCV:")
    print(rf_random.best_params_)
    print_line()"""

    # rf_classifier = RandomForestClassifier(**rf_random.best_params_)

    start_testing = time.time()
    predictions = rf_classifier.predict(X_test)
    end_testing = time.time()

    test_accuracy = accuracy_score(y_test, predictions)
    test_precision = precision_score(y_test, predictions, zero_division=np.nan)
    test_recall = recall_score(y_test, predictions, zero_division=np.nan)
    print(f"Test accuracy: {test_accuracy} - precision: {test_precision} - recall: {test_recall}")

    zd_predictions = rf_classifier.predict(X_zd)
    zd_accuracy = accuracy_score(y_zd, zd_predictions)
    zd_precision = precision_score(y_zd, zd_predictions, zero_division=np.nan)
    zd_recall = recall_score(y_zd, zd_predictions, zero_division=np.nan)
    print(f"Zero-Day accuracy: {zd_accuracy} - precision: {zd_precision} - recall: {zd_recall}")
    print_line()

    test_performance = get_model_performance(predictions.astype(int), y_test, train_time=end_training - start_training,
                                             test_time=end_testing - start_testing)
    test_performance["zero_day_detection_rate"] = zd_recall

    y_train_pred = rf_classifier.predict(X_train)
    y_train_pred = (y_train_pred > 0.5).astype(int).flatten()

    train_performance = get_model_performance(y_train_pred, y_train, prefix="train_")

    stats = pd.DataFrame(data={"modelname": modelname, "training date": datetime.now().strftime("%H-%M_%d-%m-%Y"),
                               "n_estimators": rf_params["n_estimators"], "max_depth": rf_params["max_depth"],
                               "max_features": rf_params["max_features"],
                               "min_samples_split": rf_params["min_samples_split"],
                               "min_samples_leaf": rf_params["min_samples_leaf"], "bootstrap": rf_params["bootstrap"],
                               "n_bins": rf_params["n_bins"]})

    return pd.concat([stats, train_performance, test_performance], axis=1)
