import typing
from mlflow.models.signature import infer_signature
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import json


def data_loader(path: str):
    dataset = pd.read_csv(path)
    target_column = 'income'
    y_training = dataset[target_column]
    x_training = dataset.drop(target_column, axis=1)

    return x_training, y_training


def train_random_forest_model(data_path: str,
                              parameters=None):

    if parameters is None:
        parameters = dict(n_estimators=100, max_depth=4, criterion='gini',
                          min_sample_leaf=1)

    x_training, y_training = data_loader(data_path)

    ordinal_features = x_training.select_dtypes(include="number").columns
    categorical_features = x_training.select_dtypes(include="object").columns

    ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    x_encoder = ColumnTransformer(transformers=[('ord', ordinal_transformer, ordinal_features),
                                                ('cat', categorical_transformer, categorical_features)])

    rf_clf = RandomForestClassifier(n_estimators=parameters['n_estimators'],
                                    max_depth=parameters['max_depth'],
                                    criterion=parameters['criterion'],
                                    min_samples_leaf=parameters['min_sample_leaf'],
                                    random_state=42)

    rf_pipeline = Pipeline(steps=[("preprocessing", x_encoder), ("rf_model", rf_clf)])
    rf_pipeline.fit(x_training, y_training)
    return rf_pipeline


def model_metrics(clf, data_path):

    x_test, y_test = data_loader(data_path)
    metrics = classification_report(y_test, clf.predict(x_test), output_dict=True)

    return metrics


def package_sklearn_model(x_sample, clf):

    signature = infer_signature(x_sample, clf.predict(x_sample))
    input_example = {}
    for i in x_sample.columns:
        input_example[i] = x_sample[i][0]

    mlflow.sklearn.save_model(clf, "best_model", signature=signature, input_example=input_example)

    return


def convert_sklearn_onnx(clf):

    return

