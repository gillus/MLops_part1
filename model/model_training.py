from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import json
import os
import joblib
from data.datamanager import data_loader


def train_random_forest_model(data_path: str,
                              parameters=None):

    if parameters is None:
        if os.path.exists('./params.json'):
            parameters = json.load(open("params.json", "r"))
        else:
            parameters = dict(n_estimators=100, max_depth=4, criterion='gini',
                              min_sample_leaf=1)
    print(parameters)
    x_training, y_training = data_loader(data_path)

    ordinal_features = x_training.select_dtypes(include="number").columns
    categorical_features = x_training.select_dtypes(include="object").columns

    ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    x_encoder = ColumnTransformer(transformers=[('ord', ordinal_transformer, ordinal_features),
                                                ('cat', categorical_transformer, categorical_features)])

    rf_clf = RandomForestClassifier(n_estimators=parameters['n_estimators'],
                                    max_depth=parameters['max_depth'],
                                    criterion=parameters['criterion'],
                                    min_samples_leaf=parameters['min_sample_leaf'],
                                    random_state=42)

    rf_pipeline = Pipeline(steps=[("preprocessing", x_encoder), ("rf_model", rf_clf)])
    rf_pipeline.fit(x_training, y_training)

    joblib.dump(rf_pipeline, 'model.pkl')

    return rf_pipeline


if __name__ == '__main__':

    train_random_forest_model('./data/adult_training.csv')