import pandas as pd
import mlflow
from itertools import product
import warnings
from model.model_training import train_random_forest_model, model_metrics
import argparse
import json


def grid_search_random_forest():

    max_depth = [10,15]
    criterion = ['gini','entropy']
    min_samples_leaf = [5,10]
    n_estimators= [50,100]

    parameters = product(max_depth, criterion, min_samples_leaf,n_estimators)
    parameters_list = list(parameters)
    print('Number of experiments:',len(parameters_list))

    # Hyperparameter search
    results = []
    best_precision = float("inf")
    warnings.filterwarnings('ignore')

    for i, param in enumerate(parameters_list):
        print('Running experiment number ', i)
        with mlflow.start_run(run_name='Adult_random_forest'):
            # Logging each experiment's inputs
            mlflow.log_param('param-depth', param[0])
            mlflow.log_param('param-criterion', param[1])
            mlflow.log_param('param-min_samples_leaf', param[2])
            mlflow.log_param('param-n_estimators', param[3])

            try:
                clf = train_random_forest_model(data_path='../data/adult_training.csv')
                metrics = model_metrics(clf, data_path='../data/adult_validation.csv')
                # Logging each experiment's metrics
                mlflow.log_metric("accuracy", metrics['>50K']['precision'])
                mlflow.log_metric("F1", metrics['>50K']['f1-score'])
                mlflow.log_artifact(json.dump(metrics))
                if metrics['>50K']['precision'] < best_precision:
                    best_model = clf
                    best_f1 = metrics['>50K']['f1-score']
                    best_param = param
                results.append([param, metrics['>50K']['precision']])

            except ValueError:
                print('bad parameter combination:', param)
                continue

    mlflow.end_run()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="experiment_name")
    args, leftovers = parser.parse_known_args()

    grid_search_random_forest(args.name)