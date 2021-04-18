import pandas as pd


def data_loader():
    training_csv_path = './data/adult_training.csv'
    training_dataset = pd.read_csv(training_csv_path)

    target_column = 'income'
    y_training = training_dataset[target_column]
    x_training = training_dataset.drop(target_column, axis=1)
    return y_training, x_training

