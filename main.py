from model.model_training import train_random_forest_model


if __name__ == '__main__':
    clf = train_random_forest_model('./data/adult_training.csv')