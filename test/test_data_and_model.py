import pytest
from model.model_training import data_loader, model_metrics
import joblib


@pytest.fixture
def adult_test_dataset():
    path = './data/adult_test.csv'
    x, y = data_loader(path)
    return x, y


def test_model_metrics(adult_test_dataset):
    clf = joblib.load('./model.pkl')
    print(clf.predict(adult_test_dataset[0]))
    return