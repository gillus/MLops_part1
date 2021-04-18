import pytest
from model.utils import model_metrics
from data.datamanager import data_loader
import joblib


@pytest.fixture
def adult_test_dataset():
    path = './data/adult_test.csv'
    x, y = data_loader(path)
    return x, y, path


def test_dataloader(adult_test_dataset):
    x, y, _ = adult_test_dataset
    n_unique = x.nunique(axis=0).values
    assert n_unique.min() > 1


def test_model_metrics(adult_test_dataset):
    clf = joblib.load('./model.pkl')
    metrics = model_metrics(clf, adult_test_dataset[2])
    assert metrics['>50K']['precision'] > 0.7
    assert metrics['>50K']['recall'] > 0.1
