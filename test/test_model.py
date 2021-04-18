import pytest
from model.model_training import data_loader, model_metrics
import sys
sys.path.append('../')


@pytest.fixture
def adult_test_dataset():
    path = './data/adult_test.csv'
    x, y = data_loader(path)
    return x, y


def test_model_metrics(adult_test_dataset):
    print(adult_test_dataset[0].shape)
    return