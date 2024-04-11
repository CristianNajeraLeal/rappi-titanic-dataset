from src.fit_model.fit_model import FitModel
import numpy as np


def test_init():
    model = FitModel()
    assert model._cv == 2
    assert model._random_state == 42
    assert model._bin_labels == ['child', 'teen', 'teen_plus', 'young_adult', 'adult', 'elder']
    assert model._bin_values == [-np.inf, 10, 20, 30, 40, 50, np.inf]
    assert model._param_grid == {
        'predictive_modeling__n_estimators': [10, 50],
        'predictive_modeling__max_depth': [None, 10],
        'predictive_modeling__min_samples_split': [2, 5],
        'predictive_modeling__min_samples_leaf': [1, 2]
    }
    assert model._median_age == 28
