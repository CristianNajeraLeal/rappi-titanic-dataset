import pandas as pd
from src.fit_model.fit_model import FitModel


def test_get_x_y():
    data = {'Survived': [1, 0, 1], 'Age': [22, 38, 26], 'Sex': ['male', 'female', 'female']}
    df = pd.DataFrame(data)
    X, y = FitModel._get_x_y(df)

    # Asserting X does not contain 'Survived' and y only contains 'Survived'
    assert 'Survived' not in X.columns and 'Age' in X.columns and 'Sex' in X.columns
    assert list(y.columns) == ['Survived'] and len(y) == 3
