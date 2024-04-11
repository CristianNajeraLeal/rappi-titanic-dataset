import pandas as pd
from src.fit_model.fit_model import FitModel


def test_create_family_category():
    data = {'SibSp': [1, 0, 3], 'Parch': [0, 1, 2], 'Age': [22, 38, 26]}
    df = pd.DataFrame(data)
    df_transformed = FitModel.create_family_category(df)

    # Check if 'small_family' is created correctly and 'SibSp', 'Parch', 'family_size' are removed
    assert 'small_family' in df_transformed.columns
    assert 'SibSp' not in df_transformed.columns and 'Parch' not in df_transformed.columns and 'family_size' not in df_transformed.columns
    assert df_transformed['small_family'].dtype == bool
    assert df_transformed['small_family'].iloc[0] == True and df_transformed['small_family'].iloc[2] == False
