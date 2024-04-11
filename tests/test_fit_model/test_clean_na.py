import pandas as pd
import numpy as np
from src.fit_model.fit_model import FitModel


def test_clean_na():
    data = {
        'Embarked': ['S', np.nan, 'C', 'Q', np.nan],
        'Fare': [72, 30, 15, 7.25, 7.75]
    }
    df = pd.DataFrame(data)
    cleaned_df = FitModel._clean_na(df)

    # Assert DataFrame size is reduced to 3, removing 2 rows with NaN in 'Embarked'
    assert cleaned_df.shape[0] == 3
    # Assert there are no NaN values in the 'Embarked' column of the cleaned DataFrame
    assert cleaned_df['Embarked'].isnull().sum() == 0
