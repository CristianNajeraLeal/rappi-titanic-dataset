import pandas as pd
import numpy as np
from src.fit_model import FitModel


def test_log_fare():
    # Prepare the data
    data = {'Fare': [10, 20, 30]}
    df = pd.DataFrame(data)

    # Expected outcome
    expected_fare_log = np.log(df['Fare'] + 1)
    expected_df = pd.DataFrame({'fare_log': expected_fare_log})

    # Perform the operation
    result_df = FitModel.log_fare(df)

    # Verify the results
    pd.testing.assert_frame_equal(result_df, expected_df)
