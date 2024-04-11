import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from src.fit_model.fit_model import FitModel


def test_create_age_bins():
    model = FitModel()
    data = {'Age': [5, 17, 25, 55, None]}
    df = pd.DataFrame(data)
    df_transformed = model.create_age_bins(df)

    assert 'Age' in df_transformed.columns, "DataFrame should contain the 'Age_Bin' column after processing"
    assert all(bin_label in model._bin_labels for bin_label in df_transformed['Age']), "All age bins should be one of the predefined labels"
