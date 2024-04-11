from sklearn.compose import ColumnTransformer
from src.fit_model import FitModel

def test_column_transformer():
    column_transformer = FitModel._column_transformer()

    # Check if the output is an instance of ColumnTransformer
    assert isinstance(column_transformer, ColumnTransformer), "Output should be an instance of ColumnTransformer"

    # Optionally, check the configurations of the transformers
    transformers = column_transformer.transformers
    expected_names = ['ohe_engineering', 'min_max', 'columns_to_drop']
    for transformer, expected_name in zip(transformers, expected_names):
        assert transformer[0] == expected_name, f"Expected transformer name '{expected_name}', but got '{transformer[0]}'"
