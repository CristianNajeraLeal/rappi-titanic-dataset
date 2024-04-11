from sklearn.preprocessing import FunctionTransformer
from src.fit_model import FitModel

def test_function_transformers():
    model = FitModel()
    transformers = model._function_transformers()

    # Check if the output is a tuple
    assert isinstance(transformers, tuple), "Output should be a tuple"

    # Check the length of the tuple
    assert len(transformers) == 3, "There should be exactly three transformers"

    # Check if all elements are instances of FunctionTransformer
    for transformer in transformers:
        assert isinstance(transformer, FunctionTransformer), "All elements should be instances of FunctionTransformer"
