from sklearn.pipeline import Pipeline
from src.fit_model import FitModel


def test_pipeline():
    model = FitModel()
    pipeline = model._pipeline()

    # Check if the output is a Pipeline instance
    assert isinstance(pipeline, Pipeline), "Output should be an instance of Pipeline"

    # Check the steps in the pipeline
    steps = [step[0] for step in pipeline.steps]
    expected_steps = ['age_binning', 'family_categorizing', 'fare_transformer', 'feature_engineering', 'predictive_modeling']
    assert steps == expected_steps, f"Pipeline steps should be {expected_steps}"
