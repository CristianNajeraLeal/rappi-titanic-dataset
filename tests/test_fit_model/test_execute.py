import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.fit_model import FitModel


def test_execute():
    model = FitModel()
    # Mock DataFrame
    data = {'Embarked': ['S', 'C', 'S', 'S', 'C'],
            'Survived': [1, 0, 0, 0, 1],
            'Age': [22, 28, 21, 20, 29],
            'Parch': [1, 2, 3, 4, 5],
            'SibSp': [1, 2, 3, 4, 5],
            'PassengerId': [123, 456, 798, 753, 745],
            'Name': ['A', 'B', 'C', 'D', 'E'],
            'Ticket': ['A', 'B', 'C', 'D', 'E'],
            'Cabin': [None, 'A', None, 'A', None],
            'Fare': [7.25, 71.83, 5.5, 4.5, 80.4],
            'Sex': ['female', 'male', 'male', 'male', 'female']}
    n_samples = 50
    expanded_data = {key: (values * (n_samples // len(values) + 1))[:n_samples] for key, values in data.items()}
    df = pd.DataFrame(expanded_data)

    # Execute the method
    result = model.execute(df)

    # Check if the output is a GridSearchCV instance
    assert isinstance(result, GridSearchCV), "Output should be an instance of GridSearchCV"
    # Optionally, verify that the result uses the correct scoring method
    assert result.scoring == "accuracy", "Scoring method should be accuracy"
