"""
This module provides functionalities for preparing data, transforming it, and fitting a
predictive model using a machine learning pipeline for the Titanic dataset.
"""


import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


class FitModel:
    """
    A class designed to encapsulate the entire process of data cleaning, feature engineering,
        and model fitting using a RandomForest classifier for the Titanic dataset.

    Attributes:
        _cv (int): Number of cross-validation folds.
        _random_state (int): Random state for reproducibility.
        _bin_labels (list): Labels for age categorization.
        _bin_values (list): Bin edges for age categorization.
        _param_grid (dict): Parameters to be explored during grid search.
        _median_age (int): Median age value for imputing missing ages.
    """

    def __init__(self):
        """
        Initializes the FitModel class with default settings for the machine learning pipeline.
        """
        self._cv = 2
        self._random_state = 42
        self._bin_labels = ['child', 'teen', 'teen_plus', 'young_adult', 'adult', 'elder']
        self._bin_values = [-np.inf, 10, 20, 30, 40, 50, np.inf]
        self._param_grid = {
            'predictive_modeling__n_estimators': [10, 50],
            'predictive_modeling__max_depth': [None, 10],
            'predictive_modeling__min_samples_split': [2, 5],
            'predictive_modeling__min_samples_leaf': [1, 2]
        }
        self._median_age = 28

    @staticmethod
    def _clean_na(df):
        """
        Remove rows from the dataframe where the 'Embarked' column is NA.

        :param df: A pandas DataFrame.
        :return: A DataFrame with NA rows in the 'Embarked' column removed.
        """
        df = df[df['Embarked'].notna()]
        return df

    @staticmethod
    def _get_x_y(df):
        """
        Separate the dataframe into feature vectors and target vector.

        :param df: A pandas DataFrame.
        :return: A tuple (x, y) where x is all columns except 'Survived', and y is 'Survived'.
        """
        x = df.drop(columns=['Survived'])
        y = df[['Survived']]
        return x, y

    @staticmethod
    def create_family_category(df):
        """
        Creates a new binary column 'small_family' based on family size calculated
        from 'Parch' and 'SibSp'.

        :param df: A pandas DataFrame.
        :return: The DataFrame with the new column and without the original family size columns.
        """
        df['family_size'] = df['Parch'] + df['SibSp'] + 1
        df['small_family'] = df['family_size'].isin([2, 3, 4])
        df.drop(columns=['Parch', 'SibSp', 'family_size'], inplace=True)
        return df

    def create_age_bins(self, df):
        """
        Categorizes the 'Age' column into bins.

        :param df: A pandas DataFrame to transform.
        :return: The transformed DataFrame with 'Age' replaced by categorized bins.
        """
        df.fillna({'Age': self._median_age}, inplace=True)
        age_bins = pd.cut(df['Age'], bins=self._bin_values, labels=self._bin_labels)
        df_age_bins = pd.DataFrame(age_bins)
        df.drop(columns=['Age'], inplace=True)
        df = pd.concat([df, df_age_bins], axis=1)
        return df

    @staticmethod
    def log_fare(df):
        """
        Applies a logarithmic transformation to the 'Fare' column to reduce skewness.

        :param df: A pandas DataFrame.
        :return: The DataFrame with the transformed 'Fare' column.
        """
        df['fare_log'] = np.log(df['Fare'] + 1)
        df.drop(columns=['Fare'], inplace=True)
        return df

    def _function_transformers(self):
        """
        Creates and returns function transformers for age binning, family categorization,
        and fare transformation.

        :return: A tuple of FunctionTransformer objects.
        """
        age_binner = FunctionTransformer(self.create_age_bins, validate=False)
        family_category = FunctionTransformer(self.create_family_category, validate=False)
        fare_transformer = FunctionTransformer(self.log_fare, validate=False)
        return age_binner, family_category, fare_transformer

    @staticmethod
    def _column_transformer():
        """
        Configures and returns a ColumnTransformer for the pipeline.

        :return: An instance of ColumnTransformer configured for one-hot encoding, scaling,
                    and dropping columns.
        """
        return ColumnTransformer(
            transformers=[
                ('ohe_engineering',
                 OneHotEncoder(handle_unknown='ignore'),
                 ['Age', 'Sex', 'Embarked']),
                ('min_max', MinMaxScaler(), ['fare_log']),
                ('columns_to_drop', 'drop', ['PassengerId', 'Name', 'Ticket', 'Cabin'])
            ],
            remainder='passthrough'
        )

    def _pipeline(self):
        """
        Constructs and returns the full preprocessing and classification pipeline.

        :return: An instance of sklearn.pipeline.Pipeline configured with all transformations
                    and a classifier.
        """
        age_binner, family_category, fare_transformer = self._function_transformers()
        column_transformer = self._column_transformer()
        pipeline = Pipeline(steps=[('age_binning', age_binner),
                                   ('family_categorizing', family_category),
                                   ('fare_transformer', fare_transformer),
                                   ('feature_engineering', column_transformer),
                                   ('predictive_modeling',
                                    RandomForestClassifier(random_state=42))])
        return pipeline

    def _grid_search(self, df):
        """
        Executes grid search to find the best model parameters.

        :param df: A pandas DataFrame to use for training.
        :return: A GridSearchCV object fitted to the data.
        """
        x, y = self._get_x_y(df=df)
        pipeline = self._pipeline()
        grid_search = GridSearchCV(estimator=pipeline,
                                   scoring="accuracy",
                                   param_grid=self._param_grid,
                                   cv=self._cv,
                                   verbose=0,
                                   n_jobs=-1)
        grid_search.fit(x, y)
        return grid_search

    def execute(self, df):
        """
        Cleans the data and performs grid search to optimize the model.

        :param df: A pandas DataFrame to process and model.
        :return: A fitted GridSearchCV object containing the optimized model.
        """
        df = self._clean_na(df)
        grid_search = self._grid_search(df)
        return grid_search
