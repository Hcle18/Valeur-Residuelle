from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from typing import Optional

def create_transformer_pipeline(categorical_features: list[str],
                    special_impute_cat_features: Optional[dict[str]] = None,
                    numerical_features: list[str] = None,
                    special_impute_num_features: dict[str] = None
                    ):
    """
    Create a preprocessing pipeline for categorical & numerical variables
    Args:
        categorical_features (list): List of categorical column names
        special_impute_features (dict): Dictionary of {column: value} for special imputation
        numerical_features (list): List of numerical column names
        special_impute_num_features (dict): Dictionary of {column: strategy} for special imputation
    Returns:
        full_pipeline (ColumnTransformer): A ColumnTransformer object that applies the specified transformations
        to the categorical and numerical features.
    """

    pipelines_cat = {}
    pipelines_num = {}
    # Check if special impute features are provided, if not, initialize as empty dict
    if special_impute_cat_features is None:
        special_impute_cat_features = {}
    if special_impute_num_features is None:
        special_impute_num_features = {}

    # ====================== Categorical Features =================== #
    # Transformation Pipeline for special imputation categorical features
    for feature, value in special_impute_cat_features.items():
        if feature not in categorical_features:
            pipelines_cat[feature] = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='constant', fill_value = value)),
                    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                ]
            )
    # Transformation Pipeline for other categorical features
    other_cat_features = [cat for cat in categorical_features if cat not in special_impute_cat_features.keys()]
    if other_cat_features:
        pipelines_cat['regular_cat'] = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
            ]
        )
    
    # ====================== Numerical Features =================== #
    # Transformation Pipeline for special imputation numerical features
    for num_feature, value in special_impute_num_features.items():
        if num_feature not in numerical_features:
            pipelines_num[num_feature] = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='constant', fill_value=value)),
                    ('std_scaler', StandardScaler())
                ]
            )
    # Transformation Pipeline for other numerical features 
    other_num_features = [num for num in numerical_features if num not in special_impute_num_features.keys()]
    if other_num_features:
        pipelines_num['regular_num'] = Pipeline(
            [
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler())
            ]
        )

    # Create transformers list for ColumnTransformer
    transformers = []

    # Add special imputation features
    for feature in special_impute_cat_features.keys():
        if feature not in categorical_features:
            transformers.append((feature, pipelines_cat[feature], [feature]))
    # Add regular cat features
    if other_cat_features:
        transformers.append(('regular_cat', pipelines_cat['regular_cat'], other_cat_features))

    # Add special imputation num features
    for num_feature in special_impute_num_features.keys():
        if num_feature not in numerical_features:
            transformers.append((num_feature, pipelines_num[num_feature], [num_feature]))
    # Add regular num features
    if other_num_features:
        transformers.append(('regular_num', pipelines_num['regular_num'], other_num_features))

    full_pipeline = ColumnTransformer(transformers, remainder='passthrough')

    return full_pipeline

if __name__ == "__main__":
    # Example usage
    categorical_features = ['cat2', 'cat3']
    numerical_features = ['num2']
    special_impute_cat_features = {'cat1': '1'}
    special_impute_num_features = {'num1': 'mode'}

    pipeline = create_transformer_pipeline(
        categorical_features=categorical_features,
        special_impute_cat_features=special_impute_cat_features,
        numerical_features=numerical_features,
        special_impute_num_features=special_impute_num_features
    )

    print(pipeline)