import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from helpers import get_variable_list


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = get_variable_list(variables)


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        
        X = X.copy()
        
        for var in self.variables:
            X[var+'_na'] = np.where(X[var].isnull(), 1, 0)
        
        return X


# Categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = get_variable_list(variables)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X = X.copy()
        
        for var in self.variables:
            X[var] = X[var].fillna('Missing')

        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = get_variable_list(variables)

    def fit(self, X, y=None):
        
        self.imputer_dict_ = {}
        
        for feature in self.variables:
            
            # We use could use mode to avoid outlier influence
            # Outliers are neglectable for this dataset

            self.imputer_dict_[feature] = X[feature].mean()
        
        return self

    def transform(self, X):

        X = X.copy()
        
        for feature in self.variables:
            
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)

        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = get_variable_list(variables)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X = X.copy()
        
        for var in self.variables:
            X[var] = X[var].str[0]
        
        return X

# Frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.percentage = tol
        self.variables = get_variable_list(variables)

    def fit(self, X, y=None):

        self.encoder_dict_ = {}
        
        for var in self.variables:
            
            variable_frequencies = pd.Series(X[var].value_counts() / np.float(len(X)))
            
            self.encoder_dict_[var] = list(variable_frequencies[variable_frequencies >= self.percentage].index)

        return self


    def transform(self, X):

        X = X.copy()

        for var in self.variables:
            X[var] = np.where(X[var].isin(self.encoder_dict_[var]), X[var], 'Rare')

        return X



# String to numbers categorical encoder
# TODO: Review this function, make sure it's clear
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = get_variable_list(variables)

    def fit(self, X, y=None):

        # Compute dummy subtable
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns # drop_first -> k-1 dummies

        return self

    def transform(self, X):
        
        X = X.copy()
        
        # Add dummies
        X = pd.concat([X, pd.get_dummies(X[self.variables], drop_first=True)], axis=1)
        
        # Drop real labels
        X.drop(labels=self.variables, axis=1, inplace=True)

        # Check and add missing dummies
        for var in self.dummies:
            if var not in X.columns: X[var] = 0
        
        return X