from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
from config import *


titanic_pipe = Pipeline(

   	[
         ('categorical_imputer', pp.CategoricalImputer(variables=CATEGORICAL_VARS)),
   		('missing_indicator', pp.MissingIndicator(variables=NUMERICAL_VARS)),
   		('numerical_imputer', pp.NumericalImputer(variables=NUMERICAL_VARS)),
   		('extract_first_letter', pp.ExtractFirstLetter(variables=CABIN_VAR)),
   		('rare_label_encoding', pp.RareLabelCategoricalEncoder(tol=0.05, variables=CATEGORICAL_VARS)),
   		('categorical_encoding', pp.CategoricalEncoder(variables=CATEGORICAL_VARS)),
   		('scaler', StandardScaler()),
        ('model', LogisticRegression(C=0.0005, random_state=0))
   	]
  
   )