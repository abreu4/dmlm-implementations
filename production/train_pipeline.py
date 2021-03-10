import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from pipeline import titanic_pipe
import config

# TODO: Find out where along the pipeline is optimal to store the seed
seed = 0

def run_training():

    # read training data
    data = pd.read_csv(config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.1, # solution runs 0.2
        random_state=seed)

    # fit pipeline
    titanic_pipe.fit(X_train, y_train)
    
    # save pipeline
    joblib.dump(titanic_pipe, config.PIPELINE_NAME)

if __name__ == '__main__':
    run_training()