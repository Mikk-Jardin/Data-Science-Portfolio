import sys
import os
import glob

# Import data processing functions
from data.unzip import unzip
from data.preprocess import preprocess_data

# Import model building functions
from models.model import build_model, evaluate_model, save_model

### Set random seed variables for better reproducibility
# Set a seed value
seed_value= 42
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

def main():
    if len(sys.argv) == 3:
        data_filepath, model_filepath = sys.argv[1:]

        # Preprocess data
        print(f"Preprocessing data from {data_filepath}...")
        train_dataset, test_dataset = preprocess_data(data_filepath)

        # Build and train model
        print("Building and training model...")
        clf = build_model(train_dataset)

        # Evaluate model
        evaluate_model(clf, test_dataset)

        # Save model
        print(f"Saving model to {model_filepath}")
        save_model(clf, model_filepath)

        print("Model trained and saved.")

        print("Finance Complaint Classifier built.")

    else:
        print("Please provide filepath of the csv file of data"\
              "to be used for training as the first argument"\
              "and the filepath to the folder where the model"\
              "will be saved as the second argument."\
              "Example:\n python train_model.py ./data/complaints_30k.zip"\
              "./models")
        
if __name__ == '__main__':
    main()