import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

def preprocess_data(data):
    """
    Takes a csv file and prepares data for training on model.

    Parameters
    ----------
    data (csv): csv file of data to be used for training.

    Returns
    -------
    Batched and prefetched train and test datasets.
    """
    # Load data as pandas dataframe
    df = pd.read_csv(data)

    # Split data into input (X) and ouput (y) data
    X = df['consumer_complaint_narrative']
    y = df['product']

    # Create train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Convert inputs (X) to lists
    X_train_list = X_train.tolist()
    X_test_list = X_test.tolist()

    # OneHot encode output (y) data
    encoder = OneHotEncoder(sparse_output=False) # initialize encoder
    y_train_oh = encoder.fit_transform(y_train.to_numpy().reshape(-1,1))
    y_test_oh = encoder.transform(y_test.to_numpy().reshape(-1,1))

    # Turn input and output data into tensorslice datasets
    train_dataset_ts = tf.data.Dataset.from_tensor_slices((X_train_list, y_train_oh))
    test_dataset_ts = tf.data.Dataset.from_tensor_slices((X_test_list, y_test_oh))

    # Turn tensorslice dataset into prefetched data for faster processing
    train_dataset_pf = train_dataset_ts.batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset_pf = test_dataset_ts.batch(32).prefetch(tf.data.AUTOTUNE)

    return train_dataset_pf, test_dataset_pf