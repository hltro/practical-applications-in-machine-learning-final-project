import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score

# All pages

# Page C:


def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    df = None
    data = None
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        data = st.file_uploader(
            'Upload a Dataset', type=['csv', 'txt'])

        if (data):
            df = pd.read_csv(data)
    if df is not None:
        st.session_state['data'] = df
    return df


def compute_recall(y_true, y_pred):
    recall = -1
    recall = recall_score(y_true, y_pred)
    return recall


def compute_precision(y_true, y_pred):
    precision = -1
    precision = precision_score(y_true, y_pred)
    return precision


def compute_accuracy(y_true, y_pred):
    accuracy = -1
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def apply_threshold(probabilities, threshold):
    return np.array([1 if p[1] >= threshold else -1 for p in probabilities])
