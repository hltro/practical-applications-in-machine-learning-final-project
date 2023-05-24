from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helper_functions import fetch_dataset, compute_recall, compute_accuracy, apply_threshold, compute_precision
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, accuracy_score
from pages.B_Train_Model import split_dataset
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

random.seed(10)
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <New York Crime Analysis>")

#############################################

st.title('Test Model')

#############################################

map_metrics = {'recall': compute_recall,
               'accuracy': compute_accuracy, 'precision': compute_precision}


def compute_eval_metrics(X, y_true, model, metrics):
    """
    This function computes one or more metrics (precision, recall, accuracy) using the model

    Input:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evaluate performance (string); 'precision', 'recall', 'accuracy'
    Output:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {'precision': -1,
                   'recall': -1,
                   'accuracy': -1}
    y_pred = model.predict(X)

    for metric_name in metrics:
        if metric_name == 'precision':
            # Compute precision using precision_score
            precision = precision_score(y_true, y_pred)
            metric_dict['precision'] = precision
        elif metric_name == 'recall':
            # Compute recall using recall_score
            recall = recall_score(y_true, y_pred)
            metric_dict['recall'] = recall
        elif metric_name == 'accuracy':
            # Compute accuracy using accuracy_score
            accuracy = accuracy_score(y_true, y_pred)
            metric_dict['accuracy'] = accuracy

    return metric_dict


def plot_roc_curve(X_train, X_val, y_train, y_val, trained_models, model_names):
    fig = make_subplots(rows=len(trained_models), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)
    df = pd.DataFrame()
    threshold_values = np.linspace(0.5, 1, num=100)

    for i, model_name in enumerate(model_names):
        model = trained_models[i]

        train_precision_all = []
        train_recall_all = []
        val_precision_all = []
        val_recall_all = []

        for threshold in threshold_values:
            proba_train = model.predict_proba(X_train)
            proba_val = model.predict_proba(X_val)

            pred_train = apply_threshold(proba_train, threshold)
            pred_val = apply_threshold(proba_val, threshold)

            precision_train = precision_score(
                y_train, pred_train, average='weighted', zero_division=1)
            precision_val = precision_score(
                y_val, pred_val, average='weighted', zero_division=1)

            recall_train = recall_score(
                y_train, pred_train, average='weighted', zero_division=1)
            recall_val = recall_score(
                y_val, pred_val, average='weighted', zero_division=1)

            train_precision_all.append(precision_train)
            train_recall_all.append(recall_train)
            val_precision_all.append(precision_val)
            val_recall_all.append(recall_val)

        df[model_name + " Train Precision"] = train_precision_all
        df[model_name + " Train Recall"] = train_recall_all
        df[model_name + " Validation Precision"] = val_precision_all
        df[model_name + " Validation Recall"] = val_recall_all

        fig.add_trace(go.Scatter(x=train_recall_all,
                      y=train_precision_all, name=model_name + "Train"), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=val_recall_all,
                      y=val_precision_all, name=model_name + "Validation"), row=i+1, col=1)
        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text='Precision', row=i+1, col=1)

    return fig, df


def plot_curve_cv(estimator, X, y, cv, model_name):
    fig, ax = plt.subplots()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        estimator.fit(X_train, y_train)
        y_proba = estimator.predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, alpha=0.3, label='Fold %d (AUC = %0.2f)' %
                (i+1, roc_auc))

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, 'b-',
            label='Mean (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('CV Curve - {}'.format(model_name))
    ax.legend(loc="lower right")

    return fig


def plot_learning_curve(estimator, X, y, cv, train_sizes):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation')
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color='r')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color='g')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True)

    return fig


def plot_train_test_curve(estimator, X, y, cv, train_sizes):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation')
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color='r')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color='g')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title(
        'Train/test Split Curve - {}'.format(estimator.__class__.__name__))
    ax.legend(loc='best')
    ax.grid(True)

    return fig


def restore_data_splits(df):
    """
    This function restores the training and validation/test datasets from the training page using st.session_state
                Note: if the datasets do not exist, re-split using the input df

    Input: 
        - df: the pandas dataframe
    Output: 
        - X_train: the training features
        - X_val: the validation/test features
        - y_train: the training targets
        - y_val: the validation/test targets
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    # Restore train/test dataset
    if ('X_train' in st.session_state):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.write('Restored train data ...')
    if ('X_val' in st.session_state):
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        st.write('Restored test data ...')
    if (X_train is None):
        # Select variable to explore
        numeric_columns = list(df.select_dtypes(include='number').columns)
        feature_select = st.selectbox(
            label='Select variable to predict',
            options=numeric_columns,
        )
        X = df.loc[:, ~df.columns.isin([feature_select])]
        Y = df.loc[:, df.columns.isin([feature_select])]

        # Split train/test
        st.markdown(
            '### Enter the percentage of test data to use for training the model')
        number = st.number_input(
            label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

        X_train, X_val, y_train, y_val = split_dataset(
            X, Y, number, feature_select, 'TF-IDF')
        st.write('Restored training and test data ...')
    return X_train, X_val, y_train, y_val

#############################################


df = None
df = fetch_dataset()

if df is not None:
    # Restore dataset splits
    X_train, X_val, y_train, y_val = restore_data_splits(df)

    st.markdown("## Select Models for Evaluation")
    # method_options = ['precision', 'recall', 'accuracy']
    # method_select = st.multiselect(
    #     label='Select a method for evaluation', options=method_options)

    model_options = ['Logistic Regression', 'Stochastic Gradient Descent with Logistic Regression',
                     'Stochastic Gradient Descent with Logistic Regression using GridSearchCV', 'K-Nearest Neighbors',
                     'Random Forest']
    trained_models = [
        model for model in model_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained classification models for evaluation',
        options=trained_models
    )
    if (model_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button(
            'Evaluate your selected classification model(s)')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('## Review Classification Model Performance')

            plot_options = ['Train/Test Split', 'Cross Validation', 'ROC Curve',
                            'Learning Curve', 'Evaluation Metrics Results']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=plot_options
            )

            # Plot ROC curves
            if 'Cross Validation' in review_plot:
                st.markdown("## Cross Validation Performance")
                trained_select = [st.session_state[model]
                                  for model in model_select]

                cv_value = st.number_input(
                    'Enter the number of cross-validation folds', min_value=2, max_value=10, value=5, step=1)

                figs = []

                for i, model_name in enumerate(model_select):
                    estimator = trained_select[i]
                    fig = plot_curve_cv(estimator, X_train, y_train, StratifiedKFold(
                        n_splits=cv_value, shuffle=True, random_state=42), model_name)
                    figs.append(fig)

                for fig in figs:
                    st.pyplot(fig)

            # Plot train/test split curves
            if 'Train/Test Split' in review_plot:
                st.markdown("## Train/Test Split Performance")
                trained_select = [st.session_state[model]
                                  for model in model_select]

                train_sizes = st.multiselect(
                    'Select training set sizes for learning curve',
                    options=[0.1, 0.3, 0.5, 0.7, 1.0],
                    default=[0.1, 0.3, 0.5, 0.7, 1.0]
                )

                figs = []

                for i, model_name in enumerate(model_select):
                    estimator = trained_select[i]
                    fig = plot_train_test_curve(
                        estimator, X_train, y_train, cv=3, train_sizes=train_sizes)
                    figs.append(fig)

                for fig in figs:
                    st.pyplot(fig)

            # Plot ROC curves
            if 'ROC Curve' in review_plot:
                st.markdown("## ROC Curves")
                trained_select = [st.session_state[model]
                                  for model in model_select]
                fig, df = plot_roc_curve(
                    X_train, X_val, y_train, y_val, trained_select, model_select)
                st.plotly_chart(fig)

            # Plot learning curves
            if 'Learning Curve' in review_plot:
                st.markdown("## Learning Curves")
                estimators = ['Logistic Regression',
                              'Random Forest', 'K-Nearest Neighbors']
                cv_values = [3, 5, 10]
                train_sizes = [10, 50, 150]

                estimator_select = st.selectbox(
                    label='Select Estimator',
                    options=estimators
                )
                cv_select = st.selectbox(
                    label='Select Cross-Validation (CV)',
                    options=cv_values
                )
                sizes_select = st.selectbox(
                    label='Select training set sizes',
                    options=train_sizes
                )

                if st.button('Generate Learning Curve'):
                    if estimator_select == 'Logistic Regression':
                        estimator = LogisticRegression()
                    if estimator_select == 'Random Forest':
                        estimator = RandomForestClassifier()
                    if estimator_select == 'K-Nearest Neighbors':  # Handle KNN separately
                        estimator = NearestNeighbors(n_neighbors=5)

                    fig = plot_learning_curve(
                        estimator, X_train, y_train, train_sizes=train_sizes, cv=cv_select
                    )
                    st.pyplot(fig)

            # Compute evaluation metrics
            if 'Evaluation Metrics Results' in review_plot:
                st.markdown("## Evaluation Metrics Results")
                models = [st.session_state[model]
                          for model in model_select]

                train_result_dict = {}
                val_result_dict = {}

                # Select multiple metrics for evaluation
                metric_select = st.multiselect(
                    label='Select metrics for classification model evaluation',
                    options=['precision', 'recall', 'accuracy']
                )
                if (metric_select):
                    st.session_state['metric_select'] = metric_select
                    st.write(
                        'You selected the following metrics: {}'.format(metric_select))

                    for idx, model in enumerate(models):
                        train_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_train, y_train, model, metric_select)
                        val_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_val, y_val, model, metric_select)

                    st.markdown('### Predictions on the training dataset')
                    st.dataframe(train_result_dict)

                    st.markdown('### Predictions on the validation dataset')
                    st.dataframe(val_result_dict)

    # Select a model to deploy from the trained models
    st.markdown("## Choose your Deployment Model")
    model_select = st.selectbox(
        label='Select the model you want to deploy', options=st.session_state['trained_models'])

    if (model_select):
        st.write('You selected the model: {}'.format(model_select))
        st.session_state['deploy_model'] = st.session_state[model_select]

    st.write('Continue to Deploy Model')
