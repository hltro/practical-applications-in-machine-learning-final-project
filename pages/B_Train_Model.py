import numpy as np
import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
from sklearn.model_selection import train_test_split
from pages.A_Explore_Preprocess_Data import remove_nans
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <New York Crime Analysis>")

#############################################

st.title('Train Model')


def split_dataset(X, y, test_size, random_state=45):
    """
    This function splits the dataset into the train data and the test/validation data.

    Input: 
        - X: training features
        - y: training targets
        - test_size: the ratio of test samples
        - random_state: determines random number generation for reproducibility
    Output: 
        - X_train: training features
        - X_val: test/validation features
        - y_train: training targets
        - y_val: test/validation targets
    """
    try:
        # Split data into train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        train_percentage = len(X_train) / (len(X_train) + len(X_val)) * 100
        test_percentage = len(X_val) / (len(X_train) + len(X_val)) * 100

        # Print dataset split result
        st.markdown('The training dataset contains {0:.2f} observations ({1:.2f}%) and the test/validation dataset contains {2:.2f} observations ({3:.2f}%).'.format(
            len(X_train), train_percentage, len(X_val), test_percentage))

        # Save state of train and test splits in st.session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_val'] = X_val
        st.session_state['y_train'] = y_train
        st.session_state['y_val'] = y_val

        return X_train, X_val, y_train, y_val

    except Exception as e:
        print('Exception occurred:', str(e))
        return None, None, None, None


def train_logistic_regression(X_train, y_train, model_name, params, random_state=42):
    """
    This function trains the model with logistic regression and stores it in st.session_state[model_name].

    Input:
        - X_train: training features (review features)
        - y_train: training targets
        - model_name: (string) model name
        - params: a dictionary with lg hyperparameters: max_iter, solver, tol, and penalty
        - random_state: determines random number generation for centroid initialization
    Output:
        - lg_model: the trained model
    """
    lg_model = None

    # Check input shapes
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    try:
        lg_model = LogisticRegression(
            max_iter=params['max_iter'],
            solver=params['solver'],
            tol=params['tol'],
            penalty=params['penalty'],
            random_state=random_state)

        lg_model.fit(X_train, np.ravel(y_train))
        st.session_state[model_name] = lg_model
        return lg_model

    except Exception as e:
        print("Error: ", str(e))


def train_sgd_classifer(X_train, y_train, model_name, params, random_state=42):
    """
    This function trains a classification model with stochastic gradient descent

    Input:
        - X_train: training features
        - y_train: training targets
        - model_name: (string) model name
        - params: a dictionary of the hyperparameters to tune during cross validation
        - random_state: determines random number generation for centroid initialization
    Output:
        - ridge_cv: the trained model
    """
    sgd_model = None
    try:
        sgd_model = SGDClassifier(
            random_state=random_state,
            loss=params['loss'],
            penalty=params['penalty'],
            alpha=params['alpha'],
            max_iter=params['max_iter'],
            tol=params['tol'])
        sgd_model.fit(X_train, np.ravel(y_train))
        st.session_state[model_name] = sgd_model
        return sgd_model

    except Exception as e:
        print("Error: ", str(e))


def train_sgdcv_classifer(X_train, y_train, model_name, params, cv_params, random_state=42):
    """
    This function trains a classification model with stochastic gradient descent and cross validation

    Input:
        - X_train: training features
        - y_train: training targets
        - model_name: (string) model name
        - params: a dictionary of the SGD hyperparameters
        - cv_params: a dictionary of the hyperparameters to tune during cross validation
        - random_state: determines random number generation for centroid initialization
    Output:
        - sgdcv_model: the trained model
    """
    sgdcv_model = None
    # Add code here
    try:
        sgdcv_model = GridSearchCV(estimator=SGDClassifier(
            random_state=random_state), param_grid=params, cv=cv_params['n_splits'])
        sgdcv_model.fit(X_train, np.ravel(y_train))
        st.session_state['cv_results'] = sgdcv_model.cv_results_
        st.session_state[model_name] = sgdcv_model.best_estimator_

        return sgdcv_model.best_estimator_

    except Exception as e:
        print("Error: ", str(e))


def train_knn(X_train, y_train, model_name, params):
    """
    This function trains the model with K-Nearest Neighbors (KNN) and stores it in st.session_state[model_name].

    Input:
        - X_train: training features
        - y_train: training targets
        - model_name: (string) model name
        - params: a dictionary with KNN hyperparameters: n_neighbors, weights, and algorithm
    Output:
        - knn_model: the trained KNN model
    """
    knn_model = None
    try:
        knn_model = KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            algorithm=params['algorithm']
        )
        knn_model.fit(X_train, y_train)
        st.session_state[model_name] = knn_model
        return knn_model

    except Exception as e:
        print("Error: ", str(e))


def train_random_forest(X_train, y_train, model_name, params):
    """
    This function trains the model with Random Forest Classifier and stores it in st.session_state[model_name].

    Input:
        - X_train: training features
        - y_train: training targets
        - model_name: (string) model name
        - params: a dictionary with Random Forest hyperparameters: n_estimators, max_depth, etc.
    Output:
        - rf_model: the trained Random Forest model
    """
    rf_model = None
    try:
        rf_model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=params['random_state']
        )
        rf_model.fit(X_train, y_train)
        st.session_state[model_name] = rf_model
        return rf_model

    except Exception as e:
        print("Error: ", str(e))


#############################################
df = None
df = fetch_dataset()

if df is not None:
    # Display dataframe as table
    st.dataframe(df)

    # Select variable to predict
    st.markdown('### Select variable to predict')

    feature_predict_select = st.multiselect(
        label='Select variables to predict',
        options=list(df.select_dtypes(include='number').columns),
        default=list(df.select_dtypes(include='number').columns)[
            :2],  # set a default selection
        key='feature_predict_multiselect',
    )

    # Select input features
    st.markdown('### Select input features')

    feature_input_select = st.multiselect(
        label='Select features for classification input',
        options=[f for f in list(df.select_dtypes(
            include='number').columns) if f != feature_predict_select],
        default=list(df.select_dtypes(include='number').columns)[
            :2],  # set a default selection
        key='feature_input_multiselect',
    )

    st.write('You selected input {} and output {}'.format(
        feature_input_select, feature_predict_select))

    df = remove_nans(df)

    # Convert feature_input_select and feature_predict_select to lists if they are not already
    if not isinstance(feature_input_select, list):
        feature_input_select = [feature_input_select]
    if not isinstance(feature_predict_select, list):
        feature_predict_select = [feature_predict_select]

    # Select the input and output features from the DataFrame
    X = df.loc[:, df.columns.isin(feature_input_select)]
    Y = df.loc[:, df.columns.isin(feature_predict_select)]

    # Split dataset
    st.markdown('### Split dataset into Train/Validation/Test sets')
    st.markdown(
        '#### Enter the percentage of validation/test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    X_train, X_val, y_train, y_val = split_dataset(X, Y, number)

    # Train models
    st.markdown('### Train models')
    classification_methods_options = ['Logistic Regression', 'Stochastic Gradient Descent with Logistic Regression',
                                      'Stochastic Gradient Descent with Logistic Regression using GridSearchCV', 'K-Nearest Neighbors',
                                      'Random Forest']

    trained_models = [
        model for model in classification_methods_options if model in st.session_state]

    # Collect ML Models of interests
    classification_model_select = st.multiselect(
        label='Select regression model for prediction',
        options=classification_methods_options,
    )
    st.write('You selected the follow models: {}'.format(
        classification_model_select))

    # Select parameters for logistic regression
    if (classification_methods_options[0] in classification_model_select or classification_methods_options[0] in trained_models):
        st.markdown('#### ' + classification_methods_options[0])

        lg_col1, lg_col2 = st.columns(2)

        with (lg_col1):
            # solver: algorithm to use in the optimization problem
            solvers = ['liblinear', 'lbfgs', 'newton-cg',
                       'newton-cholesky', 'sag', 'saga']
            lg_solvers = st.selectbox(
                label='Select solvers for logistic regression',
                options=solvers,
                key='lg_reg_solver_multiselect'
            )
            st.write('You select the following solver(s): {}'.format(lg_solvers))

            # penalty: 'l1' or 'l2' regularization
            lg_penalty_select = st.selectbox(
                label='Select penalty for SGD',
                options=['l1', 'l2'],
                key='lg_penalty_multiselect'
            )
            st.write('You select the following penalty: {}'.format(
                lg_penalty_select))

        with (lg_col2):
            # tolerance: stopping criteria for iterations
            lg_tol = st.text_input(
                label='Input a tolerance value',
                value='0.0001',
                key='lg_tol_textinput'
            )
            lg_tol = float(lg_tol)
            st.write('You select the following tolerance value: {}'.format(lg_tol))

            # max_iter: maximum iterations to run the LG until convergence
            lg_max_iter = st.number_input(
                label='Enter the number of maximum iterations on training data',
                min_value=1000,
                max_value=5000,
                value=1000,
                step=100,
                key='lg_max_iter_numberinput'
            )
            st.write('You set the maximum iterations to: {}'.format(lg_max_iter))

        lg_params = {
            'max_iter': lg_max_iter,
            'penalty': lg_penalty_select,
            'tol': lg_tol,
            'solver': lg_solvers,
        }
        if st.button('Logistic Regression Model'):
            train_logistic_regression(
                X_train, y_train, classification_methods_options[0], lg_params)

        if classification_methods_options[0] not in st.session_state:
            st.write('Logistic Regression Model is untrained')
        else:
            st.write('Logistic Regression Model trained')

    # Select parameters for Stochastic Gradient Descent with Logistic Regression
    if (classification_methods_options[1] in classification_model_select or classification_methods_options[2] in trained_models):
        st.markdown('#### ' + classification_methods_options[1])

        # Loss: 'log' is logistic regression, 'hinge' for Support Vector Machine
        sdg_loss_select = 'log_loss'

        sgd_col1, sgd_col2 = st.columns(2)

        with (sgd_col1):
            # max_iter: maximum iterations to run the iterative SGD
            sdg_max_iter = st.number_input(
                label='Enter the number of maximum iterations on training data',
                min_value=1000,
                max_value=5000,
                value=1000,
                step=100,
                key='sgd_max_iter_numberinput'
            )
            st.write('You set the maximum iterations to: {}'.format(sdg_max_iter))

            # penalty: 'l1' or 'l2' regularization
            sdg_penalty_select = st.selectbox(
                label='Select penalty for SGD',
                options=['l2', 'l1'],
                key='sdg_penalty_multiselect'
            )
            st.write('You select the following penalty: {}'.format(
                sdg_penalty_select))

        with (sgd_col2):
            # alpha=0.001: Constant that multiplies the regularization term. Ranges from [0 Inf)
            sdg_alpha = st.text_input(
                label='Input one alpha value',
                value='0.001',
                key='sdg_alpha_numberinput'
            )
            sdg_alpha = float(sdg_alpha)
            st.write('You select the following alpha value: {}'.format(sdg_alpha))

            # tolerance: stopping criteria for iterations
            sgd_tol = st.text_input(
                label='Input a tolerance value',
                value='0.01',
                key='sgd_tol_textinput'
            )
            sgd_tol = float(sgd_tol)
            st.write('You select the following tolerance value: {}'.format(sgd_tol))

        sgd_params = {
            'loss': sdg_loss_select,
            'max_iter': sdg_max_iter,
            'penalty': sdg_penalty_select,
            'tol': sgd_tol,
            'alpha': sdg_alpha,
        }

        if st.button('Train Stochastic Gradient Descent Model'):
            train_sgd_classifer(
                X_train, y_train, classification_methods_options[1], sgd_params)

        if classification_methods_options[1] not in st.session_state:
            st.write('Stochastic Gradient Descent Model is untrained')
        else:
            st.write('Stochastic Gradient Descent Model trained')

    # Select parameters for Stochastic Gradient Descent with Logistic Regression using Cross Validation
    if (classification_methods_options[2] in classification_model_select or classification_methods_options[2] in trained_models):
        st.markdown('#### ' + classification_methods_options[2])

        # Loss: "squared_error": Ordinary least squares, huber": Huber loss for robust regression, "epsilon_insensitive": linear Support Vector Regression.
        sdgcv_loss_select = 'log_loss'

        sdg_col, sdgcv_col = st.columns(2)

        # Collect Parameters
        with (sdg_col):
            # max_iter: maximum iterations to run the iterative SGD
            sdgcv_max_iter = st.number_input(
                label='Enter the number of maximum iterations on training data',
                min_value=1000,
                max_value=5000,
                value=1000,
                step=100,
                key='sgdcv_max_iter_numberinput'
            )
            st.write('You set the maximum iterations to: {}'.format(sdgcv_max_iter))

            # penalty: 'l1' or 'l2' regularization
            sdgcv_penalty_select = st.selectbox(
                label='Select penalty for SGD',
                options=['l2', 'l1'],
                # default='l1',
                key='sdgcv_penalty_select'
            )
            st.write('You select the following penalty: {}'.format(
                sdgcv_penalty_select))

            # tolerance: stopping criteria for iterations
            sgdcv_tol = st.text_input(
                label='Input a tolerance value',
                value='0.01',
                key='sdgcv_tol_numberinput'
            )
            sgdcv_tol = float(sgdcv_tol)
            st.write(
                'You select the following tolerance value: {}'.format(sgdcv_tol))

        # Collect Parameters
        with (sdgcv_col):
            # alpha=0.01: Constant that multiplies the regularization term. Ranges from [0 Inf)
            sdgcv_alphas = st.text_input(
                label='Input alpha values, separate by comma',
                value='0.001,0.0001',
                key='sdgcv_alphas_textinput'
            )
            sdgcv_alphas = [float(val) for val in sdgcv_alphas.split(',')]
            st.write(
                'You select the following alpha value: {}'.format(sdgcv_alphas))

            sgdcv_params = {
                'loss': [sdgcv_loss_select],
                'max_iter': [sdgcv_max_iter],
                'penalty': [sdgcv_penalty_select],
                'tol': [sgdcv_tol],
                'alpha': sdgcv_alphas,
            }

            st.markdown('Select SGD Cross Validation Parameters')
            # n_splits: number of folds
            sgdcv_cv_n_splits = st.number_input(
                label='Enter the number of folds',
                min_value=2,
                max_value=len(df),
                value=3,
                step=1,
                key='sdgcv_cv_nsplits'
            )
            st.write('You select the following split value(s): {}'.format(
                sgdcv_cv_n_splits))

            sgdcv_cv_params = {
                'n_splits': sgdcv_cv_n_splits,
            }

        if st.button('Train Stochastic Gradient Descent Model with Cross Validation'):
            train_sgdcv_classifer(
                X_train, y_train, classification_methods_options[2], sgdcv_params, sgdcv_cv_params)

        if classification_methods_options[2] not in st.session_state:
            st.write(
                'Stochastic Gradient Descent Model with Cross Validation is untrained')
        else:
            st.write(
                'Stochastic Gradient Descent Model with Cross Validation trained')

# Select parameters for K-Nearest Neighbors
if (classification_methods_options[3] in classification_model_select or classification_methods_options[3] in trained_models):
    st.markdown('#### K-Nearest Neighbors')

    knn_col, knncv_col = st.columns(2)

    # Collect Parameters
    with (knn_col):
        # n_neighbors: Number of neighbors to consider
        knn_n_neighbors = st.number_input(
            label='Enter the number of neighbors',
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key='knn_n_neighbors_numberinput',
            format='%d'
        )
        st.write('You set the number of neighbors to: {}'.format(knn_n_neighbors))

        # weights: Weight function used in prediction
        knn_weights = st.selectbox(
            label='Select weight function',
            options=['uniform', 'distance'],
            key='knn_weights_select'
        )
        st.write('You select the following weight function: {}'.format(knn_weights))

        # algorithm: Algorithm used to compute nearest neighbors
        knn_algorithm = st.selectbox(
            label='Select algorithm',
            options=['auto', 'ball_tree', 'kd_tree', 'brute'],
            key='knn_algorithm_select'
        )
        st.write('You select the following algorithm: {}'.format(knn_algorithm))

    if st.button('Train K-Nearest Neighbors Model'):
        train_knn(X_train, y_train, 'K-Nearest Neighbors',
                  {'n_neighbors': knn_n_neighbors, 'weights': knn_weights, 'algorithm': knn_algorithm})

    if 'K-Nearest Neighbors' not in st.session_state:
        st.write('K-Nearest Neighbors Model is untrained')
    else:
        st.write('K-Nearest Neighbors Model trained')


# Select parameters for Random Forest
if (classification_methods_options[4] in classification_model_select or classification_methods_options[4] in trained_models):
    st.markdown('#### Random Forest')

    rf_col, rfcv_col = st.columns(2)

    # Collect Parameters
    with (rf_col):
        # n_estimators: Number of trees in the forest
        rf_n_estimators = st.number_input(
            label='Enter the number of estimators',
            min_value=10,
            max_value=100,
            value=100,
            step=10,
            key='rf_n_estimators_numberinput',
            format='%d'
        )
        st.write('You set the number of estimators to: {}'.format(rf_n_estimators))

        # max_depth: Maximum depth of each decision tree
        rf_max_depth = st.number_input(
            label='Enter the maximum depth',
            min_value=1,
            max_value=10,
            value=1,  # Set a default value for no maximum depth
            step=1,
            key='rf_max_depth_numberinput',
            format='%d'
        )
        st.write('You set the maximum depth to: {}'.format(rf_max_depth))

    if st.button('Train Random Forest Model'):
        y_train = np.ravel(y_train)  # Reshape y_train into 1D array
        train_random_forest(X_train, y_train, 'Random Forest', {
                            'n_estimators': rf_n_estimators, 'max_depth': rf_max_depth, 'random_state': None})

    if 'Random Forest' not in st.session_state:
        st.write('Random Forest Model is untrained')
    else:
        st.write('Random Forest Model trained')

    st.write('#### Continue to Test Model')
