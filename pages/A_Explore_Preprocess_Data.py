import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
import plotly.express as px
from itertools import combinations
from pandas.plotting import scatter_matrix
import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
from sklearn.preprocessing import OrdinalEncoder


feature_lookup = {
    'CMPLNT_NUM': '**CMPLNT_NUM** - Randomly generated persistent ID for each complaint',
    'CMPLNT_FR_DT': '**CMPLNT_FR_DT** - Exact date of occurrence for the reported event (or starting date of occurrence, if CMPLNT_TO_DT exists)',
    'CMPLNT_FR_TM': '**CMPLNT_FR_TM** - Exact time of occurrence for the reported event (or starting time of occurrence, if CMPLNT_TO_TM exists)',
    'CMPLNT_TO_DT': '**CMPLNT_TO_DT** - Ending date of occurrence for the reported event, if exact time of occurrence is unknown',
    'CMPLNT_TO_TM': '**CMPLNT_TO_TM** - Ending time of occurrence for the reported event, if exact time of occurrence is unknown',
    'ADDR_PCT_CD': '**ADDR_PCT_CD** - The precinct in which the incident occurred',
    'RPT_DT': '**RPT_DT** - Date event was reported to police',
    'KY_CD': '**KY_CD** - Three digit offense classification code',
    'OFNS_DESC': '**OFNS_DESC** - Description of offense corresponding with key code',
    'PD_CD': '**PD_CD** - Three digit internal classification code (more granular than Key Code)',
    'PD_DESC': '**PD_DESC** - Description of internal classification corresponding with PD code (more granular than Offense Description)',
    'CRM_ATPT_CPTD_CD': '**CRM_ATPT_CPTD_CD** - Indicator of whether crime was successfully completed or attempted, but failed or was interrupted prematurely',
    'LAW_CAT_CD': '**LAW_CAT_CD** - Level of offense: felony, misdemeanor, violation',
    'BORO_NM': '**BORO_NM** - The name of the borough in which the incident occurred',
    'LOC_OF_OCCUR_DESC': '**LOC_OF_OCCUR_DESC** - Specific location of occurrence in or around the premises; inside, opposite of, front of, rear of',
    'PREM_TYP_DESC': '**PREM_TYP_DESC** - Specific description of premises; grocery store, residence, street, etc.',
    'JURIS_DESC': '**JURIS_DESC** - Description of the jurisdiction code',
    'JURISDICTION_CODE': '**JURISDICTION_CODE** - Jurisdiction responsible for incident. Either internal, like Police(0), Transit(1), and Housing(2); or external(3), like Correction, Port Authority, etc.',
    'PARKS_NM': '**PARKS_NM** - Name of NYC park, playground or greenspace of occurrence, if applicable (state parks are not included)',
    'HADEVELOPT': '**HADEVELOPT** - Name of NYCHA housing development of occurrence, if applicable',
    'HOUSING_PSA': '**HOUSING_PSA** - Development Level Code',
    'X_COORD_CD': '**X_COORD_CD** - X-coordinate for New York State Plane Coordinate System, Long Island Zone, NAD 83, units feet (FIPS 3104)',
    'Y_COORD_CD': '**Y_COORD_CD** - Y-coordinate for New York State Plane Coordinate System, Long Island Zone, NAD 83, units feet (FIPS 3104)',
    'SUSP_AGE_GROUP': '**SUSP_AGE_GROUP** - Suspect’s Age Group',
    'SUSP_RACE': '**SUSP_RACE** - Suspect’s Race Description',
    'SUSP_SEX': "**SUSP_SEX** - Suspect’s Sex Description",
    'TRANSIT_DISTRICT': "**TRANSIT_DISTRICT** - Transit district in which the offense occurred.",
    'Latitude': "**Longitude** - Midblock Latitude coordinate for Global Coordinate System, WGS 1984, decimal degrees (EPSG 4326)",
    'Longitude': "**Longitude** - Midblock Longitude coordinate for Global Coordinate System, WGS 1984, decimal degrees (EPSG 4326)",
    'Lat_Lon': "**Lat_Lon** - Geospatial Location Point (latitude and Longitude combined)",
    'PATROL_BORO': "**PATROL_BORO** - The name of the patrol borough in which the incident occurred",
    'STATION_NAME': "**STATION_NAME** - Transit station name",
    'VIC_AGE_GROUP': "**VIC_AGE_GROUP** - Victim’s Age Group",
    'VIC_RACE': "**VIC_RACE** - Victim’s Race Description",
    'VIC_SEX': "**VIC_SEX** - Victim’s Sex Description"
}

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <New York Crime Analysis>")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################


# Helper Function
def display_features(df, feature_lookup):
    """
    This function displayes feature names and descriptions (from feature_lookup).

    Inputs:
    df (pandas.DataFrame): The input DataFrame to be whose features to be displayed.
    feature_lookup (dict): A dictionary containing the descriptions for the features.
    """
    for idx, col in enumerate(df.columns):
        for f in feature_lookup:
            if f in df.columns:
                st.markdown('Feature %d - %s' % (idx, feature_lookup[col]))
                break
            else:
                st.markdown('Feature %d - %s' % (idx, col))
                break


def user_input_features(df):
    """
    Input: pnadas dataframe containing dataset
    Output: dictionary of sidebar filters on features
    """
    numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
    side_bar_data = {}
    for feature in numeric_columns:
        try:
            f = st.sidebar.slider(str(feature), float(df[str(feature)].min()), float(
                df[str(feature)].max()), float(df[str(feature)].mean()))
        except Exception as e:
            print(e)
        side_bar_data[feature] = f
    return side_bar_data


def compute_correlation(X, features):
    """
    This function computes pair-wise correlation coefficents of X and render summary strings 
        with description of magnitude and direction of correlation

    Input: 
        - X: pandas dataframe 
        - features: a list of feature name (string), e.g. ['age','height']
    Output: 
        - correlation: correlation coefficients between one or more features
        - summary statements: a list of summary strings where each of it is in the format: 
            '- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}'
    """
    correlation = None
    cor_summary_statements = []

    correlation = X[features].corr()
    feature_pairs = combinations(features, 2)

    for f1, f2 in feature_pairs:
        cor = correlation[f1][f2]
        magnitude = ""
        direction = ""
        if (abs(cor) > 0.5):
            magnitude = 'strongly'
        else:
            magnitude = 'weakly'

        if (cor < 0):
            direction = 'negatively'
        else:
            direction = 'positively'

        summary = "- Features {} and {} are {} {} correlated: {}".format(
            f1, f2, magnitude, direction, cor.round(2))
        cor_summary_statements.append(summary)

    return correlation, cor_summary_statements


def summarize_missing_data(df, top_n=3):
    """
    Input: 
        - df: the pandas dataframe
        - top_n: top n features with missing values, default value is 3
    Output: 
        - a dictionary containing the following keys and values: 
            - 'num_categories': counts the number of features that have missing values
            - 'average_per_category': counts the average number of missing values across features
            - 'total_missing_values': counts the total number of missing values in the dataframe
            - 'top_missing_categories': lists the top n features with missing values
    """
    out_dict = {'num_categories': 0,
                'average_per_category': 0,
                'total_missing_values': 0,
                'top_missing_categories': []}

    # Used for top categories with missing data
    missing_column_counts = df[df.columns[df.isnull().any()]].isnull().sum()
    max_idxs = np.argsort(missing_column_counts.to_numpy())[::-1][:top_n]

    # Compute missing statistics
    num_categories = df.isna().any(axis=0).sum()
    average_per_category = df.isna().sum().sum()/len(df.columns)
    total_missing_values = df.isna().sum().sum()
    top_missing_categories = df.columns[max_idxs[:top_n]].to_numpy()

    out_dict['num_categories'] = num_categories
    out_dict['average_per_category'] = average_per_category
    out_dict['total_missing_values'] = total_missing_values
    out_dict['top_missing_categories'] = top_missing_categories

    # Display missing statistics
    # st.write(out_dict)
    return out_dict


def impute_dataset(X, impute_method):
    """
    This function imputes the NaN in the dataframe with three possible ways

    Input: 
        - X: the pandas dataframe
        - impute_method: the method to impute the NaN in the dataframe, options are -
            - 'Zero': to replace NaN with zero
            - 'Mean': to replace NaN with the mean of the corressponding feature column
            - 'Median': to replace NaN with the median of the corressponding feature column
    Output: 
        - X: the updated dataframe
    """
    numeric_columns = list(X.select_dtypes(['float', 'int']).columns)
    if impute_method == 'Zero':
        for feature in numeric_columns:
            X[feature].fillna(value=0, inplace=True)
    elif impute_method == 'Mean':
        for feature in numeric_columns:
            X[feature].fillna(value=X[feature].mean(), inplace=True)
    elif impute_method == 'Median':
        for feature in numeric_columns:
            X[feature].fillna(value=X[feature].median(), inplace=True)
    return X


def remove_nans(df):
    df.dropna(inplace=True)
    return df


def replace_unknown_nans(df):
    """
    This function replaces all 'UNKNOWN' values in the dataframe with NaN.

    Input:
        - df: pandas dataframe

    Output:
        - df: updated dataframe with 'UNKNOWN' replaced by NaN
    """
    df.replace('UNKNOWN', np.nan, inplace=True)
    return df


def one_hot_encode_features(df, features):
    """
    This function performs one-hot encoding on the given features using pd.get_dummies

    Input:
        - df: the pandas dataframe
        - features: list of features to perform one-hot encoding
    Output:
        - df: dataframe with one-hot encoded features
    """

    df = pd.get_dummies(df, columns=features)  # one-hot encoding
    st.write('Features {} have been one-hot encoded.'.format(features))
    return df


def integer_encode_features(df, features):
    """
    This function performs integer-encoding on the given features using OrdinalEncoder()

    Input: 
        - df: the pandas dataframe
        - features: the list of features to perform integer-encoding
    Output: 
        - df: dataframe with integer-encoded features
    """

    enc = OrdinalEncoder()
    df[features] = enc.fit_transform(df[features])
    st.write('Features {} have been integer encoded.'.format(features))
    return df


def remove_features(df, columns_to_remove):
    """
    This function removes specified columns from the dataframe.

    Input:
        - df: pandas dataframe
        - columns_to_remove: list of column names to be removed

    Output:
        - df: updated dataframe with specified columns removed
    """
    df.drop(columns=columns_to_remove, inplace=True)
    return df

# Rare categories in the dataset need to be dealt with to avoid shapes issue on Page B


def remove_rare_categories(df, column, threshold):
    value_counts = df[column].value_counts()
    rare_categories = value_counts[value_counts < threshold].index
    df = df[~df[column].isin(rare_categories)]
    return df


def get_value_counts(df, selected_columns):
    value_counts = {}
    for column in selected_columns:
        value_counts[column] = df[column].value_counts()
    return value_counts


#############################################
df = None
df = fetch_dataset()

if df is not None:
    # Display original dataframe
    st.success('You have uploaded the dataset.', icon="✅")
    st.info('View initial data with missing values or invalid inputs', icon="ℹ️")

    st.dataframe(df)
    with st.expander("See explanation"):
        display_features(df, feature_lookup)

    # Inspect the dataset
    st.markdown("""---""")
    st.markdown('### Data Visualization')
    # Specify Input Parameters

    # add a map visualization of data
    df_map = df[['Latitude', 'Longitude']].dropna()
    df_map.rename(columns={"Latitude": "latitude",
                  "Longitude": "longitude"}, inplace=True)
    df_map['latitude'].astype('str').astype('float')
    df_map['longitude'].astype('str').astype('float')
    st.map(df_map)

    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    chart_select = st.sidebar.selectbox(
        label='type',
        options=['Bar Chart', 'Scatterplots',
                 'Histogram', 'Lineplots', 'Boxplot']
    )
    st.write(chart_select)

    numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
    X = df

    # Draw plots including Bar Chart, Scatterplots, Histogram, Lineplots, Boxplot
    if (chart_select == 'Scatterplots'):
        try:
            xvalues = st.sidebar.selectbox('X', options=numeric_columns)
            yvalues = st.sidebar.selectbox('Y', options=numeric_columns)
            side_bar_data = user_input_features(df)
            plot = px.scatter(x=df[xvalues], y=df[yvalues])
            st.write(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Histogram':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = user_input_features(df)
            plot = px.histogram(data_frame=df, x=x_values, range_x=[
                                X[x_values].min(), side_bar_data[x_values]])
            st.write(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Lineplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = user_input_features(df)
            plot = px.line(df, x=x_values, y=y_values, range_x=[X[x_values].min(
            ), side_bar_data[x_values]], range_y=[X[y_values].min(), side_bar_data[y_values]])
            st.write(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Boxplot':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = user_input_features(df)
            plot = px.box(df, x=x_values, y=y_values, range_x=[X[x_values].min(
            ), side_bar_data[x_values]], range_y=[X[y_values].min(), side_bar_data[y_values]])
            st.write(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Bar Chart':
        try:
            categorical_column = st.sidebar.selectbox(
                'Categorical Column', options=list(df.select_dtypes('object').columns))
            side_bar_data = user_input_features(df)
            plot = px.bar(df, x=df[categorical_column].value_counts(
            ).index, y=df[categorical_column].value_counts())
            st.write(plot)
        except Exception as e:
            print(e)

    st.markdown("""---""")
    st.markdown("### Looking for Correlations")

    # Collect features for correlation analysis using multiselect
    select_features = st.multiselect(
        'select some features',
        options=numeric_columns
    )

    correlation = compute_correlation(df, select_features)

    if select_features:
        fig = scatter_matrix(df[select_features], figsize=(12, 8))
        st.pyplot(fig[0][0].get_figure())

    with st.expander("See explanation"):
        st.write(correlation)

    st.markdown("""---""")
    st.markdown('### Handle Missing Values')

    # Show summary of missing values including
    missing_data_summary = summarize_missing_data(df)
    col1, col2, col3 = st.columns(3)

    col1.metric("Total # of features with missing values",
                missing_data_summary['num_categories'])
    col2.metric("Total # of missing values",
                missing_data_summary['total_missing_values'])
    col3.metric("Average # of missing values across features",
                missing_data_summary['average_per_category'])

    # Replace all 'UNKNOWN' values with NaN
    st.markdown("""---""")
    st.markdown('### Replace all UNKNOWN values with NaN')
    if st.button('Replace UNKNOWNs'):
        df = replace_unknown_nans(df)
        st.success('UNKNOWNs replaced', icon="✅")
    else:
        st.info('Dataset might contain UNKNOWN', icon="ℹ️")

    # Handle NaNs
    remove_nan_col, impute_col = st.columns(2)

    with (remove_nan_col):
        # Remove Nans
        st.markdown('### Remove NaNs')
        if st.button('Remove NaNs'):
            df = remove_nans(df)
            st.success('Nans removed', icon="✅")
        else:
            st.info('Dataset might contain Nans', icon="ℹ️")

    with (impute_col):
        # Clean dataset
        st.markdown('### Impute Data')
        st.markdown('Transform Missing Values to 0, Mean, or Median')

        # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
        impute_method = st.selectbox(
            'Select cleaning method',
            ('None', 'Zero', 'Mean', 'Median')
        )

        # Call impute_dataset function to resolve data handling/cleaning problems
        if (impute_method):
            df = impute_dataset(df, impute_method)

    # Value counts for selected columns
    st.markdown('### Value Counts for Selected Columns')
    # Get a list of column names
    columns = df.columns.tolist()

    # Select columns for value counts
    selected_columns = st.multiselect('Select Columns', columns)

    # Perform value counts
    if selected_columns:
        value_counts = get_value_counts(df, selected_columns)
        for column, counts in value_counts.items():
            st.write(f"Value Counts for {column}:")
            st.write(counts)
    else:
        st.write("No columns selected")

    # Some feature selections/engineerings here
    st.markdown("""---""")
    st.markdown('### Remove Irrelevant/Useless Features')
    columns_to_remove = st.multiselect(
        'Select columns to remove', options=df.columns.tolist())

    if st.button('Remove Columns'):
        updated_df = remove_features(df, columns_to_remove)
        st.success('Columns removed successfully', icon="✅")
    else:
        updated_df = df
        st.write('No columns removed')

    st.markdown("""---""")
    # Handle Text and Categorical Attributes
    st.markdown('### Handling Non-numerical Features')
    string_columns = list(df.select_dtypes(['object']).columns)

    int_col, one_hot_col = st.columns(2)

    # Perform Integer Encoding
    with (int_col):
        text_features_select_int = st.multiselect(
            'Select text features for integer encoding',
            string_columns,
        )
        if text_features_select_int and st.button('Integer Encode features'):
            if 'integer_encode' not in st.session_state:
                st.session_state['integer_encode'] = {}
            for feature in text_features_select_int:
                if feature not in st.session_state['integer_encode']:
                    st.session_state['integer_encode'][feature] = True
                else:
                    st.session_state['integer_encode'][feature] = True
            df = integer_encode_features(df, text_features_select_int)

    # Perform One-hot Encoding
    with (one_hot_col):
        text_features_select_onehot = st.multiselect(
            'Select text features for one-hot encoding',
            string_columns,
        )
        if (text_features_select_onehot and st.button('One-hot Encode Feature')):
            # if 'one_hot_encode' not in st.session_state:
            #     st.session_state['one_hot_encode'] = {}
            # if text_features_select_onehot not in st.session_state['one_hot_encode']:
            #     st.session_state['one_hot_encode'][text_features_select_onehot] = True
            # else:
            #    st.session_state['one_hot_encode'][text_features_select_onehot] = True
            df = one_hot_encode_features(df, text_features_select_onehot)

    # # Remove outliers
    # st.markdown('### Remove outliers')

    # # Normalize your data if needed
    # st.markdown('### Normalize data')

    st.markdown('### You have preprocessed the dataset:')
    st.dataframe(df)
    st.session_state['data'] = df

    st.write('Continue to Train Model')
