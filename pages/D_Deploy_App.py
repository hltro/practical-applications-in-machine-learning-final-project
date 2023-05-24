import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

#############################################

st.title('Deploy Application')

#############################################
enc = OrdinalEncoder()


def deploy_model(df):
    """
    Deploy trained regression model in df
    Input: 
        - df: pandas dataframe with trained regression model
    Output: 
        - house_price: predicted house price
    """
    likelihood = None
    model = None
    if ('deploy_model' in st.session_state):
        model = st.session_state['deploy_model']
        # Test model
        if (model):
            likelihood = model.predict(df)

    # return house_price
    return likelihood


def decode_integer(original_df, decode_df, feature_name):
    """
    Decode integer integer encoded feature

    Input: 
        - original_df: pandas dataframe with feature to decode
        - decode_df: dataframe with feature to decode 
        - feature: feature to decode
    Output: 
        - decode_df: Pandas dataframe with decoded feature
    """
    original_df[[feature_name]] = enc.fit_transform(
        original_df[[feature_name]])
    decode_df[[feature_name]] = enc.inverse_transform(
        st.session_state['X_train'][[feature_name]])
    return decode_df


df = None
if 'data' in st.session_state:
    df = st.session_state['data']
else:
    st.write(
        '### The <project> Application is under construction. Coming to you soon.')

# Deploy App
if df is not None:
    st.markdown('### <New York Crime Analysis>')

    # Input users input features
    user_input = {}
    original_dataset = pd.read_csv("dataset/data_template.csv")
    decode_df = pd.DataFrame()

    st.markdown('## Please enter the precise location')
    latitude = 0
    if ('Latitude' in st.session_state['X_train'].columns):
        latitude = st.number_input(
            'What is the latitude?',  min_value=40.0,  max_value=44.0, step=0.01, format="%.4f")
        if (latitude):
            user_input['Latitude'] = latitude

    longitude = 0
    if ('Longitude' in st.session_state['X_train'].columns):
        longitude = st.number_input(
            'What is the longitude?',  min_value=-75.0, max_value=-70.0, step=0.01, format="%.4f")
        if (latitude):
            user_input['Longitude'] = longitude

    VICTIM_AGE_GROUP = 0
    if ('VICTIM_AGE_GROUP' in st.session_state['X_train'].columns):
        st.markdown('## Please select your age group')
        user_options = None
        if ('integer_encode' in st.session_state and st.session_state['integer_encode'].get('VICTIM_AGE_GROUP')):
            decode_df = decode_integer(
                original_dataset, decode_df, 'VICTIM_AGE_GROUP')
            user_options = decode_df['VICTIM_AGE_GROUP'].unique()

        user_age_group = st.selectbox(
            'Please select your age group',
            options=user_options)

        if (user_age_group):
            if ('integer_encode' in st.session_state and st.session_state['integer_encode'].get('VICTIM_AGE_GROUP')):
                user_input['VICTIM_AGE_GROUP'] = enc.transform([[user_age_group]])[
                    0]
            else:
                user_input[VICTIM_AGE_GROUP] = 1

    VICTIM_RACE = 0
    if ('VICTIM_RACE' in st.session_state['X_train'].columns):
        if ('VICTIM_RACE' in st.session_state['X_train'].columns):
            st.markdown('## Please select your race')
            user_options = None
            if ('integer_encode' in st.session_state and st.session_state['integer_encode'].get('VICTIM_RACE')):
                decode_df = decode_integer(
                    original_dataset, decode_df, 'VICTIM_RACE')
                user_options = decode_df['VICTIM_RACE'].unique()

        VICTIM_RACE = st.selectbox(
            'Please select your race',
            options=user_options)
        # ('ASIAN / PACIFIC ISLANDER', 'BLACK', 'WHITE', 'AMERICAN INDIAN/ALASKAN NATIVE','WHITE HISPANIC', 'BLACK HISPANIC'))
        if (VICTIM_RACE):
            if ('integer_encode' in st.session_state and st.session_state['integer_encode'].get('VICTIM_RACE')):
                user_input['VICTIM_RACE'] = enc.transform([[VICTIM_RACE]])[0]
            else:
                user_input[VICTIM_RACE] = 1

    VICTIM_SEX = 0
    if ('VICTIM_SEX' in st.session_state['X_train'].columns):
        if ('VICTIM_SEX' in st.session_state['X_train'].columns):
            st.markdown('## Please select your sex')
            user_options = None
            if ('integer_encode' in st.session_state and st.session_state['integer_encode'].get('VICTIM_SEX')):
                decode_df = decode_integer(
                    original_dataset, decode_df, 'VICTIM_SEX')
                user_options = decode_df['VICTIM_SEX'].unique()

        VICTIM_SEX = st.selectbox(
            'Please select your sex',
            options=user_options)
        if (VICTIM_SEX):
            if ('integer_encode' in st.session_state and st.session_state['integer_encode'].get('VICTIM_SEX')):
                user_input['VICTIM_SEX'] = enc.transform([[VICTIM_SEX]])[0]
            else:
                user_input[VICTIM_SEX] = 1

    # st.write(user_input)
    selected_features_df = pd.DataFrame.from_dict(user_input, orient='index').T
    # st.write(selected_features_df)

    # Deploy model
    st.markdown("""---""")
    if ('deploy_model' in st.session_state and st.button('Predict The likelihood of attack', type='primary')):
        likelihood = deploy_model(selected_features_df)
        if (likelihood is not None):
            # Display likelihood
            if (likelihood > 0.5):
                st.snow()
                st.warning(
                    'The area is not very safe, please pay close attention to your surroundings.', icon="⚠️")

            else:
                st.success(
                    'The area seems to be very safe, you are unlikely to be attacked', icon="✅")
                st.balloons()
