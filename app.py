import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# Set page configuration
st.set_page_config(page_title="Bank Churn Prediction", layout="centered")

# --- 1. LOAD DATA AND MODEL ---
@st.cache_data
def load_data(path):
    """Loads the bank churn dataset."""
    df = pd.read_csv(path)
    # Drop the columns that were removed in the notebook
    df = df.drop(columns=[
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ])
    return df

@st.cache_resource
def load_model(path):
    """Loads the trained machine learning model."""
    model = joblib.load(path)
    return model

# Load the necessary files
try:
    df = load_data('BankChurners.csv')
    model = load_model('customer_churn_model.pkl')
except FileNotFoundError:
    st.error("Error: Make sure `BankChurners.csv` and `best_model.pkl` are in the same directory as this script.")
    st.stop()


# --- 2. PREPROCESSING FUNCTION (CORRECTED) ---
def preprocess_input(data, original_df):
    """
    Preprocesses user input to match the model's training data format.
    This function replicates the encoding steps from the notebook.
    """
    input_df = data.copy()

    # Label Encoding for Gender (0 for 'F', 1 for 'M')
    input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x == 'M' else 0)

    # One-Hot Encoding for categorical features
    ohe_cols = ['Education_Level', 'Marital_Status', 'Card_Category']
    
    # CORRECTED: Let the encoder infer categories from the original dataframe.
    # This ensures it sorts them alphabetically and drops the same 'first'
    # category as the model was trained on.
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Fit on the original dataframe to learn the correct category order
    ohe.fit(original_df[ohe_cols])

    # Transform the user input dataframe
    encoded_features = ohe.transform(input_df[ohe_cols])
    encoded_feature_names = ohe.get_feature_names_out(ohe_cols)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=input_df.index)

    # Combine with other features and drop original categorical columns
    input_df = pd.concat([input_df, encoded_df], axis=1)
    input_df = input_df.drop(columns=ohe_cols)
    
    # Drop columns that were not used in the model training
    input_df = input_df.drop(columns=['CLIENTNUM', 'Income_Category'])

    # Ensure final column order matches the model's training columns
    model_features = [
        'Customer_Age', 'Gender', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Education_Level_Doctorate', 'Education_Level_Graduate',
        'Education_Level_High School', 'Education_Level_Post-Graduate',
        'Education_Level_Uneducated', 'Education_Level_Unknown',
        'Marital_Status_Married', 'Marital_Status_Single', 'Marital_Status_Unknown',
        'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver'
    ]
    
    return input_df[model_features]

# --- 3. USER INTERFACE (UI) ---

st.title("💳 Bank Customer Churn Prediction App")
st.write("Enter the customer's details in the sidebar on the left to predict their churn status.")

# --- SIDEBAR FOR USER INPUTS ---
st.sidebar.header("Customer Input Features")

def user_input_features():
    """Creates sidebar widgets and returns a dataframe of user inputs."""
    customer_age = st.sidebar.slider('Age', 26, 73, 46)
    gender = st.sidebar.selectbox('Gender', ('M', 'F'))
    dependent_count = st.sidebar.slider('Number of Dependents', 0, 5, 2)
    education_level = st.sidebar.selectbox('Education Level', sorted(df['Education_Level'].unique()))
    marital_status = st.sidebar.selectbox('Marital Status', sorted(df['Marital_Status'].unique()))
    income_category = st.sidebar.selectbox('Income Category', df['Income_Category'].unique()) # Note: This feature is dropped later.
    card_category = st.sidebar.selectbox('Card Category', sorted(df['Card_Category'].unique()))
    months_on_book = st.sidebar.slider('Months on Book', 13, 56, 36)
    total_relationship_count = st.sidebar.slider('Total Products Held', 1, 6, 4)
    months_inactive_12_mon = st.sidebar.slider('Months Inactive (Last 12 Months)', 0, 6, 2)
    contacts_count_12_mon = st.sidebar.slider('Contacts (Last 12 Months)', 0, 6, 2)
    credit_limit = st.sidebar.number_input('Credit Limit', 1438.0, 34516.0, 8634.0)
    total_revolving_bal = st.sidebar.number_input('Total Revolving Balance', 0, 2517, 1162)
    avg_open_to_buy = st.sidebar.number_input('Average Open to Buy Credit Line', 3.0, 34516.0, 7469.0)
    total_amt_chng_q4_q1 = st.sidebar.slider('Change in Transaction Amount (Q4 vs Q1)', 0.0, 3.4, 0.76)
    total_trans_amt = st.sidebar.number_input('Total Transaction Amount (Last 12 Months)', 510, 18484, 4404)
    total_trans_ct = st.sidebar.slider('Total Transaction Count (Last 12 Months)', 10, 139, 64)
    total_ct_chng_q4_q1 = st.sidebar.slider('Change in Transaction Count (Q4 vs Q1)', 0.0, 3.72, 0.71)
    avg_utilization_ratio = st.sidebar.slider('Average Card Utilization Ratio', 0.0, 1.0, 0.27)

    data = {
        'CLIENTNUM': 0, # Placeholder, will be dropped
        'Customer_Age': customer_age,
        'Gender': gender,
        'Dependent_count': dependent_count,
        'Education_Level': education_level,
        'Marital_Status': marital_status,
        'Income_Category': income_category, # Placeholder, will be dropped
        'Card_Category': card_category,
        'Months_on_book': months_on_book,
        'Total_Relationship_Count': total_relationship_count,
        'Months_Inactive_12_mon': months_inactive_12_mon,
        'Contacts_Count_12_mon': contacts_count_12_mon,
        'Credit_Limit': credit_limit,
        'Total_Revolving_Bal': total_revolving_bal,
        'Avg_Open_To_Buy': avg_open_to_buy,
        'Total_Amt_Chng_Q4_Q1': total_amt_chng_q4_q1,
        'Total_Trans_Amt': total_trans_amt,
        'Total_Trans_Ct': total_trans_ct,
        'Total_Ct_Chng_Q4_Q1': total_ct_chng_q4_q1,
        'Avg_Utilization_Ratio': avg_utilization_ratio
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- 4. PREDICTION AND OUTPUT ---
st.subheader('Prediction')

# Create a predict button
if st.sidebar.button('Predict Churn'):
    # Preprocess the user input
    processed_input = preprocess_input(input_df, df)
    
    # Get prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    st.subheader('Prediction Result')

    # Note: In your notebook, 0 = Attrited, 1 = Existing.
    if prediction[0] == 0: 
        st.error('**Prediction: Customer is likely to CHURN.**')
        st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")
    else: 
        st.success('**Prediction: Customer is likely to STAY.**')
        st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")
else:
    st.info("Click 'Predict Churn' in the sidebar to view the result.")


# NOTE: The code is big because the data preprocessing and encoding was done outside of pipeline with pipeline the code could be shrinked