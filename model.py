import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.impute
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import env
import acquire
import prepare

def get_summary(X, y, f, t):
    '''
    Returns classification report of a model
    '''
    y_pred = f.predict(X)
    y_pred_proba = f.predict_proba(X)
    y_pred_proba = pd.DataFrame(y_pred_proba, columns = ['no churn', 'churn'])
    yhat = (y_pred_proba > t).astype(int)
    print(classification_report(y, yhat.churn))


# Create our dataframes for exporting our CSV

def X_label_encode(df):
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df.gender)
    df['online_security'] = le.fit_transform(df.online_security)
    df['online_backup'] = le.fit_transform(df.online_backup)
    df['device_protection'] = le.fit_transform(df.device_protection)
    df['tech_support'] = le.fit_transform(df.tech_support)
    df['streaming_tv'] = le.fit_transform(df.streaming_tv)
    df['streaming_movies'] = le.fit_transform(df.streaming_movies)
    df['paperless_billing'] = le.fit_transform(df.paperless_billing)
    df['churn'] = le.fit_transform(df.churn)
    return df

def one_hot_encoder(df):
    one_hot = OneHotEncoder(sparse = False)
    payment_encoded = one_hot.fit_transform(df[['payment_type']])
    payment_labels = list(df.payment_type.value_counts().sort_index().index)
    payment_encoded_df = pd.DataFrame(payment_encoded, columns = payment_labels, index = df.index)

    internet_encoded = one_hot.fit_transform(df[['internet_service_type']])
    internet_labels = list(df.internet_service_type.value_counts().sort_index().index)
    internet_encoded_df = pd.DataFrame(internet_encoded, columns = internet_labels, index = df.index)

    contract_encoded = one_hot.fit_transform(df[['contract_type']])
    contract_labels = list(df.contract_type.value_counts().sort_index().index)
    contract_encoded_df = pd.DataFrame(contract_encoded, columns = contract_labels, index = df.index)

    df = df.join([payment_encoded_df, internet_encoded_df, contract_encoded_df])

    return df

def drop_service_types(df):
    return df.drop(columns=['contract_type', 'internet_service_type', 'payment_type'])

def get_prediction():
    # Get the Data
    df = acquire.get_telco_data()

    # Prepare the Data
    df = prepare.fill_na(df)
    df = prepare.drop_na(df)
    df['total_charges'] = df['total_charges'].astype('float')
    df['tenure_years'] = df.tenure/12
    df = prepare.phone_lines(df)
    df = prepare.partner_dependents(df)
    df = prepare.drop_columns(df)

    # Add Features
    #df = features.create_features(df)

    # Encode DataFrame
    df = X_label_encode(df)
    df = one_hot_encoder(df)
    df = drop_service_types(df)

    # Select features to be used in the model

    X = df.drop(columns=['customer_id', 'churn'])
    y = df.churn
    
    # Create and fit the model
    logit = LogisticRegression(C=10, random_state = 123)

    #Fit the model to the training data
    logit.fit(X, y)

    #Estimate whether or not a customer would churn, using the training data (threshold of 0.5)
    y_pred = logit.predict(X)

    # Estimate the probability of customer churn, using the training data
    y_pred_proba = logit.predict_proba(X)
    y_pred_proba = pd.DataFrame(y_pred_proba, columns = ['no churn', 'churn'], index = X.index)
    

    # Create a DataFrame to hold predictions
    prediction = pd.DataFrame(
        {'Custumer_ID': df.customer_id,
         'Model_Predictions': logit.predict(X),
         'Model_Probabilities': logit.predict_proba(X)[:,1]
        })

    # Generate csv
    prediction.to_csv('prediction_csv.csv')

    return prediction