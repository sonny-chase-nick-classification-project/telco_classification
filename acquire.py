import pandas as pd
import numpy as np 
from env import host, user, password

# function to contact database

def get_db_url(db_name):
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"

# function to query db and return df

def get_telco_data():
    url = get_db_url('telco_churn')
    query = """
    SELECT
        *
    FROM
        customers
    JOIN contract_types USING(contract_type_id)
    JOIN internet_service_types USING(internet_service_type_id)
    JOIN payment_types USING(payment_type_id)
    """
    telco = pd.read_sql(query,url)
    return telco