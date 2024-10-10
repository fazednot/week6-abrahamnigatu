from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def create_aggregate_features(data):
    data['Total_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('sum')
    data['Average_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('mean')
    return data

def encode_categorical(data):
    encoder = OneHotEncoder()
    categorical_columns = ['ProductCategory', 'ChannelId']
    encoded_data = encoder.fit_transform(data[categorical_columns]).toarray()
    return pd.concat([data, pd.DataFrame(encoded_data)], axis=1)

def scale_numeric(data):
    scaler = StandardScaler()
    numeric_columns = ['Amount', 'Value']
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data
