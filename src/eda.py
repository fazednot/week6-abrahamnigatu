import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv('data.data.csv'csv')

def perform_eda(data):
    # Summary statistics
    print(data.info())
    print(data.describe())

    # Visualize distributions
    data.hist(figsize=(12, 8))
    plt.show()

    # Correlation heatmap
    sns.heatmap(data.corr(), annot=True)
    plt.show()

    # Check for missing values
    print(data.isnull().sum())
