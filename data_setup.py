import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def convertOceanProximity(df):
    # One-hot encode ocean_proximity instead of ordinal encoding
    return pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity')

def loadData():
    # Load the dataset
    df = pd.read_csv('data/housing.csv')

    le_ocean = LabelEncoder()
    df['ocean_proximity'] = le_ocean.fit_transform(df['ocean_proximity'])
    #print(df.info())
    # Feature engineering
    #df['rooms_per_household'] = df['total_rooms'] / df['households']
    #df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    #df['population_per_household'] = df['population'] / df['households']
    
    # One-hot encode ocean_proximity
    #ocean_proximity_encoded = convertOceanProximity(df)
    #df = df.drop('ocean_proximity', axis=1)
    #df = pd.concat([df, ocean_proximity_encoded], axis=1)
    
    # Drop rows with missing total_bedrooms
    #df = df.dropna(subset=['total_bedrooms'])
    #df = df.drop(['total_bedrooms'], axis=1)
    
    return df

def showCorrelation(df):
    corr_matrix = df.select_dtypes(include=np.number).corr()

    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix,
                annot=True,
                linewidths=0.5,
                fmt=".2f",
                cmap="YlGnBu")
    plt.title('Correlation Matrix')
    plt.show()

#showCorrelation(loadData())
loadData()