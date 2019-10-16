#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, ElasticNet, SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
import seaborn as sns

def clean_data(data):
    print('cleaning data')
    unnecessary = ['ï»¿Instance', 'Hair Color']
    data.drop(columns=unnecessary, inplace=True) #drop instance and hair colour
    #remove integers from columns with strings
    data['Gender'] = remove_int(data, 'Gender')
    data['Country'] = remove_int(data, 'Country')
    data['Profession'] = remove_int(data, 'Profession')
    data['University Degree'] = remove_int(data, 'University Degree')
    #remove unknowns in gender to only have 3 genders
    data = rem_unknown(data)
    #fill NaNs 
    data['University Degree'] = data['University Degree'].fillna('No') #assuming unfilled degree is no degree
    data['Gender'] = data['Gender'].fillna('female') #assuming if unfilled gender should be female
    data['Profession'] = data['Profession'].fillna('Unemployed') # assuming unfilled profession is unemployed
    #take median for numerical columns
    data['Year of Record'] = data['Year of Record'].fillna(data['Year of Record'].median())
    data['Age'] = data['Age'].fillna(data['Age'].median())
    #one hot encode string columns
    data = onehot_encode(data, 'Gender')
    data.drop(columns=['Gender'], inplace=True)
    data = onehot_encode(data, 'Country')
    data.drop(columns=['Country'], inplace=True)
    data = onehot_encode(data, 'University Degree')
    data.drop(columns=['University Degree'], inplace=True)
    data = onehot_encode(data, 'Profession')
    data.drop(columns=['Profession'], inplace=True)
    #use z score to standarize numerical columns
    data['Body Height [cm]'] = z_score(data, 'Body Height [cm]')
    data['Age'] = z_score(data, 'Age')
    data['Year of Record'] = z_score(data, 'Year of Record')
    data['Size of City'] = z_score(data, 'Size of City')
    return data

def clean_pred(data, empty):
    print('cleaning prediction data')
    #begin cleaning like training data
    data = rem_unknown(data)
    data['University Degree'] = data['University Degree'].fillna('No')
    data['Gender'] = data['Gender'].fillna('female')
    data['Profession'] = data['Profession'].fillna('Unemployed')
    data['Year of Record'] = data['Year of Record'].fillna(data['Year of Record'].median())
    data['Age'] = data['Age'].fillna(data['Age'].median())
    empty['Body Height [cm]'] = z_score(data, 'Body Height [cm]')
    empty['Wears Glasses'] = data['Wears Glasses']
    empty['Age'] = z_score(data, 'Age')
    empty['Year of Record'] = z_score(data, 'Year of Record')
    empty['Size of City'] = z_score(data, 'Size of City')
    #map each of the string columns as 0s and 1s to the relevant columns in a dataframe matching the training data
    print('Mapping Profession')
    for row in range(0, len(data)):
        if 'Profession_' + str(data.loc[row, 'Profession']) in list(empty.columns):
            empty.loc[row, ['Profession_' + str(data.loc[row, 'Profession'])]] = 1
    print('Mapping Gender')
    for row in range(0, len(data)):
        if 'Gender_' + str(data.loc[row, 'Gender']) in list(empty.columns):
            empty.loc[row, ['Gender_' + str(data.loc[row, 'Gender'])]] = 1
    print('Mapping Country')
    for row in range(0, len(data)):
        if 'Country_' + str(data.loc[row, 'Country']) in list(empty.columns):
            empty.loc[row, ['Country_' + str(data.loc[row, 'Country'])]] = 1
    print('Mapping Degree')
    for row in range(0, len(data)):
        if 'University Degree_' + str(data.loc[row, 'University Degree']) in list(empty.columns):
            empty.loc[row, ['University Degree_' + str(data.loc[row, 'University Degree'])]] = 1
    return empty

def remove_int(data, category):
    for row in range(0, len(data[category])):
        try:
            int(data.loc[row, category])
            data.loc[row, category]=np.nan
        except ValueError:
            pass
    return data[category]

#def normalize(data, category):
#    data[category]=(data[category]-data[category].min())/(data[category].max()
#                                                 -data[category].min())
#    return data[category]

def z_score(data, category):
    data[category] = (data[category]-data[category].mean())/data[category].std()
    return data[category]

#def label_encode(category):
    #label = LabelEncoder()
    #data[category] = label.fit_transform(data[category])

def rem_unknown(data):
    for row in range(0, len(data['Gender'])):
        if data.loc[row, 'Gender'] == 'unknown':
            data.loc[row, 'Gender']=np.nan
    return data

def onehot_encode(data, category):
    #onehot = OneHotEncoder()
    #reshape = data[category].values.reshape(-1,1)
    #X = onehot.fit_transform(reshape).toarray()
    #dataOneHot = pd.DataFrame(X, columns = [category+str(int(i)) for i in range(X.shape[1])])
    #use panda get dummies as it keeps the name from the entries as the column names
    dataOneHot = pd.get_dummies(data[category], prefix=category)
    data = pd.concat([data, dataOneHot], axis=1)
    return data


# In[66]:


def main():
    file = open('training_data.csv')
    data = pd.read_csv(file)
    #cleaning training data
    data = clean_data(data)
    #break into X and Y 
    X = data.drop(columns='Income in EUR')
    y = data['Income in EUR']
    #take a test size of 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=10)
    #remove outliers in income
    count = 0
    for row in range(0, len(data['Income in EUR'])):
        if data.loc[row, 'Income in EUR'] < -100 or data.loc[row, 'Income in EUR'] > 4000000:
            data.loc[row, 'Income in EUR'] = np.nan
    data.dropna(inplace=True)
    X = data.drop(columns='Income in EUR')
    y = data['Income in EUR']
    #re-split data after outliers have been removed (want to keep full test set)
    X_train, V_test, y_train, v_test = train_test_split(X, y, train_size=0.7, random_state=10)
    
    #final regressor used = Bayesian ridge (others were tested)
    regressor = linear_model.BayesianRidge() #RMSE = 83334.12842653893
    #regressor = LinearRegression() #RMSE = 238390549829182.34
    #regressor = LogisticRegression() #needs ints
    #regressor = Ridge() #Gets lots of Minus Numbers R2 = 0.67 RMSE = 86329.59510217133
    #regressor = RidgeCV() #RMSE = 86584.3213956372
    #regressor = ElasticNet() #RMSE = 149068.04889159356
    #regressor = svm.LinearSVR() #RMSE = 164139.2729824018
    #regressor = linear_model.LassoLars() #RMSE = 87304.59745003439
    #regressor = linear_model.TheilSenRegressor() # needs ints
    #regressor = Lasso(alpha=0.1) #RMSE = 
    #regressor = RandomForestRegressor(max_depth=30, random_state=2, n_estimators=100) #RMSE >100000
    #regressor = SGDRegressor(max_iter=1000, tol=1e-3) #RMSE = 88260.7692238782
    
    #train regressor on training data
    print('training...')
    regressor.fit(X_train, y_train)
    #predict on test data and print RMSE
    print('predicting...')
    y_pred_test = regressor.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    
    file = open('prediction_data.csv')
    pred_data = pd.read_csv(file)
    pred_data = pred_data.drop(columns='Income')
    #create dataframe with columns matching training data
    empty = np.zeros((len(pred_data), len(X_train.columns)))
    empty = pd.DataFrame(empty, columns = X_train.columns)
    #clean prediction data using empty dataframe
    pred_data = clean_pred(pred_data, empty)
    y_pred = regressor.predict(pred_data)
    #assuming all income should be positive
    y_pred = abs(y_pred)
    #write out results
    file = open('prediction.csv', 'w', newline='')
    pd.DataFrame(y_pred).to_csv(file)

if __name__ == "__main__":
    main()


# In[ ]:




