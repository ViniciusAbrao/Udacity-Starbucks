# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:01:59 2020

@author: vinic_000
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

# load in the data
df = pd.read_csv('./training.csv')

# we want to offer promo only to a person that will purchase later
purchase=df.loc[df['purchase'] > 0,:]
purchase_and_promo=purchase.loc[purchase['Promotion']=='Yes']
value_one_index=purchase_and_promo.index.values 
promos=np.zeros(df.shape[0])
for i in range(df.shape[0]):
    if i in value_one_index:
        promos[i]=1
df['labels']=promos

# Separate majority and minority classes
df_majority = df[df.labels==0]
df_minority = df[df.labels==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=df_majority.shape[0]    # to match majority class
                                ) 
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.labels.value_counts()

scaler=MinMaxScaler()
rescaled_data=scaler.fit_transform(df_upsampled[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']])
X_train, X_test, y_train, y_test = train_test_split(rescaled_data, 
                                                    df_upsampled['labels'], test_size=0.2)


clf = SVC(kernel='rbf',class_weight='balanced', C=10.0)
clf.fit(X_train, y_train)

#with open('classifier.pkl', 'rb') as file:
#    clf = pickle.load(file)
    

y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
precision=precision_score(y_test,y_pred)
print(precision)
recall=recall_score(y_test,y_pred)
print(recall)
matrix=confusion_matrix(y_test,y_pred)
print(matrix)


def promotion_strategy(df):
    '''
    INPUT 
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion_df - np.array with the values
                   'Yes' or 'No' related to whether or not an 
                   individual should recieve a promotion 
                   should be the length of df.shape[0]
                
    Ex:
    INPUT: df
    
    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2
    
    OUTPUT: promotion
    
    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and 
    the last should not.
    '''
    
    datas=scaler.transform(df)
    y_pred=clf.predict(datas)
    
    promotion=[]
    for i in range(df.shape[0]):
        if y_pred[i]==1:
            promotion.append('Yes')
        else:
            promotion.append('No')
                
    promotion=np.array(promotion)
    
    return promotion


def score(df, promo_pred_col = 'Promotion'):
    n_treat       = df.loc[df[promo_pred_col] == 'Yes',:].shape[0]
    n_control     = df.loc[df[promo_pred_col] == 'No',:].shape[0]
    n_treat_purch = df.loc[df[promo_pred_col] == 'Yes', 'purchase'].sum()
    n_ctrl_purch  = df.loc[df[promo_pred_col] == 'No', 'purchase'].sum()
    irr = n_treat_purch / n_treat - n_ctrl_purch / n_control
    nir = 10 * n_treat_purch - 0.15 * n_treat - 10 * n_ctrl_purch
    return (irr, nir)
    

def test_results(promotion_strategy):
    test_data = pd.read_csv('Test.csv')
    df = test_data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']]
    promos = promotion_strategy(df)
    score_df = test_data.iloc[np.where(promos == 'Yes')]    
    irr, nir = score(score_df)
    print("Nice job!  See how well your strategy worked on our test data below!")
    print()
    print('Your irr with this strategy is {:0.4f}.'.format(irr))
    print()
    print('Your nir with this strategy is {:0.2f}.'.format(nir))
    
    print("We came up with a model with an irr of {} and an nir of {} on the test set.\n\n How did you do?".format(0.0188, 189.45))
    return irr, nir

test_results(promotion_strategy)

with open('new_classifier.pkl', 'wb') as file:
    pickle.dump(clf, file)
    
    
