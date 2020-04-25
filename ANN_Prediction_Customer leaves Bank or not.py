# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np;
import pandas as pd
import matplotlib.pyplot as plot
dataset=pd.read_csv("F:\Documents\AI_ML\Advance DS\Keras\Deep_Learning_A_Z\Artificial_Neural_Networks\Churn_Modelling.csv")
#retrieve important features for the Prediction
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Label Encoding for Categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1=  LabelEncoder()
X[:,1]= labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=  LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
#One hot encoding for the Nominal Variable
ct = ColumnTransformer(

    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],

    remainder='passthrough')

X=ct.fit_transform(X)
X=X[:,1:]
#train test split
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#Feature scaling
#Standard Scalar

from sklearn.preprocessing import  StandardScaler
standardscaler= StandardScaler()
X_train=standardscaler.fit_transform(X_train)
X_test=standardscaler.transform(X_test)

#ANN - import modules
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()
#input lyer 
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))
#Hidden layer
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
#Output Layer
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

y_pred=classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
