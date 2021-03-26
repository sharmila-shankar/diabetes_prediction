from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def home(request):
    return render(request,'index.html',{"predicted":""})

def predict(request):

    prg = float(request.GET['prg'])
    gl = float(request.GET['gl'])
    bp = float(request.GET['bp'])
    st = float(request.GET['st'])
    ins = float(request.GET['ins'])
    bmi = float(request.GET['bmi'])
    dpf = float(request.GET['dpf'])
    age = float(request.GET['age'])

    #Reading the dataframe
    rawdata = staticfiles_storage.path('diabetes2.csv')
    ds = pd.read_csv(rawdata)
    
    # Separating features and target
    x = ds.iloc[:, :-1].values
    y = ds.iloc[:, -1].values

    
    # Split dataframe into train and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 54)

    # Creating the AI model
    knn = KNeighborsClassifier(n_neighbors=3) 
    knn.fit(x_train,y_train)
    
    #Prediction
    result = np.array([[prg, gl, bp, st, ins, bmi, dpf, age]])
    #result = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    y_pred = knn.predict(result)

    if y_pred == 0:
        res = 'No'
    else:
        res = 'Yes'
    
    return render(request,'index.html',{"predicted":res, "prg":prg, "gl":gl, "bp":bp, "st":st, "ins":ins, "bmi":bmi, "dpf":dpf, "age":age})