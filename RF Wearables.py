# Coding: utf-8 
"""
Created on Wed Apr  1 18:42:04 2020
@author: Stress Wearables Group - Saturdays AI Madrid

"""

## LIBRERIAS ##
import os
from pathlib import Path

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import math 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## ESTABLECER DIRECTORIO
path = os.getcwd()
print(path)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# os.path.abspath(os.path.dirname(sys.argv[0]))

# DATOS UTILIZADOS : WESAD #

###################
## LECTURA DATOS ##
###################

root_eda = Path('DATA','Processed','WESAD','EDA')
root_hrv = Path('DATA','Processed','WESAD','HRV')

# EDA #
for i in range(2,18):
    name ='S' + str(i) + '-eda.xlsx'
    if  i == 2 :
        eda = pd.read_excel(root_eda/name)
        len_eda = [eda.shape[0]]
    elif i == 12 :
        eda = eda
    else : 
        aux = pd.read_excel(root_eda/name)
        eda = eda.append(aux)
        len_eda.append(aux.shape[0])

eda = eda.drop(['condition'],axis=1)
eda = eda.sort_values(['subject id','Time'],ascending=[1,1]) 

# root_eda2 = Path('DATA','Raw','WESAD','EDA')
# name = 'S' + '2' + '.txt'
# eda_raw = pd.read_csv(root_eda2/name, sep=" ", header=None)
  
# HRV #
for i in range(2,18):
    name ='S' + str(i) + '-hrv.xlsx'
    if  i == 2 :
        hrv = pd.read_excel(root_hrv/name)
        len_hrv = [hrv.shape[0]]
    elif i == 12 :
        hrv = hrv
    else : 
        aux = pd.read_excel(root_hrv/name)
        hrv = hrv.append(aux)     
        len_hrv.append(aux.shape[0])
      
hrv = hrv.drop(['condition'],axis=1)
hrv = hrv.sort_values(['subject id','Time'],ascending=[1,1])   
            
########################
# PREPARACION DE DATOS #
########################

incremento = 0.1
minimo = 7.0
max_eda = max(eda['Time'])
max_hrv = max(hrv['Time'])
maximo = max(max_eda ,max_hrv)
maximo = math.floor(maximo*10)/10   
rango = np.arange(minimo, maximo + incremento, incremento)

## EDA ##
eda_sel = 0
for sujeto in range(2,18) :
    
    if sujeto == 12 : # Sujeto 12 no existe en el dataset
        eda_sel = eda_sel
    else :   
        # maximo = max(eda[eda['subject id']==sujeto]['Time'])
        for t in rango :
            if t == minimo : 
                aux = pd.DataFrame(eda[(eda['Time']>=t) & (eda['Time']<t+0.1) & (eda['subject id']==sujeto)].mean(axis=0)).transpose()
                aux['Time'] = t
                eda_suj = aux
            else :
                # aux = eda[(eda['Time']>=t) & (eda['Time']<t+0.1) & (eda['subject id']==sujeto)]
                # media = pd.DataFrame(aux.mean(axis=0)).transpose()
                aux = pd.DataFrame(eda[(eda['Time']>=t) & (eda['Time']<t+0.1) & (eda['subject id']==sujeto)].mean(axis=0)).transpose()
                aux['Time'] = t
                eda_suj = eda_suj.append(aux)
      
        if sujeto == 2 :
            eda_sel = eda_suj
        else :
            eda_sel = eda_sel.append(eda_suj)        

## HRV ##
hrv_sel = 0
for sujeto in range(2,18) :
    
    if sujeto == 12 : # Sujeto 12 no existe en el dataset
        hrv_sel = hrv_sel
    else :   
        for t in rango :
            if t == minimo : 
                aux = pd.DataFrame(hrv[(hrv['Time']>=t) & (hrv['Time']<t+0.1) & (hrv['subject id']==sujeto)].mean(axis=0)).transpose()
                aux['Time'] = t
                hrv_suj = aux
            else :
                # aux = hrv[(hrv['Time']>=t) & (hrv['Time']<t+0.1) & (hrv['subject id']==sujeto)]
                # media = pd.DataFrame(aux.mean(axis=0)).transpose()
                aux = pd.DataFrame(hrv[(hrv['Time']>=t) & (hrv['Time']<t+0.1) & (hrv['subject id']==sujeto)].mean(axis=0)).transpose()
                aux['Time'] = t
                hrv_suj = hrv_suj.append(aux)
      
        if sujeto == 2 :
            hrv_sel = hrv_suj
        else :
            hrv_sel = hrv_sel.append(hrv_suj)       

## JOIN ##
# data = pd.merge(eda_sel, hrv_sel, on=['Time','subject id','SSSQ'], how='left',suffixes=('_eda','_hrv'))
# data = data.dropna()

eda_sel2 = eda_sel.add_prefix('eda_')
hrv_sel2 = hrv_sel.add_prefix('hrv_')

data = pd.concat([eda_sel2,hrv_sel2], sort='False', axis=1)

data = data.drop(['hrv_Time','hrv_SSSQ','hrv_subject id'], axis=1)
data.rename(columns={'eda_Time':'Time', 'eda_SSSQ':'SSSQ', 'eda_subject id':'subject id'}, inplace=True)
data = data.sort_values(['subject id','Time'], ascending=[1,1])   

data = data.dropna()

##########
# MODELO #
##########

## ANALISIS ##
data.dtypes

## TRATAMIENTO DE DATOS ##
data_x = data.drop(['Time','subject id','SSSQ'],axis=1)
data_y = data['SSSQ']
data_x_std = StandardScaler().fit_transform(data_x)

## MODELIZACIÓN ##
# PCA
m_pca = PCA(n_components=15)
comp = m_pca.fit_transform(data_x)
comp_std = m_pca.fit_transform(data_x_std)

PCA = pd.DataFrame(comp)
PCA_std = pd.DataFrame(comp_std)

# SPLIT
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(PCA_std, data_y, test_size=0.2, random_state=42)

# RANDOM FOREST
# Sin PCA


rf = RandomForestClassifier()
modelo = rf.fit(x_train, y_train)

print ("Modelo entrenado :: ", modelo)
prediccion = modelo.predict(x_test)

# for i in range(0, 5):
#     print ("Estres real :: {} y Estres Predecido :: {}".format(list(y_test)[i], prediccion[i]))

print ("Accuracy Set Training  :: ", accuracy_score(y_train, modelo.predict(x_train)))
print ("Accuracy Set Test  :: ", accuracy_score(y_test, prediccion))

# Con PCA
rf = RandomForestClassifier()
modelo_pca = rf.fit(x_train_pca, y_train_pca)

print ("Modelo entrenado :: ", modelo_pca)
prediccion_pca = modelo_pca.predict(x_test_pca)

print ("Accuracy Set Training  :: ", accuracy_score(y_train_pca, modelo_pca.predict(x_train_pca)))
print ("Accuracy Set Test  :: ", accuracy_score(y_test_pca, prediccion_pca))
