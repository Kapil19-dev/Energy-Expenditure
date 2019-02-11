import pickle
import os 
from keras.models import load_model
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import csv

model_name = 'T2'
trained_models_location ='/home/s1931628/EE_model/EE_standard/'
patient_folder = '/home/s1931628/seqEE/standard_protocol/'
diagrams_folder = '/home/s1931628/standard_protocol/plots/'
csv_folder ='/home/s1931628/standard_protocol/'

get_all_patients_rsquared, get_all_patients_rms = {}, {} 

for i in range(5, 37):
    if len(str(i)) ==1: 
        name = 'GOTOV0'+str(i)
    else:
        name = 'GOTOV'+str(i)   
        
    if os.path.exists(trained_models_location+ model_name +'_'+name+'_.hdf5') == True: 
        print('Predicting test for patient', name)
        with open(patient_folder+name+'.pkl','rb') as f:
            X_train, y_train, bmi_train, X_val, y_val, bmi_val, X_test, y_test, bmi_test, scaler = pickle.load(f)
        
        part = model_name+'_'+name
        print('loading model....')
        model = load_model(trained_models_location+part+ '_.hdf5')
        yhat = model.predict([X_test, bmi_test])
        
#         print('Inversing data.....')
        X_test = X_test.reshape(-1, X_test.shape[2])
        inv_yhat = np.empty((X_test.shape[0], 1))
        inv_yhat.fill(np.nan)
        inv_yhat[:yhat.shape[0]] = yhat

        inv_yhat = np.concatenate((X_test, inv_yhat), axis=1)
        inv_yhat = np.ma.array(inv_yhat, mask=np.isnan(inv_yhat))
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 2]

        y_test = y_test.reshape(len(y_test), 1)
        inv_ytest = np.empty((X_test.shape[0], 1))
        inv_ytest.fill(np.nan)
        inv_ytest[:y_test.shape[0]] = y_test

        inv_ytest = np.concatenate((X_test, inv_ytest), axis=1)
        inv_ytest = np.ma.array(inv_ytest, mask=np.isnan(inv_ytest))
        inv_ytest = scaler.inverse_transform(inv_ytest)
        inv_ytest = inv_ytest[:, 2]

        inv_yhat = inv_yhat[~np.isnan(inv_yhat)]
        inv_ytest = inv_ytest[~np.isnan(inv_ytest)]

        rsquared = r2_score(inv_ytest, inv_yhat)
        print('rsquared...', rsquared)

        rms = sqrt(mean_squared_error(inv_ytest, inv_yhat))
        print('rms...', rms)
        
        plt.figure(figsize=(15,8))
        plt.plot(inv_ytest, label='True_EE')
        plt.plot(inv_yhat, label='Predicted_EE')
        plt.legend(loc='upper left')
        plt.savefig(diagrams_folder+name+'.pdf')
        plt.close()
        
        if name in get_all_patients_rms: 
            get_all_patients_rms[name].append(rms)
            get_all_patients_rsquared[name].append(rsquared)
        else:
            get_all_patients_rms[name] = rms
            get_all_patients_rsquared[name] = rsquared
        
    else: 
        print(False)
        
with open(csv_folder+'rms.csv', 'w') as f: 
    writer = csv.writer(f)
    for k, v in get_all_patients_rms.items():
        writer.writerow([k, v])
        
with open(csv_folder+'rsquared.csv', 'w') as f: 
    writer = csv.writer(f)
    for k, v in get_all_patients_rsquared.items():
        writer.writerow([k, v])