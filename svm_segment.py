    
import pickle
import tensorflow as tf
import pandas as pd
import random
import glob
import os
from itertools import islice
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

class Sequences_EE(object): 
    
    def __init__(self, SEQUENCE_SIZE, downsample_rate): 
        self.SEQUENCE_SIZE = SEQUENCE_SIZE
        self.downsample_rate = downsample_rate
      
    @classmethod 
    def SQSumSq(cls, df):
        return (df**2).sum(axis=1)**0.5
    
    @classmethod
    def read_data(cls, name, folder1 = '/data/gotov_data/geneActive/', folder2='/data/gotov_data/data2/cosmed/'):
        df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        ankle = ['ankle_x', 'ankle_y', 'ankle_z']
        wrist = ['wrist_x', 'wrist_y', 'wrist_z'] 
        data_to_select_random_ = []
        df_gene = pd.DataFrame()
        df_cos = pd.DataFrame()

        print('loading all datafiles...')
        for data in glob.glob(folder1+'*csv'):
            data = os.path.splitext(os.path.basename(data))[0]
            if data in ['GOTOV03','GOTOV16','GOTOV23','GOTOV04', 
                        'GOTOV02','GOTOV19', 'GOTOV12','GOTOV18', 
                        'GOTOV27','GOTOV30']:
                continue
            else:
                df = pd.read_csv(folder1+data+".csv", header = 0, index_col = None, low_memory=False)

                df_cosmed = pd.read_csv(folder2+data+'.csv', index_col=None)
                df_cosmed = df_cosmed[['EEm', 'time']]
                df_cosmed.rename(columns={'time':'time_cosmed'})

                df = df.drop('labels', axis=1).dropna(axis=0, how='any')

                time = df.time

                df_ankle = df[ankle]
                df_ankle = cls.SQSumSq(df_ankle)

                df_wrist = df[wrist]
                df_wrist = cls.SQSumSq(df_wrist)

                df = pd.concat([time, df_ankle, df_wrist], axis=1)
                df = df.rename(columns={0:'ankle',1:'wrist'})

                inv_yhat = np.empty((df.shape[0], 2))
                inv_yhat.fill(np.nan)
                inv_yhat[:df_cosmed.shape[0]] = df_cosmed
                df_cosmed = pd.DataFrame(inv_yhat, columns=['EEm','time_cos'])

                df = pd.concat([df, df_cosmed], axis=1)
                df['participant'] = data

                df_gene = df_gene.append(df)
                data_to_select_random_.append(data)
                
        if name in data_to_select_random_: data_to_select_random_.remove(name)
        
        validation_data = ['GOTOV07','GOTOV09','GOTOV08','GOTOV10', 
                            'GOTOV11','GOTOV17', 'GOTOV20','GOTOV28', 
                            'GOTOV21','GOTOV29', 'GOTOV31','GOTOV33', 
                            'GOTOV35']

        validation_data = random.sample(validation_data, 1)

        print('Getting val and train data.....')
        for data in data_to_select_random_: 
            if data in validation_data:
                df_val = df_val.append(df_gene.query('participant == "'+data+'"'))
            else:
                df_train = df_train.append(df_gene.query('participant == "'+data+'"'))

        print('Getting test data.....')
        df_test = df_test.append(df_gene.query('participant == "'+name+'"'))

        print('Done creating all dataframes.....')
        return df_train, df_val, df_test 



    @classmethod
    def to_sequences_data(cls, SEQUENCE_SIZE, obs, yobs, EEm_time, protocol, downsample_rate):
        x, y, = [], []
        for i in EEm_time:
            df = []
            window = obs.sort_index().loc[:i].iloc[-SEQUENCE_SIZE:]
            for g, dff in window.groupby(np.arange(len(window)) // 4980):
                key = str(round((1/downsample_rate[g]), 3)) + 'S'
                df.append(dff.resample(key).mean())
            window = pd.concat(df).iloc[-874:]
            window = window.values.reshape(-1, window.shape[0], window.shape[1])
            after_window = yobs[i]
            x.append(window)
            y.append(after_window)
        BMI = np.empty((len(y), np.array(protocol).shape[1]))
        BMI = protocol*len(y)

        return x, y, BMI
    
    @classmethod 
    def sequence_builder(cls, geneActive, protocol, SEQUENCE_SIZE, downsample_rate):
        seqX, seqY, seqB = [], [], []
        cols = ['time_cos', 'EEm']
        for data in geneActive['participant'].unique(): 
            print('Print building sequence for participants.....',data)
            df_geneActive = geneActive.query('participant == "'+data+'"')
            
            protocol_part = protocol[protocol.participant==data]
            protocol_part = protocol_part.drop('participant', axis=1).values.tolist()
            
            df_geneA = df_geneActive.drop(cols, axis=1)
            df_cosmed = df_geneActive[cols]
            
            df_geneA.set_index('time', inplace=True)
            df_geneA.index = pd.to_datetime(df_geneA.index, unit = "ms")

            df_cosmed.set_index('time_cos', inplace=True)
            df_cosmed.index = pd.to_datetime(df_cosmed.index, unit = "ms")

            df_cosmed['tf'] = df_cosmed['EEm'].notnull()
            EEm_times = df_cosmed.index[df_cosmed['tf']].tolist()

            x_values = df_geneA.drop(['participant'], axis=1)
            y_values = df_cosmed['EEm']

            x, y, bmi = cls.to_sequences_data(SEQUENCE_SIZE, x_values, y_values, EEm_times, protocol_part, downsample_rate)
            seqX.append(x)
            seqY.append(y)
            seqB.append(bmi)

        x = np.vstack(seqX)
        bmi = np.vstack(seqB)
        y = [item for sublist in seqY for item in sublist]
        y = np.array(y)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        print(x.shape, y.shape, bmi.shape)
        print('Done building sequences....')
        return x, y, bmi
    
    @classmethod
    def standardizing_data(cls, name, Xtrain, Xval, Xtest, SEQUENCE_SIZE, downsample_rate): 
        print('Scaling data....')

        #variables to drop and use as columns later 
        cols = [ 'time_cos', 'participant','time']

        cols_df = Xtrain.columns.tolist()
        cols_df = cols_df[1:] + cols_df[:1]

        #normalizing geneActive sensor measurements 
        X_train = Xtrain.drop(cols, axis=1)
        y_train = Xtrain[cols].values

        X_val = Xval.drop(cols,  axis=1)
        y_val = Xval[cols].values

        X_test = Xtest.drop(cols,  axis=1)
        y_test = Xtest[cols].values

        scaler = StandardScaler()

        X_train = X_train.values.astype('float32')
        X_train = np.ma.array(X_train, mask=np.isnan(X_train))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        X_val = X_val.values.astype('float32')
        X_val = np.ma.array(X_val, mask=np.isnan(X_val))
        X_val = scaler.transform(X_val)

        X_test = X_test.values.astype('float32')
        X_test = np.ma.array(X_test, mask=np.isnan(X_test))
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        X_train = pd.concat([X_train, y_train], axis=1)
        X_train.columns = cols_df

        X_val = pd.DataFrame(X_val)
        y_val = pd.DataFrame(y_val)
        X_val = pd.concat([X_val, y_val], axis=1)
        X_val.columns = cols_df 

        X_test = pd.DataFrame(X_test)
        y_test = pd.DataFrame(y_test)
        X_test = pd.concat([X_test, y_test], axis=1)
        X_test.columns = cols_df 

        protocol_file = pd.read_csv('participants_infoData.csv')
        protocol_file = protocol_file.drop(['bmi', 'trial_date'], axis=1)

        encoder = LabelEncoder()
        protocol_file.sex = encoder.fit_transform(protocol_file.sex)

        protocol_columns = protocol_file.columns.tolist()
        protocol_columns =  protocol_columns[2:]+ protocol_columns[:-3]

        age_wt_ht = protocol_file.drop(protocol_columns[3:], axis=1).values
        part_sex = protocol_file.drop(protocol_columns[:-2], axis=1).values 

        scaler2 = StandardScaler()
        age_wt_ht = scaler2.fit_transform(age_wt_ht)

        age_wt_ht = pd.DataFrame(age_wt_ht)
        part_sex = pd.DataFrame(part_sex)
        protocol_file = pd.concat([age_wt_ht, part_sex], axis=1)
        protocol_file.columns = protocol_columns

        print('Done scaling data....')

        X_train, y_train, bmi_train = cls.sequence_builder(X_train, protocol_file, SEQUENCE_SIZE, downsample_rate)
        X_val, y_val, bmi_val = cls.sequence_builder(X_val, protocol_file, SEQUENCE_SIZE, downsample_rate)
        X_test, y_test, bmi_test = cls.sequence_builder(X_test, protocol_file, SEQUENCE_SIZE, downsample_rate)

        with open('/home/s1931628/seqEE/svm_standard_protocol/'+name+'.pkl','wb') as f:
            pickle.dump((X_train, y_train, bmi_train, X_val, y_val, bmi_val, X_test, y_test, bmi_test, scaler), f)
            
    def run_all(self): 
        for i in range(31,37):
            if i in [12, 19, 23, 16]: 
                continue
            else: 
                if len(str(i)) == 1:
                    name = 'GOTOV0'+str(i)
                else: 
                    name = 'GOTOV'+str(i)
            df_train, df_val, df_test = self.read_data(name)
            self.standardizing_data(name, df_train, df_val, df_test, self.SEQUENCE_SIZE, self.downsample_rate)


