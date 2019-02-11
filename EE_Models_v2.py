from keras.layers import LSTM,Dropout, Activation, GRU, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.regularizers import l2, l1



class Model1: 
    
    def __init__(self, 
                 name, 
                 X_train,
                 y_train,
                 bmi_train,
                 X_val, y_val, 
                 bmi_val,
                 n_batch = 32, n_epochs=100): 
        self.n_batch = n_batch
        self.X_train = X_train
        self.bmi_train = bmi_train
        self.bmi_val = bmi_val
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.name = name 
        self.n_epochs = n_epochs

        
    def try_1(self):
        print('Creating functional api model')
        
        #input1 accelerometer sensor measurements 
        input_1 = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        hidden_1 = GRU(32, return_sequences=True, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(input_1)
        hidden_2 = GRU(256, return_sequences=True, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(hidden_1)
        hidden_3 = GRU(32, return_sequences=False, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(hidden_2)
        hidden_4 = Dense(32, activation='relu')(hidden_3)
        
        #input2 BMI of participants 
        input_2 = Input(shape=(self.bmi_train.shape[1],))
        hidden_8 = Dense(32)(input_2)
        
        #merge
        con = concatenate([hidden_4, hidden_8])
        x3 = Dense(32, activation='relu')(con)
        x3 = Dropout(0.2)(x3) #another change
        x3 = Dense(16, activation='relu')(x3) #changes made
        x3 = Dropout(0.2)(x3)
        output = Dense(1, activation='linear')(x3)
    
        model = Model(inputs=[input_1, input_2], outputs=output)
        plot_model(model, to_file='T2_fuctional_api.pdf')
        
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/EE/'+self.name+'/GRUbatchsize256', histogram_freq=0, write_graph=True, write_images=True)
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/EE_model/'+ self.name+'_.hdf5', 
                                       save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        history = model.fit(x=[self.X_train, self.bmi_train], y=self.y_train, batch_size=self.n_batch, epochs=self.n_epochs, verbose=1, validation_data=([self.X_val, self.bmi_val], self.y_val), shuffle=True, callbacks=[checkpointer, tbCallBack])
        return history
   
    
    def try_2(self):
        print('Creating functional api model')
        
        #input1 accelerometer sensor measurements 
        input_1 = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        hidden_1 = GRU(32, return_sequences=True, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(input_1)
        hidden_2 = GRU(256, return_sequences=True, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(hidden_1)
        hidden_3 = GRU(32, return_sequences=False, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(hidden_2)
        hidden_4 = Dense(32, activation='relu')(hidden_3)
        
        #input2 BMI of participants 
        input_2 = Input(shape=(self.bmi_train.shape[1],))
        hidden_8 = Dense(32)(input_2)
        
        #merge
        con = concatenate([hidden_4, hidden_8])
        x3 = Dense(32, activation='relu')(con)
        x3 = Dropout(0.2)(x3) #another change
        x3 = Dense(16, activation='relu')(x3) #changes made
        x3 = Dropout(0.2)(x3)
        output = Dense(1, activation='linear')(x3)
    
        model = Model(inputs=[input_1, input_2], outputs=output)
        plot_model(model, to_file='T2_fuctional_api.pdf')
        
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/EE/'+self.name+'/GRUbatchsize256', histogram_freq=0, write_graph=True, write_images=True)
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/EE_model/'+ self.name+'_.hdf5', 
                                       save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        history = model.fit(x=[self.X_train, self.bmi_train], y=self.y_train, batch_size=self.n_batch, epochs=self.n_epochs, verbose=1, validation_data=([self.X_val, self.bmi_val], self.y_val), shuffle=True, callbacks=[checkpointer, tbCallBack])
        return history
   
    def try_3(self):
        print('Creating functional api model')
        
        #input1 accelerometer sensor measurements 
        input_1 = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        hidden_1 = GRU(32, return_sequences=True, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(input_1)
        hidden_2 = GRU(256, return_sequences=True, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(hidden_1)
        hidden_3 = GRU(32, return_sequences=False, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(hidden_2)
        hidden_4 = Dense(32, activation='relu')(hidden_3)
        
        #input2 BMI of participants 
        input_2 = Input(shape=(self.bmi_train.shape[1],))
        hidden_8 = Dense(32)(input_2)
        
        #merge
        con = concatenate([hidden_4, hidden_8])
        x3 = Dense(32, activation='relu')(con)
        x3 = Dropout(0.2)(x3) #another change
        x3 = Dense(16, activation='relu')(x3) #changes made
        x3 = Dropout(0.2)(x3)
        output = Dense(1, activation='linear')(x3)
    
        model = Model(inputs=[input_1, input_2], outputs=output)
        plot_model(model, to_file='T2_fuctional_api.pdf')
        
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/EE/'+self.name+'/GRUbatchsize256', histogram_freq=0, write_graph=True, write_images=True)
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/EE_model/'+ self.name+'_.hdf5', 
                                       save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        history = model.fit(x=[self.X_train, self.bmi_train], y=self.y_train, batch_size=self.n_batch, epochs=self.n_epochs, verbose=1, validation_data=([self.X_val, self.bmi_val], self.y_val), shuffle=True, callbacks=[checkpointer, tbCallBack])
        return history
    
    def try_4(self):
        print('Creating functional api model')
        
        #input1 accelerometer sensor measurements 
        input_1 = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        hidden_1 = GRU(32, return_sequences=True, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(input_1)
        hidden_2 = GRU(256, return_sequences=True, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(hidden_1)
        hidden_3 = GRU(32, return_sequences=False, kernel_regularizer=l2(l=0.0001), recurrent_dropout=0.5)(hidden_2)
        hidden_4 = Dense(32, activation='relu')(hidden_3)
        
        #input2 BMI of participants 
        input_2 = Input(shape=(self.bmi_train.shape[1],))
        hidden_8 = Dense(32)(input_2)
        
        #merge
        con = concatenate([hidden_4, hidden_8])
        x3 = Dense(32, activation='relu')(con)
        x3 = Dropout(0.2)(x3) #another change
        x3 = Dense(16, activation='relu')(x3) #changes made
        x3 = Dropout(0.2)(x3)
        output = Dense(1, activation='linear')(x3)
    
        model = Model(inputs=[input_1, input_2], outputs=output)
        plot_model(model, to_file='T2_fuctional_api.pdf')
        
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/EE/'+self.name+'/GRUbatchsize256', histogram_freq=0, write_graph=True, write_images=True)
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/EE_model/'+ self.name+'_.hdf5', 
                                       save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        history = model.fit(x=[self.X_train, self.bmi_train], y=self.y_train, batch_size=self.n_batch, epochs=self.n_epochs, verbose=1, validation_data=([self.X_val, self.bmi_val], self.y_val), shuffle=True, callbacks=[checkpointer, tbCallBack])
        return history

    
   