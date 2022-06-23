import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from annotated_text import annotated_text


start='2010-01-02'
end='2022-06-22'


st.set_page_config(page_title='Stock Trend forecaster')


st.title('Stock Trend forecaster')



user_input=st.text_input('Stock','TSLA')
df=pdr.DataReader(user_input,'yahoo',start,end)
df.head()


#Desctibing data

st.subheader('Data 2010-2022')
st.write(df.describe())

#VIZ

st.subheader('Historical closing price')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)





st.subheader('Closing price with 100MA&200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='100EMA')
plt.plot(ma200,'g',label='200EMA')
plt.plot(df.Close,'gray')
plt.legend()
st.pyplot(fig)

#splitiing data into traing and testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])



from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)




#loading the ML model

model=load_model('keras_model.h5')

#testing part
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)



x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
    
    
x_test,y_test=np.array(x_test),np.array(y_test)

y_p=model.predict(x_test)
scaler=scaler.scale_


scale_factor= 1/scaler[0]
y_predicted=y_p * scale_factor
y_test=y_test * scale_factor


st.subheader('Predicted vs Actual')  
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Actual Price')
plt.plot(y_predicted,'r',label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

    
    



    


