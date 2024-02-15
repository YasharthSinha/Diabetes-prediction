#libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score




diabetes_data= pd.read_csv("C:/Users/KIIT/python/DP/diabetes.csv")

a = diabetes_data.drop(columns = 'Outcome', axis=1)
b = diabetes_data['Outcome']

scaler= StandardScaler()
a=scaler.fit_transform(a)

a_train, a_test, b_train, b_test = train_test_split(a,b,test_size=0.2,stratify=b,random_state=2)

classif= svm.SVC(kernel='linear')
classif.fit(a_train, b_train)

a_train_prediction= classif.predict(a_train)
train_data_accuracy= accuracy_score(a_train_prediction, b_train)
print('Accuracy = ',train_data_accuracy) 

a_test_prediction= classif.predict(a_test)
test_data_accuracy= accuracy_score(a_test_prediction, b_test)
print('Accuracy = ',test_data_accuracy) 

input_data= (5,166,72,19,175,25.8,0.587,51)
input_data_npa = np.asarray(input_data)
input_data_reshaped = input_data_npa.reshape(1,-1)

std_data= scaler.fit_transform(input_data_reshaped)
predict= classif.predict(std_data)
if(predict[0]==0): print("Not Diabetic")
else  : print("Diabetic")