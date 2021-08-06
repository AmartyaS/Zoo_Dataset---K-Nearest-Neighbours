# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 01:40:22 2021

@author: ASUS
"""
#Importing all the required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC

#Loading up the data file
file=pd.read_csv("D:\Data Science Assignments\R-Assignment\KNN_Classifier\Zoo.csv")
file.head()
file.shape

#Data Manipulation
def norm(i):  #Normalising the values in the dataframe
    x=((i-i.min())/(i.max()-i.min()))
    return x
data=norm(file.iloc[:,1:17])
data['type']=file.type
data.head()

#Splitting the data file into training  and testing data 
train,test=train_test_split(data,test_size=0.3)
train.type.value_counts() #Checking the quality of the data distribution
test.type.value_counts()

#Checking the model with k-value=4
n1=KNC(n_neighbors=4)
n1.fit(train.iloc[:,0:16],train.loc[:,'type'])
train_acc1=np.mean(n1.predict(train.iloc[:,0:16])==train.loc[:,'type'])
test_acc1=np.mean(n1.predict(test.iloc[:,0:16])==test.loc[:,'type'])
test_acc1        #Testing Accuracy
train_acc1      #Training Accuracy

#Checking the model with k-value=5
n2=KNC(n_neighbors=5)
n2.fit(train.iloc[:,0:16],train.loc[:,'type'])
train_acc2=np.mean(n2.predict(train.iloc[:,0:16])==train.loc[:,'type'])
test_acc2=np.mean(n2.predict(test.iloc[:,0:16])==test.loc[:,'type'])
train_acc2      #Training Accuracy
test_acc2       #Testing Accuracy

#Testing the model with a range of k-values
acc=[]
for i in range(1,20):
    n=KNC(n_neighbors=i)
    n.fit(train.iloc[:,0:16],train.loc[:,'type'])
    train_acc=np.mean(n.predict(train.iloc[:,0:16])==train.loc[:,'type'])
    test_acc=np.mean(n.predict(test.iloc[:,0:16])==test.loc[:,'type'])
    acc.append([i,train_acc,test_acc])
accuracy=pd.DataFrame(acc)
accuracy.shape
accuracy.columns=["I_Values","Training_Accuracy","Testing_Accuracy"]
print(accuracy)


#Plotting Training Accuracy Values
plt.plot(range(1,20),accuracy.Training_Accuracy,"ro-",scalex=True)
plt.xlabel("KNeighbour Values")
plt.ylabel("Accuracy")
plt.title("Accuracy of model with Training Dataset ")


#Plotting the Testing accuracy values
plt.plot(range(1,20),accuracy.Testing_Accuracy,"go-",scalex=True)
plt.xlabel("KNeighbour Values")
plt.ylabel("Accuracy")
plt.title("Accuracy of model with Testing Dataset ")
