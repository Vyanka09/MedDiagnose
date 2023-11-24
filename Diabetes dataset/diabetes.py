import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import random

df= pd.read_csv('D:\Major Project\Healthcure\Diabetes dataset\diabetes.csv')


print(df.head())
print(df.tail())

print(df.describe())

print(df.info())

df_no = df[df['Outcome']==0]
df_yes = df[df['Outcome']==1]
print(df_no)


# concat
df = pd.concat([df_no, df_yes])
df = df.copy(deep = True)
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

print(df.isnull().sum())

df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)

print(df['Outcome'].value_counts())

sns.countplot(df['Outcome']) 
plt.show()
sns.distplot(df['Age'])
plt.show()
plt.figure(figsize=(10,5)) 
sns.heatmap(df.corr(),annot=True,cmap='rainbow')
plt.show()

print(df.groupby('Outcome').mean())
x= df.drop(columns ='Outcome', axis=1) 
y =df['Outcome']
print(x)
print(y)
random.seed(42)
x_train ,x_test , y_train, y_test= train_test_split(x,y,test_size=0.3,stratify=y,random_state=1) 

print(x.shape,x_train.shape,x_test.shape)
# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 42)
ranfor.fit(x_train, y_train)

predictiontraining = ranfor.predict(x_test) 
accuracytraining = accuracy_score(predictiontraining,y_test) 

predictiontest= ranfor.predict(x_test) 
accuracytesting = accuracy_score(predictiontest,y_test)


from sklearn.metrics import confusion_matrix 
import seaborn as sns
sns.heatmap(confusion_matrix(y_test,predictiontest),annot=True,fmt="d") 
print(confusion_matrix(y_test,predictiontest)) 
plt.title("Diabetes prediction",fontSize=14) 
plt.show()

TP= confusion_matrix(y_test,predictiontest) [0,0] 
FN = confusion_matrix(y_test,predictiontest) [0,1] 
FP = confusion_matrix(y_test,predictiontest) [1,0] 
TN = confusion_matrix(y_test,predictiontest) [1,1] 
accuracy = (TP+TN)/(TP+FP+FN+TN) 
precision = TP/(TP+FP) 
recall = TP/(TP+FN) 
fscore= 2* precision* recall/(precision+recall)
print("Accuracy on testing dataset: ",accuracy)
print("Precision on testing dataset: ",precision) 
print("Recall on testing dataset: ",recall) 
print("f-score on testing dataset ",fscore)

import pickle
filename = 'diabetes.sav'
pickle.dump(ranfor, open(filename, 'wb'))