import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

dataset = pd.read_csv("D:\Major Project\Healthcure\Heart disease dataset\heart.csv")

print(dataset.shape)

print(dataset.head(5))
print(dataset.describe())

print(dataset.info())

info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])

print(dataset.corr()["target"].abs().sort_values(ascending=False))

from sklearn.model_selection import train_test_split

predictors = dataset.drop(["target","trestbps","chol","fbs","restecg","sex","slope"],axis=1)
target = dataset["target"]
random.seed(4242)
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
from sklearn.metrics import accuracy_score



import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,learning_rate=0.1,max_iterations=10)
xgb_model.fit(X_train, Y_train)

Y_pred_xgb = xgb_model.predict(X_test)
print(Y_pred_xgb.shape)

score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)
print(score_xgb)

from sklearn.metrics import confusion_matrix 
import seaborn as sns

import matplotlib.pyplot as plt
predictiontest= xgb_model.predict(X_test)
sns.heatmap(confusion_matrix(Y_test,predictiontest),annot=True,fmt="d") 
plt.title("Heart Disease prediction",fontSize=14) 
plt.show()
import pickle
filename = 'heart_disease_pickle.dat'
pickle.dump(xgb_model, open(filename, 'wb'))