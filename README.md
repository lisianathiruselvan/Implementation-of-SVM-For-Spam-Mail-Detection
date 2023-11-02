# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: T.LISIANA
RegisterNumber:  212222240053
*/
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy  
```

## Output:

## Result:
![image](https://github.com/lisianathiruselvan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119389971/10c0483e-12ac-415f-a06b-3c8fe9538d00)


## Data.head():
![image](https://github.com/lisianathiruselvan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119389971/e3a88c04-bee2-461a-a56c-a8ded982cdb6)


## data.info():
![image](https://github.com/lisianathiruselvan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119389971/27c1dfb6-fa50-4180-be96-243da0e8e0ce)


## data.isnull().sum():
![image](https://github.com/lisianathiruselvan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119389971/cf1c0c0d-f9e4-4e00-a7fb-28c24e89746e)


## Y prediction value:
![image](https://github.com/lisianathiruselvan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119389971/cc19f6bf-af8e-4634-b8c2-bcd96ae686f1)


## Accuracy value:
![image](https://github.com/lisianathiruselvan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119389971/db5ad61a-7f4d-4fe0-b2ab-ae973254ae03)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
