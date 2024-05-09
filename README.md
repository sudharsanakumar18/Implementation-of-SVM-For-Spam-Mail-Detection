# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import neccessary libraries required.
2. Load the dataset using pd.read_csv.
3. Use CountVectorizer to convert text data into a matrix of token counts.
4. Create an SVM model with a linear kernel.
5. print the accuracy and classification report.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Purajith S
RegisterNumber: 212223040158
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as t
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
df=pd.read_csv("/content/spam.csv",encoding='ISO-8859-1')
df.head()

vectorizer=CountVectorizer()
x=vectorizer.fit_transform(df['v2'])
y=df['v1']
x_train,x_test,y_train,y_test=t(x,y,test_size=0.25,random_state=42)
model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)
predictions=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,predictions))
print("Classification report:")
print(classification_report(y_test,predictions))
```

## Output:
i.) Dataset:

![image](https://github.com/shoaib3136/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117919362/77a5d64f-0501-41e3-bcea-f72aa1bc93e4)

ii.) Accuracy and Classification report:

![image](https://github.com/shoaib3136/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117919362/cf83afc3-3982-430e-8939-78791e2a0cb2)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
