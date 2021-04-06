import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('data.csv')
df.head(7)
df['diagnosis'].value_counts()

#visualize
sns.countplot(df['diagnosis'], label = 'count')

df.dtypes

#Encode the data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)


#pair plot
sns.pairplot(df.iloc[:,1:5], hue = 'diagnosis')

#first 5
df.head(5)

#Splitting the data set
X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values

#Splitting the data set into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = X_test.reshape(-1, 1)
X_test = sc.fit_transform(X_test)

#Creating a function for the models
def models(X_train, Y_train):
  #Logistic Regression
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)

  #Decision Tree
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  #Random Forest Classifier
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)

  #Accuracy of models
  print('[0]Logistic Regression Accuracy:', log.score(X_train, Y_train))
  print('[1]Decision Tree Classifier Accuracy:', tree.score(X_train, Y_train))
  print('[2]Random Forest Classifier Accuracy:', forest.score(X_train, Y_train))

  return log, tree, forest

model = models(X_train, Y_train)

#Results
pred = model[2].predict(X_test)
print(pred)
print()
print(Y_test)