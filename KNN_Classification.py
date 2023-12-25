import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import requests
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
    else:
        print(f"error")


path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"

download(path, "teleCust100t.csv")
data = pd.read_csv('teleCust100t.csv')
print(data)

#see how many of each class in the dataset
class_data = data['custcat'].value_counts()
print(class_data)

#we can explore data by visualization techniques:
data.hist(column='income', bins=50)
plt.show()

#define feature sets
#To use scikit-learn library, have to convert the Pandas data frame to Numpy array
X = data[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values

#for labels
y = data[['custcat']].values
print(y)

#Normalize data
#to ensure that all features contribute equally to the result
# gradient descent-based algorithms to perform efficiently
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
y_train = y_train.ravel()
y_test = y_test.ravel()
print('Train set: ', X_train.shape, y_train.shape)
print('Test set: ', X_test.shape, y_test.shape)

#Model
k = 4
#Train the model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(neigh)

#Predict
yhat = neigh.predict(X_test)
print(yhat)

# Model Prediction
yhat = neigh.predict(X_test)
print(yhat)

#Accuracy evaluation
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#predict K value by reserving a part of data for testing the accuracy of the model
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)


plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


#Model
k = 9
#Train the model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(neigh)

#Predict
yhat = neigh.predict(X_test)
print(yhat)

# Model Prediction
yhat = neigh.predict(X_test)
print(yhat)

#Accuracy evaluation
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
