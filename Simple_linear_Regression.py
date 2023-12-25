import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import linear_model
from sklearn.metrics import r2_score
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
    else:
        print(f"Fail")


download(path, "FuelConsumption.csv")
df = pd.read_csv("FuelConsumption.csv")
print(df)

#function that generate summarization of data
print(df.describe())

#select some features to explore more
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(9))

#plot each of these features:
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
#use histogram to check whether continuous data or discrete data
print(viz.hist())
(plt.show())

# plotting each of these feature against the emission to see how linear their relationship is
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
plt.show()

#Creating train and test dataset
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
print(train)
print(test)

#Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Modeling using sklearn package to model data
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
print('Coefficients: θ1 ', regr.coef_)
print('Intercept: θ0 ', regr.intercept_)

#plotting fit line over the data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='red')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], 'b')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

#Evaluation
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error:  %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of square (MSE): %.2f" % np.mean((test_y_ -test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))