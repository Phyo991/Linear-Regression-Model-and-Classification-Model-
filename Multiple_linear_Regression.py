import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np


path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'

def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
    else:
        print(f"error")


download(path, "FuelConsumption.csv")
data = pd.read_csv("FuelConsumption.csv")
print(data)

column = data.keys()
print(column)
#select some feature that can use for prediction --> selection independent variables and also dependent variable
cdf = data[['ENGINESIZE',
            'CYLINDERS',
            'FUELCONSUMPTION_CITY',
            'FUELCONSUMPTION_HWY',
            'FUELCONSUMPTION_COMB',
            'CO2EMISSIONS']]

# then plot emission value base on independent variables:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.title('CO2emission base on Engine size')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

#split data for training and test
#by creating musk to select random rows using the np.random.rand()
msk = np.random.rand(len(data)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.title('Train Data')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

#fit line or  hyperplane  -->
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
#The coefficients
print('Coefficients: θ1 ', regr.coef_)
print('Intercept: θ0 ', regr.intercept_)

#Prediction
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))
print("Variance score: %.2f" % regr.score(x, y))


# Set up the figure and axis for 3D plot
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

# Assuming 'ENGINESIZE', 'FUELCONSUMPTION_COMB', and 'CO2EMISSIONS' are columns in your 'train' DataFrame.
x = train['ENGINESIZE']
y = train['FUELCONSUMPTION_COMB']
z = train['CO2EMISSIONS']
typical_cylinders = np.median(train['CYLINDERS'])
# Plotting the actual points
ax.scatter(x, y, z, color='red', label='Actual data', s=50)

# Creating a meshgrid for the plane with a finer resolution
x_surf, y_surf = np.meshgrid(np.linspace(x.min(), x.max(), 50),
                             np.linspace(y.min(), y.max(), 50))

z_surf = regr.predict(np.array([x_surf.ravel(), np.full(x_surf.size, typical_cylinders), y_surf.ravel()]).T).reshape(x_surf.shape)

# Plotting the hyperplane
surf = ax.plot_surface(x_surf, y_surf, z_surf, cmap='coolwarm', alpha=0.6, edgecolor='none')

# Labels and titles
ax.set_xlabel('Engine Size', fontsize=12)
ax.set_ylabel('Fuel Consumption Comb', fontsize=12)
ax.set_zlabel('CO2 Emissions', fontsize=12)
plt.title('3D Plot with Multiple Regression Hyperplane', fontsize=15)

# Adding a color bar
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()