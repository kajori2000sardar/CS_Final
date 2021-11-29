import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from LR import LinearRegression

#Loding the data
df = pd.read_csv('insurance.csv')

# cleaning
df.dropna(inplace=True)
df = df.drop(["region"], axis=1)
#df.head()

# changing sex and smoker value to binary
dummy1 = pd.get_dummies(df.sex)
df = pd.concat([df,dummy1], axis = 1)
df = df.drop(["sex","female"], axis = 1)
df = df.rename(columns = {"male":"sex-male"})
dummy2 = pd.get_dummies(df.smoker)
df = pd.concat([df,dummy2], axis = 1)
df = df.drop(["smoker","no"], axis = 1)
df = df.rename(columns={"yes":"smoker"})
#df.head()

#Preparing the data
x = np.array(df.drop("charges", axis=1))
y = np.array(df["charges"])

sc = StandardScaler()
x = sc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

regressor = LinearRegression(x_train, y_train)

#Training the model with .fit method
cost_history = regressor.fit(20000, 0.00001)

plt.clf()
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Itteration count")
plt.ylabel("Charge")
plt.savefig("charge_history.png")


#Prediciting the values
y_pred = regressor.predict(x_test)
y_train_pred = regressor.predict(x_train)

# Initial Coefficients

plt.clf()
plt.scatter(y_pred, y_test)
plt.scatter(y_train_pred, y_train,color='orange')
plt.xlabel('Predicted charges')
plt.ylabel('Actual charges')
plt.grid()
#plt.savefig("linear.png")