from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVR

from Further_Data_Preparation_and_Segregation import x_train, x_test, y_train, y_test

# Model Training
regressor = LinearRegression()
model = regressor.fit(x_train, y_train)

# Model Prediction
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)