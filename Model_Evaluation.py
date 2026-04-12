import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from Further_Data_Preparation_and_Segregation import x, y, x_train, x_test, y_train, y_test
from Model_Building import y_pred, y_pred1, regressor

# Model Evaluation
training_r2 = r2_score(y_train, y_pred)
training_rmse = np.sqrt(mean_squared_error(y_train, y_pred))

test_r2 = r2_score(y_test, y_pred1)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred1))

        # Validation 
cv_mean = cross_val_score(estimator = regressor, X = x, y = y, cv = 3)

print(f"Training R2: {training_r2}")
print(f"Training RMSE: {training_rmse}")
print(f"Test R2: {test_r2}")
print(f"Test RMSE: {test_rmse}")
print(f"CV Mean: {cv_mean.mean()} \n CV Scores: {cv_mean}")