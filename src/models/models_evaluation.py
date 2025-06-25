import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def model_evaluation(model, X_test, y_test, model_name):
    """
    Evaluate the model using various metrics.

    Parameters:
    - model: The trained model to evaluate.
    - X_test: The test features.
    - y_test: The true labels for the test set.
    - model_name: Name of the model for logging purposes.

    Returns:
    - A dataframe containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    r2_adjusted = 1 - ((1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

    print(f"Evaluation results for {model_name}:")
    print("---------------------------------")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")
    print(f"Adjusted R-squared (Adj R2): {r2_adjusted}")

    return pd.DataFrame ([mae, mse, rmse, r2, r2_adjusted], index=['MAE', "MSE", "RMSE", "R2", "ADJUSTED R2"], columns = [model_name])