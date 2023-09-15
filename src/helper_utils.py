import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error,
                             r2_score)

def project_metrics(y_true, y_pred):
    
    mad, hmse, hmae = heteroskedasticity_standard_errors(y_true, y_pred)
    
    losses = dict(
        # Mean absolute error, MAE
        mae = mean_absolute_error(y_true = y_true, y_pred = y_pred),
    
        # Mean squared error, MSE
        mse = mean_squared_error(y_true = y_true, y_pred = y_pred),
        
        # Root mean squared error, RMSE
        rmse = mean_squared_error(y_true = y_true, y_pred = y_pred, squared=False),
        
        # Mean absolute deviation, MAD
        mad = mad,
        
        # HMSE
        hmse = hmse,
        
        # HMAE
        hmae = hmae,
        
        # Mean absolute percentage error, MAPE
        mape = mean_absolute_percentage_error(y_true = y_true, y_pred = y_pred),
        
        # R2
        r2 = r2_score(y_true = y_true, y_pred = y_pred)
    
    )
    
    return losses


def heteroskedasticity_standard_errors(y_true, y_pred):
    
    # Estimate residuals
    residuals = y_true - y_pred
    
    # Estimate Mean absolute deviation
    mad = np.mean(np.abs(residuals))
    
    # Estimate heteroskedasticity-robust standard errors
    model = sm.OLS(y_true, sm.add_constant(y_pred))
    results = model.fit(cov_type='HC3')
    
    # Calculate HMSE
    hmse = np.sqrt(np.mean((residuals / np.sqrt(results.cov_HC3[0, 0])) ** 2))

    # Calculate HMAE
    hmae = np.mean(np.abs(residuals / np.sqrt(results.cov_HC3[0, 0])))
    
    return mad, hmse, hmae
