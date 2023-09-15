import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error,
                             r2_score)

def project_metrics(y_true, y_pred):
    
    mad, hmse, hmae, aic, bic, r2 = heteroskedasticity_standard_errors(y_true, y_pred)
    
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
        
        r2_hetro = r2,
        
        # AIC
        aic = aic,
        
        # BIC
        bic = bic,
        
        # R2
        r2 = r2_score(y_true = y_true, y_pred = y_pred),
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
    
    # aic
    aic = results.aic

    # bic
    bic = results.bic
    
    # r-squared
    r2 = results.rsquared
    
    return mad, hmse, hmae, aic, bic, r2

### PREPARE CONFIGURATION FOR TRAINING
def prepare_configuration(specs):
    from helper_config import Config
    
    config = Config()
    config.fold = -1
    config.data_dir = specs.get('data_dir', '../inputs')
    config.models_dir = specs.get('models_dir', '../models')
    
    config.model_name = specs['model_name']
    config.learning_rate = 1e-5
    config.train_batch_size = 8
    config.num_epochs = 1
    config.early_stop_count = 10
    config.save_checkpoint = True
    
    config.experiment_config['model_type'] = specs['model_type']
    config.experiment_config['use_commodity_prices'] = specs['use_commodity_prices']
    config.experiment_config['window'] = specs['window']
    
    config.apply_seed(seed = 3407)
    
    return config