import os
import random
import pandas as pd
import numpy as np
import pickle
from pyts.image import GramianAngularField
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

try:
    import talib
except:
    pass

import warnings
warnings.filterwarnings('ignore')

### To estimate parameters
from helper_models import GarchModel

### Prepare Dataset
class Preprocess:
    
    def __init__(self, df, config, asset_index_name):
        self.df = df
        self.config = config
        self.dest_path = os.path.join(self.config.data_dir, 'prep_data', f'window_{self.config.experiment_config["window"]}', asset_index_name)
        os.makedirs(self.dest_path, exist_ok=True)
        
    def estimate_log_returns(self):
        self.df.rename(columns = {'Close' : 'Price'}, inplace = True)
        self.df.dropna(subset = ['Price'], how = 'any', axis = 0, inplace = True)
        self.df['Price'] = self.df['Price'].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) else float(x))
        
        # convert string date to pandas datetime object to sort in the ascending order
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values(by = 'Date', ascending = True, inplace = True)
        self.df.reset_index(inplace = True, drop = True)        # drop index for tracking
        
        #self.df = self.df.head(60)
        
        # estimate log-returns and mu
        self.log_returns = np.log(self.df['Price'] / self.df['Price'].shift(1)).dropna()
        
    def calculate_rv(self, log_returns):
        
        log_returns = np.array(log_returns)
        
        # average log returns
        avg= np.mean(log_returns)
        
        # variance
        variance = np.array([(r - avg)**2 for r in log_returns])
        
        # realized volatility
        rv = np.sqrt(variance)

        return rv
    
    def estimate_realized_volatility(self):
        window      = self.config.experiment_config['window']
        step_size   = self.config.step_size
        rv_list     = []
        for i in range(len(self.log_returns), -1, -step_size):
            sub_seq = self.log_returns[i-window : i].values.tolist()
            if len(sub_seq) == window: 
                rv_list.append([i, self.calculate_rv(sub_seq)])      
        
        return rv_list
        
    def estimate_garch_parameters(self):
        self.garch_model = GarchModel(log_returns=self.log_returns, config=self.config)
        
        # Estimate garch, egarch, and gjr-garch parameters
        # The returns values are list of parameters with each item [sequence_index, omega, alpha, beta, gamma]
        garch_params        = self.garch_model.garch_params()
        egarch_params       = self.garch_model.egarch_params()
        gjr_garch_params    = self.garch_model.gjr_garch_params()
        
        return garch_params, egarch_params, gjr_garch_params
    
    def combine(self):
        
        # log returns of a stock / index
        log_returns = self.log_returns
        log_rtn_df = pd.DataFrame(log_returns).reset_index()
        log_rtn_df.columns = ['index', 'log_returns']
        
        # realized volatlity        (target)
        rv_list = self.estimate_realized_volatility()
        rv_df = pd.DataFrame(rv_list, columns = ['index', 'realized_volatility'])
        
        # garch, egarch, and gjr_garch_params
        garch_params, egarch_params, gjr_garch_params = self.estimate_garch_parameters()
        garch_df        = pd.DataFrame(garch_params , columns = ['index', 'garch_omega', 'garch_alpha', 'garch_beta', 'garch_gamma'])
        egarch_df       = pd.DataFrame(egarch_params, columns = ['index', 'egarch_omega', 'egarch_alpha', 'egarch_beta', 'egarch_gamma'])
        gjr_garch_df    = pd.DataFrame(gjr_garch_params, columns = ['index', 'gjr_garch_omega', 'gjr_garch_alpha', 'gjr_garch_beta', 'gjr_garch_gamma'])
        
        # merge
        final_df = log_rtn_df.merge(rv_df, on = 'index', how = 'inner')
        final_df = final_df.merge(garch_df, on = 'index', how = 'inner')
        final_df = final_df.merge(egarch_df, on = 'index', how = 'inner')
        final_df = final_df.merge(gjr_garch_df, on = 'index', how = 'inner')
        
        # normalize
        norm_final_df = self.normalize(final_df.copy())
        
        # append original data
        final_df.set_index('index', inplace = True)
        final_df = final_df.merge(self.df, right_index = True, left_index = True)
        
        norm_final_df.set_index('index', inplace = True)
        norm_final_df = norm_final_df.merge(self.df, right_index = True, left_index = True)
        
        # save
        final_df.to_csv(os.path.join(self.dest_path, 'final_data.csv'))
        norm_final_df.to_csv(os.path.join(self.dest_path, f'norm_data.csv'))
        
        return final_df
    
    def normalize(self, df):
        
        scaler = StandardScaler()
        features = [c for c in df.columns if c not in ['index', 'realized_volatility']]
        df[features] = scaler.fit_transform(df[features])
        
        tgt_scaler = StandardScaler()
        df['realized_volatility'] = tgt_scaler.fit_transform(df['realized_volatility'].values.reshape(-1,1))
        
        # save
        self.save_scalers(features_scaler=scaler, target_scaler=tgt_scaler)
        
        return df
    
    def save_scalers(self, features_scaler, target_scaler):
        with open(os.path.join(self.dest_path, 'features_scaler.pkl'), 'wb') as f:
            pickle.dump(features_scaler, f)
            
        with open(os.path.join(self.dest_path, 'target_scaler.pkl'), 'wb') as f:
            pickle.dump(target_scaler, f)
            
            
### DATASET
class MyDataset(Dataset):
    def __init__(self, df, com_df, config):
        self.df = df
        print(df.shape)
        self.com_df = com_df        # commodity_prices
        self.config = config
        self.exp_config = self.config.experiment_config
        # prepare dataset
        self.data = self.prepare_data()
    
    def prepare_data(self):
        
        # Experiment configuration
        use_commodity_prices = self.exp_config['use_commodity_prices']
        hybrid_model         = self.exp_config['hybrid_model']
        model_type           = self.exp_config['model_type'][0]
        garch_models         = self.exp_config['model_type'][1:] if hybrid_model else ''
        window               = self.exp_config['window']
        step_size            = self.config.step_size
        n_forward            = 1
        
        # list of features
        features = ['log_returns']
        
        # target
        target = 'realized_volatility'
        
        # append commodity prices
        if use_commodity_prices:
            self.df = self.df.merge(self.com_df, on = 'Date', how = 'inner')
            features += ['log_returns_oil', 'log_returns_gold']
            
        # Append garch model features
        if 'garch' in garch_models:
            features += ['garch_omega', 'garch_alpha', 'garch_beta']
            
        if 'egarch' in garch_models:
            features += ['egarch_omega', 'egarch_alpha', 'egarch_beta', 'egarch_gamma']
            
        if 'gjr_garch' in garch_models:
            features += ['gjr_garch_omega', 'gjr_garch_alpha', 'gjr_garch_beta', 'gjr_garch_gamma']
            
        ## sort by date
        self.df.sort_values(by = 'Date', ascending = False)
        
        data = []
        for i in range(len(self.df)-window-n_forward, -1, -step_size):
            patch = self.df[features][i:i+window]
            if len(patch) == window:
                data.append({
                    'date'    : str(self.df['Date'].values[i + window]),
                    'inputs'  : self.df[features][i:i+window].values,
                    'target'  : self.df[target].values[i + window],
                    })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        sample = self.data[idx]
        
        date = sample['date']
        inputs = torch.tensor(sample['inputs'], dtype=torch.float32)
        target = torch.tensor(sample['target'], dtype=torch.float32)
        
        return (date, inputs, target)
    
    
def get_dataloaders(config, fold = 1):
    
    ## Experiment details
    mkt_index = config.experiment_config['index']
    window    = config.experiment_config['window']
    use_commodity_prices = config.experiment_config['use_commodity_prices']
    
    # folds
    folds = config.folds[fold]
    (train_start, train_end) = folds['train']
    (valid_start, valid_end) = folds['valid']
    
    # read index data
    df = pd.read_csv(os.path.join(config.data_dir, 'prep_data', f'window_{window}', mkt_index, 'norm_data.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by = 'Date', ascending = False, inplace = True)
    
    # split train and validation datasets
    train = df[(df['Date'] >= train_start) & (df['Date'] < train_end)]
    valid = df[(df['Date'] >= valid_start) & (df['Date'] < valid_end)]
    
    com_train, com_valid = pd.DataFrame(), pd.DataFrame()
    if use_commodity_prices:
        # read crude oil and gold data
        oil = pd.read_csv(os.path.join(config.data_dir, 'prep_data', f'window_{window}', 'Crude_Oil_WTI Futures', 'norm_data.csv'))
        gold = pd.read_csv(os.path.join(config.data_dir, 'prep_data', f'window_{window}', 'Gold_Futures', 'norm_data.csv'))
        
        oil = oil[['Date', 'log_returns']]
        gold = gold[['Date', 'log_returns']]
        
        com_df = oil.merge(gold, on = 'Date', suffixes = ['_oil', '_gold'])
        com_df['Date'] = pd.to_datetime(com_df['Date'])
        com_df.sort_values(by = 'Date', ascending = False, inplace = True)
        
        # split train and validation datasets
        com_train = com_df[(com_df['Date'] >= train_start) & (com_df['Date'] < train_end)]
        com_valid = com_df[(com_df['Date'] >= valid_start) & (com_df['Date'] < valid_end)]
    
    # dataloaders
    train_loader = DataLoader(MyDataset(df = train, com_df = com_train, config = config),
                              batch_size    = config.train_batch_size,
                              shuffle       = True,
                              drop_last     = False)
    
    valid_loader = DataLoader(MyDataset(df = valid, com_df = com_valid, config = config),
                              batch_size    = config.valid_batch_size,
                              shuffle       = False,
                              drop_last     = False)
    
    print(len(train_loader.dataset), len(valid_loader.dataset))
        
    return train, train_loader, valid_loader, None
        
        
if __name__ == '__main__':
    from helper_config import Config
    config = Config()
    config.data_dir = 'inputs'
    
    file = 'ATX.csv'
    df = pd.read_csv(f'inputs/data/{file}')
    p = Preprocess(df, config, asset_index_name=file.replace('.csv', ''))
    p.estimate_log_returns()
    final_df = p.combine()
    print(final_df.head())
    
    # config.experiment_config = dict(
    #     index                   = 'ATX',
    #     use_commodity_prices    = False,
    #     hybrid_model            = False,
    #     model_type              = ['LSTM'],
    #     window                  = 14
    # )
    
    # config.fold = 1
    # _ = get_dataloaders(config, fold = 1)