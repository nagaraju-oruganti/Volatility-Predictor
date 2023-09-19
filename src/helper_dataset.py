import os
import random
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        
        # estimate log-returns and mu
        self.log_returns = np.log(self.df['Price'] / self.df['Price'].shift(1)).dropna()
        
    def calculate_rv(self, log_returns):
        
        log_returns = np.array(log_returns)
        
        # average log returns
        avg= np.mean(log_returns)
        
        # variance
        variance = np.array([r**2 for r in log_returns])        # 100 x mean(log_rtn - mean(log_rn))^2
        
        # realized volatility
        rv = np.sqrt(np.mean(variance) * 252)       # annualized

        return rv
    
    def estimate_realized_volatility(self):
        window      = self.config.experiment_config['window']
        step_size   = self.config.step_size
        rv_list     = []
        for i in range(len(self.log_returns), -1, -step_size):
            sub_seq = self.log_returns[i: i + window].values.tolist()
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
        
        def __make_dataset(series, name):
            d = pd.DataFrame(series).reset_index()
            d.columns = ['index', name]
            return d
            
        # log returns of a stock / index
        log_rtn_df = __make_dataset(self.log_returns, name = 'log_returns')
        log_rtn_df['log_returns'] = (log_rtn_df['log_returns']) ** 2              # 100 x log_rtn^2
        
        # realized volatlity        (target)
        rv_list = self.estimate_realized_volatility()
        rv_df = pd.DataFrame(rv_list, columns = ['index', 'realized_volatility'])
        
        # garch, egarch, and gjr_garch_params
        garch_params, egarch_params, gjr_garch_params = self.estimate_garch_parameters()
        garch_df        = __make_dataset(garch_params, name = 'garch_volatility')
        egarch_df       = __make_dataset(egarch_params, name = 'egarch_volatility')
        gjr_garch_df    = __make_dataset(gjr_garch_params, name = 'gjr_garch_volatility')
        
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
        
        scaler = StandardScaler()   #MinMaxScaler(feature_range=(-1, 1))
        features = [c for c in df.columns if c not in ['index', 'realized_volatility']]
        df[features] = scaler.fit_transform(df[features])
        
        tgt_scaler = StandardScaler()   #MinMaxScaler(feature_range=(-1, 1))
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
        self.com_df = com_df        # commodity_prices
        self.config = config
        self.exp_config = self.config.experiment_config
        # prepare dataset
        self.data = self.prepare_data()
    
    def prepare_data(self):
        
        # Experiment configuration
        use_commodity_prices = self.exp_config['use_commodity_prices']
        hybrid_config        = self.exp_config['model_type']
        window               = self.exp_config['window']
        step_size            = self.config.step_size
        n_forward            = 1
        
        # list of features
        features = ['log_returns', 'realized_volatility']
        
        # target
        target = 'realized_volatility'
        
        # append commodity prices
        if use_commodity_prices:
            self.df = self.df.merge(self.com_df, on = 'Date', how = 'inner')
            features += ['log_returns_oil', 'log_returns_gold']
            
        # Append garch model features
        if 'garch'      in hybrid_config: features.append('garch_volatility')
        if 'egarch'     in hybrid_config: features.append('egarch_volatility')
        if 'gjr_garch'  in hybrid_config: features.append('gjr_garch_volatility')
        
        ## sort by date
        self.df.sort_values(by = 'Date', ascending = False, inplace = True)
        
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

def prepare_dataset(config, window, mkt_index):
    
    filepath = os.path.join(config.data_dir, 'prep_data', f'window_{window}', mkt_index, 'final_data.csv')
    if not os.path.exists(filepath):
        # prepare
        df = pd.read_csv(f'{config.data_dir}/data/{mkt_index}.csv')
        p = Preprocess(df, config, asset_index_name=mkt_index)
        p.estimate_log_returns()
        _ = p.combine()
    
def get_dataloaders(config, fold = 1):
    
    ## Experiment details
    mkt_index = config.experiment_config['index']
    window    = config.experiment_config['window']
    use_commodity_prices = config.experiment_config['use_commodity_prices']
    
    ## Prepare dataset
    prepare_dataset(config, window, mkt_index)
    prepare_dataset(config, window, 'Gold_Futures')
    prepare_dataset(config, window, 'Crude_Oil_WTI Futures')
    
    # folds
    folds = config.splits if fold == -1 else config.folds[fold]         # if fold is -1 then we are training on the entire dataset minus test samples
    (train_start, train_end) = folds['train']
    (valid_start, valid_end) = folds['valid']
    
    # read index data
    df = pd.read_csv(os.path.join(config.data_dir, 'prep_data', f'window_{window}', mkt_index, 'final_data.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by = 'Date', ascending = False, inplace = True)
    
    # split train and validation datasets
    train = df[(df['Date'] >= train_start) & (df['Date'] < train_end)]
    valid = df[(df['Date'] >= valid_start) & (df['Date'] < valid_end)]
    
    com_train, com_valid = pd.DataFrame(), pd.DataFrame()
    if use_commodity_prices:
        # read crude oil and gold data
        oil = pd.read_csv(os.path.join(config.data_dir, 'prep_data', f'window_{window}', 'Crude_Oil_WTI Futures', 'final_data.csv'))
        gold = pd.read_csv(os.path.join(config.data_dir, 'prep_data', f'window_{window}', 'Gold_Futures', 'final_data.csv'))
        
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
    
    print(f'train samples: {len(train_loader.dataset)}')
    print(f'valid samples: {len(valid_loader.dataset)}')
        
    return train, train_loader, valid_loader, None