import random, os
import numpy as np
import torch

### SEED EVERYTHING
def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    print(f'set seed to {seed}')

class Config:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### repo
    data_dir = '../data'
    models_dir = '../models'
    
    ### Train and test splits
    splits = {
        'train': ('01-01-2013', '12-31-2020'),
        'test' : ('01-01-2021', '12-31-2022')
    }
    
    folds = {
        1 : {
            'train': ('01-01-2013', '12-31-2020'),
            'valid': ('01-01-2021', '12-31-2022')
            },
        2 : {
            'train': ('01-01-2013', '12-31-2017'),
            'valid': ('01-01-2018', '12-31-2018')
            },
        3 : {
            'train': ('01-01-2013', '12-31-2018'),
            'valid': ('01-01-2019', '12-31-2019')
            },
        3 : {
            'train': ('01-01-2013', '12-31-2019'),
            'valid': ('01-01-2020', '12-31-2020')
            },
        }
    # Periodicity window
    step_size = 1
    default_value = -10000
    
    # GARCH configurations
    garch       = dict(p = 1, q = 1, o = 0, vol = 'garch')
    egarch      = dict(p = 1, q = 1, o = 1, vol = 'egarch')
    gjr_garch   = dict(p = 1, q = 1, o = 1, vol = 'garch')
    
    # Experiment configuation
    experiment_config = dict(
        index                   = 'ATX',
        use_commodity_prices    = False,
        hybrid_model            = False,
        model_type              = ['LSTM'],
        window                  = 14
    )
    
    # Modeling
    fold = 1
    model_name = 'baseline'
    train_batch_size = 16
    valid_batch_size = 32
    iters_to_accumlate = 1
    learning_rate = 5e-5
    num_epochs = 200
    save_epoch_wait = 1
    early_stop_count = 20
    save_checkpoint = False
    
    # model inputs / configuration
    input_size = experiment_config['window']
    hidden_size = 64
    num_layers = 3
    output_size = 1
    