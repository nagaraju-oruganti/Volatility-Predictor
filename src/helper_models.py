import os
import pandas as pd
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Surpress warnings
import warnings
warnings.filterwarnings('ignore')

# GARCH MODELS
from arch import arch_model

#### GARCH MODELS
class GarchModel:
    
    ### DOCUMENTATION ON GARCH
    ### https://readthedocs.org/projects/arch/downloads/pdf/latest/
    
    
    def __init__(self, log_returns, config):
        self.log_returns = 100 * log_returns
        self.config = config
    
    def estimate_volatility(self, garch_config):
        
        # garch configuration
        vol = garch_config['vol']
        p   = garch_config['p']
        q   = garch_config['q']
        o   = garch_config['o']
        
        # create model
        model       = arch_model(self.log_returns, vol = vol, p = p, q = q, o = o, dist='normal')
        results     = model.fit(update_freq=-1, disp = False, show_warning=False)
        volatility  = results.conditional_volatility
        
        return volatility
    
    def garch_params(self):
        return self.estimate_volatility(garch_config=self.config.garch)
    
    def egarch_params(self):
        return self.estimate_volatility(garch_config=self.config.egarch)
    
    def gjr_garch_params(self):
        return self.estimate_volatility(garch_config=self.config.gjr_garch)
    
    
### Treditional LSTM MODEL
class LSTMModel(nn.Module):
    
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        
        self.config = config
        
        self.input_size  = self.config.input_size
        self.hidden_size = self.config.hidden_size
        self.num_layers  = self.config.num_layers
        self.output_size = self.config.output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size  = self.input_size, 
                            hidden_size = self.hidden_size, 
                            num_layers  = self.num_layers,
                            batch_first =True)
        
        # Fully connected output layer
        self.fc = nn.Linear(in_features = self.hidden_size, 
                            out_features= self.output_size)
        
        # loss
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x, y = None):
        
        x = x.permute(0, 2, 1)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.config.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.config.device)
        
        # Forward pass through LSTM layers
        out, _ = self.lstm(x, (h0, c0))
        
        # output
        out = self.fc(out[:, -1, :])        # output from the last time step
        loss = self.loss_fn(out, y)
        if y is None:
            return out
        
        return out, loss
    
### PEEPHOLE LSTM
class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PeepholeLSTM, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        
        # LSTM weights and biases
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x, prev_hidden_state, prev_cell_state):
        
        # Input gate
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(prev_hidden_state) + prev_cell_state)
        
        # Forget gate
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(prev_hidden_state) + prev_cell_state)
        
        # Cell state update
        c_tilda = torch.tanh(self.W_c(x) + self.U_c(prev_hidden_state))
        c_t = f_t * prev_cell_state + i_t * c_tilda
        
        # Output gate
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(prev_hidden_state) + c_t)
        
        # Hidden state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t
    
class MultiLayerPeepholeLSTM(nn.Module):
    def __init__(self, config):
        super(MultiLayerPeepholeLSTM, self).__init__()
        
        self.config = config
        
        self.input_size  = self.config.input_size
        self.hidden_size = self.config.hidden_size
        self.num_layers  = self.config.num_layers
        self.output_size = self.config.output_size
        
        ## Peephole LSTM Layers
        self.layers = nn.ModuleList([PeepholeLSTM(input_size=self.input_size, hidden_size=self.hidden_size)])
        for _ in range(1, self.num_layers):
            self.layers.append(PeepholeLSTM(input_size=self.input_size, hidden_size=self.hidden_size))
            
        # Fully connected output layer
        self.fc = nn.Linear(in_features = self.hidden_size, 
                            out_features= self.output_size)
        
        # loss
        self.loss_fn = nn.MSELoss()
        
        
    def forward(self, x, y):

        x = x.permute(0, 2, 1) 
        
        # Initialize hidden and cell states
        h0 = torch.zeros(x.size(0), x.size(1), self.hidden_size).to(self.config.device)
        c0 = torch.zeros(x.size(0), x.size(1), self.hidden_size).to(self.config.device)
        
        # Forward pass through Peephole LSTM layers
        for layer in self.layers:
            h0, c0 = layer(x, prev_hidden_state = h0, prev_cell_state = c0)
        
        # output
        out = self.fc(h0[:, -1, :])        # output from the last time step
        
        if y is None:
            return out
        
        loss = self.loss_fn(out, y)
        
        return out, loss

### Gated Recurrent Unit (GRU)
class MultiLayerGRU(nn.Module):
    def __init__(self, config):
        super(MultiLayerGRU, self).__init__()
        
        self.config = config
        
        self.input_size  = self.config.input_size
        self.hidden_size = self.config.hidden_size
        self.num_layers  = self.config.num_layers
        self.output_size = self.config.output_size
        
        ## Note: Unlike LSTM, GRU do not have argument to process multiple layers
        # First GRU layer
        self.gru1 = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first= True)
        
        # Additional GRU layers
        self.gru_layers = nn.ModuleList([nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True) \
            for _ in range(1, self.num_layers)])
    
        # Fully connected output layer
        self.fc = nn.Linear(in_features = self.hidden_size, 
                            out_features= self.output_size)
        
        # loss
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x, y = None):
        
        x = x.permute(0, 2, 1)
        
        # forward pass through first GRU layer
        out, hidden = self.gru1(x)
        
        # forward pass through the additional GRU layers
        for layer in self.gru_layers:
            out, hidden = layer(out)
        
        # output
        out = self.fc(out[: , -1, :])        # output from the last time step

        if y is None:
            return out
        loss = self.loss_fn(out, y)
        return out, loss
        
if __name__ == '__main__':
    
    from helper_config import Config
    config = Config()
    
    df = pd.read_csv('inputs/data/ATX.csv')
    df.rename(columns = {'Close' : 'Price'}, inplace = True)
    df.dropna(subset = ['Price'], how = 'any', axis = 0, inplace = True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by = 'Date', ascending = True, inplace = True)
    df.reset_index(inplace = True, drop = True)
    
    # log returns
    log_returns = 100 * np.log(df['Price'] / df['Price'].shift(1)).dropna()

    # garch models
    garch_model = GarchModel(log_returns, config = config)
    params = garch_model.garch_params()
    print(params[:10])