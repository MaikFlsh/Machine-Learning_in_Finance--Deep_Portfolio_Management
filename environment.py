import numpy as np
import pandas as pd

from utils import portfolio


class Environment:
    
    def __init__(self, prices = './data/portfolio.csv', capital = 1e6):       
        self.prices = prices  
        self.capital = capital  
        self.data = self.load_data()

    def load_data(self):
        data =  pd.read_csv(self.prices)
        try:
            data.index = data['Datum']
            data = data.drop(columns = ['Datum'])
        except:
            data.index = data['datum']
            data = data.drop(columns = ['datum'])            
        return data
    
    def preprocess_state(self, state):
        return state
    
    def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        
        decision_making_state = self.data.iloc[t-lookback:t]
        decision_making_state = decision_making_state.pct_change().dropna()

        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)

    def get_reward(self, action, action_t, reward_t, alpha = 0.01):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])

        data_period = self.data[action_t:reward_t]
        weights = action
        returns = data_period.pct_change().dropna()
      
        sharpe = local_portfolio(returns, weights)[-1]
        sharpe = np.array([sharpe] * len(self.data.columns))          
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]
        
        return np.dot(returns, weights), rew
        


