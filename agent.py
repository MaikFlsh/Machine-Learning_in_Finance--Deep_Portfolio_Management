import scipy.optimize as sco
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.models import load_model
from scipy.cluster.hierarchy import dendrogram, linkage        
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import pylab
import datetime as dt


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from environment import portfolio



class AutoencoderAgent:

    def __init__(
                     self, 
                     portfolio_size,
                     allow_short = True,
                     encoding_dim = 5
                 ):
        
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.encoding_dim = encoding_dim
        
        
    def model(self):
        # Autoencoder Modeldefinition
        input_img = Input(shape=(self.portfolio_size, ))
        encoded = Dense(self.encoding_dim, activation='relu', kernel_regularizer=regularizers.l2(1e-6))(input_img)# Encoder
        decoded = Dense(self.portfolio_size, activation= 'linear', kernel_regularizer=regularizers.l2(1e-6))(encoded)# Decoder 
        autoencoder = Model(input_img, decoded)# Autoencoder Modell
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
        

    def act(self, returns):
        #Autoencoder - Fit und Prognose
        data = returns
        autoencoder = self.model()
        autoencoder.fit(data, data, shuffle=False, epochs=25, batch_size=32, verbose=False)
        reconstruct = autoencoder.predict(data)
        
        communal_information = []

        for i in range(0, len(returns.columns)):
            diff = np.linalg.norm((returns.iloc[:,i] - reconstruct[:,i])) 
            communal_information.append(float(diff))

        optimal_weights = np.array(communal_information) / sum(communal_information)

        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            optimal_weights += np.abs(np.min(optimal_weights))
            optimal_weights /= sum(optimal_weights)
            
        return optimal_weights
    
    
    
class HRPAgent:

    def __init__(
                     self, 
                     portfolio_size,
                     allow_short = True,
                 ):
        
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size, )
        import numpy as np
        import pandas as pd
        from scipy.cluster.hierarchy import dendrogram, linkage        
        from scipy.cluster.hierarchy import cophenet
        from scipy.spatial.distance import pdist
        import pylab

    def getIVP(cov, **kargs):
        # Inverse Varianz des Portfolis berechnen
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp


    def getClusterVar(cov,cItems):
        # Varianz des Clusters berechnen
        from agent import HRPAgent
        cov_=cov.loc[cItems,cItems] # matrix slice
        w_= HRPAgent.getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar

    def correlDist(corr):
        #1. Schritt 
        # Distanzmatrix aus der Kovarianzmatrix bestimmen 
        dist = ((1 - corr) / 2.)**.5  # distance matrix
        return dist

    def getQuasiDiag(link):
        # 2. Schritt Quasi Diagonalisierung
        # Geclusterte Elemente nach der Distanz sortieren
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # Anzahl der ursprülgichen Elemente
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  
            df0 = sortIx[sortIx >= numItems]  # Clustersuche
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # Element 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = sortIx.append(df0)  # Element 2
            sortIx = sortIx.sort_index()  # Sortieren 
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()


    def getRecBipart(cov, sortIx):
        from agent import HRPAgent
        # 3. Schritt  - Rekursive Bisektion
        w = pd.Series(1, index=sortIx) # Gewichtung der Cluster Varianz zunächst 1 
        cItems = [sortIx]  # Zunächst alle Elemente in einem Cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # Bisektion 
            for i in range(0, len(cItems), 2):  # zerlegen in einzelne Paare
                cItems0 = cItems[i]  # Cluster 1
                cItems1 = cItems[i + 1]  # Cluster 2
                cVar0 = HRPAgent.getClusterVar(cov, cItems0) # Varianz berechnen für Cluster 1
                cVar1 = HRPAgent.getClusterVar(cov, cItems1) # Varianz berechnen für Cluster 1
                alpha = 1 - cVar0 / (cVar0 + cVar1) 
                w[cItems0] *= alpha  # Gewichtung Varianz aus Cluster 1 
                w[cItems1] *= 1 - alpha  # Gewichtung Varianz aus Cluster 2
        return w



    def getHRP(cov, corr):
        from agent import HRPAgent
        # Konstruktion des hierarchischen Portfolios
        dist = HRPAgent.correlDist(corr) # 1. Schritt: Cluster erstellen
        link = linkage(dist, 'single') # Cluster mit min Distanz erstellen 
        sortIx = HRPAgent.getQuasiDiag(link) #2. Schritt: Quasi Diagonalisierung
        sortIx = corr.index[sortIx].tolist()
        hrp = HRPAgent.getRecBipart(cov, sortIx) #3. Schritt: Rekursive Bisektion
        return hrp.sort_index()


    def act(self, returns):
        from agent import HRPAgent
        corr = returns.corr()
        cov = returns.cov()
        optimal_weights = HRPAgent.getHRP(cov, corr) 
        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            optimal_weights += np.abs(np.min(optimal_weights))
            optimal_weights /= sum(optimal_weights)
        return optimal_weights

class SmoothingAgent:

    def __init__(
                     self, 
                     portfolio_size,
                     allow_short = True,
                     forecast_horizon = 252,
                 ):
        
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.forecast_horizon = forecast_horizon

    def act(self, timeseries):

        optimal_weights = []
        
        for asset in timeseries.columns:
            ts = timeseries[asset]
            fit1 = Holt(ts).fit()  # Modelfit exp. Glätten
            forecast = fit1.forecast(self.forecast_horizon) # Prognose
            prediction = forecast.values[-1] - forecast.values[0]
            optimal_weights.append(prediction)

        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            optimal_weights += np.abs(np.min(optimal_weights))
            optimal_weights /= sum(optimal_weights)
            
        return optimal_weights


class GARCHAgent:

    def __init__(
                     self, 
                     portfolio_size,
                     allow_short = True,
                     forecast_horizon = 252,
                 ):
        
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.forecast_horizon = forecast_horizon
        from arch import arch_model

    def act(self, timeseries):
        from arch import arch_model
        optimal_weights = []
        
        for asset in timeseries.columns:
            ts = timeseries[asset]
            model = arch_model(ts, vol='Garch', p=1, q=1,mean='constant', dist='Normal')# GARCH(1,1) Model definieren
            fit1 = model.fit(disp='off')
            forecast = fit1.forecast(horizon=self.forecast_horizon)# Prognose des GARCH Modells
            prediction = np.sum(forecast.mean.iloc[-1].values)
            optimal_weights.append(prediction)

        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            optimal_weights += np.abs(np.min(optimal_weights))
            optimal_weights /= sum(optimal_weights)
            
        return optimal_weights

   