"""

"""

import numpy as np

"""
 
  ######   ######## 
 ##    ##     ##    
 ##           ##    
 ##   ####    ##    
 ##    ##     ##    
 ##    ##     ##    
  ######      ##    
 
"""

class GroundTruth:
    def __init__(self, F, Q, init_state):
        self.F = F
        self.Q = Q
        self.state = init_state
        return
    
    def update(self):
        w = np.random.multivariate_normal(np.array([0, 0, 0, 0]), self.Q)
        self.state = self.F@self.state + w
        return self.state

"""
 
  ######  ######## ##    ##  ######   #######  ########  
 ##    ## ##       ###   ## ##    ## ##     ## ##     ## 
 ##       ##       ####  ## ##       ##     ## ##     ## 
  ######  ######   ## ## ##  ######  ##     ## ########  
       ## ##       ##  ####       ## ##     ## ##   ##   
 ##    ## ##       ##   ### ##    ## ##     ## ##    ##  
  ######  ######## ##    ##  ######   #######  ##     ## 
 
"""

class SensorAbs:
    def measure(self, ground_truth):
        raise NotImplementedError

class SensorPure(SensorAbs):
    def __init__(self, n, m, H, R):
        self.n = n
        self.m = m
        self.H = H
        self.R = R
        return
    
    def measure(self, ground_truth):
        v = np.random.multivariate_normal(np.array([0, 0]), self.R)
        return self.H@ground_truth + v

"""
 
 ######## #### ##       ######## ######## ########   ######  
 ##        ##  ##          ##    ##       ##     ## ##    ## 
 ##        ##  ##          ##    ##       ##     ## ##       
 ######    ##  ##          ##    ######   ########   ######  
 ##        ##  ##          ##    ##       ##   ##         ## 
 ##        ##  ##          ##    ##       ##    ##  ##    ## 
 ##       #### ########    ##    ######## ##     ##  ######  
 
"""

class FilterAbs:
    def predict(self):
        raise NotImplementedError
    
    def update(self, measurement):
        raise NotImplementedError

class KFilter(FilterAbs):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov):
        self.n = n
        self.m = m
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x = init_state
        self.P = init_cov
        return
    
    def predict(self):
        self.x = self.F@self.x
        self.P = self.Q + (self.F@self.P@self.F.T)
        return self.x, self.P
    
    def update(self, measurement):
        S = (self.H@self.P@self.H.T) + self.R
        invS = np.linalg.inv(S)

        K = self.P@self.H.T@invS

        self.x = self.x + K@(measurement - self.H@self.x)
        self.P = self.P - (K@S@K.T)
        return self.x, self.P
