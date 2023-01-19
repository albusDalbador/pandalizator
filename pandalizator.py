 
from distutils.log import error
import cv2
import numpy as np 
import matplotlib.pyplot as plt 


class Pandalizator():

    def __init__(self,eta=0.01,n_iter=20,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self,Images,target_values):
        rgen = np.random.RandomState(self.random_state)
        
        self.weights_ = rgen.normal(loc=0.0,scale=0.1,size=X.shape)
        self.bias_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for sample,target in zip(Images,target_values):
                update = self.eta * (target - self.predict(sample))
                self.weights_ += update * sample
                self.bias_ += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    
    def net_input(self,sample):
        return np.dot(sample,self.weights_) + self.bias_

    
    def predict(self,sample):
        return np.where(self.net_input(sample) >= 0.0, 1, 0)
