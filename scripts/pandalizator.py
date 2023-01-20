 
from distutils.log import error
import numpy as np 
import matplotlib.pyplot as plt 
import json 
from json import JSONEncoder


# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, numpy.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)


class Pandalizator():

    def __init__(self,eta=0.01,n_iter=20,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self,Images,target_values):
        rgen = np.random.RandomState(self.random_state)
        
        self.weights_ = rgen.normal(loc=0.0,scale=0.1,size=len(Images[0]))
        self.bias_ = np.float_(0.)
        self.errors_ = []

        for it in range(self.n_iter):
            print('iteration  ',it)
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

    
    def save_model_to_file(self,filename='model.json'):
        data = {
            "bias": self.bias_,
            "weights": self.weights_.tolist()
        }

        json_object = json.dumps(data)
        
        with open(filename,'w') as output:
            output.write(json_object)


    def read_model_from_file(self,filename='model.json'):
        with open(filename,'r') as input:
            data = json.load(input)
            self.bias_ = data['bias']
            self.weights_ = np.array(data['weights'])
