from .pandalizator import Pandalizator
from .proceed_dataset import *

import matplotlib.pyplot as plt


def evaluate_model(model):
    test_dataset,test_target = get_test_data()

    errors = 0
    for i in range(len(test_target)):
        if not test_target[i] == model.predict(test_dataset[i]):
            errors += 1
    
    print(errors/len(test_target) * 100, "%% blędów")

def fit_model(model):
    training_dataset,training_target = get_training_data()

    model.fit(training_dataset,training_target)
    model.save_model_to_file()

    plt.plot(range(1,len(model.errors_)+1),model.errors_)
    return model

if __name__ == '__main__':  


    pandalizator = Pandalizator(eta=0.05,n_iter=20)
    
    pandalizator = fit_model(pandalizator)
    evaluate_model(pandalizator)

    plt.show()

