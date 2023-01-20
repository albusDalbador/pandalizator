from pandalizator import Pandalizator
from proceed_dataset import *

import matplotlib.pyplot as plt




if __name__ == '__main__':  

    training_dataset,training_target = get_training_data()

    pandalizator = Pandalizator(eta=0.2,n_iter=20)
    
    pandalizator.fit(training_dataset,training_target)
    pandalizator.save_model_to_file()
    
    plt.plot(range(1,len(pandalizator.errors_)+1),pandalizator.errors_)
    plt.show()

