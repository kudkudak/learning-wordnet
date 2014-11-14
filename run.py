#TODO: add randomized sampling

from model import *
from data_api import *

#1.3 mln przykladow na minute (u mnie 600k przykladow ok.)

def train_model():

    X = prepare_experiment_data(dataset="Wordnet", CV=0)

    u = TensorKnowledgeLearner(range(11), 3,  X["U"], X["E"])
    #t = SGD(u, lr=0.3, num_updates=200, L2=0, valid_freq=1)

    t = Scipy(u, num_updates=10, L2=0.0001)
    for i in range(5):
        batches_train = generate_batches(E=X, batch_size=50000, seed=i)
        t.train_scipy(batches_train, X["X_test"], num_updates=1)

    return u



train_model()