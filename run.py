#TODO: add randomized sampling

from model import *
from data_api import *

@cached_FS()
def train_model():

    X = prepare_experiment_data(dataset="Wordnet", multiplication=10, CV=0, force_reload=True)

    u = TensorKnowledgeLearner(0, 3,  X["U"], X["E"])

    t = SGD(u, lr=0.5, num_updates=20, valid_freq=2)

    batches_train, batches_test = generate_batches(X["X"], X["X_test"], batch_size=10000)

    t.train(batches_train, batches_test)

    return u


train_model()