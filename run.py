from model import *
from data_api import *

X = prepare_experiment_data(dataset="Wordnet", multiplication=10, CV=0)

u = TensorKnowledgeLearner(0, 3,  X["U"], X["E"])

t = SGD(u, lr=0.01)

batches_train, batches_test = generate_batches(X["X"][0], X["X_test"][0], batch_size=None, tr_batch_count=30)

t.train(batches_train, batches_test)
