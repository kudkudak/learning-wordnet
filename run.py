from model import *
from data_api import *


X = prepare_experiment_data(dataset="Wordnet", multiplication=10, CV=0)



u = TensorKnowledgeLearner(0, 3,  X["U"], X["E"])



t = SGD(u)


t.train(X["X"], X["X_test"])
