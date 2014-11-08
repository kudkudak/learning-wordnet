#Todo: random batch sampling

from config import *
from utils import *

import scipy
import scipy.io

import theano.tensor as T
import theano.sparse
from theano.sparse import *

import numpy as np

class TensorKnowledgeLearner(object):
    def __init__(self, R, k, embedding_matrix, entity_matrix):
        """
        """
        self.k = k # Slices count
        self.R = R # Relation index
        self.embedding_size = embedding_matrix.shape[1]

        init_range = 2.0

        # Setup params
        #Tensor matrix
        W = np.random.uniform(low=-init_range, high=init_range, size=(self.embedding_size, self.embedding_size,k))
        #Neural matrix
        V = np.random.uniform(low=-init_range, high=init_range, size=(k, 2*self.embedding_size))
        #Bias
        b = np.random.uniform(low=-init_range, high=init_range, size=(k,))
        #Concatenation
        u = np.random.uniform(low=-init_range, high=init_range, size=(k,))

        self.E, self.U, self.W = theano.sparse.sharedvar.sparse_constructor(entity_matrix, name="E"), \
        theano.shared(embedding_matrix, name="U"), theano.shared(W, name="W")
        self.V, self.b, self.u = theano.shared(V, name="V"), theano.shared(b, name="b"), theano.shared(u, name="u")

        self.params = [self.U, self.W, self.V, self.b, self.u]

        self.input = T.lmatrix()

        self.inputs = [self.input] # For trainer
    @property
    def cost(self):
        EU = theano.sparse.dot(self.E, self.U)

        emb1 = EU[self.input[:,0]]

        emb2 = EU[self.input[:,2]]

        emb3 = EU[self.input[:,3]]

        emb4 = EU[self.input[:,4]]

        return T.mean(T.maximum(0.0, 1.0 - (self.f_prop(emb1, emb2) - self.f_prop(emb3, emb4)))) # The margin should be > 1

    @property
    def monitors(self):
        return []

    def f_prop(self, ent1, ent2):

        #Calculated raw activation of the network
        def raw_activation(idx, tensor_slice, e1, e2):
            return theano.dot(theano.dot(e1[idx:idx+1,:], self.W[:,:,tensor_slice]), e2[idx:idx+1].T) +\
            theano.dot(self.V[tensor_slice:tensor_slice+1], T.vertical_stack(e1[idx:idx+1].T, e2[idx:idx+1].T)) +\
            self.b[tensor_slice] # Bias part

        #Calculated raw activation of the network
        def raw_activation_fast(sl, e1, e2):
            return T.batched_dot(theano.dot(e1, self.W[:,:,sl]), e2) +\
            theano.dot(self.V[sl:sl+1], T.vertical_stack(e1.T, e2.T)) +\
            self.b[sl] # Bias part

        def tensor_output_fast(e1, e2):
            return T.add(*[T.mul(self.u[i],T.tanh(raw_activation_fast(i, e1, e2))) for i in range(self.k)])


        #Tanh nonlinearity and concatenation
        def tensor_output(idx, e1, e2):
            return T.add(*[self.u[i]*T.tanh(raw_activation(idx, i, e1, e2)) for i in range(self.k)])


        # #Calculate predictions
        # values, _ = theano.scan(fn=lambda idx: tensor_output(idx, ent1, ent2),\
        #                          sequences=[theano.tensor.arange(ent1.shape[0])])

        return tensor_output_fast(ent1, ent2)


class SGD(object):
    '''This is a base class for all trainers.'''

    def __init__(self, network, lr=0.01, momentum=0.4, epochs=0, num_updates=10,  valid_freq=10):
        super(SGD, self).__init__()

        self.valid_freq = valid_freq
        self.num_updates = num_updates

        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.params = network.params

        self.cost = network.cost
        self.cost_exprs = [self.cost]
        self.cost_names = ['Cost']
        for name, monitor in network.monitors:
            self.cost_names.append(name)
            self.cost_exprs.append(monitor)


        self.f_eval = theano.function(
            network.inputs, self.cost_exprs)

        self.shapes = [p.get_value(borrow=True).shape for p in self.params]
        self.counts = [np.prod(s) for s in self.shapes]
        self.starts = np.cumsum([0] + self.counts)[:-1]
        self.dtype = self.params[0].get_value().dtype

        self.best_cost = 1e100
        self.best_iter = 0
        self.best_params = [p.get_value().copy() for p in self.params]

        self.f_learn = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=list(self.learning_updates()), mode='FAST_RUN')

    def flat_to_arrays(self, x):
        x = x.astype(self.dtype)
        return [x[o:o+n].reshape(s) for s, o, n in
                zip(self.shapes, self.starts, self.counts)]

    def arrays_to_flat(self, arrays):
        x = np.zeros((sum(self.counts), ), self.dtype)
        for arr, o, n in zip(arrays, self.starts, self.counts):
            x[o:o+n] = arr.ravel()
        return x

    def set_params(self, targets):
        for param, target in zip(self.params, targets):
            param.set_value(target)

    def learning_updates(self):
        for param in self.params:
            delta = self.lr * T.grad(self.cost, param)
            velocity = theano.shared(
                np.zeros_like(param.get_value()), name=param.name + '_vel')
            yield velocity, self.momentum * velocity - delta
            yield param, param + velocity

    def evaluate(self, iteration, valid_set):
        costs = list(zip(
            self.cost_names,
            np.mean([self.f_eval(x) for x in valid_set], axis=0)))
        return costs

    def train_minibatch(self, x):
        return self.f_learn(x)

    def train(self, train_set, valid_set=None, **kwargs):
        iteration = 0
        while iteration < self.num_updates:
            if iteration % self.valid_freq == 0:
                print(self.evaluate(iteration, valid_set))

            # costs = list(zip(
            #     self.cost_names,
            #     np.mean([self.train_minibatch(x) for x in train_set], axis=0)))
            batch_res = []
            for batch in train_set:
                batch_res.append(self.train_minibatch(batch))
                print("Trained batch")

            costs = list(zip(
                self.cost_names,
                np.mean(batch_res, axis=0)))

            print(iteration)
            print(costs)

            iteration += 1

            