#Todo: random batch sampling

from config import *
from utils import *

import scipy
import scipy.io

import theano.tensor as T
import theano.sparse
from theano.sparse import *

import numpy as np




from theano import ProfileMode




"""

        self.W, self.V, self.b, self.u = [],[],[],[]
        self.embedding_size_t = theano.shared(self.embedding_size)
        self.E = theano.sparse.sharedvar.sparse_constructor(entity_matrix, name="E")
        self.U = theano.shared(embedding_matrix, name="U")

        for r in R:
            # Setup params
            #Tensor matrix
            W = np.random.uniform(low=-init_range, high=init_range, size=(self.embedding_size, self.embedding_size, k, len(self.R)))
            #Neural matrix
            V = np.random.uniform(low=-init_range, high=init_range, size=(2*self.embedding_size, k, len(self.R)))
            #Bias
            b = np.random.uniform(low=-init_range, high=init_range, size=(k, len(self.R)))
            #Concatenation
            u = np.random.uniform(low=-init_range, high=init_range, size=(k, len(self.R)))


            self.W.append(theano.shared(W, name="W"))
            self.V.append(theano.shared(V, name="V"))
            self.b.append(theano.shared(b, name="b"))
            self.u.append(theano.shared(u, name="u"))
"""





class TensorKnowledgeLearner(object):
    def __init__(self, R, k, embedding_matrix, entity_matrix):
        """
        """


        self.k = k # Slices count
        self.R = R # Relation indexes
        self.R_dict = {}
        for i,r in enumerate(R):
            self.R_dict[r] = i
        self.embedding_size = embedding_matrix.shape[1]

        init_range = 0.1

        print(theano.config.floatX)

        # Setup params
        #Tensor matrix
        W = np.random.uniform(low=-init_range, high=init_range, size=(self.embedding_size, self.embedding_size, k, len(self.R)))
        #Neural matrix
        V = np.random.uniform(low=-init_range, high=init_range, size=(2*self.embedding_size, k, len(self.R)))
        #Bias
        b = np.random.uniform(low=-init_range, high=init_range, size=(k, len(self.R)))
        #Concatenation
        u = np.random.uniform(low=-init_range, high=init_range, size=(k, len(self.R)))

        self.embedding_size_t = theano.shared(self.embedding_size)
        self.E, self.U, self.W = \
            theano.sparse.sharedvar.sparse_constructor(entity_matrix, name="E"), \
            theano.shared(np.asarray(embedding_matrix, dtype=theano.config.floatX), name="U"), \
            theano.shared(np.asarray(W, dtype=theano.config.floatX), name="W")
        self.V, self.b, self.u = theano.shared(np.asarray(V, dtype=theano.config.floatX), name="V"), \
                                 theano.shared(np.asarray(b, dtype=theano.config.floatX), name="b"), \
                                 theano.shared(np.asarray(u, dtype=theano.config.floatX), name="u")

        self.params = [self.U, self.W, self.V, self.b, self.u]

        self.EU = theano.shared(np.zeros(shape=(entity_matrix.shape[0], embedding_matrix.shape[1]), dtype=theano.config.floatX), name="EU")

        self.input = T.lmatrix()

        self.inputs = [self.input] # For trainer

    @property
    def f_prop(self):
        EU = theano.sparse.dot(self.E, self.U) # It is only expression, not evaluated yet

        values, _ = theano.scan(fn=lambda idx: self.activation(EU[self.input[idx,0]],\
                                                               EU[self.input[idx,2]],\
                                                               self.R_dict[self.input[idx, c["REL_IDX"]]]),\
                                  sequences=[theano.tensor.arange(self.input.shape[0])])

        return theano.function([self.input], values)

    def update_EU(self):
        self.EU.set_value(T.cast(theano.sparse.dot(self.E, self.U), theano.config.floatX).eval())

    def cost(self, use_old_EU=False):
        R = self.input[0, c["REL_IDX"]] ## IMPORTANT BATCH IS CONSISTENT FOR COST

        EU = self.EU
        if not use_old_EU:
            EU = T.cast(theano.sparse.dot(self.E, self.U), theano.config.floatX) #duzo niepotrzebne.. 20k batch ,

        self.margin = 1.0 - (self.activation(EU[self.input[:,0]], EU[self.input[:,2]], R) -
                             self.activation(EU[self.input[:,3]], EU[self.input[:,4]], R))

        #return T.sum(T.maximum(0.0, self.margin))
        return T.sum(self.margin[(self.margin > 0).nonzero()])
         # The margin should be > 1

    @property
    def monitors(self):
        return [["Incorrect percentage", T.sum(self.margin > 0.0) / (0.01+self.margin.shape[0]*self.margin.shape[1])]]

    def activation(self, ent1, ent2, R):
        #
        # #Calculated raw activation of the network
        # def raw_activation(idx, tensor_slice, e1, e2):
        #     return theano.dot(theano.dot(e1[idx:idx+1,:], self.W[:,:,tensor_slice, R]), e2[idx:idx+1].T) +\
        #     theano.dot(self.V[:,tensor_slice:tensor_slice+1, R], T.vertical_stack(e1[idx:idx+1].T, e2[idx:idx+1].T)) +\
        #     self.b[tensor_slice, R] # Bias part

        #Calculated raw activation of the network
        def raw_activation_fast(sl, e1, e2):
            return T.batched_dot(theano.dot(e1, self.W[:,:,sl,R]), e2) +\
            theano.dot(T.reshape(self.V[:,sl,R],(1,-1)), T.vertical_stack(e1.T, e2.T)) +\
            self.b[sl,R] # Bias part

        def tensor_output_fast(e1, e2):
            #return T.add(*[T.mul(self.u[i,R],T.tanh(raw_activation_fast(i, e1, e2))) for i in range(self.k)])
            return T.mul(self.u[0,R],T.tanh(raw_activation_fast(0, e1, e2))) + \
                   T.mul(self.u[1,R],T.tanh(raw_activation_fast(1, e1, e2))) \
                   +T.mul(self.u[2,R],T.tanh(raw_activation_fast(2, e1, e2)))
        #
        #
        # #Tanh nonlinearity and concatenation
        # def tensor_output(idx, e1, e2):
        #     return T.add(*[self.u[i]*T.tanh(raw_activation(idx, i, e1, e2)) for i in range(self.k)])


        # #Calculate predictions
        # values, _ = theano.scan(fn=lambda idx: tensor_output(idx, ent1, ent2),\
        #                          sequences=[theano.tensor.arange(ent1.shape[0])])

        return tensor_output_fast(ent1, ent2)

import time

class SGD(object):
    '''This is a base class for all trainers.'''

    def __init__(self, network, profile=False, lr=0.3, momentum=0.4, epochs=0, num_updates=10,  valid_freq=10, L2=0.0001, compile=True):
        self.profile = profile


        self.network = network

        self.valid_freq = valid_freq
        self.num_updates = num_updates


        self.lr = np.float32(lr)
        self.momentum = np.float32(momentum)


        self.epochs = epochs
        self.params = network.params

        self.cost = network.cost() + np.float32(L2)*T.sum([(p**2).sum() for p in self.params])
        self.grads = T.grad(self.cost, self.params)

        # Expressions evaluated for training
        self.cost_exprs_update = [self.cost, network.cost()]
        self.cost_names = ['L2 cost', "Network cost"]
        for name, monitor in network.monitors:
            self.cost_names.append(name)
            self.cost_exprs_update.append(monitor)

        # Expressions when propagating
        self.cost_exprs_evaluate = [network.cost(use_old_EU=True) + np.float32(L2)*T.sum([(p**2).sum() for p in self.params]), \
                                network.cost(use_old_EU=True)]
        for name, monitor in network.monitors:
            self.cost_exprs_evaluate.append(monitor)


        self.shapes = [p.get_value(borrow=True).shape for p in self.params]
        self.counts = [np.prod(s) for s in self.shapes]
        self.starts = np.cumsum([0] + self.counts)[:-1]
        self.dtype = self.params[0].get_value().dtype

        self.best_cost = 1e100
        self.best_iter = 0
        self.best_params = [p.get_value().copy() for p in self.params]

        mode = "FAST_RUN"
        if self.profile:
            mode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())


        if compile:
            self.f_eval = theano.function(
            network.inputs, self.cost_exprs_evaluate)

            self.f_learn_R = {}
            for r in network.R:
                f_learn = theano.function(
                    network.inputs,
                    self.cost_exprs_update,
                    updates=list(self.learning_updates(R=r)),mode=mode)

                self.f_learn_R[r] = f_learn

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

    def learning_updates(self, R=0):
        # This code computes updates only for given R, so it drops last dimension. Plus soe theano magic to circumvent its graph comp.
        grads = self.grads
        for i, param in enumerate(self.params):
            if i==0:
                delta = self.lr * grads[i]
                velocity = theano.shared(
                    np.zeros_like(param.get_value(), dtype=theano.config.floatX), name=param.name +'_vel')

                yield velocity, T.cast(self.momentum * velocity - delta, theano.config.floatX)
                yield param, param + velocity
            else:

                if len(self.shapes[i])==2:
                    delta = self.lr * grads[i][:,R]
                    velocity = theano.shared(
                        np.zeros(shape=self.shapes[i][0:-1], dtype=theano.config.floatX), name=param.name + str(R)+'_vel')
                    subgrad = T.set_subtensor(param[:,R], param[:,R] + velocity)
                    yield velocity, self.momentum * velocity - delta
                    yield param, subgrad
                if len(self.shapes[i])==3:
                    delta = self.lr * grads[i][:,:,R]
                    velocity = theano.shared(
                        np.zeros(shape=self.shapes[i][0:-1], dtype=theano.config.floatX), name=param.name + str(R)+'_vel')
                    subgrad = T.set_subtensor(param[:,:,R], param[:,:,R] + velocity)

                    yield velocity, self.momentum * velocity - delta
                    yield param, subgrad
                if len(self.shapes[i])==4:
                    delta = self.lr * grads[i][:,:,:,R]
                    velocity = theano.shared(
                        np.zeros(shape=self.shapes[i][0:-1], dtype=theano.config.floatX), name=param.name + str(R)+'_vel')
                    subgrad = T.set_subtensor(param[:,:,:,R], param[:,:,:,R] + velocity)
                    yield velocity, self.momentum * velocity - delta
                    yield param, subgrad


    def evaluate(self, iteration, valid_set):
        print("Evaluating")

        costs = list(zip(
            self.cost_names,
            np.mean([self.f_eval(x.reshape(1,-1)) for x in valid_set], axis=0)))

        print(costs)

        return True

    def train_minibatch(self, x, R=None):
        if R is not None:
            return self.f_learn_R[x[0,c["REL_IDX"]]](x)
        else:
            return self.f_learn(x)

    def train(self, train_batches, valid_set=None, num_updates=1000000):
        iteration = 0

        while iteration < min(num_updates, self.num_updates):
            batch_res = []

            time_start = time.time()
            for id, batch in enumerate(train_batches):
                if id%4 == 0 and id>0:
                    print(str(id) +" "+ str(time.time() - time_start))
                    time_start = time.time()
                    if self.profile:
                        ProfileMode.print_summary()

                batch_res.append(self.train_minibatch(batch, R=True))

            costs = list(zip(
                self.cost_names,
                np.mean(batch_res, axis=0)))

            print(iteration)
            print(costs)

            iteration += 1

            if iteration % self.valid_freq == 0:
                self.network.update_EU() #This is a hack to use calculate embeddings rather than reculaculate for every batch
                print(valid_set[0].shape)
                print(self.evaluate(iteration, valid_set))

import sys

class Scipy(SGD):
    '''General trainer for neural nets using `scipy.optimize.minimize`.'''

    METHODS = ('l-bfgs-b', 'cg', 'dogleg', 'newton-cg', 'trust-ncg')

    def __init__(self, network, num_updates=10, L2=0.0001, method = 'l-bfgs-b'):
        SGD.__init__(self,network=network, L2=L2, num_updates=num_updates, compile=False)

        self.method = method

        logging.info('compiling gradient function')

        #TODO: poprawic szybkosc
        self.f_eval = theano.function(network.inputs, self.cost_exprs_update)
        self.f_eval_fast = theano.function(network.inputs, self.cost_exprs_evaluate)
        self.f_grad = theano.function(network.inputs, T.grad(self.cost, self.params))

    @timed
    def function_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x))
        return np.mean([self.f_eval(x)[0] for x in train_set]).astype(np.float64) #lbfgs fortran code wants float64. meh

    @timed
    def gradient_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x))
        grads = [[] for _ in range(len(self.params))]
        for x in train_set:
            for i, g in enumerate(self.f_grad(x)):
                grads[i].append(np.asarray(g))
        G = self.arrays_to_flat([np.mean(g, axis=0) for g in grads]).astype(np.float64) #lbfgs fortran code wants float64. meh
        return G

    def train_scipy(self, train_set, valid_set=None, num_updates=100000):
        current_batch = 0

        def display(x):
            self.set_params(self.flat_to_arrays(x))
            costs = self.f_eval(train_set[current_batch])
            cost_desc = ' '.join(
                '%s=%.6f' % el for el in zip(self.cost_names, costs))
            print('scipy %s %i %s' %
                  (self.method, i + 1, cost_desc))
            sys.stdout.flush()

        print("Training on "+str(len(train_set))+" batches")

        for i in range(min(self.num_updates, num_updates)):

            for batch_id, batch in enumerate(train_set):
                current_batch = current_batch

                try:
                    res = scipy.optimize.minimize(
                        fun=self.function_at,
                        jac=self.gradient_at,
                        x0=self.arrays_to_flat(self.best_params),
                        args=([batch], ),
                        method=self.method,
                        callback=display,
                        options=dict(maxiter=5),
                    )
                except KeyboardInterrupt:
                    print('interrupted!')
                    break

                @timed
                def set():
                    self.set_params(self.flat_to_arrays(res.x))

                set()

                try:
                    if not self.evaluate(i, valid_set):
                        print('patience elapsed, bailing out')
                        break
                except KeyboardInterrupt:
                    print('interrupted!')
                    break


                sys.stdout.flush()
                sys.stderr.flush()

        self.set_params(self.best_params)

