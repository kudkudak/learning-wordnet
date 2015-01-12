#Todo: TensorKnowledgeLearner -> to many objects that shared E and U matrices

#Todo: random batch sampling

from config.config import *
from utils import *

import scipy
import scipy.io

import theano.tensor as T
import theano.sparse
from theano.sparse import *

import numpy as np




from theano import ProfileMode




def createNetworks(entity_matrix, embedding_matrix, R=range(11), k=3, only_metric=False):
    #Common
    E, U= \
            theano.sparse.sharedvar.sparse_constructor(entity_matrix, name="E"), \
            theano.shared(np.asarray(embedding_matrix, dtype=theano.config.floatX), name="U")

    EU = theano.shared(np.zeros(shape=(entity_matrix.shape[0], embedding_matrix.shape[1]), dtype=theano.config.floatX), name="EU")

    embedding_size = embedding_matrix.shape[1]


    networks = []
    for r in R:
        networks.append(TensorKnowledgeLearner(R=r, k=k, E=E, U=U, EU=EU, embedding_size=embedding_size, only_metric=only_metric))

    return networks

def saveNetworks(networks, file):
    params = []
    for n in networks:
        params.append(n.dump_params())
    import cPickle
    cPickle.dump({"network_params":params, "embedding_matrix":networks[0].U.get_value(borrow=True)}, open(file, "w"))



class TensorKnowledgeLearner(object):
    def __init__(self, R, k, E, U, EU, embedding_size, only_metric):
        self.k = k # Slices count
        self.R = R
        self.embedding_size = embedding_size

        init_range = 0.00007
        init_range_W = 0.01
        # Setup params
        #Tensor matrix
        W = np.random.uniform(low=-init_range_W, high=init_range_W, size=(self.embedding_size, self.embedding_size, k))
        #Neural matrix
        V = np.random.uniform(low=-init_range, high=init_range, size=(2*self.embedding_size, k))
        #Bias
        b = np.random.uniform(low=-init_range, high=init_range, size=(k,))
        #Concatenation
        u = np.random.uniform(low=-init_range, high=init_range, size=(k, ))

        self.embedding_size_t = theano.shared(self.embedding_size)
        self.W = theano.shared(np.asarray(W, dtype=theano.config.floatX), name="W")

        self.E, self.U, self.EU = E, U, EU # Shared among networks

        self.V, self.b, self.u = theano.shared(np.asarray(V, dtype=theano.config.floatX), name="V"+str(R)), \
                                 theano.shared(np.asarray(b, dtype=theano.config.floatX), name="b"+str(R)), \
                                 theano.shared(np.asarray(u, dtype=theano.config.floatX), name="u"+str(R))


        if only_metric == True:
            self.params = [self.W, self.U, self.u]
        else:
            self.params = [self.W, self.U, self.V, self.b, self.u]


        self.input = T.lmatrix()

        self.inputs = [self.input] # For trainer

    def dump_params(self):
        return [p.get_value(borrow=True) for p in self.params if p!=self.U]

    def load_params(self, params):
        assert(len(params)==len(self.params)-1)
        id_params = 0
        for id, p in enumerate(self.params):
            if(id == 1):
                continue #skip setting global U
            self.params[id].set_value(params[id_params])
            id_params += 1

    @property
    def f_prop(self):
        EU = theano.sparse.dot(self.E, self.U) # It is only expression, not evaluated yet

        values, _ = theano.scan(fn=lambda idx: self.activation(EU[self.input[idx,0]],\
                                                               EU[self.input[idx,2]]),\
                                  sequences=[theano.tensor.arange(self.input.shape[0])])

        return theano.function([self.input], values)

    def update_EU(self):
        self.EU.set_value(T.cast(theano.sparse.dot(self.E, self.U), theano.config.floatX).eval())

    def cost(self, use_old_EU=False):
        EU = self.EU
        if not use_old_EU:
            EU = T.cast(theano.sparse.dot(self.E, self.U), theano.config.floatX) #duzo niepotrzebne.. 20k batch ,

        self.margin = 1.0 - (self.activation(EU[self.input[:,0]], EU[self.input[:,2]]) -
                             self.activation(EU[self.input[:,3]], EU[self.input[:,4]]))


        #return theano.tensor.constant(0.0)
        ##return T.sum(T.maximum(0.0, self.margin))
        #indexes = (self.margin > 0).nonzero()
        return T.sum(T.maximum(0.0, self.margin))
        #return (1.0/(self.margin.shape[1]))*T.sum(self.margin[indexes])
         # The margin should be > 1

    def f_prop(self):
        return self.activation(self.EU[self.input[:,0]], self.EU[self.input[:,2]])

    @property
    def monitors(self):
        return [["Margin shape", self.margin.shape[1]], \
                ["Incorrect percentage", T.sum(self.margin > 0.0) / (0.01+self.margin.shape[0]*self.margin.shape[1])],
                ["Mean W weight", T.mean(T.mean(abs(self.W)))],
                ["Mean U weight", T.mean(T.mean(abs(self.U)))],
                ["Mean V weight", T.mean(T.mean(abs(self.V)))],
                ["Mean W activation / Mean linear activation", T.mean(abs(self.Wact)/abs(self.Vact))],
                ["Incorrect sum", T.sum(self.margin > 0.0)]
                ]

    def activation(self, ent1, ent2):
        def raw_activation_fast(sl, e1, e2):
            self.Wact = T.batched_dot(theano.dot(e1, self.W[:,:,sl]), e2)
            self.Vact = 1e-4 * theano.dot(T.reshape(self.V[:,sl],(1,-1)), T.vertical_stack(e1.T, e2.T)) +\   //1e-4 reflects change of scale
            self.b[sl] # Bias part
            return self.Wact + self.Vact

        def tensor_output_fast(e1, e2):
            #return T.add(*[T.mul(self.u[i,R],T.tanh(raw_activation_fast(i, e1, e2))) for i in range(self.k)])
            if self.k == 3:
                return T.mul(self.u[0],T.tanh(raw_activation_fast(0, e1, e2))) + \
                    T.mul(self.u[1],T.tanh(raw_activation_fast(1, e1, e2))) \
                    +T.mul(self.u[2],T.tanh(raw_activation_fast(2, e1, e2)))
            else:
                return T.mul(self.u[0],T.tanh(raw_activation_fast(0, e1, e2))) + \
                    T.mul(self.u[1],T.tanh(raw_activation_fast(1, e1, e2))) \
                    +T.mul(self.u[2],T.tanh(raw_activation_fast(2, e1, e2))) \
                    +T.mul(self.u[3],T.tanh(raw_activation_fast(3, e1, e2)))


        return tensor_output_fast(ent1, ent2)

import time

class SGD(object):
    '''This is a base class for all trainers.'''

    def __init__(self, network, profile=False, lr=0.3, momentum=0.4, epochs=0, num_updates=10,  valid_freq=10, L2=0.00005,L1=0, compile=True):
        self.profile = profile


        self.network = network

        self.valid_freq = valid_freq
        self.num_updates = num_updates


        self.lr = np.float32(lr)
        self.momentum = np.float32(momentum)


        self.epochs = epochs
        self.params = network.params

        print(self.params[2:3])

        self.regul= np.float32(L2*1.0)*T.sum([(p**2).sum() for p in self.params[3:]]) \
                    + np.float32(L2*1.0)*(self.params[0]**2).sum()\
                    + np.float32(L2*0.0)*(self.params[1]**2).sum()\
                    + np.float32(L2*4e1)*((self.params[2]**2).sum())\
                    + np.float32(L1*1.0)*T.sum([abs(p).sum() for p in self.params[3:]]) \
                    + np.float32(L1*1.0)*abs(self.params[0]).sum()\
                    + np.float32(L1*0.0)*abs(self.params[1]).sum()\
                    + L1*4e1*(abs(self.params[2]).sum())

        self.cost = network.cost() + self.regul
        self.grads = T.grad(self.cost, self.params)

        # Expressions evaluated for training
        self.cost_exprs_update = [self.cost, network.cost()]
        self.cost_names = ['L2 cost', "Network cost"]
        for name, monitor in network.monitors:
            self.cost_names.append(name)
            self.cost_exprs_update.append(monitor)

        # Expressions when propagating
        self.cost_exprs_evaluate = [network.cost(use_old_EU=True) + self.regul, \
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

            self.f_learn = theano.function(
                network.inputs,
                self.cost_exprs_update,
                updates=list(self.learning_updates()),mode=mode)


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
        # This code computes updates only for given R, so it drops last dimension. Plus soe theano magic to circumvent its graph comp.
        grads = self.grads
        for i, param in enumerate(self.params):
            delta = self.lr * grads[i]
            velocity = theano.shared(
                np.zeros_like(param.get_value(), dtype=theano.config.floatX), name=param.name + str(self.network.R)+'_vel')

            yield velocity, T.cast(self.momentum * velocity - delta, theano.config.floatX)
            yield param, param + velocity

    @timed
    def train_minibatch(self, x):
        return self.f_learn(x)






from collections import OrderedDict

class HintonIsGod(SGD):
    '''This is a base class for all trainers.'''

    def __init__(self, network, max_scaling=1e5, decay=0.9, profile=False,L1=0, lr=0.3, momentum=0.4, epochs=0, num_updates=10,  valid_freq=10, L2=0.00005, compile=True):
        SGD.__init__(self,network=network, profile=profile, lr=lr, momentum=momentum, epochs=epochs, num_updates=num_updates,  valid_freq=valid_freq,
                     L2=L2, L1=L1, compile=False)

        self.decay = theano.shared(decay, 'decay')
        self.epsilon = 1. / max_scaling
        self.mean_square_grads = OrderedDict()

        if compile:
            self.f_eval = theano.function(
                network.inputs, self.cost_exprs_evaluate)

            self.f_learn = theano.function(
                network.inputs,
                self.cost_exprs_update,
                updates=list(self.learning_updates()),mode="FAST_RUN")

    def learning_updates(self):
        # This code computes updates only for given R, so it drops last dimension. Plus soe theano magic to circumvent its graph comp.
        grads = self.grads
        for i, param in enumerate(self.params):
            mean_square_grad = theano.shared(
                np.zeros_like(param.get_value(), dtype=theano.config.floatX), name=param.name + str(self.network.R)+'_msg')

            mean_square_grad.name = 'mean_square_grad_' + param.name

            # Store variable in self.mean_square_grads for monitoring.
            self.mean_square_grads[param.name] = mean_square_grad

            # Accumulate gradient
            new_mean_squared_grad = (self.decay * mean_square_grad +
                                     (1 - self.decay) * T.sqr(grads[i]))

            # Compute update
            scaled_lr = self.lr
            rms_grad_t = T.sqrt(new_mean_squared_grad)
            rms_grad_t = T.maximum(rms_grad_t, self.epsilon)
            delta_x_t = - scaled_lr * grads[i] / rms_grad_t

            # Apply update
            yield mean_square_grad, T.cast(new_mean_squared_grad, dtype=theano.config.floatX)
            yield param, T.cast(param + delta_x_t, dtype=theano.config.floatX)










class AdaDelta(SGD):
    '''This is a base class for all trainers.'''

    def __init__(self, network, decay=0.95, profile=False,L1=0, lr=0.3, momentum=0.4, epochs=0, num_updates=10,  valid_freq=10, L2=0.00005, compile=True):
        SGD.__init__(self,network=network, profile=profile, lr=lr, momentum=momentum, epochs=epochs, num_updates=num_updates,  valid_freq=valid_freq,
                     L2=L2, L1=L1, compile=False)

        self.decay = decay

        assert(0 <= self.decay <= 1)

        if compile:
            self.f_eval = theano.function(
                network.inputs, self.cost_exprs_evaluate)

            self.f_learn = theano.function(
                network.inputs,
                self.cost_exprs_update,
                updates=list(self.learning_updates()),mode="FAST_RUN")

    def learning_updates(self):
        # This code computes updates only for given R, so it drops last dimension. Plus soe theano magic to circumvent its graph comp.
        grads = self.grads
        for i, param in enumerate(self.params):

            mean_square_grad = theano.shared(
                np.zeros_like(param.get_value(), dtype=theano.config.floatX), name=param.name + str(self.network.R)+'_msg')

            mean_square_dx = theano.shared(
                np.zeros_like(param.get_value(), dtype=theano.config.floatX), name=param.name + str(self.network.R)+'_dx')


            # Accumulate gradient
            new_mean_squared_grad = (
                self.decay * mean_square_grad +
                (1 - self.decay) * T.sqr(grads[i])
            )

            # Compute update
            epsilon = self.lr
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - (rms_dx_tm1 / rms_grad_t) * grads[i]

            # Accumulate updates
            new_mean_square_dx = (
                self.decay * mean_square_dx +
                (1 - self.decay) * T.sqr(delta_x_t)
            )

            # Apply update
            yield mean_square_grad, T.cast(new_mean_squared_grad, dtype=theano.config.floatX)
            yield mean_square_dx, T.cast(new_mean_square_dx, dtype=theano.config.floatX)
            yield param,  param + 1*T.cast(delta_x_t, dtype=theano.config.floatX)

import sys

class Scipy(SGD):
    '''General trainer for neural nets using `scipy.optimize.minimize`.'''

    METHODS = ('l-bfgs-b', 'cg', 'dogleg', 'newton-cg', 'trust-ncg')

    def __init__(self, network, num_updates=10, L2=0.0001, L1=0,  method = 'l-bfgs-b'):
        SGD.__init__(self,network=network, L2=L2, L1=L1, num_updates=num_updates, compile=False)

        self.method = method

        logging.info('compiling gradient function')

        self.f_eval = theano.function(network.inputs, self.cost_exprs_update)
        self.f_grad = theano.function(network.inputs, T.grad(self.cost, self.params))


    @timed
    def function_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x.astype(np.float32)))
        return np.mean([self.f_eval(y)[0] for y in train_set]).astype(np.float64) #lbfgs fortran code wants float64. meh

    @timed
    def gradient_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x.astype(np.float32)))
        grads = [[] for _ in range(len(self.params))]
        for y in train_set:
            for i, g in enumerate(self.f_grad(y)):
                grads[i].append(np.asarray(g))
        G = self.arrays_to_flat([np.mean(g, axis=0) for g in grads]).astype(np.float64) #lbfgs fortran code wants float64. meh
        return G

    def train_minibatch(self, x):
        def display(p):
            self.set_params(self.flat_to_arrays(p.astype(np.float32)))
            costs = self.f_eval(x)
            cost_desc = ' '.join(
                '%s=%.6f' % el for el in zip(self.cost_names, costs))
            print('scipy %s %s' %
                  (self.method, cost_desc))
            sys.stdout.flush()


        try:
            res = scipy.optimize.minimize(
                fun=self.function_at,
                jac=self.gradient_at,
                x0=self.arrays_to_flat(self.best_params),
                args=([x], ),
                method=self.method,
                callback=display,
                options=dict(maxiter=2),
            )
        except KeyboardInterrupt:
            print('interrupted!')

        @timed
        def set():
            params = self.flat_to_arrays(res.x.astype(np.float32))
            self.set_params(params)

        set()


        return []

