from model import *
from data_api import *
import sys

X = prepare_experiment_data(dataset="Wordnet", CV=0)

params = cPickle.load(open(sys.argv[1], "r"))


R = range(11)
networks = createNetworks(entity_matrix=X["E"], embedding_matrix=params["embedding_matrix"], k=4, only_metric=False)
pars = params["network_params"]

for i, n in enumerate(networks):
    n.load_params(pars[i])


X_test = split_per_relation(X["X_test"], rel=range(11))

best_splits = []
accs_history = []
networks[0].update_EU()
for r in range(11):
    test_values = np.linspace(-4.5,4.5,num=50)
    f_eval = theano.function(networks[r].inputs, networks[r].f_prop())
    accuracies = []
    for split in test_values:
        results_pos = f_eval(X_test[r][:,0:3]).reshape(-1)
        results_neg = f_eval(X_test[r][:,[3,1,4]]).reshape(-1)
        accuracies.append((sum(results_pos > split) + sum(results_neg < split)) /(2*float(X_test[r].shape[0])))
    best_splits.append(test_values[np.argmax(accuracies)])
    accs_history.append(accuracies)

print(best_splits)


accs = []
global_acc = 0
for r in range(11):

    f_eval = theano.function(networks[r].inputs, networks[r].f_prop())
    results_pos = f_eval(X_test[r][:,0:3]).reshape(-1)
    results_neg = f_eval(X_test[r][:,[3,1,4]]).reshape(-1)
    cnt = (sum(results_pos > best_splits[r]) + sum(results_neg < best_splits[r]))
    accs.append(cnt /(2*float(X_test[r].shape[0])))
    global_acc += cnt
global_acc = global_acc / (2*float(X["X_test"].shape[0]))

print(global_acc)
print(accs)
