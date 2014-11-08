"""
This file contains functions retrieving data from disk
"""
from config import *
from utils import *

import scipy
import scipy.io

REL_IDX=1

def dataset_to_dir(dataset="Wordnet"):
    if dataset == "Wordnet":
        return c["WORDNET_DIR"]
    else:
        return ""

def get_data_desc(dataset="Wordnet"):
    data_dir = dataset_to_dir(dataset=dataset)
    def entity_to_idx(entity_filename):
        d = {}
        with open(entity_filename, "r") as f:
            for l in f:
                d[l.strip("\n")] = len(d)
        return d
    def rel_to_idx(rel_filename):
        d = {}
        with open(rel_filename, "r") as f:
            for l in f:
                d[l.strip("\n")] = len(d)
        return d

    embed = scipy.io.loadmat(os.path.join(data_dir, "initEmbed.mat"))



    # Build inverse dict for words and validate

    word_to_embedding = {w[0]: idx for (idx,w) in enumerate(embed['words'][0])}
    embedding = embed['We']

    basic = {"ent": entity_to_idx(os.path.join(data_dir, "entities.txt")), \
            "rel": rel_to_idx(os.path.join(data_dir, "relations.txt")),
            "word_to_embedding":word_to_embedding,\
            "U":embedding.T
            }

    return basic


import itertools
def get_test_data(dataset="Wordnet"):
    data_desc = get_data_desc(dataset=dataset)
    ent, rel = data_desc['ent'], data_desc['rel']

    data = []
    row = []
    Y = []
    with open(os.path.join(dataset_to_dir(dataset), "test_all.txt"), "r") as f:
        for i, l in enumerate(f):
            row = l.strip("\n").split()
            e1, r, e2, y = l.strip("\n").split()
            if i%2 == 0:
                data.append([ent[e1], rel[r], ent[e2]])
            else:
                data[-1].append(ent[e1])
                data[-1].append(ent[e2])

            #Y.append(y)

    #indices = np.random.permutation(range(X.shape[0]))

    return np.array(data, dtype="int32") #[indices], Y[indices]
import itertools
def get_train_raw_data(dataset="Wordnet"):
    data_desc = get_data_desc(dataset=dataset)
    ent, rel = data_desc['ent'], data_desc['rel']

    data = []
    row = []
    with open(os.path.join(dataset_to_dir(dataset), "train.txt"), "r") as f:
        for l in f:
            e1, r, e2 = l.strip("\n").split()
            data.append([ent[e1], rel[r], ent[e2]])
    return data

#Can be used to corrupt both training and testing data
def get_corrupted_randomly_data(data, dataset="Wordnet", multiplication=10):
    X = np.zeros(shape=(len(data)*multiplication, 5), dtype="int32") # 4 because we will add corrupted entity
    data_desc = get_data_desc(dataset=dataset)
    ent, rel = data_desc['ent'], data_desc['rel']

    idx = 0
    batch_size = len(data)
    for batch in range(multiplication):
        for id, row in enumerate(data):
            X[batch_size*batch + id, 0:3] = row

            if np.random.randint(2) == 0:
                X[batch_size*batch + id, 3] = np.random.randint(0, len(ent)) # why not corrupt relation? especially slightly.
                X[batch_size*batch + id, 4] = row[2]
            else:
                X[batch_size*batch + id, 4] = np.random.randint(0, len(ent)) # why not corrupt relation? especially slightly.
                X[batch_size*batch + id, 3] = row[0]


            #TODO: explore better corruption schemas taking into consideration proximity of relations or something like that
            #Also proximity of entities.
            #But we can delearn good relation. what then? damn!
    # Each data --> 2 data entries
    # number of entries = multiplication * data_size

    return np.random.permutation(X)



@timed
@cached_FS(use_cPickle=True)
def prepare_experiment_data(dataset="Wordnet", multiplication=10, CV=0, batch_size=20000):
    assert(CV == 0)

    #TODO: add corruption for test data aswell and do CV
    train_data = get_train_raw_data(dataset=dataset)
    X = get_corrupted_randomly_data(train_data, dataset=dataset, multiplication=multiplication)
    data_desc = get_data_desc(dataset=dataset)

    test_data = get_test_data(dataset=dataset)

    # Prepare sparse matrix E
    data = []
    indices = []
    indptr = [0] # shape len(data_desc["entity"]) + !
    import itertools

    for e,idx in data_desc["ent"].iteritems():
        words = e[2:].split("_")[0:-1]
        indexes = [data_desc["word_to_embedding"].get(w, data_desc["word_to_embedding"]["unknown"]) for w in words]
        data += [1 for i in xrange(len(words))]
        indices += indexes
        indptr.append(indptr[-1] + len(indexes))

    print(len(indptr))

    E = scipy.sparse.csr_matrix((np.array(data, dtype="float64"), np.array(indices, dtype="int64"), np.array(indptr, dtype="int64")),\
                                shape=(len(data_desc["ent"]), data_desc["U"].shape[0]))

    ent, rel = data_desc['ent'], data_desc['rel']

    X_rel = []
    X_test_rel = []
    for rel_idx in range(len(rel)):
        X_rel.append(X[X[:,REL_IDX]==rel_idx])
        X_test_rel.append(test_data[test_data[:,REL_IDX]==rel_idx])


    return {"X":X_rel, "X_test":X_test_rel, "U":data_desc["U"], "E":E, "ent":ent, "rel":rel}



def generate_batches(X, X_test, batch_size=100, tr_batch_count = None, randomize=0):
    if tr_batch_count is not None:
        batch_size = X.shape[0]/tr_batch_count

    M = X_test
    eq = M.shape[0] - M.shape[0]%batch_size
    if eq > 0:
        X_test_batches=np.split(M[0:eq], eq/batch_size)
        X_test_batches.append(M[eq:])
    else:
        X_test_batches = [X_test]

    M = X
    eq = M.shape[0] - M.shape[0]%batch_size
    if eq > 0:
        X_batches=np.split(M[0:eq], eq/batch_size)
        X_batches.append(M[eq:])
    else:
        X_batches = [X]

    return X_batches, X_test_batches
##TODO: add corruption schema get_corrupted_allowable, and then we can work on both train and test data and perform cross validation.


##TODO: add corruption schema get_corrupted_allowable, and then we can work on both train and test data and perform cross validation.

if __name__ == "__main__":
    X = get_corrupted_randomly_data(data=get_train_raw_data(dataset="Wordnet"), dataset="Wordnet")
    w = get_train_raw_data(dataset="Wordnet")
    #Does get_corrupted work correctly? (without permutation should print same thing)
    print(X[0])
    print(X[len(w)])
    E = prepare_experiment_data(dataset="Wordnet", multiplication=10, CV=0, force_reload=True, use_mmap=True)