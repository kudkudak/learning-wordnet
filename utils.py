"""
This file contains utility functions
"""

import time
import os
import cPickle
import numpy
from config.config import logger, c
import pickle
import numpy as np
import glob
import mmap
import pandas as pd
from scipy import sparse
import theanets


# This will be used when used cache_ram=True
mem_storage = {}

def timed(func):
    """ Decorator for easy time measurement """

    def timed(*args, **dict_args):
        tstart = time.time()
        result = func(*args, **dict_args)
        tend = time.time()
        print("{0} ({1}, {2}) took {3:2.4f} s to execute".format(func.__name__, len(args), len(dict_args), tend - tstart))
        return result

    return timed


cache_dict = {}


def cached_in_memory(func):
    global cache_dict

    def func_caching(*args, **dict_args):

        key = (func.__name__, args, frozenset(dict_args.items()))
        if key in cache_dict:
            return cache_dict[key]
        else:
            returned_value = func(*args, **dict_args)
            cache_dict[key] = returned_value
            return returned_value

    return func_caching


def theanets_load(key, theanonets_type):
    file_name_network = os.path.join(c["CACHE_DIR"], key + ".network.pkl")
    file_name_network_structure = os.path.join(c["CACHE_DIR"], key + ".network.structure.pkl")
    e = None
    with open(file_name_network_structure, "r") as f:
        e = theanets.Experiment(theanonets_type, **pickle.load(f))
    e.load(file_name_network)
    return e


def theanets_check(key):
    file_name_network = os.path.join(c["CACHE_DIR"], key + ".network.pkl")
    return os.path.exists(file_name_network)


def theanets_save(key, val):
    file_name_network = os.path.join(c["CACHE_DIR"], key + ".network.pkl")
    file_name_network_structure = os.path.join(c["CACHE_DIR"], key + ".network.structure.pkl")
    with open(file_name_network_structure, "w") as f:
        pickle.dump({"layers": val.network.layers, "hidden_activation": val.network.hidden_activation,
                     "output_activation": val.network.output_activation
                     }, f)
    val.network.save(file_name_network)


from sklearn.externals import joblib

import os

def scikit_load(key):
    dir = os.path.join(c["CACHE_DIR"], key)
    file_name = os.path.join(os.path.join(c["CACHE_DIR"],dir), key + ".pkl")
    return joblib.load(file_name)

def scikit_check(key):
    dir = os.path.join(c["CACHE_DIR"], key)
    return len(glob.glob(os.path.join(os.path.join(c["CACHE_DIR"],dir), key + ".pkl*"))) > 0

def scikit_save(key, val):
    dir = os.path.join(c["CACHE_DIR"], key)
    os.system("mkdir "+dir)
    file_name = os.path.join(dir, key + ".pkl")
    joblib.dump(val, file_name)


def scipy_csr_load(key):
    file_name = os.path.join(c["CACHE_DIR"], key + ".npz")
    f = np.load(file_name)
    return sparse.csr_matrix((f["arr_0"], f["arr_1"], f["arr_2"]), shape=f["arr_3"])


def scipy_csr_check(key):
    return os.path.exists(os.path.join(c["CACHE_DIR"], key + ".npz"))


def scipy_csr_save(key, val):
    file_name = os.path.join(c["CACHE_DIR"], key)
    np.savez(file_name, val.data, val.indices, val.indptr, val.shape)


def pandas_save_fnc(key, val):
    file_name = os.path.join(c["CACHE_DIR"] + key + ".msg")
    val.to_msgpack(file_name)


def pandas_check_fnc(key):
    return os.path.exists(os.path.join(c["CACHE_DIR"] + key + ".msg"))


def pandas_load_fnc(key):
    file_name = os.path.join(c["CACHE_DIR"] + key + ".msg")
    return pd.read_msgpack(file_name)


def numpy_save_fnc(key, val):
    if isinstance(val, tuple):
        raise "Please use list to make numpy_save_fnc work"
    # Note - not using savez because it is reportedly slow.
    if isinstance(val, list):
        logger.info("Saving as list")
        save_path = os.path.join(c["CACHE_DIR"], key)
        save_dict = {}
        for id, ar in enumerate(val):
            save_dict[str(id)] = ar
        np.savez(save_path, **save_dict)
    else:
        logger.info("Saving as array " + str(val.shape))
        np.save(os.path.join(c["CACHE_DIR"], key + ".npy"), val)


def numpy_check_fnc(key):
    return len(glob.glob(os.path.join(c["CACHE_DIR"], key + ".np*"))) > 0


def numpy_load_fnc(key):
    if os.path.exists(os.path.join(c["CACHE_DIR"], key + ".npz")):
        # Listed numpy array

        savez_file = np.load(os.path.join(c["CACHE_DIR"], key + ".npz"))

        ar = []

        for k in sorted(list((int(x) for x in savez_file))):
            logger.info("Loading " + str(k) + " from " + str(key) + " " + str(savez_file[str(k)].shape))
            ar.append(savez_file[str(k)])
        return ar
    else:
        return np.load(os.path.join(c["CACHE_DIR"], key + ".npy"))


import hashlib
import sys


def generate_key(func_name, args, dict_args_original, skip_args):
    args_concat = [v for key, v in sorted(dict_args_original.iteritems()) if key not in skip_args]




    # Get serialized arguments (function name, or string of v if is not reference checked in ugly way
    args_serialized = \
        '_'.join(sorted([
            v.__name__
            if hasattr(v, '__call__')
            else
            (str(v) if len(str(v)) < 200 else hashlib.md5(str(v)).hexdigest())
            for v in args_concat if hasattr(v, '__call__') or str(v).find("0x") == -1]))



    logger.info("Serialized args to " + args_serialized)

    key = func_name + "_" + ''.join((a for a in args_serialized if a.isalnum() or a in "!@#$%^&**_+-"))

    full_key = func_name + "(" + "".join([str(k)+"="+(str(v) if len(str(v))<200 else hashlib.md5(str(v)).hexdigest())
                                          for k,v in sorted(dict_args_original.iteritems()) if key not in skip_args])

    if len(key) > 400:
        key = key[0:400]

    return key, full_key


def cached_FS(save_fnc=None, load_fnc=None, check_fnc=None, skip_args=None, cache_ram=False, use_cPickle=False):
    """
    To make it work correctly please pass parameters to function as dict

    @param save_fnc, load_fnc function(key, returned_value)
    @param check_fnc function(key) returning True/False
    """
    if not skip_args:
        skip_args = {}

    def cached_HDD_inner(func):
        def func_caching(*args, **dict_args):
            """
            Use mmap only for files around 1/2 memory
            """
            if len(args) > 0:
                raise Exception("For cached_FS functions pass all args by dict_args (ensures cache resolution)")


            # ## Dump special arguments
            dict_args_original = dict(dict_args)
            dict_args_original.pop("use_mmap", None)
            dict_args_original.pop("force_reload", None)

            key, fullkey = generate_key(func.__name__, args, dict_args_original, skip_args)

            # For retrievel
            if not os.path.exists(os.path.join(c["CACHE_DIR"], "cache_dict.txt")):
                with open(os.path.join(c["CACHE_DIR"], "cache_dict.txt"),"w") as f:
                    f.write("File containing hashes explanation for refernce\n ==== \n")

            with open(os.path.join(c["CACHE_DIR"], "cache_dict.txt"),"a") as f:
                f.write(key + "\n" + fullkey +"\n")


            if cache_ram and key in mem_storage:
                print("Reading from cache ram")
                return mem_storage[key]

            logger.info("Checking key " + key)
            cache_file_default = os.path.join(c["CACHE_DIR"], str(key) + ".cache.pkl")
            exists = os.path.exists(cache_file_default) if check_fnc is None else check_fnc(key)
            if exists and not "force_reload" in dict_args:
                logger.info("Loading (pickled?) file")

                if load_fnc:
                    return load_fnc(key)
                else:
                    with open(cache_file_default, "r") as f:
                        if "use_mmap" in dict_args:
                            g = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                            obj = cPickle.load(g) if use_cPickle  else pickle.load(g)
                            g.close()
                            return obj
                        else:
                            obj = cPickle.load(f) if use_cPickle else pickle.load(f)
                            return obj
            else:
                logger.info("Cache miss or force reload. Caching " + key)
                returned_value = func(*args, **dict_args_original)
                if save_fnc:
                    save_fnc(key, returned_value)
                else:
                    with open(cache_file_default, "w") as f:
                        if use_cPickle:
                            cPickle.dump(returned_value, f)
                        else:
                            pickle.dump(returned_value, f)

                if cache_ram:
                    mem_storage[key] = returned_value

                return returned_value

        return func_caching

    return cached_HDD_inner


cached_FS_list_np = cached_FS(numpy_save_fnc, numpy_load_fnc, numpy_check_fnc)

if __name__ == "__main__":
    @timed
    @cached_FS()
    def check_pickle(k=10, d=20, single=False):
        x = np.ones(shape=(3000, 1000))
        y = np.ones(shape=(10, 10))
        return [x, y] if not single else x

    @timed
    @cached_FS(save_fnc=numpy_save_fnc, load_fnc=numpy_load_fnc, check_fnc=numpy_check_fnc)
    def check_np(k=10, d=20, single=False):
        x = np.ones(shape=(3000, 1000))
        y = np.ones(shape=(10, 10))
        return [x, y] if not single else x

    print(check_pickle(single=False)[0].shape)
    print(check_np(single=False)[0].shape)

    """
        results: check_pickle - file 84kb, 5.37 to save 0.6351 to load
                 check_np - file 23kB, 0.69 to save, 0.0849 to load
    """
