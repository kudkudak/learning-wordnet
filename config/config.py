'''
Basic config for whole project. Modify those variables to reflect changes
'''
import os
import sys
import logging





import os
base_dir = "/home/moje/Projekty_big/diamentowy_grant/tfml-2014-wordnet-prediction/knowledge-learner"



# Logger
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('knowledge_learner')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(funcName)s - %(asctime)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False
ch_file = logging.FileHandler(os.path.join(base_dir, "knowledge_learner.log"))
ch_file.setLevel(level = logging.INFO)
ch_file.setFormatter(formatter)
logger.addHandler(ch_file)




# Configurations
c = {
    "CACHE_DIR" : os.path.join(base_dir, "cache"),
    "TRAIN_FILE": "train.txt",
    "WORDNET_DIR":os.path.join(base_dir, "data/Wordnet"),
    "WORDNET_DEV_SIZE":5218,
    "WORDNET_RELATIONS":11,
    "BASE_DIR":base_dir,
    "TEST_FILE" : "test_all.txt",
    "WORDNET_TEST_SIZE": 21088,
    "WORDNET_DEV_SIZE": 5218,
    "REL_IDX":1
}
