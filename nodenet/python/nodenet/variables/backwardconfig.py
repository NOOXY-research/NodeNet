# nodenet/lconfiables/backwardconfig.py
# Description:
# "backwardconfig.py" provide parameters for backwardconfig.
# Copyright 2018 NOOXY. All Rights Reserved.

import nodenet.learningalgorithm as la
from . import learningconfig as lconf

backward_config = {
    'learning_algorithm' : la.adam,
    'learning_configuration' : lconf.adam_config,
    'trace' : False,
    'dropout_keep' : 1,
}

backward_training_config = {
    'learning_algorithm' : la.adam,
    'learning_configuration' : lconf.adam_config,
    'trace' : True,
    'dropout_keep' : 1,
}

backward_dropout_config = {
    'learning_algorithm' : la.adam,
    'learning_configuration' : lconf.adam_config,
    'trace' : False,
    'dropout_keep' : 0.6,
}

backward_dropout_training_config = {
    'learning_algorithm' : la.adam,
    'learning_configuration' : lconf.adam_config,
    'trace' : True,
    'dropout_keep' : 0.6,
}
