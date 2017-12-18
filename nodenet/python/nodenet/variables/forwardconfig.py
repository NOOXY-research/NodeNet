# nodenet/variables/forwardconfig.py
# Description:
# "forwardconfig.py" provide parameters for forwardconfig.
# Copyright 2018 NOOXY. All Rights Reserved.

forward_config = {
    'trace' : False,
    'dropout_keep' : 1,
}

forward_training_config = {
    'trace' : True,
    'dropout_keep' : 1,
}

forward_dropout_config = {
    'trace' : False,
    'dropout_keep' : 0.6,
}

forward_dropout_training_config = {
    'trace' : True,
    'dropout_keep' : 0.6,
}
