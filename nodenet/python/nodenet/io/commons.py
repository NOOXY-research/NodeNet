# nodenet/io/commons.py
# Description:
# "commons.py" provide access of files and folders.
# Copyright 2018 NOOXY. All Rights Reserved.

import os
import pickle

def makedirs(dirspath):
    if not os.path.exists(dirspath):
        os.makedirs(dirspath)

# Save object to file
def dopickle(obj, filename):
    btyes = pickle.dumps(obj)
    f = open(filename, 'wb')
    f.write(btyes)
    f.close()

# Load object to file
def unpickle(filename):
    f = open(filename, 'rb')
    btyes = f.read()
    obj = pickle.loads(btyes)
    f.close()
    return obj
