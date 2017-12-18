# nodenet/interface/consoles/colors.py
# Description:
# "colors.py" provide color py indexing.
# Copyright 2018 NOOXY. All Rights Reserved.

colors = {
    'end' : 0,
    'Bold' : 1,
    'Dim' : 2,
    'Red' : 31,
    'Green' : 32,
    'Yellow' : 33,
    'Blue' : 34
}

def color(color):
    return encode(colors[color])

def encode(index):
    return '\e['+str(index)+'m'
