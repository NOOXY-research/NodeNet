# nodenet/interface/consoles/commons.py
# Description:
# "commons.py" provide commons console function that can be use widely.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
from . import colors
import nodenet.variables as var
import subprocess as sp

def logo():
    printer('')
    printer(colors.color('Bold')+colors.color('Red')+'88b 88  dP\'Yb   dP\'Yb  Yb  dP Yb  dP  TM'+colors.color('end'))
    printer(colors.color('Bold')+colors.color('Blue')+'88Yb88 dP   Yb dP   Yb  YbdP   YbdP  '+colors.color('end'))
    printer(colors.color('Bold')+colors.color('Green')+'88 Y88 Yb   dP Yb   dP  dPYb    88   '+colors.color('end'))
    printer(colors.color('Bold')+colors.color('Yellow')+'88  Y8  YbodP   YbodP  dP  Yb   88 '+colors.color('end')+'  nodenet. ')
    printer('')
    printer(var.nodenet['Copyright'])
    printer('')
    printer(var.nodenet['Version'])
    printer('For more information or update ->'+var.nodenet['Website']+'.')
    printer('')

def printer(string):
    sp.call('echo -e "'+string+'"',shell=True)

def log(tag_name, message, msg_color=None, tag_color=None):
    message = message.replace('\n', '\n'+tag(tag_name))
    tagstr = tag(tag_name)
    if msg_color is not None:
        message = colors.color(msg_color)+message+colors.color('end')
    if tag_color is not None:
        tagstr =colors.color(tag_color)+tagstr+colors.color('end')
    printer(tagstr+message)

def tag(tag_name):
    tag_length = 10
    padding_left = int((tag_length-len(tag_name))/2)
    padding_right = tag_length-padding_left-len(tag_name)
    string = '['+'.'*padding_left+tag_name+'.'*padding_right+'] '
    return string
