# nodenet/interface/console/input.py
# Description:
# "input.py" provide console's input interface.
# Copyright 2018 NOOXY. All Rights Reserved.

def input_integer(prompt=True, string=''):
    if prompt:
        print('Input integer number.')

    if string == '':
        value = input('>>>\n')
    else:
        value = string

    return int(value)

def input_float(prompt=True, string=''):
    if prompt:
        print('Input float number.')

    if string == '':
        value = input('>>>\n')
    else:
        value = string

    return float(value)

def input_string(prompt=True, string=''):
    if prompt:
        print('Input string')

    if string == '':
        value = input('>>>\n')
    else:
        value = string

    return value

def input_list(seperator=' ', prompt=True, string=''):
    if prompt:
        print('Input series of values. (list)')
        print('Enter values one by one seperated by "'+str(seperator)+'".')
        print('e.g. >>>1'+str(seperator)+'2'+str(seperator)+'3')

    if string == '':
        value = input('>>>\n')
    else:
        value = string

    return value.split(seperator)

def input_dict(seperator=' ', assignop='=', prompt=True, string=''):
    if prompt:
        print('Input series of values and its tags. (dict)')
        print('Enter values and tags one by one seperated by "'+str(seperator)+'".')
        print('And assign values by operator "'+str(assignop)+'".')
        print('e.g. >>>age'+str(assignop)+'18'+str(seperator)+'height'+str(assignop)+'175')

    if string == '':
        value = input('>>>\n')
    else:
        value = string

    result = {}
    for x in value.split(seperator):
        pair = x.split(assignop)
        result[pair[0]] = pair[1]
    return result

def complex_input(handler_dict, seperator=' ', prompt=True, string=''):
    if string == '':
        value = input('>>>\n')
    else:
        value = string

    handler_dict[value.split(seperator)[0]](value.replace(value.split(seperator)[0]+seperator, ''))
