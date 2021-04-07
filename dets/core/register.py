from collections import OrderedDict


import inspect
import importlib


WORKSPACE = {}


def register(clas):
    '''register
    '''    
    assert clas.__name__ not in WORKSPACE, ''

    WORKSPACE[clas.__name__] = clas
    # importlib.import_module(clas.__module__)

    return clas 


def create(name):
    '''create
    '''
    pass

