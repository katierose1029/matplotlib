from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from traitlets import *

import six
from collections import OrderedDict, namedtuple

import re
import warnings
import inspect
import numpy as np
import matplotlib
import matplotlib.cbook as cbook
from matplotlib.cbook import mplDeprecation
from matplotlib import docstring, rcParams
from matplotlib.transforms import (Bbox, IdentityTransform, TransformedBbox,
                         TransformedPatchPath, TransformedPath, Transform)
from matplotlib.path import Path
from functools import wraps
from contextlib import contextmanager


class TraitProxy(TraitType):

    def __init__(self, trait):
        self.__trait = trait

    def instance_init(self, obj):
        self.__trait.instance_init(obj)

    def class_init(self, cls, name):
        self.__trait.class_init(cls, name)

    def set(self, obj, val):
        self.__trait.set(obj, val)

    def get(self, obj, cls):
        return self.__trait.get(obj, cls)

    def __getattr__(self, name):
        return getattr(self.__trait, name)

class Perishable(TraitProxy):

    def set(self, obj, val):
        super(Perishable, self).set(obj, val)
        obj.stale = True

class TransformTrait(TraitType):

    #TODO: assure that an instance of transform is being passed
    default_value = None
    allow_none = True
    info_text = 'matplotlib.transforms.Transform'


    #Question: why use the get_transform function as the validate?
    # I understand that there is a logic involving how to handle None and
    #returning IdentityTransform() if there is None, but at the time,
    #how does that validate it?
    def validate(self, obj, value):
        if value is None:
            return IdentityTransform()
        elif (not isinstance(value, Transform) and hasattr(value, '_as_mpl_transform')):
            # TO DO: finish this
            # return trans
            # self._transform = self._transform._as_mpl_transform(self.axes)
            return value

class PathTrait(TraitType):

    #TODO: assure that an instance of path is being passed.
    default_value = None
    allow_none = True
    info_text = 'matplotlib.path.Path'
    def validate(self, obj, value):
        if value is None:
            return IdentityTransform()
        elif (not isinstance(value, Path) and hasattr(value, '_as_mpl_transform')):
            # TO DO: finish this
            # return trans
            # self._transform = self._transform._as_mpl_transform(self.axes)
            return value



#this is in traits but for some reason, my code could not pick up on it?
class Callable(TraitType):
    """A trait which is callable.

    Notes
    -----
    Classes are callable, as are instances
    with a __call__() method."""

    info_text = 'a callable'

    def validate(self, obj, value):
        if six.callable(value):
            return value
        else:
            self.error(obj, value)