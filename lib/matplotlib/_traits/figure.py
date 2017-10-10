"""
matplotlib._traits.figure.Figure

Note: from base Figure class
The figure module provides the top-level
:class:`~matplotlib.artist.Artist`, the :class:`Figure`, which
contains all the plot elements.  The following classes are defined

:class:`SubplotParams`
    control the default spacing of the subplots

:class:`Figure`
    top level container for all plot elements
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import warnings

import numpy as np

from matplotlib import rcParams
from matplotlib import docstring
from matplotlib import __version__ as _mpl_version

import matplotlib.artist as martist
from matplotlib.artist import Artist, allow_rasterization
# import matplotlib._traits.artist as martist
# from matplotlib._traits.artist import Artist, allow_rasterization

import matplotlib.cbook as cbook

from matplotlib.cbook import Stack, iterable

from matplotlib import image as mimage
from matplotlib.image import FigureImage

import matplotlib.colorbar as cbar

from matplotlib.axes import Axes, SubplotBase, subplot_class_factory
from matplotlib.blocking_input import BlockingMouseInput, BlockingKeyMouseInput
from matplotlib.gridspec import GridSpec
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
from matplotlib.projections import (get_projection_names,
                                    process_projection_requirements)
from matplotlib.text import Text, _process_text_args
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
                                   TransformedBbox)
from matplotlib.backend_bases import NonGuiException

docstring.interpd.update(projection_names=get_projection_names())


from traitlets import HasTraits, Any, Instance, Unicode, Float, Bool, Int, validate, observe, default

# for monkey patching
import matplotlib.figure as b_figure

def _stale_figure_callback(self,val):
	if self.figure:
		self.figure.stale = val

class AxesStack(Stack):
	def __init__(self):
        Stack.__init__(self)
        self._ind = 0

class SubplotParams(object):
	def __init__(self, left=None, bottom=None, right=None, top=None,wspace=None, hspace=None):
		self.validate = True
        self.update(left, bottom, right, top, wspace, hspace)
	
class Figure(Artist, HasTraits, b_figure.Figure):


    figsize = instance("matplotlib.figure.figsize", default_value=rcParams['figure.figsize'], allow_none = True)
    dpi = instance("matplotlib.figure.dpi", default_value=dpi = rcParams['figure.dpi'], allow_none = True)
    facecolor = instance("matplotlib.figure.facecolor", default_value=rcParams['figure.facecolor'], allow_none = True)
    edgecolor = instance("matplotlib.figure.edgecolor", default_value=rcParams['figure.edgecolor'], allow_none = True)
    linewidth = Float(default_value = 0.0)
    frameon = Bool(default_value=rcParams['figure.frameon'],allow_none = True)
    #subplotpars = instance("TODO:", default_value = None, allow_none = True);
    tight_layout = instance("matplotlib.figure.autolayout", default_value=None)
    #Artist.__init__(self) #is this necessary?

    # TODO: Only required if parameters are passed in by user (or when validating?)
    ''' 
    if not np.isfinite(figsize).all():
        raise ValueError('figure size must be finite not '
                         '{}'.format(figsize))
    '''

    '''
    @default("figsize")
    def _figsize_default(self,proposal):
        from matplotlib.figure import figsize
    '''

    #TODO: Finish up the validates. Check which ones need default functions (and possibly observe function)
    @validate("figsize")
    def _figsize_validate(self, proposal):
        if not self.figsize == proposal.value:
            self.figsize = proposal.value

    @validate("dpi")
    def _dpi_validate(self, proposal):
        return None

    @validate("facecolor")
    def _facecolor_validate(self, proposal):
        return None

    @validate("edgecolor")
    def _edgecolor_validate(self, proposal):
        return None

    @validate("linewidth")
    def _linewidth_validate(self, proposal):
        return None

    @validate("frameon")
    def _frameon_validate(self, proposal):
        return None

    #@validate("subplotpars") TODO: make a validate for subplot

    @validate("tight_layout")
    def _tight_layout_validate(self, proposal):
        return None
