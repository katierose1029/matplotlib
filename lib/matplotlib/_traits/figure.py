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

#for monkey patching
import matplotlib.figure as b_figure
