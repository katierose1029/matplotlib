"""
matplotlib.lines.Line2D refactored in traitlets
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import warnings
import numpy
import numpy as np

from matplotlib import artist, colors as mcolors, docstring, rcParams
from .artist import Artist, allow_rasterization
import matplotlib.artist as b_artist

from matplotlib._traits.artist import Artist, allow_rasterization
from matplotlib.cbook import (
    iterable, is_numlike, ls_mapper, ls_mapper_r, STEP_LOOKUP_MAP)
from matplotlib.markers import MarkerStyle
# from matplotlib.path import Path
# from matplotlib.transforms import Bbox, IdentityTransform
from matplotlib.transforms import Bbox, TransformedPath, IdentityTransform
# Imported here for backward compatibility, even though they don't
# really belong.
from numpy import ma
from matplotlib import _path
from matplotlib.markers import (
    CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
    CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE,
    TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN)

from traitlets import HasTraits, Any, Instance, Unicode, Float, Bool, Int, validate, observe, default, Type, List
from .traits import PathTrait, TransformedPathTrait

#for monkey patching into base lines
import matplotlib.lines as b_Line2D


def _get_dash_pattern(style):
    """Convert linestyle -> dash pattern
    """
    # go from short hand -> full strings
    if isinstance(style, six.string_types):
        style = ls_mapper.get(style, style)
    # un-dashed styles
    if style in ['solid', 'None']:
        offset, dashes = None, None
    # dashed styles
    elif style in ['dashed', 'dashdot', 'dotted']:
        offset = 0
        dashes = tuple(rcParams['lines.{}_pattern'.format(style)])
    #
    elif isinstance(style, tuple):
        offset, dashes = style
    else:
        raise ValueError('Unrecognized linestyle: %s' % str(style))

    # normalize offset to be positive and shorter than the dash cycle
    if dashes is not None and offset is not None:
        dsum = sum(dashes)
        if dsum:
            offset %= dsum

    return offset, dashes

def _scale_dashes(offset, dashes, lw):
    if not rcParams['lines.scale_dashes']:
        return offset, dashes

    scaled_offset = scaled_dashes = None
    if offset is not None:
        scaled_offset = offset * lw
    if dashes is not None:
        scaled_dashes = [x * lw if x is not None else None
                         for x in dashes]

    return scaled_offset, scaled_dashes

def segment_hits(cx, cy, x, y, radius):
    """
    Determine if any line segments are within radius of a
    point. Returns the list of line segments that are within that
    radius.
    """
    # Process single points specially
    if len(x) < 2:
        res, = np.nonzero((cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2)
        return res

    # We need to lop the last element off a lot.
    xr, yr = x[:-1], y[:-1]

    # Only look at line segments whose nearest point to C on the line
    # lies within the segment.
    dx, dy = x[1:] - xr, y[1:] - yr
    Lnorm_sq = dx ** 2 + dy ** 2  # Possibly want to eliminate Lnorm==0
    u = ((cx - xr) * dx + (cy - yr) * dy) / Lnorm_sq
    candidates = (u >= 0) & (u <= 1)
    #if any(candidates): print "candidates",xr[candidates]

    # Note that there is a little area near one side of each point
    # which will be near neither segment, and another which will
    # be near both, depending on the angle of the lines.  The
    # following radius test eliminates these ambiguities.
    point_hits = (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2
    #if any(point_hits): print "points",xr[candidates]
    candidates = candidates & ~(point_hits[:-1] | point_hits[1:])

    # For those candidates which remain, determine how far they lie away
    # from the line.
    px, py = xr + u * dx, yr + u * dy
    line_hits = (cx - px) ** 2 + (cy - py) ** 2 <= radius ** 2
    #if any(line_hits): print "lines",xr[candidates]
    line_hits = line_hits & candidates
    points, = point_hits.ravel().nonzero()
    lines, = line_hits.ravel().nonzero()
    #print points,lines
    return np.concatenate((points, lines))

def _mark_every_path(markevery, tpath, affine, ax_transform):
    """
    Helper function that sorts out how to deal the input
    `markevery` and returns the points where markers should be drawn.

    Takes in the `markevery` value and the line path and returns the
    sub-sampled path.
    """
    # pull out the two bits of data we want from the path
    codes, verts = tpath.codes, tpath.vertices

    def _slice_or_none(in_v, slc):
        '''
        Helper function to cope with `codes` being an
        ndarray or `None`
        '''
        if in_v is None:
            return None
        return in_v[slc]

    # if just a float, assume starting at 0.0 and make a tuple
    if isinstance(markevery, float):
        markevery = (0.0, markevery)
    # if just an int, assume starting at 0 and make a tuple
    elif isinstance(markevery, int):
        markevery = (0, markevery)
    # if just an numpy int, assume starting at 0 and make a tuple
    elif isinstance(markevery, np.integer):
        markevery = (0, markevery.item())

    if isinstance(markevery, tuple):
        if len(markevery) != 2:
            raise ValueError('`markevery` is a tuple but its '
                'len is not 2; '
                'markevery=%s' % (markevery,))
        start, step = markevery
        # if step is an int, old behavior
        if isinstance(step, int):
            #tuple of 2 int is for backwards compatibility,
            if not(isinstance(start, int)):
                raise ValueError('`markevery` is a tuple with '
                    'len 2 and second element is an int, but '
                    'the first element is not an int; '
                    'markevery=%s' % (markevery,))
            # just return, we are done here

            return Path(verts[slice(start, None, step)],
                        _slice_or_none(codes, slice(start, None, step)))

        elif isinstance(step, float):
            if not (isinstance(start, int) or
                    isinstance(start, float)):
                raise ValueError('`markevery` is a tuple with '
                    'len 2 and second element is a float, but '
                    'the first element is not a float or an '
                    'int; '
                    'markevery=%s' % (markevery,))
            #calc cumulative distance along path (in display
            # coords):
            disp_coords = affine.transform(tpath.vertices)
            delta = np.empty((len(disp_coords), 2),
                             dtype=float)
            delta[0, :] = 0.0
            delta[1:, :] = (disp_coords[1:, :] -
                                disp_coords[:-1, :])
            delta = np.sum(delta**2, axis=1)
            delta = np.sqrt(delta)
            delta = np.cumsum(delta)
            #calc distance between markers along path based on
            # the axes bounding box diagonal being a distance
            # of unity:
            scale = ax_transform.transform(
                np.array([[0, 0], [1, 1]]))
            scale = np.diff(scale, axis=0)
            scale = np.sum(scale**2)
            scale = np.sqrt(scale)
            marker_delta = np.arange(start * scale,
                                     delta[-1],
                                     step * scale)
            #find closest actual data point that is closest to
            # the theoretical distance along the path:
            inds = np.abs(delta[np.newaxis, :] -
                            marker_delta[:, np.newaxis])
            inds = inds.argmin(axis=1)
            inds = np.unique(inds)
            # return, we are done here
            return Path(verts[inds],
                        _slice_or_none(codes, inds))
        else:
            raise ValueError('`markevery` is a tuple with '
                'len 2, but its second element is not an int '
                'or a float; '
                'markevery=%s' % (markevery,))

    elif isinstance(markevery, slice):
        # mazol tov, it's already a slice, just return
        return Path(verts[markevery],
                    _slice_or_none(codes, markevery))

    elif iterable(markevery):
        #fancy indexing
        try:
            return Path(verts[markevery],
                    _slice_or_none(codes, markevery))

        except (ValueError, IndexError):
            raise ValueError('`markevery` is iterable but '
                'not a valid form of numpy fancy indexing; '
                'markevery=%s' % (markevery,))
    else:
        raise ValueError('Value of `markevery` is not '
            'recognized; '
            'markevery=%s' % (markevery,))

class Line2D(b_artist.Artist, HasTraits):
    """
    A line - the line can have both a solid linestyle connecting all
    the vertices, and a marker at each vertex.  Additionally, the
    drawing of the solid line is influenced by the drawstyle, e.g., one
    can create "stepped" lines in various styles.
    """

    #lineStyles
    lineStyles = _lineStyles = {  # hidden names deprecated
        '-':    '_draw_solid',
        '--':   '_draw_dashed',
        '-.':   '_draw_dash_dot',
        ':':    '_draw_dotted',
        'None': '_draw_nothing',
        ' ':    '_draw_nothing',
        '':     '_draw_nothing',
    }

    #drawStyles_l
    _drawStyles_l = {
        'default':    '_draw_lines',
        'steps-mid':  '_draw_steps_mid',
        'steps-pre':  '_draw_steps_pre',
        'steps-post': '_draw_steps_post',
    }

    #drawStyles_s
    _drawStyles_s = {
        'steps': '_draw_steps_pre',
    }

    # drawStyles should now be deprecated.
    drawStyles = {}
    drawStyles.update(_drawStyles_l)
    drawStyles.update(_drawStyles_s)
    # Need a list ordered with long names first:
    drawStyleKeys = list(_drawStyles_l) + list(_drawStyles_s)

    # Referenced here to maintain API.  These are defined in
    # MarkerStyle
    markers = MarkerStyle.markers
    filled_markers = MarkerStyle.filled_markers
    fillStyles = MarkerStyle.fillstyles

    zorder = 2
    validCap = ('butt', 'round', 'projecting')
    validJoin = ('miter', 'round', 'bevel')


    linewidth=Float(allow_none=True, default_value=rcParams['lines.linewidth'])
    linestyle = Unicode(allow_none=True, default_value=rcParams['lines.linestyle'])
    #TODO: not sure if this is correct?
    # color=Unicode(allow_none=True, default_value=None)
    color=Unicode(allow_none=True, default_value=rcParams['lines.color'])
    #TODO: check if import statement is in default function; set defaulty value there
    marker=Unicode(allow_none=True)
    # marker=Instance('matplotlib.markers',allow_none=False)
    # marker=Instance('matplotlib.markers.MarkerStyle',allow_none=False)
    markersize=Float(allow_none=True,default_value=rcParams['lines.markersize'])
    markeredgewidth=Float(allow_none=True,default_value=None)
    #TODO: not sure if this is correct?
    markerfacecolor=Unicode(allow_none=True, default_value=None)
    # same applies for the alternative face color
    markerfacecoloralt=Unicode(allow_none=True, default_value='none')
    #TODO: this gets passed into marker so I want to assume same for color however only accepts the following strings: ['full' | 'left' | 'right' | 'bottom' | 'top' | 'none']
    fillstyle=Unicode(allow_none=True, default_value=None)
    antialiased=Bool(default_value=rcParams['lines.antialiased'])
    # accepts: ['butt' | 'round' | 'projecting']
    dash_capstyle=Unicode(allow_none=True, default_value=rcParams['lines.dash_capstyle'])
    # accepts: ['butt' | 'round' | 'projecting']
    solid_capstyle=Unicode(allow_none=True, default_value=rcParams['lines.solid_capstyle'])
    # accepts: ['miter' | 'round' | 'bevel']
    dash_joinstyle=Unicode(allow_none=True, default_value=rcParams['lines.dash_joinstyle'])

    # accepts: ['miter' | 'round' | 'bevel']
    solid_joinstyle=Unicode(allow_none=True, default_value=rcParams['lines.solid_joinstyle'])
    pickradius=Int(allow_none=True, default_value=5)
    #TODO: assure this attribute works
    # accepts: ['default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post']
    drawstyle=Unicode(allow_none=True, default_value='default')
    #TODO: assure this attribute works
    markevery=Any(allow_none=True, default_value=None)
    verticalOffset = None   #only found once in the original lines code so not sure what to do with this
    ind_offset = Int(allow_none=True,default_value=0)
    invalidx=Bool(default_value=True)
    invalidy=Bool(default_value=True)
    #TODO: assure this works because I am not sure of the default value
    # path = PathTrait(allow_none=False, default_value=Path([(0.0,0.0),(1.0,0.0),(1.0,1.0),(1.0,0.0)])) #TODO: fix this
    path=Instance('matplotlib.path.Path', allow_none=False)
    # print("isinstance(path, Path):", isinstance(path, Path))
    # print("isinstance(path, PathTrait):", isinstance(path, PathTrait))

    # transformed_path=Instance('matplotlib.transforms.TransformedPath', allow_none=True) #default_value set in default function
    # TransformedPath(path, self.get_transform())
    # transformed_path=TransformedPathTrait(allow_none=False) #TODO: assure this works
    transformed_path=Instance('matplotlib.transforms.TransformedPath', allow_none=False)
    # print("isinstance(transformed_path, TransformedPath):", isinstance(transformed_path, TransformedPath))
    # print("isinstance(transformed_path, TransformedPathTrait):", isinstance(transformed_path, TransformedPathTrait))

    subslice=Bool(default_value=False)  # used in subslicing; only x is needed
    #TODO: assure numpy.array is imported in default function & assure this works
    # x_filled=Instance('numpy.array', allow_none=True, default_value=None)
    x_filled=Instance('numpy.array', allow_none=True)


    #TODO: figure this out
    dashSeq = None
    # dashSeq = Instance('')

    dashOffset=Int(allow_none=True, default_value=None)
    # unscaled dash + offset; this is needed scaling the dash pattern by linewidth
    #TODO: figure this out
    us_dashSeq = None #NOTE: could be an Instance?
    us_dashOffset=Int(allow_none=True, default_value=None)


    xorig = numpy.asarray([])
    yorig = numpy.asarray([])
    x = numpy.array([])
    y = numpy.array([])
    xy = None

    def set_data(self, *args):
        """
        Set the x and y data

        ACCEPTS: 2D array (rows are x, y) or two 1D arrays
        """
        if len(args) == 1:
            x, y = args[0]
        else:
            x, y = args

        self.set_xdata(x)
        self.set_ydata(y)

    def set_xdata(self, x):
        """
        Set the data np.array for x

        ACCEPTS: 1D array
        """
        self.xorig = x
        self.invalidx = True
        self.stale = True

    def set_ydata(self, y):
        """
        Set the data np.array for y

        ACCEPTS: 1D array
        """
        self.yorig = y
        self.invalidy = True
        self.stale = True

    # not sure how much this will have to be refactored
    def __str__(self):
        if self._label != "":
            return "Line2D(%s)" % (self._label)
        elif self.x is None:
            return "Line2D()"
        elif len(self.x) > 3:
            return "Line2D((%g,%g),(%g,%g),...,(%g,%g))"\
                % (self.x[0], self.y[0], self.x[0],
                   self.y[0], self.x[-1], self_y[-1])
        else:
            return "Line2D(%s)"\
                % (",".join(["(%g,%g)" % (x, y) for x, y
                             in zip(self.x, self.y)]))

    def __init__(self, xdata, ydata,
                 linewidth=rcParams['lines.linewidth'],
                 linestyle=rcParams['lines.linestyle'],
                 color=rcParams['lines.color'],
                 marker=rcParams['lines.marker'],
                 markersize=rcParams['lines.markersize'],
                 markeredgewidth=None,
                 markeredgecolor=None,
                 markerfacecolor=None,
                 markerfacecoloralt='none',
                 fillstyle=None,
                 antialiased=rcParams['lines.antialiased'],
                 dash_capstyle=rcParams['lines.dash_capstyle'],
                 solid_capstyle=rcParams['lines.solid_capstyle'],
                 dash_joinstyle=rcParams['lines.dash_joinstyle'],
                 solid_joinstyle=rcParams['lines.solid_joinstyle'],
                 pickradius=5,
                 drawstyle='default',
                 markevery=None,
                 **kwargs
                 ):

        Artist.__init__(self)

        if not iterable(xdata):
            raise RuntimeError('xdata must be a sequence')
        if not iterable(ydata):
            raise RuntimeError('ydata must be a sequence')

        self.set_data(xdata, ydata)

    #linewidth validate
    @validate("linewidth")
    def _linewidth_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.linewidth']
        #to assure we are dealing with a FLOAT
        proposal.value=float(proposal.value) #watch out for recursion on this line
        return proposal.value
    #linewidth observer
    @observe("linewidth", type="change")
    def _linewidth_observe(self, change):
        self.stale = True
        #NOTE: this line may cause recursion error
        self.dashOffset, self.dashSeq = _scale_dashes(self.us_dashOffset, self.us_dashSeq, self.linewidth)

    #linestyle validate
    @validate("linestyle")
    def _linestyle_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.linestyle']

        if isinstance(proposal.value, six.string_types):
            # ds, ls = self._split_drawstyle_linestyle(linestyle)
            # if ds is not None and drawstyle is not None and ds != drawstyle:
            #     raise ValueError("Inconsistent drawstyle ({0!r}) and "
            #                      "linestyle ({1!r})".format(drawstyle,
            #                                                 linestyle)
            #                      )
            ds, ls = self._split_drawstyle_linestyle(proposal.value)
            if ds is not None and self.drawstyle is not None and ds != drawstyle:
                raise ValueError("Inconsistent drawstyle ({0!r}) and "
                                 "linestyle ({1!r})".format(self.drawstyle,
                                                            proposal.value)
                                 )
            # linestyle = ls

            #NOTE: this line may cause error
            if ds is not None:
                # drawstyle = ds
                self.drawstyle = ds
                return proposal.value
        # return proposal.value

    #color validate
    @validate("color")
    def _color_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.color']
        return proposal.value
    #color observer
    @observe("color", type="change")
    def _color_observe(self, change):
        self.stale = True

    #marker default
    @default("marker")
    def _marker_default(self):
        #TODO: assure these import statements work properly
        import matplotlib.markers
        return rcParams['lines.marker']
    #marker validate
    @validate("marker")
    def _marker_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.marker']
        #TODO: find a method to make the line below work
        # self.marker.set_marker(marker) #TODO: testing
        return proposal.value
    #marker observer
    @observe("marker", type="change")
    def _marker_observe(self, change):
        self.stale = True

    #markersize validate
    @validate("markersize")
    def _markersize_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.markersize']
        return proposal.value
    #markersize observer
    @observe("markersize", type="change")
    def _markersize_observe(self, change):
        self.stale = True

    #markeredgewidth validate
    @validate("markeredgewidth")
    def _markeredgewidth_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.markeredgewidth']
        return proposal.value
    #markeredgewidth observer
    @observe("markeredgewidth", type="change")
    def _markeredgewidth_observe(self, change):
        self.stale = True

    #markerfacecolor validate
    @validate("markerfacecolor")
    def _markerfacecolor_validate(self, proposal):
        return proposal.value

    #markerfacecoloralt validate
    @validate("markerfacecoloralt")
    def _markerfacecoloralt_validate(self, proposal):
        return proposal.value

    #fillstyle validate
    @validate("fillstyle")
    def _fillstyle_validate(self, proposal):
        # return self._marker.set_fillstyle(proposal.value)
        return proposal.value
    #fillstyle observer
    @observe("fillstyle", type="change")
    def _fillstyle_observe(self, change):
        self.stale = True

    #antialiased validate
    @validate("antialiased")
    def _antialiased_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.antialiased']
        return proposal.value
    #antialiased observer
    @observe("antialiased", type="change")
    def _antialiased_observe(self, change):
        self.stale = True

    #dash_capstyle validate
    @validate("dash_capstyle")
    def _dash_capstyle_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.dash_capstyle']
        return proposal.value

    #solid_capstyle validate
    @validate("solid_capstyle")
    def _solid_capstyle_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.solid_capstyle']
        return proposal.value

    #dash_joinstyle validate
    @validate("dash_joinstyle")
    def _dash_joinstyle_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.dash_joinstyle']
        proposal.value = proposal.value.lower() #not sure on this line
        #NOTE: validJoin = ('miter', 'round', 'bevel')
        if proposal.value not in self.validJoin:
            raise ValueError('dash_joinstyle validate passed "%s";\n' % (proposal.value,)
                             + 'valid joinstyles are %s' % (self.validJoin,))
        return proposal.value
    # observer
    @observe("dash_joinstyle", type="change")
    def __dash_joinstyleobserve(self, change):
        self.stale = True

    #solid_joinstyle validate
    @validate("solid_joinstyle")
    def _solid_joinstyle_validate(self, proposal):
        if proposal.value is None:
            return rcParams['lines.solid_joinstyle']
        proposal.value = proposal.value.lower() #not sure on this line
        #NOTE: validJoin = ('miter', 'round', 'bevel')
        if proposal.value not in self.validJoin:
            raise ValueError('solid_joinstyle validate passed "%s";\n' % (proposal.value,)
                     + 'valid joinstyles are %s' % (self.validJoin,))
        return proposal.value
    #solid_joinstyle observer
    @observe("solid_joinstyle", type="change")
    def _solid_joinstyle_observe(self, change):
        self.stale = True

    #pickradius validate
    @validate("pickradius")
    def _pickradius_validate(self, proposal):
        return proposal.value

    #drawstyle validate
    @validate("drawstyle")
    def _drawstyle_validate(self, proposal):
        if proposal.value is None:
            return 'default'
        if proposal.value not in self.drawStyles:
            raise ValueError('Unrecognized drawstyle {!r}'.format(proposal.value))
        return proposal.value
    #drawstyle observer
    @observe("drawstyle", type="change")
    def _drawstyle_observe(self, change):
        self.stale = True


        """Set the markevery property to subsample the plot when using markers.

        e.g., if `every=5`, every 5-th marker will be plotted.

        ACCEPTS: [None | int | length-2 tuple of int | slice |
        list/array of int | float | length-2 tuple of float]

        Parameters
        ----------
        every: None | int | length-2 tuple of int | slice | list/array of int |
        float | length-2 tuple of float
            Which markers to plot.

            - every=None, every point will be plotted.
            - every=N, every N-th marker will be plotted starting with
              marker 0.
            - every=(start, N), every N-th marker, starting at point
              start, will be plotted.
            - every=slice(start, end, N), every N-th marker, starting at
              point start, upto but not including point end, will be plotted.
            - every=[i, j, m, n], only markers at points i, j, m, and n
              will be plotted.
            - every=0.1, (i.e. a float) then markers will be spaced at
              approximately equal distances along the line; the distance
              along the line between markers is determined by multiplying the
              display-coordinate distance of the axes bounding-box diagonal
              by the value of every.
            - every=(0.5, 0.1) (i.e. a length-2 tuple of float), the
              same functionality as every=0.1 is exhibited but the first
              marker will be 0.5 multiplied by the
              display-cordinate-diagonal-distance along the line.

        Notes
        -----
        Setting the markevery property will only show markers at actual data
        points.  When using float arguments to set the markevery property
        on irregularly spaced data, the markers will likely not appear evenly
        spaced because the actual data points do not coincide with the
        theoretical spacing between markers.

        When using a start offset to specify the first marker, the offset will
        be from the first data point which may be different from the first
        the visible data point if the plot is zoomed in.

        If zooming in on a plot when using float arguments then the actual
        data points that have markers will change because the distance between
        markers is always determined from the display-coordinates
        axes-bounding-box-diagonal regardless of the actual axes data limits.
        """
    #markevery validate
    @validate("markevery")
    def _markevery_validate(self, proposal):
        #TODO: figure this out
        # if self._markevery != every:
            # self.stale = True
        # self._markevery = every
        return proposal.value
    #markevery observer
    @observe("markevery", type="change")
    def _markevery_observe(self, change):
        self.stale = True

    #ind_offset validate
    @validate("ind_offset")
    def _ind_offset_validate(self, proposal):
        return proposal.value

    #invalidx validate
    @validate("invalidx")
    def _invalidx_validate(self, proposal):
        return proposal.value

    #invalidy validate
    @validate("invalidy")
    def _invalidy_validate(self, proposal):
        return proposal.value

    #TODO: assure this works correctly
    @default("path")
    def _path_default(self):
        from matplotlib.path import Path
        print("creating default value for Path")
        verts = [
        (0., 0.), # left, bottom
        (0., 1.), # left, top
        (1., 1.), # right, top
        (1., 0.), # right, bottom
        (0., 0.), # ignored
        ]
        codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]
        return Path(verts, codes)
    #path validate
    @validate("path")
    def _path_validate(self, proposal):
        from matplotlib.path import Path
        print("isinstance(proposal.value, Path):", isinstance(proposal.value, Path))
        return proposal.value

    #transformed default
    @default("transformed_path")
    def _transformed_path_default(self):
        from matplotlib.transforms import TransformedPath
        # return TransformedPath(self.path, self.get_transform())
        #self.transform in relation to the artist
        return TransformedPath(self.path, IdentityTransform())
    @validate("transformed_path")
    def _transformed_path_validate(self, proposal):
        from matplotlib.transforms import TransformedPath
        print("isinstance(proposal.value, TransformedPath):", isinstance(proposal.value, TransformedPath))
        return proposal.value

    #subslice validate
    @validate("subslice")
    def _subslice_validate(self, proposal):
        return proposal.value

    #x_filled default
    @default("x_filled")
    def _x_filled_default(self):
        from numpy import array
        return None
    #x_filled validate
    @validate("x_filled")
    def _x_filled_validate(self, proposal):
        return proposal.value


    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred on the line.  The pick
        radius determines the precision of the location test (usually
        within five points of the value).  Use
        :meth:`~matplotlib.lines.Line2D.get_pickradius` or
        :meth:`~matplotlib.lines.Line2D.set_pickradius` to view or
        modify it.

        Returns *True* if any values are within the radius along with
        ``{'ind': pointlist}``, where *pointlist* is the set of points
        within the radius.

        TODO: sort returned indices by distance
        """
        if callable(self.contains):
            return self.contains(self, mouseevent)

        if not is_numlike(self.pickradius):
            raise ValueError("pick radius should be a distance")

        # Make sure we have data to plot
        if self.invalidy or self.invalidx:
            self.recache()
        if len(self.xy) == 0:
            return False, {}

        # Convert points to pixels
        # transformed_path = self._get_transformed_path()
        transformed_path = self.transformed_path
        path, affine = transformed_path.get_transformed_path_and_affine()
        path = affine.transform_path(path)
        xy = path.vertices
        xt = xy[:, 0]
        yt = xy[:, 1]

        # Convert pick radius from points to pixels
        if self.figure is None:
            warnings.warn('no figure set when check if mouse is on line')
            pixels = self.pickradius
        else:
            pixels = self.figure.dpi / 72. * self.pickradius

        # the math involved in checking for containment (here and inside of
        # segment_hits) assumes that it is OK to overflow.  In case the
        # application has set the error flags such that an exception is raised
        # on overflow, we temporarily set the appropriate error flags here and
        # set them back when we are finished.
        with np.errstate(all='ignore'):
            # Check for collision
            if self._linestyle in ['None', None]:
                # If no line, return the nearby point(s)
                d = (xt - mouseevent.x) ** 2 + (yt - mouseevent.y) ** 2
                ind, = np.nonzero(np.less_equal(d, pixels ** 2))
            else:
                # If line, return the nearby segment(s)
                ind = segment_hits(mouseevent.x, mouseevent.y, xt, yt, pixels)
                if self.drawstyle.startswith("steps"):
                    ind //= 2

        ind += self.ind_offset

        # Return the point(s) within radius
        return len(ind) > 0, dict(ind=ind)

    def get_window_extent(self, renderer):
        bbox = Bbox([[0, 0], [0, 0]])
        trans_data_to_xy = self.get_transform().transform
        bbox.update_from_data_xy(trans_data_to_xy(self.get_xydata()),
                                 ignore=True)
        # correct for marker size, if any
        if self._marker:
            ms = (self._markersize / 72.0 * self.figure.dpi) * 0.5
            bbox = bbox.padded(ms)
        return bbox

    # @Artist.axes.setter
    def axes(self, ax):
        # call the set method from the base-class property
        # Artist.axes.fset(self, ax)
        Artist.axes = ax # in reference to Artist with traitlets
        if ax is not None:
            # connect unit-related callbacks
            if ax.xaxis is not None:
                self._xcid = ax.xaxis.callbacks.connect('units', self.recache_always)
            if ax.yaxis is not None:
                self._ycid = ax.yaxis.callbacks.connect('units', self.recache_always)

    def recache_always(self):
        self.recache(always=True)

    def recache(self, always=False):
        from matplotlib.path import Path
        if always or self.invalidx:
            xconv = self.convert_xunits(self.xorig)
            if isinstance(self.xorig, np.ma.MaskedArray):
                x = np.ma.asarray(xconv, float).filled(np.nan)
            else:
                x = np.asarray(xconv, float)
            x = x.ravel()
        else:
            x = self.x
        if always or self.invalidy:
            yconv = self.convert_yunits(self.yorig)
            if isinstance(self.yorig, np.ma.MaskedArray):
                y = np.ma.asarray(yconv, float).filled(np.nan)
            else:
                y = np.asarray(yconv, float)
            y = y.ravel()
        else:
            y = self._y

        if len(x) == 1 and len(y) > 1:
            x = x * np.ones(y.shape, float)
        if len(y) == 1 and len(x) > 1:
            y = y * np.ones(x.shape, float)

        if len(x) != len(y):
            raise RuntimeError('xdata and ydata must be the same length')

        self.xy = np.empty((len(x), 2), dtype=float)
        self.xy[:, 0] = x
        self.xy[:, 1] = y

        self.x = self.xy[:, 0]  # just a view
        self.y = self.xy[:, 1]  # just a view

        self.subslice = False
        if (self.axes and len(x) > 1000 and self.is_sorted(x) and
                self.axes.name == 'rectilinear' and
                self.axes.get_xscale() == 'linear' and
                self.markevery is None and
                self.get_clip_on() is True):
            self._subslice = True
            nanmask = np.isnan(x)
            if nanmask.any():
                self.x_filled = self.x.copy()
                indices = np.arange(len(x))
                self.x_filled[nanmask] = np.interp(indices[nanmask],
                        indices[~nanmask], self._x[~nanmask])
            else:
                self.x_filled = self.x

        if self.path is not None:
            interpolation_steps = self.path._interpolation_steps
        else:
            interpolation_steps = 1
        xy = STEP_LOOKUP_MAP[self.drawstyle](*self.xy.T)

        self.path = Path(np.asarray(xy).T,
                          _interpolation_steps=interpolation_steps)
        # self.transformed_path = None
        self.invalidx = False
        self.invalidy = False

    def transform_path(self, subslice=None):
        """
        Puts a TransformedPath instance at self._transformed_path;
        all invalidation of the transform is then handled by the
        TransformedPath instance
        """
        # Masked arrays are now handled by the Path class itself
        if subslice is not None:
            xy = STEP_LOOKUP_MAP[self.drawstyle](*self.xy[subslice, :].T) #TODO: 10.23.17 assure that this works
            # xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy[subslice, :].T)
            path = Path(np.asarray(xy).T,
                         interpolation_steps=self.path.interpolation_steps)
            print("__traits/lines.py line 992 path: ", path)
        else:
            path = self.path
        self.transformed_path = TransformedPath(path, self.get_transform())
        print("__traits/lines.py line 996 self.transformed_path: ", self.transformed_path)

    def _is_sorted(self, x):
        """return True if x is sorted in ascending order"""
        # We don't handle the monotonically decreasing case.
        # return _path.is_sorted(x)
        return path.is_sorted(x) #TODO: test


    @allow_rasterization
    def draw(self, renderer):
        """draw the Line with `renderer` unless visibility is False"""
        if not self.get_visible():
            return

        if self.invalidy or self.invalidx:
            self.recache()
        self.ind_offset = 0  # Needed for contains() method.
        if self.subslice and self.axes:
            x0, x1 = self.axes.get_xbound()
            i0, = self.x_filled.searchsorted([x0], 'left')
            i1, = self.x_filled.searchsorted([x1], 'right')
            subslice = slice(max(i0 - 1, 0), i1 + 1)
            self.ind_offset = subslice.start
            self.transform_path(subslice)

        print("_traits/lines.py line 925 self.transformed_path: ", self.transformed_path)
        # print("_traits/lines.py line 1023 transf_path: ", transf_path)
        # transf_path = self._get_transformed_path()
        transf_path = self.transformed_path
        print("_traits/lines.py line 929 transf_path: ", transf_path)


        # if self.get_path_effects():
        #     from matplotlib.patheffects import PathEffectRenderer
        #     renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        if self.path_effects:
            from matplotlib.patheffects import PathEffectRenderer
            renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        renderer.open_group('line2d', self.get_gid())
        # print("_traits/lines.py line 1062 function self.linestyle",self.linestyle)
        if self.lineStyles[self.linestyle] != '_draw_nothing':
            tpath, affine = transf_path.get_transformed_path_and_affine()
            if len(tpath.vertices):
                gc = renderer.new_gc()
                self._set_gc_clip(gc)

                ln_color_rgba = self._get_rgba_ln_color()
                gc.set_foreground(ln_color_rgba, isRGBA=True)
                gc.set_alpha(ln_color_rgba[3])

                gc.set_antialiased(self.antialiased)
                gc.set_linewidth(self.linewidth)

                if self.is_dashed():
                    # cap = self.dashcapstyle
                    cap = self.dash_capstyle
                    # print("if self.is_dashed() cap: ", cap)
                    # join = self.dashjoinstyle
                    join = self.dash_joinstyle
                    # print("if self.is_dashed() join: ", join)
                else:
                    # cap = self.solidcapstyle
                    cap = self.solid_capstyle
                    # print("else cap: ", cap)
                    # join = self.solidjoinstyle
                    join = self.solid_joinstyle
                    # print("else join: ", join)
                gc.set_joinstyle(join)
                gc.set_capstyle(cap)
                gc.set_snap(self.get_snap())
                if self.get_sketch_params() is not None:
                    gc.set_sketch_params(*self.get_sketch_params())

                gc.set_dashes(self.dashOffset, self.dashSeq)
                renderer.draw_path(gc, tpath, affine.frozen())
                gc.restore()

        print("self.marker: ", self.marker)
        print("self.markersize: ", self.markersize)
        if self.marker and self.markersize > 0:
            gc = renderer.new_gc()
            self._set_gc_clip(gc)
            rgbaFace = self._get_rgba_face()
            rgbaFaceAlt = self._get_rgba_face(alt=True)
            edgecolor = self.get_markeredgecolor()
            if (isinstance(edgecolor, six.string_types)
                    and edgecolor.lower() == 'none'):
                gc.set_linewidth(0)
                gc.set_foreground(rgbaFace, isRGBA=True)
            else:
                gc.set_foreground(edgecolor)
                gc.set_linewidth(self._markeredgewidth)
                mec = self._markeredgecolor
                if (isinstance(mec, six.string_types) and mec == 'auto' and
                        rgbaFace is not None):
                    gc.set_alpha(rgbaFace[3])
                else:
                    gc.set_alpha(self.get_alpha())

            marker = self._marker
            tpath, affine = transf_path.get_transformed_points_and_affine()
            if len(tpath.vertices):
                # subsample the markers if markevery is not None
                markevery = self.get_markevery()
                if markevery is not None:
                    subsampled = _mark_every_path(markevery, tpath,
                                                  affine, self.axes.transAxes)
                else:
                    subsampled = tpath

                snap = marker.get_snap_threshold()
                if type(snap) == float:
                    snap = renderer.points_to_pixels(self._markersize) >= snap
                gc.set_snap(snap)
                gc.set_joinstyle(marker.get_joinstyle())
                gc.set_capstyle(marker.get_capstyle())
                marker_path = marker.get_path()
                marker_trans = marker.get_transform()
                w = renderer.points_to_pixels(self._markersize)

                if (isinstance(marker.get_marker(), six.string_types) and
                        marker.get_marker() == ','):
                    gc.set_linewidth(0)
                else:
                    # Don't scale for pixels, and don't stroke them
                    marker_trans = marker_trans.scale(w)

                renderer.draw_markers(gc, marker_path, marker_trans,
                                      subsampled, affine.frozen(),
                                      rgbaFace)

                alt_marker_path = marker.get_alt_path()
                if alt_marker_path:
                    alt_marker_trans = marker.get_alt_transform()
                    alt_marker_trans = alt_marker_trans.scale(w)
                    if (isinstance(mec, six.string_types) and mec == 'auto' and
                            rgbaFaceAlt is not None):
                        gc.set_alpha(rgbaFaceAlt[3])
                    else:
                        gc.set_alpha(self.get_alpha())

                    renderer.draw_markers(
                            gc, alt_marker_path, alt_marker_trans, subsampled,
                            affine.frozen(), rgbaFaceAlt)

            gc.restore()

        renderer.close_group('line2d')
        self.stale = False

    def _split_drawstyle_linestyle(self, ls):
        '''Split drawstyle from linestyle string

        If `ls` is only a drawstyle default to returning a linestyle
        of '-'.

        Parameters
        ----------
        ls : str
            The linestyle to be processed

        Returns
        -------
        ret_ds : str or None
            If the linestyle string does not contain a drawstyle prefix
            return None, otherwise return it.

        ls : str
            The linestyle with the drawstyle (if any) stripped.
        '''
        ret_ds = None
        for ds in self.drawStyleKeys:  # long names are first in the list
            if ls.startswith(ds):
                ret_ds = ds
                if len(ls) > len(ds):
                    ls = ls[len(ds):]
                else:
                    ls = '-'
                break

        return ret_ds, ls

    def set_dashes(self, seq):
        """
        Set the dash sequence, sequence of dashes with on off ink in
        points.  If seq is empty or if seq = (None, None), the
        linestyle will be set to solid.

        ACCEPTS: sequence of on/off ink in points
        """
        if seq == (None, None) or len(seq) == 0:
            self.set_linestyle('-')
        else:
            self.set_linestyle((0, seq))

    #TODO: this should be in the validate function
    # @docstring.dedent_interpd
    # def set_marker(self, marker):
    #     """
    #     Set the line marker
    #
    #     ACCEPTS: :mod:`A valid marker style <matplotlib.markers>`
    #
    #     Parameters
    #     ----------
    #
    #     marker: marker style
    #         See `~matplotlib.markers` for full description of possible
    #         argument
    #
    #     """
    #     self._marker.set_marker(marker)
    #     self.stale = True

    def set_transform(self, t):
        """
        set the Transformation instance used by this artist

        ACCEPTS: a :class:`matplotlib.transforms.Transform` instance
        """
        Artist.set_transform(self, t)
        self.invalidx = True
        self.invalidy = True
        self.stale = True

    def _get_rgba_face(self, alt=False):
        facecolor = self._get_markerfacecolor(alt=alt)
        # print("self.markerfacecolor: ", self.markerfacecolor)
        # facecolor = self.markerfacecolor #TODO: work in alt
        if (isinstance(facecolor, six.string_types)
                and facecolor.lower() == 'none'):
            rgbaFace = None
        else:
            rgbaFace = mcolors.to_rgba(facecolor, self.alpha)
        return rgbaFace

    def _get_rgba_ln_color(self, alt=False):
        return mcolors.to_rgba(self.color, self.alpha)

    # for testing
    def get_path(self):
        """
        Return the :class:`~matplotlib.path.Path` object associated
        with this line.
        """
        if self.invalidy or self.invalidx:
            self.recache()
        return self.path

    def is_dashed(self):
        'return True if line is dashstyle'
        return self.linestyle in ('--', '-.', ':')

    def _get_markerfacecolor(self, alt=False):
        if alt:
            fc = self.markerfacecoloralt
        else:
            fc = self.markerfacecolor
        if (isinstance(fc, six.string_types) and fc.lower() == 'auto'):
            if self.get_fillstyle() == 'none':
                return 'none'
            else:
                return self._color
        else:
            return fc

    def get_markerfacecolor(self):
        return self._get_markerfacecolor(alt=False)

lineStyles = Line2D.lineStyles
lineMarkers = MarkerStyle.markers
drawStyles = Line2D.drawStyles
fillStyles = MarkerStyle.fillstyles

#for monkey patching
b_Line2D.Line2D = Line2D

docstring.interpd.update(Line2D=artist.kwdoc(Line2D))
#TODO: print statement to see what this line does: right now returns None
# print("docstring.interpd.update(Line2D=artist.kwdoc(Line2D)): ", docstring.interpd.update(Line2D=artist.kwdoc(Line2D)))

# You can not set the docstring of an instance method,
# but you can on the underlying function.  Go figure.
docstring.dedent_interpd(Line2D.__init__)
#TODO: print statement to see what this line does:
# print("docstring.dedent_interpd(Line2D.__init__): ", docstring.dedent_interpd(Line2D.__init__))
