import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import hilbert
from pandas.tools.util import cartesian_product


class Signal3D(pd.Panel):
    """
    Represents data from a raster scan. Each point is represented by its 2-D coordnates
    (X, Y), and contains a time series signal with time base t.
    The axes are such that:

        * **axis 0 (items)**: *Y*-direction representing scan index,
        * **axis 1 (major_axis)**: time base *t* for each time series,
        * **axis 2 (minor_axis)**: *X*-direction representing scan position.

    .. autosummary::

        shift_axis
    """

    @property
    def _constructor(self):
        return Signal3D

    @property
    def _constructor_sliced(self):
        from .signal2d import Signal2D
        return Signal2D

    def operate(self, option='', axis=0):
        yout = self
        if 'e' in option:
            yout = np.abs(hilbert(yout, axis=axis))
        if 'n' in option:
            yout = yout/np.abs(yout).max().max()
        if 'd' in option:
            yout = 20*np.log10(np.abs(yout))
        return Signal3D(yout, items=self.items, major_axis=self.major_axis,
                        minor_axis=self.minor_axis)

    # _____________________________________________________________ #
    def _verify_val_and_axis(self, val, axis, assign, raise_none=True):
        """
        Internal method to verify that a value is 3-element array. If it is a scalar or
        const:`None`, the missing values are selected based on the given *axis*.

        Parameters
        ---------
        val : float, array_like
            This is the value to be tested

        axis : int
            The axis for which the *val* corresponds.

        assign : array_like, 3-elements
            The values to assigns to new *val* if the input *val* is a scalar or const:`None`.

        raise_none : bool, optional
            If const:`True`, raise an error if *val* is const:`None`.

        Returns
        -------
        : 2-tuple
            A 2-element array representing the filled *val* with missing axis value.
        """
        axis = self._make_axis_as_num(axis)

        if isinstance(axis, str):
            axis = axis.lower()

        if val is None:
            if raise_none:
                raise ValueError('value cannot be None.')
            return assign

        try:
            if len(val) != 3:
                raise ValueError('value should be either a scalar or a 3 element array.')
            if axis is None:
                return val
            else:
                raise ValueError('value must be a scalar is axis is specified.')
        except TypeError:
            if axis is None:
                return [val, val, val]
            elif axis == 0:
                return [val, assign[1], assign[2]]
            elif axis == 1:
                return [assign[0], val, assign[2]]
            elif axis == 2:
                return [assign[0], assign[1], val]
            else:
                raise ValueError('Unknown value for axis. See documentation for allowed axis '
                                 'values.')

    # _____________________________________________________________ #
    def shift_axis(self, shift, axis=None):
        """
        Applies shifting of a given axis.

        Parameters
        ----------
        shift : float
            The amount to shift the axis.

        axis : int/string, optional
            If 0/'Y'/'items', shift the Y-axis, if 1/'t'/'major_axis', shift the t-axis. If
            2/'X'/'minor_axis' shift the X-axis. If None shift all axes.

        Returns
        -------
        : Signal3D
            A copy of Signal3D with shifts axes.
        """
        shift = self._verify_val_and_axis(shift, axis, assign=[0, 0, 0], raise_none=True)
        return Signal3D(self.values, items=self.items-shift[0],
                        major_axis=self.major_axis-shift[1], minor_axis=self.minor_axis-shift[2])

    # _____________________________________________________________ #
    def scale_axes(self, scale, start=None, stop=None, axis=None):
        """
        Scales a given axis (or both) by a given amount.

        Parameters
        ----------
        scale : float, array_like
            The amount to scale the axis. If axis is specified,
            *scale* should be scalar. If no axis specified *scale* can be a scalar (scale both
            axes by the same amount), or a 2-element vector for a different scale value for each
            axis.

        start : float, optional
            The axis value at which to start applying the scaling. If not
            specified, the whole axis will be scaled starting from the first axis value. If
            *axis* is specified, *start* should be a scalar, otherwise, *start* can be either a
            scalar (scale both axes by the same factor), or a 2-element array for scaling each
            axis differently.

        stop : float, optional
            The axis value at which to end the scaling. If not specified, the axis will be
            scaled up to the last axis value. If *axis* is specified, *end* should be a
            scalar, otherwise, *end* can be either a scalar (scale both axes by the same
            factor), or a 2-element array for scaling each axis differently.

        axis : int/string, optional
            If 0, 'Y' or index, scale the index, if 1, 'X' or 'columns', scale the columns. If None
            scale both axes.

        Returns
        -------
        : Signal2D
            A copy of Signal2D with scaled axes.

        Note
        ----
        If only partial domain on the axis is specified for scaling results in non-monotonic
        axis, an exception error will occur.
        """
        start = self._verify_val_and_axis(start, axis, [self.axes[0][0], self.axes[1][0],
                                                        self.axes[2][0]], raise_none=False)
        stop = self._verify_val_and_axis(stop, axis, [self.axes[0][-1], self.axes[1][-1],
                                                      self.axes[2][-1]], raise_none=False)

        if start[0] > stop[0] or start[1] > stop[1] or start[2] > stop[2]:
            raise ValueError('start should be smaller than end.')

        scale = self._verify_val_and_axis(scale, axis, [1., 1., 1.], raise_none=True)
        newitems, newmajor, newminor = self.items.values, self.major_axis.values, \
                                       self.minor_axis.values
        newitems[(newitems >= start[0]) & (newitems <= stop[0])] *= scale[0]
        newmajor[(newmajor >= start[1]) & (newmajor <= stop[1])] *= scale[1]
        newminor[(newminor >= start[1]) & (newminor <= stop[2])] *= scale[2]
        return Signal3D(self.values, items=newitems, major_axis=newmajor, minor_axis=newminor)

    @staticmethod
    def _make_axis_as_num(axis, soft_error=False):
        """

        :param axis:
        :return:
        """
        if isinstance(axis, str):
            axis = axis.lower()

        if axis in [0, 'y', 'items']:
            return 0
        if axis in [1, 't', 'major_axis']:
            return 1
        if axis in [2, 'x', 'minor_axis']:
            return 2

        if soft_error:
            return None
        else:
            raise ValueError('Unknown axis value.')

    def bscan(self, option='max'):
        """
        Computes B-scan image generally used for Utrasound Testing.
        Slices the raster scan along the X-t axes (i.e. at a given Y location).

        Parameters :
            depth (float/string, optional) :
                Select which slice to be used for the b-scan. If 'max', the
                slice passing through the maximum point is selected.
        Returns :
            Signal2D :
                A Scan2D object representing the B-scan.
        """
        if option == 'max':
            s = self.apply(lambda x: x.max().max(), axis=(1, 2))
        elif option == 'var':
            s = self.apply(lambda x: x.var().var(), axis=(1, 2))
        else:
            raise ValueError('Unknown option.')
        print(s.idxmax())
        return self[s.idxmax()]

    def dscan(self, depth='max'):
        """
        Convenience method that return b-scans for Ultrasound Testing raster scans.
        Slices the raster scan along the Y-t axes (i.e. at a given X location).

        Parameters:
            - depth (float/sting, optional):
                Select which slice to be used for the b-scan. If 'max', the
                slice passing through the maximum point is selectd.
        Returns:
            Signal2D :
                A Scan2D object representing the D-scan.
        """
        if depth == 'max':
            idx = self.cscan().max(axis=0).idxmax()
            return self.loc[:, :, idx]
        else:
            if depth > self.minor_axis.max() or depth < self.minor_axis.min():
                raise ValueError('depth is outside range of scan in X-direction.')
            # find the nearest X-coordinate near the given depth
            ind = self.minor_axis[np.abs(self.minor_axis.values - depth).argmin()]
            return self.loc[:, :, ind]

    def cscan(self, depth='project', skew_angle=0, method='nearest', **kwargs):
        """
        Computes C-scan image generally used for Ultrasound Testing. This collapses the
        raster scan along the T direction, or takes a slice at a given T coordinate,
        resulting in an image spanning the X-Y axes.

        Parameters:
            depth (float/string, optional):
                The methodology to obtain the C-scan. Supported options are:
                'project' collapses the whole T-axis into a single value using the methodology
                specified by the option parameter. 'max' obtains the
                T-axis coordinate where the T-value is maximum. Otherwise depth
                is a float value specifying the actual T-coordinate value.

            skew_angle (float, optional) :
                Skews the X-direction to be aligned with the ultrasonic beam angle,
                before projecting the T-axis onto the X-axis.

        Returns:
            Signal2D :
                A Scan2D object representing the C-scan.
        """
        if depth == 'project':
            uskew = self.apply(lambda x: x.skew(skew_angle, axes=1, method=method, **kwargs),
                               axis=(1, 2))
            return uskew.std(axis=1).T
        elif depth == 'max':
            return self.abs().max(axis=1).T
        else:
            if depth > self.major_axis.max() or depth < self.major_axis.min():
                raise ValueError('depth is outside range of scan in T-direction.')
            # find the nearest T-coordinate near the given depth
            ind = self.major_axis[np.abs(self.major_axis.values - depth).argmin()]
            return self.loc[:, ind, :].T

    def flatten(self):
        """
        Flatten an array and its corresponding indices.

        Returns:
            Y, T, X, values (tuple):
                A 4-element tuple where each element is a flattened array of the Signal3D,
                and each representing a point with coordinates Y, T, X and its value.
        """
        yv, tv, xv = np.meshgrid(self.Y, self.t, self.X, indexing='xy')
        return np.array([yv.ravel(), tv.ravel(), xv.ravel(), self.values.ravel()])

    @property
    def axis(self):
        return self.items, self.major_axis, self.minor_axis

    @property
    def ts(self):
        return np.mean(np.diff(self.items)), np.mean(np.diff(self.major_axis)),\
               np.mean(np.diff(self.minor_axis))

    # _______________________________________________________________________#
    @property
    def x(self):
        """ Convenience property to return X-axis coordinates as ndarray. """
        return self.minor_axis.values

# _______________________________________________________________________#
    @property
    def y(self):
        """ Convenience property to return Y-axis coordinates as ndarray. """
        return self.items.values

# _______________________________________________________________________#
    @property
    def z(self):
        """ Convenience property to return t-axis coordinates as ndarray. """
        return self.major_axis.values
