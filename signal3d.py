import pandas as pd
import numpy as np
from scipy.signal import hilbert


class Signal3D(pd.Panel):
    """
    Represents data from a raster scan. Each point is represented by its 2-D coordnates
    (X, Y), and contains a time series signal with time base t.
    The axes are such that:

        * **axis 0 (items)**: *y*-axis representing scan index,
        * **axis 1 (major_axis)**: time base *t* for each time series,
        * **axis 2 (minor_axis)**: *x*-axis representing scan position.

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

    @classmethod
    def from_panel(cls, pnl):
        return cls(pnl.values, items=pnl.items, major_axis=pnl.major_axis,
                   minor_axis=pnl.minor_axis)

    def operate(self, option='', axis=0):
        """
        Operate on the signal along a given axis.

        Parameters
        ----------
        option : string/char, optional
            The possible options are (combined options are allowed):

             +--------------------+--------------------------------------+
             | *option*           | Meaning                              |
             +====================+======================================+
             | '' *(Default)*     | Return the raw signal                |
             +--------------------+--------------------------------------+
             | 'n'                | normalized signal                    |
             +--------------------+--------------------------------------+
             | 'd'                | decibel value                        |
             +--------------------+--------------------------------------+

        axis : int, optional
            Only used in the case option specified 'e' for envelop. Specifies along which axis to
            compute the envelop.

        Returns
        -------
        : Signal3D
            The modified Signal3D.
        """
        axis = self._make_axis_as_num(axis)
        yout = self
        if 'e' in option:
            yout = np.abs(hilbert(yout, axis=axis))
        if 'n' in option:
            yout = yout/np.abs(yout).max().max()
        if 'd' in option:
            yout = 20*np.log10(np.abs(yout))
        return Signal3D(yout, items=self.items, major_axis=self.major_axis,
                        minor_axis=self.minor_axis)

    def _get_other_axes(self, axis):
        axis = self._make_axis_as_num(axis)
        if axis == 0:
            return [1, 2]
        if axis == 1:
            return [0, 2]
        if axis == 2:
            return [0, 1]
        raise ValueError('Unknown axis value.')

    @staticmethod
    def _make_axis_as_num(axis, soft_error=False):
        """
        Returns the number of the axis, according to the naming conventions of the axes.
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

    def shift_axis(self, shift, axis=None):
        """
        Applies shifting of a given axis.

        Parameters
        ----------
        shift : float
            The amount to shift the axis.

        axis : int/string, optional
            If 0/'y'/'items', shift the y-axis, if 1/'t'/'major_axis', shift the t-axis. If
            2/'x'/'minor_axis' shift the x-axis. If None shift all axes.

        Returns
        -------
        : Signal3D
            A copy of Signal3D with shifts axes.
        """
        shift = self._verify_val_and_axis(shift, axis, assign=[0, 0, 0], raise_none=True)
        return Signal3D(self.values, items=self.items-shift[0],
                        major_axis=self.major_axis-shift[1], minor_axis=self.minor_axis-shift[2])

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

    def skew(self, angle, axis, skew_axes=1, start=None, stop=None, interpolate=True,
             ts=None, **kwargs):
        """
        Applies a 2-D skew for each slice along a specified axis. Uses interpolation to
        recalculate the signal values at the new coordinates.
        Note
        ----
        This does not perform a 3-D skew and interpolation, but only a 2-D skew and
        interpolation, along a specified 3rd axis.
        Parameters
        ----------
        angle : float, 2-element array
            The angle of the skew.
        axis : {0/'y'/'items', 1/'t'/'major_axis', 2/'x'/'minor_axis'}
            The axis along which to extract each slice to be skewed.
        skew_axes: {0/'y'/'index', 1/'x'/'columns', None}, optional
            Determine along which axis to apply the skew, after extracting the slice along the
            specified *axis*. The axes domain is in that of :class:`Signal2D`.
        start, stop, ts : See :meth:`Signal2D.skew()` for documentation on these arguments.
        Returns
        -------
        : Signal3D
            A copy of Signal3D after application of the skew.
        """
        other_ax = self._get_other_axes(axis)

        # Make nearest neighbor the default interpolation method.
        if 'method' not in kwargs:
            kwargs['method'] = 'nearest'
        return self.apply(lambda x: x.skew(angle, axes=skew_axes, start=start, stop=stop,
                                           ts=ts, interpolate=interpolate, **kwargs), axis=other_ax)

    def extract(self, option='max', axis=0):
        """
        Extracts a Signal2D according to a given option by slicing along a specified axis.

        Parameters
        ----------
        option : {'max', 'var', float} optional
            Select the method to find the slice
            * 'max': The maximum amplitude along the axis
            * 'var': the maximum variance along the axis.
            * scalar: A scalar can be specified to select a specific point along the axis

        axis : scalar/string, optional
            The axis along which to extract the slice.  Conventions for naming axes: 0/'y'/'items',
            1/'t'/'major_axis', or 2/'x'/'minor_axis'.

        Returns
        -------
        : Signal2D
            A Signal2D object representing the slice.
        """
        axis = self._make_axis_as_num(axis)
        other_axes = self._get_other_axes(axis)

        if option == 'max':
            s = self.apply(lambda x: x.max().max(), axis=other_axes)
            option = s.idxmax()
        elif option == 'var':
            s = self.apply(lambda x: x.var().var(), axis=other_axes)
            option = s.idxmax()

        if axis == 0:
            out = self.loc[option]
        elif axis == 1:
            out = self.loc[:, option, :]
        elif axis == 2:
            out = self.loc[:, :, option]
        else:
            raise ValueError('Unknown axis value.')

        return option, out

    def dscan(self, option='max'):
        """
        Convenience method that return D-scans from Ultrasound Testing raster scans.
        Slices the raster scan along the y-t axes (i.e. at a given x location).

        Parameters
        ----------
        option : {'max', 'var', float} optional
            Select the method to find the slice
            * 'max': The maximum amplitude along the axis
            * 'var': the maximum variance along the axis.
            * scalar: A scalar can be specified to select a specific point along the axis

        Returns
        -------
        : Signal2D
            Extracted D-scan as a Signal2D object.
        """
        return self.extract(option, axis=2)

    def bscan(self, option='max'):
        """
        Convenience method that return B-scans from Ultrasound Testing raster scans.
        Slices the raster scan along the x-t axes (i.e. at a given y location).

        Parameters
        ----------
        option : {'max', 'var', float} optional
            Select the method to find the slice
            * 'max': The maximum amplitude along the axis
            * 'var': the maximum variance along the axis.
            * scalar: A scalar can be specified to select a specific point along the axis

        Returns
        -------
        : Signal2D
            Extracted B-scan as a Signal2D object.
        """
        return self.extract(option, axis=0)

    def flatten(self):
        """
        Flatten an array and its corresponding indices.

        Returns
        -------
        x, t, x, values : numpy.ndarray
            A 4-element tuple where each element is a flattened array of the Signal3D, and each
            representing a point with coordinates y, t, x and its value.
        """
        yv, tv, xv = np.meshgrid(self.Y, self.t, self.X, indexing='xy')
        return np.array([yv.ravel(), tv.ravel(), xv.ravel(), self.values.ravel()])

    @property
    def axis(self):
        """
        Return the values of the three axes in the Signal3D.
        """
        return self.items, self.major_axis, self.minor_axis

    @property
    def ts(self):
        """
        The mean sampling interval for each of the axes.
        """
        return np.mean(np.diff(self.items)), np.mean(np.diff(self.major_axis)),\
               np.mean(np.diff(self.minor_axis))

    @property
    def x(self):
        """ Convenience property to return X-axis coordinates as ndarray. """
        return self.minor_axis.values

    @property
    def y(self):
        """ Convenience property to return Y-axis coordinates as ndarray. """
        return self.items.values

    @property
    def z(self):
        """ Convenience property to return t-axis coordinates as ndarray. """
        return self.major_axis.values
