import pandas as pd
import numpy as np
from scipy.signal import hilbert, get_window
from scipy.interpolate import griddata
from scipy.fftpack import fft2, fftfreq, fftshift
import matplotlib.pyplot as plt
from time import time
from .signal3d import Signal3D


class Signal2D(pd.DataFrame):
    """
    Extends :class:`pandas.DataFrame` class to support operations commonly required
    in radio frequency signals, and especially in ultrasonics. The axis convention used is:

        * **Axis 1 (columns)**: *X*-direction
        * **Axis 0 (index)**: *Y*-direction

    For example, in the context of ultrasonic inspection, the *X*-direction would represent
    the spatial line scan, and the *Y*-direction represents the signal time base, which can
    be scaled to represent the ultrasonic beam depth through the material.

    The class constructor is similar as that of :class:`pandas.DataFrame` with the added
    option of specifying only the sampling intervals along the *X* and *Y* directions.
    Thus, if *index* and/or *columns* are scalars, which *data* is a 2-D array, then
    the Signal2D basis are constructed starting from 0 at the given sampling intervals.

    If data input is a dictionary, usual rules from :class:`pandas.DataFrame` apply, but index
    can still be a scalar specifying the sampling interval.
    """
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        if index is not None and data is not None:
            if not hasattr(index, '__len__'):
                if isinstance(data, dict):
                    if hasattr(data[list(data.keys())[0]], '__len__'):
                        datalen = len(data[list(data.keys())[0]])
                    else:
                        datalen = 0
                elif hasattr(data, '__len__'):
                    datalen = len(data)
                else:
                    datalen = 0
                if datalen > 0:
                    index = np.arange(datalen) * index

        if columns is not None and data is not None:
            if not hasattr(columns, '__len__'):
                if isinstance(data, dict):
                    datalen = 0
                elif isinstance(data, pd.Series):
                    datalen = 0
                elif isinstance(data, pd.DataFrame):
                    datalen = data.shape[1]
                elif hasattr(data, '__len__'):
                    datalen = len(data[0])
                else:
                    datalen = 0
                if datalen > 0:
                    columns = np.arange(datalen) * columns

        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        # check for axes monotonicity
        if not self.index.is_monotonic_increasing:
            raise ValueError('Index must be monotonically increasing.')
        if not self.columns.is_monotonic_increasing:
            raise ValueError('Columns must be monotonically increasing.')

    @property
    def _constructor(self):
        return Signal2D

    @property
    def _constructor_sliced(self):
        from .signal import Signal
        return Signal

    @property
    def _constructor_expanddim(self):
        return Signal3D

    # _____________________________________________________________ #
    def __call__(self, index, columns, **kwargs):
        """
        Interpolate the axes. This function used :func:`scipy.interpolate.griddata`.

        Parameters
        ----------
        index : array_like
            New index values to compute the interpolation.

        columns : array_like
            New columns values to compute the interpolation.

        Returns
        -------
        : Signal2D
            A new Signal2D object computed at the new given axes values.

        Notes
        -----
        Other keyword arguments are passed directly to the interpolation function
        :func:`scipy.interpolate.griddata`.
        """
        index, columns = np.array(index), np.array(columns)
        if (index.ndim != 1) or (columns.ndim != 1):
            raise TypeError('New index and columns must be one dimensional arrays.')

        xg, yg = np.meshgrid(self.columns, self.index, indexing='xy')
        vals = griddata((yg.ravel(), xg.ravel()), self.values.ravel(), (index, columns), **kwargs)
        return Signal2D(vals, index=index, columns=columns)

    # _____________________________________________________________ #
    def operate(self, option='', axis=0):
        """
        Returns the signal according to a given option.

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
        : Signal2D
            The modified Signal2D.
        """
        yout = self
        if 'e' in option:
            yout = np.abs(hilbert(yout, axis=axis))
        if 'n' in option:
            yout = yout/np.abs(yout).max().max()
        if 'd' in option:
            yout = 20*np.log10(np.abs(yout))
        return Signal2D(yout, index=self.index, columns=self.columns)

    @staticmethod
    def _make_axes_as_num(axes, soft_error=False):
        """
        Takes an axis name, which could be the axis number of corresponding naming convention,
        and always return the axis number.

        Parameters
        ----------
        axes : int/string
            The name of the axis

        soft_error : bool, optional
            If :const:`True`, then an exception will be raised if the given axis name is not valid.
            Otherwise, :const:`None` will be returned

        Returns
        -------
        : int
            The axis number.
        """
        if axes is None:
            return [0, 1]
        if not hasattr(axes, '__len__'):
            axes = (axes, )

        out_ax = []
        for ax in axes:
            if isinstance(ax, str):
                ax = ax.lower()
            if ax in [0, -2, 'y', 'index']:
                out_ax.append(0)
            elif ax in [1, -1, 'x', 'columns']:
                out_ax.append(1)
            else:
                if not soft_error:
                    raise ValueError('Unknown axis value.')
        return out_ax

    def _set_val_on_axes(self, val, axes, assign):
        """
        Internal method to verify that a value is 2-element array. If it is a scalar or
        const:`None`, the missing values are selected based on the given *axis*.

        Parameters
        ---------
        val : float, array_like
            This is the value to be tested

        axes : int
            The axis for which the *val* corresponds.

        assign : array_like, 2-elements
            The values to assigns to new *val* if the input *val* is a scalar or const:`None`.

        Returns
        -------
        : 2-tuple
            A 2-element array representing the filled *val* with missing axis value.
        """
        out_vals = np.array(assign)
        if len(out_vals) != 2:
            raise ValueError('assign must be a two element sequence.')
        axes = self._make_axes_as_num(axes)
        out_vals[axes] = val
        out_vals[np.isnan(out_vals)] = np.array(assign)[np.isnan(out_vals)]
        return out_vals

    def fft(self, ssb=False, axes=(0, 1), **kwargs):
        """
        Computes the Fourier transform in two dimensions, or along a specified axis.

        Parameters
        ----------
        ssb : bool, optional
            Determines if only the single sided Fourier transform will be returned.

        axes : int, array_like, optional
            The axes along which to compute the FFT.

        Returns
        -------
        : Signal2D
            A new signal representing the Fourier transform.

        Note
        ----
        Keyword arguments can be given to the the underlying Fourier transform function
        :func:`scipy.fftpack.fft2`.
        """
        fval = fftshift(fft2(self.values, axes, **kwargs), axes=axes)
        coords = [self.axes[i].values for i in range(self.ndim)]
        for ax in axes:
            coords[ax] = fftshift(fftfreq(coords[ax].size, self.ts[ax]))
        s = Signal2D(fval, index=coords[0], columns=coords[1])

        if ssb:
            for ax in axes:
                coords[ax] = coords[ax][coords[ax] >= 0]
            s = Signal2D(s, index=coords[0], columns=coords[1])
        return s

    def window(self, index1=None, index2=None, axes=None, is_positional=False, win_fcn='hann',
               fftbins=False):
        """

        :param index1:
        :param index2:
        :param is_positional:
        :param win_fcn:
        :param fftbins:
        :return:
        """
        wind = Signal2D(0, index=self.index, columns=self.columns)

        indices = [index1, index2]
        for i in [0, -1]:
            if not is_positional:
                indices[i] = self.index.get_loc(self.self.ndices[i][0], method='nearest')
            indices[i] = self._set_val_on_axes(indices[i], axes, [i, i])


            window1 = get_window(win_fcn, wind.loc[index1[0]:index2[0]].shape[0])

        window2 = get_window(win_fcn, wind.loc[:, index1[1]:index2[1]].shape[1])
        wind.loc[index1[0]:index2[0], index1[1]:index2[1]] = np.sqrt(np.outer(window1, window2))

        if fftbins:
            if wind[-index2:-index1].size == 0:
                raise IndexError('The signal does not have values at the negative of the indices '
                                 'supplied. Disable fftbins for one-sided windowing.')
            wind[-index2:-index1] = get_window(win_fcn, len(wind[-index2:-index1]))
        return self*wind

    def filter_freq(self, cutoff, option='bp'):
        """

        :param cutoff:
        :param option:
        :return:
        """
        fval = self.fft()

    def shift_axes(self, shift, axes=None):
        """
        Shifts an axis (or both axis) by a specified amount.

        Parameters
        ----------
        shift : float, array_like
            The amount to shift the axis. If axis is specified, *shift* should be scalar. If no
            axis specified *shift* can be a scalar (shift both axes by the same amount),
            or a 2-element vector for a different shift value for each axis.

        axes : int/string, optional
            If 0  or 'Y' or 'index', shift the index, if 1 or 'X' or 'columns', shift the
            columns. If None shift both.

        Returns
        -------
        Signal2D:
            A new Signal2D with shifted axes.
        """
        shift = self._set_val_on_axes(shift, axes, [0, 0])
        return Signal2D(self.values, index=self.index-shift[0], columns=self.columns-shift[1])

    # _____________________________________________________________ #
    def scale_axes(self, scale, start=None, stop=None, axes=None):
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

        axes : int/string, optional
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
        start = self._set_val_on_axes(start, axes, [self.index[0], self.columns[0]])
        stop = self._set_val_on_axes(stop, axes, [self.index[-1], self.columns[-1]])
        if start[0] > stop[0] or start[1] > stop[1]:
            raise ValueError('start should be smaller than end.')

        scale = self._set_val_on_axes(scale, axes, [1., 1.])
        newindex, newcol = self.index.values, self.columns.values
        newindex[(newindex >= start[0]) & (newindex <= stop[0])] *= scale[0]
        newcol[(newcol >= start[1]) & (newcol <= stop[1])] *= scale[1]
        return Signal2D(self.values, index=newindex, columns=newcol)

    # _____________________________________________________________ #
    def skew(self, angle, axes=1, start=None, stop=None, ts=None, interpolate=True, **kwargs):
        """
        Applies a skew transformation on the data.

        Parameters
        ----------
        angle : float, array_like
            The angle to skew the Scan2D coordinates. If *axis* is not specified, and *angle*
            is scalar, then a skew is applied on both axes with the same angle.

        axes : integer, str, optional
            The axis along which to skew. to skew the image horizontally, axis can be 0, 'y',
            or 'index. To skew the image vertically, it can be 1, 'x', or 'columns'. If axis is
            set to None, then both axes are skewed.

        start : float, array_like, optional
            The starting coordinate (in Signal2D axes units) to apply the skew
            operation. If it is not specified, apply the skew starting from the first coordinate
            value. If it is a scalar and axis is :const:`None`, then set the start to be the same
            for both axes.

        stop : float, array_like, optional
            The stop coordinate (in Signal2D axes units) for the skew operation. Same conditions
            apply as those of *start*

        interpolate : bool, optional
            If const:`True`, realign the skewed axes on a regular grid, and use interpolation to
            recompute the values of the Signal2D at the new regular grid. The new grid will be
            computed to span the new range of the axes.Otherwise,  no realignment will occur,
            and due to the skew operation, the Signal2D values are not
            on a regular grid.

        ts : float, array_like, optional
            Only required if *interpolate* is set to :const:`True`. Specified the sampling interval
            for the new regular grid used in the interpolation. If not specified,
            then by default, the current sampling intervals of the Signal2D object will be used.

        Returns
        -------
        signal2D : Signal2D
            If *interpolate* is :const:`True`, then a new Signal2D object is returned,
            after interpolation onto a regular grid.

        X, Y : ndarray tuple
            If *interpolate* is :const:`False`, then only the new skewed coordinates are returned as
            2-D grid numpy matrices. The elements of these matrices correspond to the values in
            the current Signal2D object.
        """
        start = self._set_val_on_axes(start, axes, [self.index[0], self.columns[0]])
        stop = self._set_val_on_axes(stop, axes, [self.index[-1], self.columns[-1]])
        if start[0] > stop[0] or start[1] > stop[1]:
            raise ValueError('start should be smaller than end.')
        tan_angle = np.tan(np.deg2rad(self._set_val_on_axes(angle, axes, [0., 0.])))
        skew_matrix = [[1, tan_angle[1]], [tan_angle[0], 1]]

        x, y = np.meshgrid(self.columns, self.index, indexing='xy')
        xstart_ind = self.x.get_loc(start[1], method='nearest')
        xstop_ind = self.x.get_loc(stop[1], method='nearest')
        ystart_ind = self.y.get_loc(start[0], method='nearest')
        ystop_ind = self.y.get_loc(stop[0], method='nearest')
        # these are used to modify x and y in place.
        xslice = x[ystart_ind:ystop_ind+1, xstart_ind:xstop_ind+1]
        yslice = y[ystart_ind:ystop_ind+1, xstart_ind:xstop_ind+1]
        xskew, yskew = np.dot(skew_matrix, np.array([xslice.ravel(), yslice.ravel()]))
        xslice[:], yslice[:] = xskew.reshape(xslice.shape), yskew.reshape(yslice.shape)

        if interpolate:
            if not hasattr(ts, '__len__'):
                ts = self._set_val_on_axes(ts, axes, self.ts)

            xnew = np.arange(np.min(x), np.max(x) + ts[1], ts[1])
            ynew = np.arange(np.min(y), np.max(y) + ts[0], ts[0])
            xv, yv = np.meshgrid(xnew, ynew, indexing='xy')
            vals = griddata((y.ravel(), x.ravel()), self.values.ravel(), (yv, xv), **kwargs)

            if ('method' in kwargs) and (kwargs['method'] == 'nearest'):
                axes = self._make_axes_as_num(axes)
                if 1 in axes:
                    smin = pd.Series(np.min(x, axis=1),
                                     index=self.y).reindex(ynew, method='nearest')
                    smax = pd.Series(np.max(x, axis=1),
                                     index=self.y).reindex(ynew, method='nearest')
                    vals[xnew-smin.values.reshape(-1, 1) < 0] = np.nan
                    vals[xnew-smax.values.reshape(-1, 1) > 0] = np.nan
                if 0 in axes:
                    smin = pd.Series(np.min(y, axis=0),
                                     index=self.x).reindex(xnew, method='nearest')
                    smax = pd.Series(np.max(y, axis=0),
                                     index=self.x).reindex(xnew, method='nearest')
                    vals[ynew.reshape(-1, 1)-smin.values < 0] = np.nan
                    vals[ynew.reshape(-1, 1)-smax.values > 0] = np.nan

            return Signal2D(vals, index=ynew, columns=xnew)
        else:
            return x, y

    def flip(self, axis):
        """
        Flips the values of the uFrame without flipping corresponding X/Y-axes coordinates.

        Parameters:
            axis (int/string) :
                The axis along which to flip the values. Options are 0/'Y'/'index' or 1/'X'/columns

        Returns:
            Signal2D :
                New uFrame with axis flipped.
        """
        if axis == 'Y' or axis == 0 or axis == 'index':
            return Signal2D(self.values[::-1, :], index=self.index, columns=self.columns)
        elif axis == 'X' or axis == 1 or axis == 'columns':
            return Signal2D(self.values[:, ::-1], index=self.index, columns=self.columns)
        else:
            raise ValueError('Unknown axis value. Shoud be 0/\'Y\'/\'index\'' +
                             'or 1/\'X\'/\'columns\'')

    # _____________________________________________________________ #
    def roll(self, value, axis=None):
        """
        Circular shift of the uFrame by a given value, along a given axis.

        Parameters:
            value (float) :
                The amount (in X-Y coordinates units) by which to shift the uFrame.

            axis (string/int, optional):
                The axis along which to shift the uFrame. Options are 0/'Y'/'index'
                or 1/'X'/columns. By default, the uFrame is
                flattened before shifting, after which the original shape is restored.
                See numpy.roll for more information.

        Returns:
            Signal2D :
                The new circularly shifted uFrame.
        """
        indexes = int(np.around(value/self.Xs[axis]))
        return self._constructor(np.roll(self, indexes, axis), index=self.index,
                                 columns=self.columns)

    # _____________________________________________________________ #
    def centroid(self):
        """
        Computes the centroid of the image corresponding to the dataframe.

        Returns:
            Cx, Cy ((2,) tuple) :
                The X-coordinate, Y-coordinate of the centeroid location.
        """
        Ax, Ay = self.sum('index'), self.sum('columns')
        norm1 = Ax.sum()
        Cx = np.dot(self.columns.values, Ax.values)/norm1
        Cy = np.dot(self.index.values, Ay.values)/norm1
        return Cx, Cy

    # _____________________________________________________________ #
    def center(self, axis=None):
        """
        Move the centroid of the uFrame to the coordinates center. This will result in a
        circularly shifted uFrame.

        Parameters:
            axis (string/int, optional) :
                The axis along which to center the uFrame.
                Options are 0/'Y'/'index' or 1/'X'/columns. By default, uFrame is centered
                on both axes.

        Returns:
            Signal2D :
                The centered uFrame.
        """
        Cx, Cy = self.centroid()
        if axis == 1 or axis == 'X' or axis == 'columns' or axis is None:
            out = self.roll(np.mean(self.X) - Cx, axis=1)
        if axis == 0 or axis == 'Y' or axis == 'index' or axis is None:
            out = out.roll(np.mean(self.Y) - Cy, axis=0)
        return out

    # _____________________________________________________________ #
    def max_point(self):
        """
        Gets the X/Y coordinates of the maximum point.

        Returns:
            X, Y ((2,) tuple) :
                The X, Y coordinates of the Signal2D maximum.
        """
        X = self.max(0).idxmax()
        Y = self.loc[:, X].idxmax()
        return X, Y

    # _____________________________________________________________ #
    def flatten(self, skew_angle=0):
        """
        Flattens the Signal2D to give coordinates (X, Y, Values).

        Parameters:
            skew_angle (float, optional) :
                Angle (in degrees) to skew the X-Y coordinates before flattening.
                For example, this is used to generate True B-scan coordinates along
                the ultrasound beam angle.

        Returns:
            X, Y, Z (tuple) :
                The X-coordinates, Y-coordinates, Values of the Signal2D.
        """
        XV, YV = self.meshgrid(skew_angle=skew_angle)
        return np.array([XV.ravel(), YV.ravel(), self.values.ravel()])

    # _____________________________________________________________ #
    def meshgrid(self, indexing='xy', skew_angle=0):
        """
        Gives a meshgrid of the Signal2D coordinates.

        Parameters:
            indexing ({'xy', 'ij'}, optional) :
                Indexing of the output. There is not reason
                to change this in the context of this libary. See numpy.meshgrid for
                more information.

            skew_angle (float, optional) :
                Angle (in degrees) to skew the X-Y coordinates. For example, this is used to
                generate True B-scan coordinates along the ultrasound beam angle.

        Returns:
            X, Y (ndarray) :
                The meshgrids for the X-coordinates and Y-coordinates.
        """
        X, Y = np.meshgrid(self.X, self.Y, indexing=indexing)
        X += Y*np.sin(np.deg2rad(-1*skew_angle))
        Y = Y*np.cos(np.deg2rad(-1*skew_angle))
        return X, Y

    # _____________________________________________________________ #
    def remove_mean(self, axis=None):
        """
        Removes the mean of the uFrame along a given axis.

        Parameters:
            axis (string/int, optional):
                The axis along  which to remove the means. If axis not specified, remove the global
                mean from the uFrame.

        Returns:
            Signal2D :
                New uFrame with means subtracted along given axis.
        """
        if axis is None:
            return self - self.mean().mean()
        if axis == 'Y' or axis == 'index' or axis == 0:
            return self - self.mean(axis=0)
        elif axis == 'X' or axis == 'columns' or axis == 1:
            return (self.T - self.mean(axis=1)).T

    # _____________________________________________________________ #
    def series(self, axis=1, option='max'):
        """
        Extracts a series depending on the given option.

        Parameters:
            axis (int, optional) :
                Axis along which to extract the :class:`Signal`. Options are 0/'Y'/'index'
                or 1/'X'/'columns'.

            option (string, optional) : Currently only the option ``max`` is supported. This
                returns the signal at the maximum point in the Signal2D.

        Returns:
            Signal:
                A new :class:`Signal` object representing the extracted signal.

        """
        if option.lower() == 'max':
            X, Y = self.max_point()
            if axis == 0 or axis == 'Y' or axis == 'index':
                return self.loc[Y, :]
            elif axis == 1 or axis == 'X' or axis == 'columns':
                return self.loc[:, X]
            else:
                raise ValueError('Unknwon axis. Options are 0/\'Y\'/\'index\'' +
                                 'or 1/\'X\'/columns or None.')

    # _____________________________________________________________ #
    @property
    def ts(self):
        """ Get the signal sampling period. """
        return np.mean(np.diff(self.index)), np.mean(np.diff(self.columns))

    # _____________________________________________________________ #
    @property
    def x(self):
        """ Convenience property to return X-axis coordinates as ndarray. """
        return self.axes[1]

    # _____________________________________________________________ #
    @property
    def y(self):
        """ Convenience property to return Y-axis coordinates as ndarray. """
        return self.axes[0]

Signal2D._setup_axes(['index', 'columns'], info_axis=1, stat_axis=0,
                      axes_are_reversed=True, aliases={'rows': 0, 'y': 0, 'x': 1})