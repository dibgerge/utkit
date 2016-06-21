import pandas as pd
import numpy as np
from scipy.signal import hilbert
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
                    datalen = len(data)
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
    def __call__(self, key0, key1, **kwargs):
        pass

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

        axis : int
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

    # _____________________________________________________________ #
    def shift_axis(self, shift, axis=None):
        """
        Shifts an axis (or both axis) by a specified amount.

        Parameters
        ----------
        shift : float, array_like
            The amount to shift the axis. If axis is specified, *shift* should be scalar. If no
            axis specified *shift* can be a scalar (shift both axes by the same amount),
            or a 2-element vector for a different shift value for each axis.

        axis : int/string, optional
            If 0  or 'Y' or 'index', shift the index, if 1 or 'X' or 'columns', shift the
            columns. If None shift both.

        Returns
        -------
        Signal2D:
            A new Signal2D with shifted axes.
        """
        if axis is None:
            try:
                if len(shift) != 2:
                    raise ValueError('shift should be either a scalar or 2 element array/tuple.')
            except TypeError:
                shift = [shift, shift]
            return Signal2D(self.values, index=self.index-shift[0], columns=self.columns-shift[1])
        elif axis == 'Y' or axis == 0 or axis == 'index':
            return Signal2D(self.values, index=self.index-shift, columns=self.columns)
        elif axis == 'X' or axis == 1 or axis == 'columns':
            return Signal2D(self.values, index=self.index, columns=self.columns-shift)
        else:
            raise ValueError('Unknown axis value given. See documentation for allowed axis values.')

    # _____________________________________________________________ #
    def scale_axis(self, scale, start=None, end=None, axis=None):
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

        end : float, optional
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
        """
        if axis is None:
            try:
                if len(scale) != 2:
                    raise ValueError('scale should be either a scalar or 2 element array/tuple.')
            except TypeError:
                scale = [scale, scale]
        ynew = self.copy()
        if axis is None:
            ynew.index *= scale
            ynew.columns *= scale
        elif axis == 'Y' or axis == 0:
            ynew.index *= scale
        elif axis == 'X' or axis == 1:
            ynew.colums *= scale
        else:
            raise ValueError('Unknown axis value.')
        return ynew

    # _____________________________________________________________ #
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
            return Signal2D(self.loc[::-1, :].values, index=self.index, columns=self.columns)
        elif axis == 'X' or axis == 1 or axis == 'columns':
            return Signal2D(self.loc[:, ::-1].values, index=self.index, columns=self.columns)
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
    def Xs(self):
        """ Get the signal sampling period. """
        return np.mean(np.diff(self.index)), np.mean(np.diff(self.columns))

    # _____________________________________________________________ #
    @property
    def X(self):
        """ Convenience property to return X-axis coordinates as ndarray. """
        return self.axes[1].values

    # _____________________________________________________________ #
    @property
    def Y(self):
        """ Convenience property to return Y-axis coordinates as ndarray. """
        return self.axes[0].values
