import pandas as pd
import numpy as np
from scipy.signal import hilbert
from scipy.interpolate import InterpolatedUnivariateSpline



class Scan2D(pd.DataFrame):
    """
    Extends :class:`pandas.DataFrame` class to support operations commonly required
    in ultrasonics. The axis convention used is:

        * **Axis 1 (columns)**: *X*-direction
        * **Axis 0 (index)**: *Y*-direction

    For example, in the context of ultrasonic inspection, the *X*-direction would represent
    the spatial line scan, and the *Y*-direction represents the signal time base, which can
    be scaled to represent the ultrasonic beam depth through the material.

    The class constructor is similar as that of :class:`pandas.DataFrame` with the added
    option of specifying only the sampling intervals along the *X* and *Y* directions.
    Thus, if *index* and/or *columns* are scalars, which *data* is a 2-D array, then
    the Scan2D basis are constructed starting from 0 at the given sampling intervals.

    .. autosummary::

        resample_axis
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        # Consider the case when index is scalar, then it is Ts
        if not hasattr(index, '__len__') and hasattr(data, '__len__') and index is not None:
            index = np.arange(data.shape[0])*index

        if not hasattr(columns, '__len__') and hasattr(data, '__len__') and index is not None:
            columns = np.arange(data.shape[1])*columns
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self._fdomain = None
        self._interp_fnc = None
        self._interp_s = None

    @property
    def _constructor(self):
        return Scan2D

    # _constructor_sliced = utkit.Signal

    @property
    def _constructor_expanddim(self):
        return Scan3D

    # _____________________________________________________________ #
    def __call__(self, option=''):
        """
        Returns the signal according to a given option.

        Parameters:
            options (string, char) :
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
        """
        yout = self
        if 'e' in option:
            yout = np.abs(hilbert(yout, axis=0))
        if 'n' in option:
            yout = yout/np.abs(yout).max().max()
        if 'd' in option:
            yout = 20*np.log10(np.abs(yout))
        return Scan2D(yout, index=self.index, columns=self.columns)

    # _____________________________________________________________ #
    def resample_axis(self, start=None, end=None, step=None, axis=0, fill='min'):
        """
        Resamples Frame along a given axis. Currently only downsampling at integer multiples of
        current sampling rate is supported (to make this method fast).
        If upsampling, or non-integer multiples of sampling frequency are required, consider using
        interpolation functions.

        Parameters:
            start (float, optional) :
                The start index along the given axis. Default is the minimum index of the axis.

            end (float, optional):
                The end index along the given axis. Default is the maximum index of the axis.

            step (float, optional):
                The sampling step size. If this step size is smaller than current sampling step
                size, it will be defaulted to current step size. If step is not a multiple of
                current sampling step size, it will be rounded to the nearest multiple. Defaults to
                current step size if not specified.

            axis (int, optional):
                The axis to resample, take values of 0 (index/Y-axis) or 1 (columns/X-axis).

            fill (string/float, optional):
                Value used to fill the uFrame if padding is required. Supported options:

                 +--------------------+--------------------------------------+
                 | *fill*             | Meaning                              |
                 +====================+======================================+
                 | 'min' *(Default)*  | Minimum value in current uFrame      |
                 +--------------------+--------------------------------------+
                 | 'max'              | Maximum value in current uFrame      |
                 +--------------------+--------------------------------------+
                 | 'mean'             | Mean value of the uFrame             |
                 +--------------------+--------------------------------------+
                 | (float)            | A custom scalar                      |
                 +--------------------+--------------------------------------+

        Returns:
            Scan2D:
                A new uFrame with its axis resampled.
        """
        if fill == 'min':
            fill = self.min().min()
        elif fill == 'max':
            fill = self.max().max()
        elif fill == 'mean':
            fill = self.mean().mean()

        if start is None and end is None and step is None:
            return self
        if start is None:
            start = self.axes[axis].min()
        if end is None:
            end = self.axes[axis].max()
        if step is None:
            step = 1
        else:
            step = int(np.around(step/self.Xs[axis]))
            if step == 0:
                step = 1

        # resample the original array
        ynew = self
        if axis == 1:
            ynew = self.loc[:, start:end:step]
        elif axis == 0:
            ynew = self.loc[start:end:step, :]
        else:
            raise ValueError('Unknown axis value.')

        new_axis = ynew.axes[axis].values

        # pad the start of the dataframe if needed
        npad_start = int((ynew.axes[axis].min() - start)/ynew.Xs[axis])
        if npad_start > 0:
            start = ynew.axes[axis].min() - npad_start*ynew.Xs[axis]
            new_axis = np.concatenate((np.arange(npad_start)*ynew.Xs[axis] + start, new_axis))

        # pad the end of the dataframe if needed
        npad_end = int((end - ynew.axes[axis].max())/ynew.Xs[axis])
        if npad_end > 0:
            end = ynew.axes[axis].max() + ynew.Xs[axis]
            new_axis = np.concatenate((new_axis, np.arange(npad_end)*ynew.Xs[axis] + end))

        # construct the new dataframe with the new axis
        if axis == 0:
            out = Scan2D(fill, index=new_axis, columns=ynew.columns)
        elif axis == 1:
            out = Scan2D(fill, index=ynew.index, columns=new_axis)
        out.loc[ynew.index.values, ynew.columns.values] = ynew.values
        return out

    # _____________________________________________________________ #
    def resample(self, xstart=None, ystart=None, xend=None, yend=None,
                 xstep=None, ystep=None, fill='min'):
        """
        Resamples the uFrame along both the X-Axis and Y-Axis. Uses the method resample_axis for
        fulfilling this purpose. Currently only downsampling is supported. See docs of
        resample_axis for more information.

        Parameters:
            xstart (float, optional):
                The X-axis (axis 1/columns) start. Default is the minimum of current X-axis.

            ystart (float, optional):
                The Y-axis (axis 0/index) start. Default is the minimum of current Y-axis.

            xend (float, optional):
                The X-axis (axis 1/columns) end. Default is the maximum of current X-axis.

            yend (float, optional):
                The Y-axis (axis 0/index) end. Default is the maximum of current Y-axis.

            xstep (float, optional):
                The X-axis sampling step size. See resample_axis for more information.

            ystep (float, optional):
                The Y-axis sampling step size. See resample_axis for more information.

            fill (string/float, optional):
                Value used to fill the uFrame if padding is required. See resample_axis for more
                information.

        Returns:
            Scan2D:
                A new uFrame with its axes resampled.
        """
        # these are not needed (duplicated from resample_axis), but they can make execution faster
        if fill == 'min':
            fill = self.min().min()
        elif fill == 'max':
            fill = self.max().max()
        elif fill == 'mean':
            fill = self.mean().mean()

        df = self.resample_axis(start=xstart, end=xend, step=xstep, axis=1, fill=fill)
        df = df.resample_axis(start=ystart, end=yend, step=ystep, axis=0, fill=fill)
        return df

    # _____________________________________________________________ #
    def interp(self, x, y, s=None):
        """
        Returns the value of the Scan2D at a given coordinates using
        Scipy SmoothBivariateSpline interpolation. In this implementation,
        the spline degree is set to 1.

        Parameters:
            x (float, array_like) :
                The X-axis value(s) over which to compute the signal value.

            y (float, array_like) :
                The Y-axis value(s) over which to compute the signal value.

            s (float, optional) :
                Smoothing factor. See the Scipy documentation for more information.

        Returns:
            (float/Scan2D) :
                The Scan2D values at the interpolation coordinates.
        """
        if self._interp_fnc is None or self._interp_s != s:
            xv, yv, zv = self.flatten()
            self._interp_fnc = InterpolatedUnivariateSpline(xv, yv, zv, kx=1, ky=1, s=s)
            self._interp_s = s
            self.apply(lambda n: InterpolatedUnivariateSpline(y, n.values))

        return self._interp_fnc(x, y)

    # _____________________________________________________________ #
    def shift_axis(self, shift, axis=None):
        """
        Applies shifting of a given axis.

        Parameters:
            shift (float):
                The amount to shift the axis.

            axis (int/string, optional):
                If 0 or 'Y', shift the index, if 1 or 'X', shift the columns. If None shift both.

        Returns:
            Scan2D:
                A copy of uFrame with shiftes axes.
        """
        ynew = self.copy()
        if axis is None:
            ynew.index -= shift[1]
            ynew.columns -= shift[0]
        elif axis == 'Y' or axis == 0 or axis == 'index':
            ynew.index -= shift
        elif axis == 'X' or axis == 1 or axis == 'columns':
            ynew.columns -= shift
        else:
            raise ValueError('Unknwon axis value.')
        return ynew

    # _____________________________________________________________ #
    def scale_axis(self, scale, axis=None):
        """
        Applies scaling of a given axis.

        Parameters:
            scale (float) :
                The amount to scale the axis.

            axis (int/string, optional):
                If 0 or 'Y', scale the index, if 1 or 'X', scale the columns. If None scale both.

        Returns:
            Scan2D:
                A copy of uFrame with scaled axes.
        """
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
            Scan2D :
                New uFrame with axis flipped.
        """
        if axis == 'Y' or axis == 0 or axis == 'index':
            return Scan2D(self.loc[::-1, :].values, index=self.index, columns=self.columns)
        elif axis == 'X' or axis == 1 or axis == 'columns':
            return Scan2D(self.loc[:, ::-1].values, index=self.index, columns=self.columns)
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
            Scan2D :
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
            Scan2D :
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
                The X, Y coordinates of the Scan2D maximum.
        """
        X = self.max(0).idxmax()
        Y = self.loc[:, X].idxmax()
        return X, Y

    # _____________________________________________________________ #
    def flatten(self, skew_angle=0):
        """
        Flattens the Scan2D to give coordinates (X, Y, Values).

        Parameters:
            skew_angle (float, optional) :
                Angle (in degrees) to skew the X-Y coordinates before flattening.
                For example, this is used to generate True B-scan coordinates along
                the ultrasound beam angle.

        Returns:
            X, Y, Z (tuple) :
                The X-coordinates, Y-coordinates, Values of the Scan2D.
        """
        XV, YV = self.meshgrid(skew_angle=skew_angle)
        return np.array([XV.ravel(), YV.ravel(), self.values.ravel()])

    # _____________________________________________________________ #
    def meshgrid(self, indexing='xy', skew_angle=0):
        """
        Gives a meshgrid of the Scan2D coordinates.

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
            Scan2D :
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
                returns the signal at the maximum point in the Scan2D.

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
