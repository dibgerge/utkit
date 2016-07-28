import pandas as pd
import numpy as np
from scipy.signal import hilbert, get_window
from scipy.interpolate import griddata
from scipy.fftpack import fft2, fftfreq, fftshift, ifft2
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
        new_xg, new_yg = np.meshgrid(columns, index, indexing='xy')
        vals = griddata((yg.ravel(), xg.ravel()), self.values.ravel(), (new_yg, new_xg), **kwargs)
        return Signal2D(vals, index=index, columns=columns)

    def reset_rate(self, ts, axes=None, **kwargs):
        """
        Re-samples the Signal to be of a specified sampling rate. The method uses interpolation
        to recompute the signal values at the new samples.

        Parameters
        ----------
        ts : float, array_like (2,)
            The sampling rate for the axes specified. If it is a scalar, and axes is
            :const:`None`, the same sampling rate will be set for both axes.

        axes : {0/'y'/'index', 1/'x'/'columns'}, optional
            The axes along which to apply the re-sampling. If it is not set, both axes will be
            re-sampled.

        Returns
        -------
        : Signal2D
            A copy of the Signal2D with the new sampling rate.
        """
        ts = self._set_val_on_axes(ts, axes, self.ts)
        indices = [np.arange(self.axes[i][0], self.axes[i][-1], ts[i]) for i in range(2)]
        return self(index=indices[0], columns=indices[1], **kwargs)

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
        axes = self._make_axes_as_num(axes)
        fval = fftshift(fft2(self.values, axes=axes, **kwargs), axes=axes)
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
        Applies a window to the signal within a given time range.

        Parameters
        ----------
        index1 : {float, int, array_like}, optional
            The start index/position of the window. Default value is minimum of index and columns.
            If *index1* is a two_element array, then it specifies the start positions for both axes.

        index2 : {float, int, array_like}, optional
            The end index/position of the window. Default value is maximum of index and columns.
            If *index2* is a two_element array, then it specifies the end positions for both axes.

        axes : {int, string, array_like}, optional
            The axes names/numbers along which to apply the window.

        is_positional : bool, optional
            Indicates whether the inputs `index1` and `index2` are positional or index units
            based. Default is :const:`False`, i.e. index units based.

        win_fcn : string/float/tuple, optional
            The type of window to create. See the function :func:`scipy.signal.get_window()` for
            a complete list of available windows, and how to pass extra parameters for a
            specific window function.

        fftbins : bool, optional
            If True, then applies a symmetric window with respect to index/columns of value 0.

        Returns
        -------
        Signal:
            The windowed Signal signal.

        Note
        ----
          If the window requires no parameters, then `win_fcn` can be a string.
          If the window requires parameters, then `win_fcn` must be a tuple
          with the first argument the string name of the window, and the next
          arguments the needed parameters. If `win_fcn` is a floating point
          number, it is interpreted as the beta parameter of the kaiser window.
        """
        indices = [index1, index2]
        for i in [0, -1]:
            if is_positional:
                indices[i] = self._set_val_on_axes(indices[i], axes, [i, i])
                indices[i] = [self.index[indices[i][0]], self.columns[indices[i][1]]]
            else:
                indices[i] = self._set_val_on_axes(indices[i], axes, [self.index[i],
                                                                      self.columns[i]])
        win2d = Signal2D(0, index=self.index, columns=self.columns)

        win1 = get_window(win_fcn, win2d.loc[indices[0][0]:indices[1][0]].shape[0])
        win2 = get_window(win_fcn, win2d.loc[:, indices[0][1]:indices[1][1]].shape[1])
        win = np.sqrt(np.outer(win1, win2))
        win2d.loc[indices[0][0]:indices[1][0], indices[0][1]:indices[1][1]] = win

        if fftbins:
            axes = self._make_axes_as_num(axes)
            if 0 in axes:
                ax0_ind = self.index[(self.index.values >= -indices[1][0]) &
                                     (self.index.values <= -indices[0][0])]
            else:
                ax0_ind = self.index
            if 1 in axes:
                ax1_ind = self.columns[(self.columns.values >= -indices[1][1]) &
                                       (self.columns.values <= -indices[0][1])]
            else:
                ax1_ind = self.columns

            if len(ax0_ind) == 0 or len(ax1_ind) == 0:
                raise IndexError('The Signal2d does not have values at the negative of the indices '
                                 'supplied. Disable fftbins for one-sided windowing.')
            win2d.loc[ax0_ind, ax1_ind] = win
        return self*win2d

    def filter_freq(self, low_freq=None, high_freq=None, axes=None, win_fcn='boxcar'):
        """
        Applies a filter in the frequency domain.

        Parameters
        ----------
        low_freq : scalar, array_like, optional
            The lower cutoff frequency for the filter. All frequencies less than this will be
            filtered out. If this is scalar, and *axes* is :const:`None`, then the same
            low_frequency will be applied for both axes.

        high_freq : scalar, array_like, optional
            The upper cutoff frequency for the filter. All frequencies higher than this will be
            filtered out. If this is scalar, and *axes* is :const:`None`, then the same
            high_frequency will be applied for both axes.

        axes : {int, string}, optional
            The axes along which to filter the 2D signal.

        win_fcn : {string, tuple}, optional
            The window type to apply for performing the filtering in the frequency domain. See the
            function :func:`scipy.signal.get_window()` for a complete list of available windows,
            and how to pass extra parameters for a specific window function.

        Returns
        -------
        : Signal2D
            The new filtered signal.
        """
        axes = self._make_axes_as_num(axes)
        fdomain = self.fft(axes=axes)
        low_freq = self._set_val_on_axes(low_freq, axes, [0.0, 0.0])
        high_freq = self._set_val_on_axes(high_freq, axes, [fdomain.index.max(),
                                                            fdomain.columns.max()])
        fdomain = fdomain.window(index1=low_freq, index2=high_freq, axes=axes, win_fcn=win_fcn,
                                 fftbins=True)
        vals = fftshift(fdomain.values, axes=axes)
        ift = ifft2(vals, axes=axes)
        return Signal2D(np.real(ift), index=self.index, columns=self.columns)

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
        : Signal2D
            A new Signal2D with shifted axes.
        """
        shift = self._set_val_on_axes(shift, axes, [0.0, 0.0])
        return Signal2D(self.values, index=self.index-shift[0], columns=self.columns-shift[1])

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

    def skew(self, angle, axes=1, start=None, stop=None, ts=None, interpolate=True, **kwargs):
        """
        Applies a skew transformation on the data.

        Parameters
        ----------
        angle : float, array_like
            The angle to skew the Scan2D coordinates. If *axis* is not specified, and *angle*
            is scalar, then a skew is applied on both axes with the same angle.

        axes : integer, str, optional
            The axis along which to skew. to skew the image vertically, axis can be 0, 'y',
            or 'index. To skew the image horizontally, it can be 1, 'x', or 'columns'. If axis is
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
            return pd.Series(np.ravel(self.values), index=[np.ravel(x), np.ravel(y)])

    def pad(self, extent, axes=None, fill=0.0, position='split'):
        """
        Adds padding along the given axes.

        Parameters
        ----------
        extent : scalar, 2-element array
            The desired extent of the axis that requires padding. If the given extent is smaller
            than the current axis extent, the signal2D will be truncated.

        axes : {0/'y'/index or 1/'x'/'columns, None}
            The axes along which to apply the padding. If :const:`None` is specified, then the
            Signal2D will be padded along both axes.

        fill : {'min', 'max', scalar}, optional
            The value to fill the padded regions:
                * 'min': Pad with values of the minimum amplitude in the signal.
                * 'max': Pad with the value of the maximum amplitude in the signal.
                * scalar: otherwise, a custom scalar value can be specified for the padding.

        position : {'start', 'end', 'split'}
            How to apply the padding to the Signal2D:
                * 'start' : apply the padding at the beginning of the axes. i.e., to the left for
                  the columns, and to the top for the index.
                * 'end' : apply the padding at the end of the axes.
                * 'split': split the padding to be half at the start and half at the end. If the
                  number of samples padded is odd, then the end will have one more sample padded
                  than the start.

        Returns
        -------
        : Signal2D
            A new Signal2D object will the axes padded. The sampling interval of the padded
            region is equal to the mean sampling rate of the corresponding axis.

        """
        extent = self._set_val_on_axes(extent, axes, np.array(self.extent) + np.array(self.ts))
        npad = np.array([int(np.ceil(extent[i]/self.ts[i] - self.shape[i])) for i in [0, 1]])
        npad_start, npad_end = npad, npad
        if position == 'end':
            npad_start = [0, 0]
        elif position == 'start':
            npad_end = [0, 0]
        elif position == 'split':
            npad_start, npad_end = np.floor(npad / 2), np.ceil(npad / 2)
        else:
            raise ValueError('Unknown value for position.')
        ax = []
        for i in [0, 1]:
            if npad_end[i] >= 0 and npad_start[i] >= 0:
                left_part = self.axes[i].values[0] - np.arange(npad_start[i], 0, -1) * self.ts[i]
                right_part = self.axes[i].values[-1] + np.arange(1, npad_end[i]+1) * self.ts[i]
                ax.append(np.concatenate((left_part, self.axes[i].values, right_part)))
            else:
                ax.append(self.axes[i][-npad_start[i]:npad_end[i]])
        if fill == 'min':
            fill = self.min().min()
        elif fill == 'max':
            fill = self.max().max()
        return self.reindex(index=ax[0], columns=ax[1], fill_value=fill)

    def pad_coords(self, index=None, columns=None, fill=0.0):
        """
        This allows padding by specifying the start and end of the coordinates for each of the axes.

        Parameters
        ----------
        index : 2-element array, optional
            Specifies the start and end of the index.

        columns : 2-element array, optional
            Specified the start and end of the columns.

        fill : float, optional
             The value to fill the padded regions:
                * 'min': Pad with values of the minimum amplitude in the signal.
                * 'max': Pad with the value of the maximum amplitude in the signal.
                * scalar: otherwise, a custom scalar value can be specified for the padding.

        Returns
        -------
        : Signal2D
            The padded signal.
        """
        if index is not None and len(index) != 2:
            raise ValueError('index_range should have a size of 2.')

        if columns is not None and len(columns) != 2:
            raise ValueError('columns_range should have a size of 2.')

        x, y = self.columns, self.index
        if index is not None:
            y = np.arange(index[0], index[1], self.ts[0])

        if columns is not None:
            x = np.arange(columns[0], columns[1], self.ts[1])

        out = self.reindex(index=y, columns=x)
        print(out)
        # out.loc[out.index < self.index[0]] = fill
        # out.loc[out.index > self.index[-1]] = fill
        # out.loc[:, out.columns < self.columns[0]] = fill
        # out.loc[:, out.columns > self.columns[-1]] = fill
        return out


        # index_start_pad, index_end_pad = self.extent[0], self.extent[0]
        # columns_start_pad, columns_end_pad = self.extent[1], self.extent[1]
        #
        # if index is not None:
        #     index_start_pad = self.extent[0] + self.index.min() - index[0]
        #     index_end_pad = self.extent[0] - (self.index.max() - index[1])
        #
        # if columns is not None:
        #     columns_start_pad = self.extent[1] + self.columns.min() - columns[0]
        #     columns_end_pad = self.extent[1] - (self.columns.max() - columns[1])
        #
        # out = self.pad([index_start_pad, columns_start_pad], axes=None, fill=fill, position='start')
        # out = out.pad([index_end_pad, columns_end_pad], axes=None, fill=fill, position='end')
        # return out

    def flip(self, axes=None):
        """
        Flips the values without flipping corresponding X/Y-axes coordinates.

        Parameters
        ----------
        axes : int/string, optional
            The axis along which to flip the values. axis can be 0/'y'/'index'. or 1/'x'/'columns'.
            If axis is set to :const:`None`, both axes will be flipped.

        Returns
        --------
        : Signal2D
            A copy of Signal2D with axis flipped.
        """
        axes = self._make_axes_as_num(axes)
        vals = self.values
        if 0 in axes:
            vals = vals[::-1, :]
        if 1 in axes:
            vals = vals[:, ::-1]
        return Signal2D(vals, index=self.index, columns=self.columns)

    def roll(self, value, axes=None):
        """
        Circular shift by a given value, along a given axis.

        Parameters
        ----------
        value : float
            The amount (in X-Y coordinates units) to shift.

        axes : string/int, optional
            The axis along which to shift. Options are 0/'Y'/'index' or 1/'X'/columns. By default,
            the uFrame is flattened before shifting, after which the original shape is restored.
            See numpy.roll for more information.

        Returns
        -------
        : Signal2D
            A copy of Signal2D after applying the circular shift.
        """
        axes = self._make_axes_as_num(axes)
        value = self._set_val_on_axes(value, axes, [0., 0.])
        out_val = self.values
        for ax in axes:
            indexes = int(np.around(value/self.ts[ax]))
            out_val = np.roll(out_val, indexes, ax)
        return Signal2D(out_val, index=self.index, columns=self.columns)

    def max_point(self):
        """
        Gets the (x, y) coordinates of the points that has the maximum amplitude.

        Returns
        -------
        x, y : ((2,) tuple)
            The (x, y) coordinates of the Signal2D maximum amplitude.
        """
        x = self.max(0).idxmax()
        y = self.loc[:, x].idxmax()
        return x, y

    def flatten(self):
        """
        Flattens the Signal2D to give coordinates (X, Y, Values).

        Returns
        -------
        x, y, z : numpy.ndarray
            The X-coordinates, Y-coordinates, Values of the Signal2D.
        """
        xv, yv = np.meshgrid(self.columns, self.index, indexing='xy')
        return np.array([xv.ravel(), yv.ravel(), self.values.ravel()])

    def remove_mean(self, axes=None):
        """
        Removes the mean along a given axis.

        Parameters
        ----------
        axes : string/int, optional:
            The axis along  which to remove the means. If axis not specified, remove the global
            mean from the uFrame.

        Returns
        -------
        : Signal2D
            A copy of signal2D with means subtracted along given axes.
        """
        axes = self._make_axes_as_num(axes)
        out = self.copy()
        for ax in axes:
            out = out - self.mean(axis=ax)
        return out

    def extract(self, axis=1, option='max'):
        """
        Extracts a 1-D Signal depending on the given option.

        Parameters
        ----------
        axis : int, optional
            Axis along which to extract the :class:`Signal`. Options are 0/'y'/'index' or
            1/'x'/'columns'.

        option : {'max'}, optional
            Currently only the option ``max`` is supported. This returns the signal at the
            maximum point in the Signal2D.

        Returns
        -------
        : Signal
            A new :class:`Signal` object representing the extracted signal.

        """
        axis = self._make_axes_as_num(axis)
        if len(axis) > 1:
            raise ValueError('axis cannot be None, or an array.')
        if option.lower() == 'max':
            x, y = self.max_point()
            if 0 in axis == 0:
                out = self.loc[y, :]
                coord = y
            elif 1 in axis:
                out = self.loc[:, x]
                coord = x
            else:
                raise ValueError('Unknown axis value.')
        else:
            raise ValueError('Unknown option value.')

        return coord, out

    @property
    def ts(self):
        """ Get the signal sampling period. """
        return np.mean(np.diff(self.index)), np.mean(np.diff(self.columns))

    @property
    def x(self):
        """ Convenience property to return X-axis coordinates as ndarray. """
        return self.axes[1]

    @property
    def y(self):
        """ Convenience property to return Y-axis coordinates as ndarray. """
        return self.axes[0]

    @property
    def extent(self):
        """ Returns the extents of the axes values"""
        return self.index.max() - self.index.min(), self.columns.max() - self.columns.min()


Signal2D._setup_axes(['index', 'columns'], info_axis=1, stat_axis=0,
                     axes_are_reversed=True, aliases={'rows': 0, 'y': 0, 'x': 1})
