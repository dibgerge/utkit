import pandas as pd
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift, ifft
from scipy.signal import get_window, hilbert, fftconvolve
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from pandas.tools.util import cartesian_product
import matplotlib.pyplot as plt


# _____________________________________________________________ #
# _____________________________________________________________ #
class uSeries(pd.Series):
    """
    Represents a time series object, by keeping track of the time index and corresponding
    values. This class extends the Pandas :class:`pandas.Series` class to make it more
    convenient to handle signals generally encountered in ultrasonics.

    The class constructor is the same as that used for :class:`pandas.Series`, with an extra
    option. If *data* is an array, and *index* is a scalar, then *index* is interpreted as
    a sampling interval, and the signal basis will be constructed as uniformly sampled
    starting from index 0.

    .. note::
        Although the intent of this class is to represent time series signals, these signals
        can also be spatial series, where each point represents the value at a given
        spatial coordinate.
        Thus the basis of the signal is reffered to as *index*, which is consistent with
        :mod:`pandas` nomenclature. The *index* would generally refer to time or space, depending
        on the nature of the signal.
    """
    def __init__(self, data=None, index=None, dtype=None, name=None,
                 copy=False, fastpath=False):
        # Consider the case when index is scalar, then it is Ts
        if not hasattr(index, '__len__') and hasattr(data, '__len__') and index is not None:
            index = np.arange(len(data))*index

        super().__init__(data=data, index=index, dtype=dtype, name=name,
                         copy=copy, fastpath=fastpath)
        self._fdomain = None
        self._interp_fnc = None
        self._interp_k = None
        self._interp_ext = None

    @property
    def _constructor(self):
        """ Override the abstract class method so that all base class methods (i.e.
        methods in pd.Series) return objects of type utkit.Series when required.
        """
        return uSeries

    # _____________________________________________________________ #
    def __call__(self, option=''):
        """
        Returns the signal according to a given option.

        Parameters:
            options (string, char) :
                The possible options are (combined options are allowed)
                 +--------------------+--------------------------------------+
                 | *option*           | Meaning                              |
                 +====================+======================================+
                 | '' *(Default)*     | Return the raw signal                |
                 +--------------------+--------------------------------------+
                 | 'n'                | normalized signal                    |
                 +--------------------+--------------------------------------+
                 | 'e'                | signal envelop                       |
                 +--------------------+--------------------------------------+
                 | 'd'                | decibel value                        |
                 +--------------------+--------------------------------------+
        """
        yout = self
        if 'e' in option:
            yout = np.abs(hilbert(yout))
        if 'n' in option:
            yout = yout/max(np.abs(yout))
        if 'd' in option:
            yout = 20*np.log10(np.abs(yout))
        return uSeries(yout, index=self.index)

    # _____________________________________________________________ #
    def normalize(self, option='max'):
        """

        Args:
            option:

        Returns:

        """
        if option == 'energy':
            return self/np.sqrt(self.energy())
        elif option == 'max':
            return self/np.max(np.abs(self))
        else:
            raise ValueError('Unknown option value.')

    # _____________________________________________________________ #
    def stretch(self, factor, n=None):
        """

        Args:
            value:

        Returns:

        """

        uf = self.fft().resize((1-factor)*self.Fs, fftbins=True)
        t = np.arange(uf.size)/uf.range
        return uSeries(np.real(ifft(uf)), index=t/factor)

    # _____________________________________________________________ #
    def interp(self, key, k=3, ext=0):
        """
        Computes the value of the :class:`uSeries` at a given index value.
        This method makes use of the SciPy interpolation method
        :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

        Parameters :
            key (float, array_like) :
                The index value over which to compute the signal value.

            k (int, optional) :
                Degree of the spline. See :class:`scipy.interpolate.InterpolatedUnivariateSpline`
                documentation for more information.

            ext (int, optional) :
                Controls the extrapolation mode. Default is to return zeros.
                See :class:`scipy.interpolate.InterpolatedUnivariateSpline` documentation
                for more information.

        Returns:
            float:
                If *key* is a float, then the value of :class:`uSeries` at given
                *key* is returned.

            uSeries:
                If *key* is a sequence, a new :class:`uSeries` with its time base
                given by *key* is returned.
        """
        if self._interp_fnc is None or self._interp_k != k or self._interp_ext != ext:
            self._interp_fnc = InterpolatedUnivariateSpline(self.index,
                                                            np.real(self.values),
                                                            k=k, ext=ext)
            self._interp_k = k
            self._interp_ext = ext

        # if not a scalar, return a uSeries object
        if hasattr(key, '__len__'):
            return uSeries(self._interp_fnc(key), index=key)
        return self._interp_fnc(key)

    # _____________________________________________________________ #
    def stcc(self, other, width, overlap=0, start=None, end=None, win_fcn='hann'):
        """
        Compute the short-time correlation coefficient of the signal.

        Parameters:
            width (float):
                Window size that will be used in computing the short-time correlation
                coefficient.
            start (float):
                Start index for which to compute the STCC.
            end (float):
                End index for which to compute the STCC.

        Returns:
            (array_like):
                The computed STCC.
        """
        y1 = self.resample(self.Ts, start=start, end=end)
        y2 = other.resample(self.Ts, start=start, end=end)

        ind1, ind2 = 0, width
        tau, tc = [], []
        while ind2 <= y1.index[-1]:
            c = fftconvolve(y1.window(index1=ind1, index2=ind2, win_fcn=win_fcn)(),
                            y2.window(index1=ind1, index2=ind2, win_fcn=win_fcn)()[::-1],
                            mode='full')
            tau.append((y1.size - np.argmax(c))*self.Ts)
            tc.append((ind1+ind2)/2.0)
            ind1 += width-overlap
            ind2 += width-overlap
        return tc, tau

    # _____________________________________________________________ #
    def resample(self, Ts=None, start=None, end=None, k=3, ext=1):
        """
        Resample the signal by changing the sampling rate, start index, or end index.

        Parameters:
            Ts (float, optional) :
                The new signal sampling rate. If not specified, defaults
                to current signal sampling rate.

            start (float, optional) :
                The start index for resampling. If not specified,
                defaults to minimum of current index.

            end (float, optional) :
                The end index for resampling. If not specified,
                defaults to maximum of current index.

            k (int, optional) :
                Degree of the spline.See SciPy documentation for more information.

            ext (int, optional) :
                Controls the extrapolation mode. Default is to return zeros.
                See the Scipy documentation for more information.

        Returns:
            uSeries object along the given interval.

        .. note::
            If the given sampling rate is not a multiple of the signal interval
            (end - start), then the interval is cut short at the end.
        """
        # if all arguments are None, don't do anything!
        if Ts is None and start is None and end is None:
            return self

        if start is None:
            start = self.index.min()
        if end is None:
            end = self.index.max()
        if Ts is None:
            Ts = self.Ts
        tout = np.arange(start, end+Ts, Ts)
        return self.interp(tout, k=k, ext=ext)

    # _____________________________________________________________ #
    def resize(self, interval, fill=0, fftbins=False):
        """

        Args:
            n:
            option:
            isindex:
            fftbins:

        Returns:

        """
        n = int(interval / self.Ts)
        # these are only used for fftbins case
        nupper = np.ceil(n / 2.0)
        nlower = np.floor(n / 2.0)
        nl = nlower if self.size % 2 == 0 else nupper
        nr = nlower if nl == nupper else nupper
        print(interval)
        if interval > 0:
            if fill == 'mean':
                fill = np.mean(self)
            elif fill == 'max':
                fill = np.max(self)
            elif fill == 'min':
                fill = np.min(self)

            if fftbins:
                left = uSeries(np.ones(nl)*fill,
                               index=self.index.min()-np.arange(nl, 0, -1)*self.Ts)
                right = uSeries(np.ones(nr)*fill,
                                index=self.index.max()+np.arange(1, nr+1)*self.Ts)
                upos = self[self.index >= 0].append(right)
                uneg = left.append(self[self.index < 0])
                unew = upos.append(uneg)
            else:
                unew = uSeries(np.ones(n)*fill,
                               index=self.index[-1]+np.arange(1, n+1)*self.Ts)
                unew = self.append(unew, verify_integrity=True)
        elif interval < 0:
            if fftbins:
                unew = self.iloc[nl:nr]
            else:
                unew = self.iloc[:n]
        else:
            unew = self
        return uSeries(unew)


    # _____________________________________________________________ #
    def align(self, other, ext=1):
        """
        Aligns the Series to the indexbase of another given Series.

        Parameters:
            other (uSeries) :
                Another uSeries whose indexbase will be used as reference.

            ext (int, optional) :
                Controls the extrapolation mode if new indexbase is larger
                than current indexbase. Default is to return zeros.
                See the Scipy documentation for more information.

        Returns:
            uSeries :
                The signal with the new time base.
        """
        return self.interp(other.index, ext=ext)

    # _____________________________________________________________ #
    def fft(self, NFFT=None, ssb=False):
        """
        Gives a convenient way to compute the Fourier transform of a time series signal.
        The Fourier transform of a time series function is defined as:

        .. math::
           \mathcal{F}(y) ~=~ \int_{-\infty}^{\infty} y(t) e^{-2 \pi j f t}\,dt

        Parameters:
          NFFT (int, optional) :
                  Specify the number of points for the FFT. The
                  default is the length of the time series signal.

          ssb (boolean, optional) :
              If true, returns only the single side band.

        Returns:
          uSeries :
              The FFT of the signal.
        """
        if NFFT is None:
            NFFT = self.size
        uf = uSeries(fft(self, n=NFFT),
                     index=fftfreq(NFFT, self.Ts))
        if ssb:
            return uf[uf.index >= 0]

        return uf

    # _____________________________________________________________ #
    def filter(self, cutoff, option='lp', win_fcn='boxcar'):
        """
        Applies a frequency domain filter to the signal. Returns a new Series,
        and keeps the current object intact.

        Parameters:
            cutoff (float or (2,) array_like) :
                 The cuttoff frequency (Hz) of the filter.
                 This is a scalar value if type  is ``'lp'`` or ``'hp'``.
                 When type is ``'bp'``, cutoff  should be a 2 element
                 list, where the first element specifies the lower
                 cutoff frequency, and the second element specifies
                 the upper cutoff frequency.

            option (string, optional) :
                The type of filter to be used.

                +--------------------+-----------------------------------------+
                | *option*           | Meaning                                 |
                +====================+=========================================+
                | 'lp' *(Default)*   | Low-pass filter                         |
                +--------------------+-----------------------------------------+
                | 'hp'               | High-pass filter                        |
                +--------------------+-----------------------------------------+
                | 'bp'               | Band-pass filter                        |
                +--------------------+-----------------------------------------+

            win_fcn (string) :
                Apply a specific window in the frequency domain. See the function
                :func:`scipy.signal.get_window` for a complete list of
                available windows, and how to pass extra parameters for a
                specific window function.

        Returns:
            uSeries :
                The filter uSeries signal.
        """
        fdomain = self.fft()
        index1 = 0
        index2 = self.Fs/2.0
        if option == 'lp':
            index2 = cutoff
        elif option == 'hp':
            index1 = cutoff
        elif option == 'bp':
            index1 = cutoff[0]
            index2 = cutoff[1]
        else:
            raise ValueError('The value for type is not recognized.')

        fdomain = fdomain.window(index1=index1, index2=index2, win_fcn=win_fcn, fftbins=True)
        return uSeries(np.real(ifft(fftshift(fdomain))), index=self.index)

    # _____________________________________________________________ #
    def window(self, index1=None, index2=None, is_positional=False,
               win_fcn='hann', fftbins=False):
        """
        Applies a window to the signal within a given time range.

        Parameters:
          index1 (float or int, optional) :
                 The start index/position of the window. Default value is minimum of index.

          index2 (float or int, optional) :
                 The end index/position of the window. Defaul value is maximum of index.

          is_position (bool, optional):
                 Indicates whether the inputs `index1` and `index2` are positional or value
                 based. Default is :const:`False`, i.e. value based.

          win_fcn (string, float, or tuple):
                 The type of window to create. See the function
                 :func:`scipy.signal.get_window()` for a complete list of
                 available windows, and how to pass extra parameters for a
                 specific window function.

        Returns:
          uSeries:
              The windowed uSeries signal.

        .. note::
          If the window requires no parameters, then `win_fcn` can be a string.
          If the window requires parameters, then `win_fcn` must be a tuple
          with the first argument the string name of the window, and the next
          arguments the needed parameters. If `win_fcn` is a floating point
          number, it is interpreted as the beta parameter of the kaiser window.
        """
        wind = uSeries(0, index=self.index)
        if is_positional:
            index1 = wind.index[index1]
            index2 = wind.index[index2]

        wind[index1:index2] = get_window(win_fcn, len(wind[index1:index2]))
        if fftbins:
            wind[-index2:-index1] = get_window(win_fcn, len(wind[-index2:-index1]))

        return self*wind

    # _____________________________________________________________ #
    def center_frequency(self, threshold=-6):
        """
        Computes the center frequency of the signal, which is the mean of the bandwidth
        limits.

        Parameters:
            threshold (float, optional) :
                Threshold value in dB, indicating the noise floor level.
                This value should be negative, as the specturm is normalized by
                its maximum values, and thus the maximum amplitude is 0 dB. Default is -6 dB.

        Returns:
          float:
              The value of the center frequency in Hz.
        """
        if threshold > 0:
            raise ValueError("threshold should be dB value <= 0.")
        fdomain = self.fft(ssb=True)
        Yn = fdomain('nd')
        frequencies = Yn[Yn >= threshold].index
        return (frequencies.max() + frequencies.min())/2

    # _____________________________________________________________ #
    def peak_frequency(self, threshold=-6):
        """
        Computes the peak frequency of the signal.

        Parameters:
            threshold (float, optional) :
                Threshold value in dB, indicating the noise floor level.
                This value should be negative, as the specturm is normalized by
                its maximum values, and thus the maximum amplitude is 0 dB. Default is -6 dB.

        Returns:
          float: The value of the center frequency in Hz.
        """
        if threshold > 0:
            raise ValueError("threshold should be dB value <= 0.")
        return self.fft(ssb=True).idxmax()

    # _____________________________________________________________ #
    def bandwidth(self, threshold=-6):
        """
        Computes the bandwidth of the signal by finding the range of frequencies
        where the signal is above a given threshold.

        Parameters:
            threshold (float, optional) :
                Units is decibel (dB) <= 0. The spectrum is normalized by its
                maximum value, and thus the maximum amplitude is 0 dB. Default is -6 dB.

        Returns:
          float :
              The total signal bandwidth.
        """
        if threshold > 0:
            raise ValueError("threshold should be dB value <= 0.")

        fdomain = self.fft(ssb=True)
        Yn = fdomain('nd')
        frequencies = Yn[Yn >= threshold].index
        return frequencies.max() - frequencies.min()

    # _____________________________________________________________ #
    def limits(self, threshold=-20):
        """
        Computes the index limits where the signal first goes above a given threshold,
        and where it last goes below this threshold.

        Parameter:
            threshold (float, optional):
                Units is dB. The value used to compute the where the signal
                first rises above and last falls below.

        Returns:
            start_index, end_index (tuple (2,)):
                A two element tuple representing the *start_index* and *end_index* of the
                signal.
        """
        if threshold > 0:
            raise ValueError("threshold should be dB value <= 0.")
        senv = self('end')
        ind = np.where(senv >= threshold)[0]
        tout = []
        # make linear interpolation to get time value at threshold
        for i in [0, -1]:
            x1, x2 = self.index[ind[i]-1], self.index[ind[i]]
            y1, y2 = senv.iloc[ind[i]-1], senv.iloc[ind[i]]
            tout.append((threshold-y1)*(x2-x1)/(y2-y1) + x1)
        return tout[0], tout[1]

    # _____________________________________________________________ #
    def max(self, option='peak', frequency=None):
        """
        Computes the maximum of the signal according to a given method.

        Paramaters:
            option (str, optional) :
                The method to be used to compute the maximum. Supported options are:

                ==================    ======================================
                *option*               Meaning
                ==================    ======================================
                abs                   Max of the signal absolute value
                env                   Max of the signal envelop
                peak *(Default)*      Max of the raw signal
                fft                   Max of the signal FFT magnitude
                ==================    ======================================

            frequency (float, optional) :
                Used only when *option* is '*ftt*. This specifies
                at what frequency to find the FFT value. If *None*, then the maximum
                value of the fft is returned.

        Returns:
          float :
              The maximum value of the specified signal form.
        """
        if option == 'peak':
            y = self
        elif option == 'abs':
            y = self.abs()
        elif option == 'env':
            y = self('e')
        elif option == 'fft':
            F = self.fft(ssb=True)
            y = F.abs()/F.size
            if frequency is not None:
                return F.abs().interp(frequency)
        else:
            raise ValueError("The value for option is unknown. Should be: 'abs',"
                             "'env', 'peak', or 'fft'.")
        return np.max(y.values)

    # _____________________________________________________________ #
    def energy(self, option='abs'):
        """
        Computes the energy of the given waveform in the specified domain.

        Paramaters:
            option (str, optional):
                The method to be used to compute the energy. Supported options are:

                =================    ======================================
                *option*               Meaning
                =================    ======================================
                abs                  Use the absolute value of the signal
                env                  Use the envelop of the signal
                =================    ======================================
        """
        if option == 'abs':
            return simps(self.values**2, x=self.index)
        elif option == 'env':
            return simps(self('e'), x=self.index)
        else:
            raise ValueError("The value for option is unknown. Should be either 'abs' or 'env'.")

    # _____________________________________________________________ #
    def entropy(self):
        return np.sum(self.values*np.log10(self.values))

    # _____________________________________________________________ #
    def remove_mean(self):
        """
        Subtracts the mean of the signal.

        Returns:
            uSeries:
                The :class:`uSeries` with its mean subtracted.
        """
        return self - self.mean()

    # _____________________________________________________________ #
    @property
    def Ts(self):
        """ Get the signal sampling period. """
        return np.mean(np.diff(self.index[self.index >= 0]))

    # _____________________________________________________________ #
    @property
    def range(self):
        """ Get the signal sampling period. """
        return self.index.max() - self.index.min()

    # _____________________________________________________________ #
    @property
    def Fs(self):
        """ Get the signal sampling frequency. """
        return 1.0/self.Ts


class uFrame(pd.DataFrame):
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
    the uFrame basis are constructed starting from 0 at the given sampling intervals.
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
        return uFrame

    # _constructor_sliced = utkit.uSeries

    @property
    def _constructor_expanddim(self):
        return RasterScan

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
        return uFrame(yout, index=self.index, columns=self.columns)

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
            uFrame:
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
            out = uFrame(fill, index=new_axis, columns=ynew.columns)
        elif axis == 1:
            out = uFrame(fill, index=ynew.index, columns=new_axis)
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
            uFrame:
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
        Returns the value of the uFrame at a given coordinates using
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
            (float/uFrame) :
                The uFrame values at the interpolation coordinates.
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
            uFrame:
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
            uFrame:
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
            uFrame :
                New uFrame with axis flipped.
        """
        if axis == 'Y' or axis == 0 or axis == 'index':
            return uFrame(self.loc[::-1, :].values, index=self.index, columns=self.columns)
        elif axis == 'X' or axis == 1 or axis == 'columns':
            return uFrame(self.loc[:, ::-1].values, index=self.index, columns=self.columns)
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
            uFrame :
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
            uFrame :
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
                The X, Y coordinates of the uFrame maximum.
        """
        X = self.max(0).idxmax()
        Y = self.loc[:, X].idxmax()
        return X, Y

    # _____________________________________________________________ #
    def flatten(self, skew_angle=0):
        """
        Flattens the uFrame to give coordinates (X, Y, Values).

        Parameters:
            skew_angle (float, optional) :
                Angle (in degrees) to skew the X-Y coordinates before flattening.
                For example, this is used to generate True B-scan coordinates along
                the ultrasound beam angle.

        Returns:
            X, Y, Z (tuple) :
                The X-coordinates, Y-coordinates, Values of the uFrame.
        """
        XV, YV = self.meshgrid(skew_angle=skew_angle)
        return np.array([XV.ravel(), YV.ravel(), self.values.ravel()])

    # _____________________________________________________________ #
    def meshgrid(self, indexing='xy', skew_angle=0):
        """
        Gives a meshgrid of the uFrame coordinates.

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
        X += Y*np.sin(np.deg2rad(skew_angle))
        Y = Y*np.cos(np.deg2rad(skew_angle))
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
            uFrame :
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
                Axis along which to extract the :class:`uSeries`. Options are 0/'Y'/'index'
                or 1/'X'/'columns'.

            option (string, optional) : Currently only the option ``max`` is supported. This
                returns the signal at the maximum point in the uFrame.

        Returns:
            uSeries:
                A new :class:`uSeries` object representing the extracted signal.

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


# _______________________________________________________________________#
# _______________________________________________________________________#
class RasterScan(pd.Panel):
    """
    Represents data from a raster scan. Each point is represented by its 2-D coordnates
    (X, Y), and contains a time series signal with time base t.
    The axes are such that:

        * **axis 0 (items)**: *Y*-direction representing scan index,
        * **axis 1 (major_axis)**: time base *t* for each time series,
        * **axis 2 (minor_axis)**: *X*-direction representing scan position.
    """

    @property
    def _constructor(self):
        return RasterScan

    _constructor_sliced = uFrame

    # _____________________________________________________________ #
    def shift_axis(self, shift, axis=None):
        """
        Applies shifting of a given axis.

        Parameters:
            shift (float) :
                The amount to shift the axis.

            axis (int/string, optional):
                If 0/'Y'/'items', shift the Y-axis, if 1/'t'/'major_axis',
                shift the t-axis. If 2/'X'/'minor_axis' shift the X-axis.
                If None shift all axes.

        Returns:
            uFrame: A copy of uFrame with shiftes axes.
        """
        ynew = self.copy()
        if axis is None:
            ynew.items -= shift[0]
            ynew.major_axis -= shift[1]
            ynew.minor_axis -= shift[2]
        elif axis == 'Y' or axis == 0 or axis == 'items':
            ynew.items -= shift
        elif axis == 't' or axis == 1 or axis == 'major_axis':
            ynew.major_axis -= shift
        elif axis == 'X' or axis == 2 or axis == 'minor_axis':
            ynew.minor_axis -= shift
        else:
            raise ValueError('Unknown value for axis.')
        return ynew

    # _____________________________________________________________ #
    def scale_axis(self, scale, axis=None):
        """
        Applies scaling of a given axis.

        Parameters:
            scale (float) :
                The amount to scale the axis.

            axis (int/string, optional):
                If 0/'Y'/'items', scale the Y-axis, if 1/'t'/'major_axis',
                scale the t-axis. If 2/'X'/'minor_axis' scale the X-axis.
                If None scale all axes.

        Returns:
            uFrame :
                A copy of uFrame with scaled axes.
        """
        ynew = self.copy()
        if axis is None:
            ynew.items *= scale
            ynew.major_axis *= scale
            ynew.minor_axis *= scale
        elif axis == 'Y' or axis == 0 or axis == 'items':
            ynew.items *= scale
        elif axis == 't' or axis == 1 or axis == 'major_axis':
            ynew.major_axis *= scale
        elif axis == 'X' or axis == 2 or axis == 'minor_axis':
            ynew.minor_axis *= scale
        else:
            raise ValueError('Unknown value for axis.')
        return ynew

    # _______________________________________________________________________#
    def bscan(self, depth='max'):
        """
        Computes B-scan image generally used for Utrasound Testing.
        Slices the raster scan along the X-t axes (i.e. at a given Y location).

        Parameters :
            depth (float/string, optional) :
                Select which slice to be used for the b-scan. If 'max', the
                slice passing through the maximum point is selected.
        Returns :
            uFrame :
                A uFrame object representing the B-scan.
        """
        if depth == 'max':
            _, Y = self.cscan().max_point()
            return self.loc[Y, :, :]
        else:
            if depth > self.items.max() or depth < self.items.min():
                raise ValueError('depth is outside range of scan in Y-direction.')
            # find the nearest Y-coordinate near the given depth
            ind = self.items[np.abs(self.items.values - depth).argmin()]
            return self.loc[ind, :, :]

    # _______________________________________________________________________#
    def dscan(self, depth='max'):
        """
        Convenience method that return b-scans for Utrasound Testing raster scans.
        Slices the raster scan along the Y-t axes (i.e. at a given X location).

        Parameters:
            - depth (float/sting, optional):
                Select which slice to be used for the b-scan. If 'max', the
                slice passing through the maximum point is selectd.
        Returns:
            uFrame :
                A uFrame object representing the D-scan.
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

    # _______________________________________________________________________#
    def cscan(self, depth='project', option='max', skew_angle=0):
        """
        Computes C-scan image generally used for Utrasound Testing. This collapses the
        raster scan along the T direction, or takes a slice at a given T coordinate,
        resulting in an image spanning the X-Y axes.

        Parameters:
            depth (float/string, optional):
                The methodology to obtain the C-scan. Supported options are:
                'project' collapses the whole T-axis into a single value using the methodology
                specified by the option parameter. 'max' obtains the
                T-axis coordinate where the T-value is maximum. Otherwise depth
                is a float value specifiying the actual T-coordinate value.

            option (string, optional) :
                This is relevant only if depth is 'project'.
                Then option specifies how the projection is done. If 'max', then
                at each X-Y point, the maximum value along the T-axis is selected.
                If 'enegry', then at each X-Y point, the total signal energy
                along the T-axis is computed.

            skew_angle (float, optional) :
                Skews the X-direction to be aligned with the ultrasonic beam angle,
                before projecting the T-axis onto the X-axis.

        Returns:
            uFrame :
                A uFrame object representing the C-scan.
        """
        if depth == 'project':
            X, t, _ = self.iloc[0, :, :].flatten(skew_angle=skew_angle)
            X = np.around(X / self.Xs[2])*self.Xs[2]
            df = uFrame(self.to_frame().values, index=[t, X], columns=self.axes[0])
            if option == 'energy':
                return uFrame(df.var(level=1)).T
            elif option == 'max':
                return uFrame(df.max(level=1)).T
            else:
                raise ValueError('Uknown value for option.')
        elif depth == 'max':
            _, ind = self.bscan().max_point()
            return self.loc[:, ind, :].T
        else:
            if depth > self.major_axis.max() or depth < self.major_axis.min():
                raise ValueError('depth is outside range of scan in T-direction.')
            # find the nearest T-coordinate near the given depth
            ind = self.major_axis[np.abs(self.major_axis.values - depth).argmin()]
            return self.loc[:, ind, :].T

    # _______________________________________________________________________#
    def flatten(self):
        """
        Flatten an array and its corresponding indices.

        Returns:
            Y, T, X, values (tuple):
                A 4-element tuple where each element is a flattened array of the RasterScan,
                and each representing a point with coordinates Y, T, X and its value.
        """
        yv, tv, xv = np.meshgrid(self.Y, self.t, self.X, indexing='xy')
        return np.array([yv.ravel(), tv.ravel(), xv.ravel(), self.values.ravel()])

    # _______________________________________________________________________#
    def _apply_1d(self, func, axis):
        """
        This function copied from Pandas Panel definition, and modified to support the
        utkit subclass heirarchy.
        """
        axis_name = self._get_axis_name(axis)
        # ax = self._get_axis(axis)
        ndim = self.ndim
        values = self.values

        # iter thru the axes
        slice_axis = self._get_axis(axis)
        slice_indexer = [0] * (ndim - 1)
        indexer = np.zeros(ndim, 'O')
        indlist = list(range(ndim))
        indlist.remove(axis)
        indexer[axis] = slice(None, None)
        indexer.put(indlist, slice_indexer)
        planes = [self._get_axis(axi) for axi in indlist]
        shape = np.array(self.shape).take(indlist)

        # all the iteration points
        points = cartesian_product(planes)

        results = []
        for i in range(np.prod(shape)):
            # construct the objectD
            pts = tuple([p[i] for p in points])
            indexer.put(indlist, slice_indexer)

            obj = uSeries(values[tuple(indexer)], index=slice_axis, name=pts)
            result = func(obj)
            results.append(result)

            # increment the indexer
            slice_indexer[-1] += 1
            n = -1
            while (slice_indexer[n] >= shape[n]) and (n > (1 - ndim)):
                slice_indexer[n - 1] += 1
                slice_indexer[n] = 0
                n -= 1

        # empty object
        if not len(results):
            return self._constructor(**self._construct_axes_dict())

        # same ndim as current
        if isinstance(results[0], uSeries):
            arr = np.vstack([r.values for r in results])
            arr = arr.T.reshape(tuple([len(slice_axis)] + list(shape)))
            tranp = np.array([axis] + indlist).argsort()
            arr = arr.transpose(tuple(list(tranp)))
            return self._constructor(arr, **self._construct_axes_dict())

        # ndim-1 shape
        results = np.array(results).reshape(shape)
        if results.ndim == 2 and axis_name != self._info_axis_name:
            results = results.T
            planes = planes[::-1]
        return self._construct_return_type(results, planes)

    # TODO: Potential new methods:
    #        * resample

    # _______________________________________________________________________#
    @property
    def Xs(self):
        """
        The sampling intervals for the three axes (X, Y, T).
        """
        return np.mean(np.diff(self.items.values)),\
            np.mean(np.diff(self.major_axis.values)),\
            np.mean(np.diff(self.minor_axis.values))

    # _______________________________________________________________________#
    @property
    def X(self):
        """ Convenience property to return X-axis coordinates as ndarray. """
        return self.minor_axis.values

# _______________________________________________________________________#
    @property
    def Y(self):
        """ Convenience property to return Y-axis coordinates as ndarray. """
        return self.items.values

# _______________________________________________________________________#
    @property
    def t(self):
        """ Convenience property to return t-axis coordinates as ndarray. """
        return self.major_axis.values
