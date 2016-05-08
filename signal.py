import pandas as pd
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift, ifft
from scipy.signal import get_window, hilbert, fftconvolve
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
# from peakutils import peak


# _____________________________________________________________ #
class Signal(pd.Series):
    """
    Represents physical signals, by keeping track of the index (time/space) and corresponding
    values. This class extends the Pandas :class:`pandas.Series` class to make it more
    convenient to handle signals generally encountered in ultrasonics.

    The class constructor is the same as that used for :class:`pandas.Series`, with an extra
    option. If *data* is an array, and *index* is a scalar, then *index* is interpreted as
    a sampling interval, and the signal basis will be constructed as uniformly sampled
    starting from index 0.
    """
    def __init__(self, data=None, index=None, dtype=None, name=None,
                 copy=False, fastpath=False):
        # Consider the case when index is scalar, then it is Ts
        if not hasattr(index, '__len__') and hasattr(data, '__len__') and index is not None:
            index = np.arange(len(data))*index

        super().__init__(data=data, index=index, dtype=dtype, name=name,
                         copy=copy, fastpath=fastpath)
        # self._fdomain = None
        # self._interp_fnc = None
        # self._interp_k = None
        # self._interp_ext = None

    @property
    def _constructor(self):
        """ Override the abstract class method so that all base class methods (i.e.
        methods in pd.Series) return objects of type utkit.Series when required.
        """
        return Signal

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
        return Signal(yout, index=self.index)

    # _____________________________________________________________ #
    def fft(self, nfft=None, ssb=False):
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
          Signal :
              The FFT of the signal.
        """
        if nfft is None:
            nfft = self.size
        uf = Signal(fft(self, n=nfft),
                    index=fftfreq(nfft, self.Ts))
        if ssb:
            return uf[uf.index >= 0]

        return uf

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
    def segment(self, thres, pulse_width, win_fcn='hann'):
        """

        :return:
        """
        peak_ind = peak.indexes(self.values, thres=thres, min_dist=int(pulse_width*self.Fs))
        wind_len = np.mean(np.diff(self.index[peak_ind]))
        parts = [self.window(index1=self.index[i]-wind_len/2.0,
                             index2=self.index[i]+wind_len/2.0,
                             win_fcn=win_fcn) for i in peak_ind]
        return parts

    # _____________________________________________________________ #
    def tof(self, other):
        """

        :param other:
        :return:
        """
        c = fftconvolve(self('n'), other('n')[::-1], mode='full')
        ind = self.size - np.argmax(c)
        return self.Ts * ind

    # _____________________________________________________________ #
    def stretch(self, factor, n=None):
        """

        Args:
            value:

        Returns:

        """

        uf = self.fft().resize((1-factor)*self.Fs, fftbins=True)
        t = np.arange(uf.size)/uf.range
        return Signal(np.real(ifft(uf)), index=t / factor)

    # _____________________________________________________________ #
    def interp(self, key, k=3, ext=0):
        """
        Computes the value of the :class:`Signal` at a given index value.
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
                If *key* is a float, then the value of :class:`Signal` at given
                *key* is returned.

            Signal:
                If *key* is a sequence, a new :class:`Signal` with its time base
                given by *key* is returned.
        """
        if self._interp_fnc is None or self._interp_k != k or self._interp_ext != ext:
            self._interp_fnc = InterpolatedUnivariateSpline(self.index,
                                                            np.real(self.values),
                                                            k=k, ext=ext)
            self._interp_k = k
            self._interp_ext = ext

        # if not a scalar, return a Signal object
        if hasattr(key, '__len__'):
            return Signal(self._interp_fnc(key), index=key)
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
            Signal object along the given interval.

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
                left = Signal(np.ones(nl) * fill,
                              index=self.index.min()-np.arange(nl, 0, -1)*self.Ts)
                right = Signal(np.ones(nr) * fill,
                               index=self.index.max()+np.arange(1, nr+1)*self.Ts)
                upos = self[self.index >= 0].append(right)
                uneg = left.append(self[self.index < 0])
                unew = upos.append(uneg)
            else:
                unew = Signal(np.ones(n) * fill,
                              index=self.index[-1]+np.arange(1, n+1)*self.Ts)
                unew = self.append(unew, verify_integrity=True)
        elif interval < 0:
            if fftbins:
                unew = self.iloc[nl:nr]
            else:
                unew = self.iloc[:n]
        else:
            unew = self
        return Signal(unew)


    # _____________________________________________________________ #
    def align(self, other, ext=1):
        """
        Aligns the Series to the indexbase of another given Series.

        Parameters:
            other (Signal) :
                Another Signal whose indexbase will be used as reference.

            ext (int, optional) :
                Controls the extrapolation mode if new indexbase is larger
                than current indexbase. Default is to return zeros.
                See the Scipy documentation for more information.

        Returns:
            Signal :
                The signal with the new time base.
        """
        return self.interp(other.index, ext=ext)



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
            Signal :
                The filter Signal signal.
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
        return Signal(np.real(ifft(fftshift(fdomain))), index=self.index)

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
          Signal:
              The windowed Signal signal.

        .. note::
          If the window requires no parameters, then `win_fcn` can be a string.
          If the window requires parameters, then `win_fcn` must be a tuple
          with the first argument the string name of the window, and the next
          arguments the needed parameters. If `win_fcn` is a floating point
          number, it is interpreted as the beta parameter of the kaiser window.
        """
        wind = Signal(0, index=self.index)
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
            Signal:
                The :class:`Signal` with its mean subtracted.
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
