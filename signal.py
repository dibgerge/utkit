import pandas as pd
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift, ifft
from scipy.signal import get_window, hilbert, fftconvolve, spectrogram, welch, coherence
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from . import peakutils
from .signal2d import Signal2D


# _____________________________________________________________ #
class Signal(pd.Series):
    """
    Represents physical signals, by keeping track of the index (time/space) and corresponding
    values. This class extends the Pandas :class:`pandas.Series` class to make it more
    convenient to handle signals generally encountered in ultrasonics or RF.

    The class constructor is the same as that used for :class:`pandas.Series`. In addition,
    if :attr:`data` is an array, and :attr:`index` is a scalar, then :attr:`index` is interpreted as
    the sampling time, and the signal basis will be constructed as uniformly sampled at intervals
    specified by the scalar :attr:`index` and starting from 0.

    For example, to define a sine wave signal at 100 kHz and sampling interval of 1 microsecond,

    .. code-block:: python

        import numpy as np
        import utkit

        Ts = 1e-6
        t = np.arange(100)*Ts
        s = utkit.Signal(np.sin(2*np.pi*100e3*t), index=t)

    the last line which calls the constructor of :class:`Signal` is also equivalent to:

    .. code-block:: python

        s = utkit.Signal(np.sin(2*np.pi*100e3*t), index=Ts)

    The class :class:`Signal` provides various methods for signal reshaping, transforms,
    and feature extraction.
    """
    def __init__(self, data=None, index=None, *args, **kwargs):
        if index is not None and data is not None:
            if not hasattr(index, '__len__') and hasattr(data, '__len__'):
                index = np.arange(len(data)) * index
            elif not hasattr(index, '__len__') and not hasattr(data, '__len__'):
                index = [index]

        super().__init__(data, index, *args, **kwargs)
        if not self.index.is_monotonic_increasing:
            raise ValueError('Index must be monotonically increasing.')

        self._interp_fnc = None
        self._interp_kwargs = {'w': None,
                               'bbox': [None, None],
                               'k': 3,
                               'ext': 1,
                               'check_finite':  False}

    @property
    def _constructor(self):
        """ Override the abstract class method so that all base class methods (i.e.
        methods in pd.Series) return objects of type utkit.Series when required.
        """
        return Signal

    # _____________________________________________________________ #
    def __call__(self, key=None, ts=None, **kwargs):
        """
        Make the Signal act like a function, where given index values are computed using an
        interpolant, and do not necessarily require to be at one of the current sample points.
        This method makes use of the SciPy interpolation method
        :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

        Parameters
        ----------
        key : float, array_like
           The index value over which to compute the signal value. Can be either a scalar or a
           sequence of indices.

        ts : float, optional
            *ts* can be specified only if *key* has not been specified. This will resample the
            signal to the sample interval specified by *ts*.

        Returns
        -------
        value : float
           If *key* is a float, then the value of :class:`Signal` at given
           *key* is returned.

        value : Signal
           If *key* is a sequence, a new :class:`Signal` with its time base
           given by *key* is returned.

        Note
        ----
        This method take additional optional keyword arguments that are passed to the interpolant
        function. To see a description of supported arguments, refer to the documentation of
        :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

        Note
        ----
        One difference in default values for the argument is that the extrapolation mode is set
        to 'zeros', instead of 'extrapolate'. This is based on the assumption that the signal
        goes to zero outside of its sampled range.
        """
        if key is None and ts is None:
            return self.copy()

        if key is not None and ts is not None:
            raise AttributeError("Only one of key or ts can be specified at one time, not both.")

        if ts is not None:
            key = np.arange(self.index[0], self.index[-1], ts)

        same_args = True
        for k in kwargs:
            if self._interp_kwargs[k] != kwargs[k]:
                same_args = False
                self._interp_kwargs[k] = kwargs[k]

        if self._interp_fnc is None or not same_args:
            self._interp_fnc = InterpolatedUnivariateSpline(self.index, self.values,
                                                            **self._interp_kwargs)
        return Signal(self._interp_fnc(key), index=key if hasattr(key, '__len__') else [key])

    # _____________________________________________________________ #
    def window(self, index1=None, index2=None, is_positional=False, win_fcn='hann', fftbins=False):
        """
        Applies a window to the signal within a given time range.

        Parameters
        ----------
        index1 : float or int, optional
            The start index/position of the window. Default value is minimum of index.

        index2 : float or int, optional
            The end index/position of the window. Defaul value is maximum of index.

        is_positional : bool, optional
            Indicates whether the inputs `index1` and `index2` are positional or value
            based. Default is :const:`False`, i.e. value based.

        win_fcn : string/float/tuple, optional
            The type of window to create. See the function
            :func:`scipy.signal.get_window()` for a complete list of
            available windows, and how to pass extra parameters for a
            specific window function.

        fftbins : bool, optional
            If True, then applies a symmetric window with respect to index of value 0.

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
        wind = Signal(0, index=self.index)
        if is_positional:
            if isinstance(index1, float) or isinstance(index2, float):
                raise ValueError('Indices are floats, are you sure you want positional indices?')
            index1 = wind.index.values[index1]
            index2 = wind.index.values[index2]

        wind[index1:index2] = get_window(win_fcn, len(wind[index1:index2]))
        if fftbins:
            if wind[-index2:-index1].size == 0:
                raise IndexError('The signal does not have values at the negative of the indices '
                                 'supplied. Disable fftbins for one-sided windowing.')
            wind[-index2:-index1] = get_window(win_fcn, len(wind[-index2:-index1]))
        return self*wind

    # _____________________________________________________________ #
    def normalize(self, option='max'):
        """
        Normalizes the signal according to a given option.

        Parameters
        ----------
        option: string, optional
            Method to be used to normalize the signal. The possible options are:

                * *'max' (Default)* : Divide by the maximum of the signal, so that the normalized
                  maximum has an amplitude of 1.

                * *'energy'*: Divide by the signal energy.

        Returns
        -------
            : Signal
                Signal with normalized amplitude.
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
        Segments the signal into a collection of signals, with each item in the collection,
        representing the signal within a given time window. This is usually useful to
        automate the extraction of multiple resolvable echoes.

        Parameters
        ----------
        thres : float
            A threshold value (in dB). Search for echoes will be only for signal values
            above this given threshold. Note that the maximum threshold is 0 dB, since
            the signal will be normalized by its maximum before searching for echoes.

        pulse_width : float
            The expected pulse_width. This should have the same units as the units of the Signal
            index. If this is not known exactly, it is generally better to specify this
            parameter to be slightly larger than the actual pulse_width.

        win_fcn : string, array_like
            The window type that will be used to window each extracted segment (or echo). See
            :func:`scipy.signal.get_window()` for a complete list of available windows,
            and how to pass extra parameters for a specific window type, if needed.

        Returns
        -------
            : list
            A list with elements of type :class:`Signal`. Each Signal element represents an
            extracted segment.
        """
        peak_ind = peakutils.indexes(self.values, thres=thres, min_dist=int(pulse_width * self.fs))
        wind_len = np.mean(np.diff(self.index[peak_ind]))
        parts = [self.window(index1=self.index[i]-wind_len/2.0,
                             index2=self.index[i]+wind_len/2.0,
                             win_fcn=win_fcn) for i in peak_ind]
        return parts

    # _____________________________________________________________ #
    def operate(self, option=''):
        """
        This is a convenience function for common operations on signals. Returns the signal
        according to a given option.

        Parameters
        ----------
        options (string, char) :
            The possible options are (combined options are allowed)
            +--------------------+--------------------------------------+
            | *option*           | Meaning                              |
            +====================+======================================+
            | '' *(Default)*     | Return the raw signal                |
            +--------------------+--------------------------------------+
            | 'n'                | normalized signal by max amplitude   |
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
        Computes the Fast Fourier transform of the signal using :func:`scipy.fftpack.fft` function.
        The Fourier transform of a time series function is defined as:

        .. math::
           \mathcal{F}(y) ~=~ \int_{-\infty}^{\infty} y(t) e^{-2 \pi j f t}\,dt

        Parameters
        ----------
        nfft : int, optional
            Specify the number of points for the FFT. The default is the length of
            the time series signal.

        ssb : boolean, optional
            If true, returns only the single side band (components corresponding to positive
            frequency).

        Returns
        -------
         : Signal
            The FFT of the signal.
        """
        if nfft is None:
            nfft = self.size

        uf = Signal(fftshift(fft(self, n=nfft)), index=fftshift(fftfreq(nfft, self.ts)))
        return uf[uf.index >= 0] if ssb else uf

    # _____________________________________________________________ #
    def limits(self, threshold=-20):
        """
        Computes the index limits where the signal first goes above a given threshold,
        and where it last goes below this threshold. Linear interpolation is used to find the
        exact point for this crossing.

        Parameters
        ----------
        threshold : float, optional
            Units is dB. The value used to compute the where the signal first rises above and
            last falls below.

        Returns
        -------
        start_index, end_index (tuple (2,)):
            A two element tuple representing the *start_index* and *end_index* of the signal.
        """
        if threshold > 0:
            raise ValueError("threshold should be dB value <= 0, which is relative to the "
                             "normalized maximum amplitude.")
        senv = self.operate('nd')
        ind = np.where(senv >= threshold)[0]
        tout = []
        # make linear interpolation to get time value at threshold
        for i in [0, -1]:
            x1, x2 = self.index[ind[i]-1], self.index[ind[i]]
            y1, y2 = senv.iloc[ind[i]-1], senv.iloc[ind[i]]
            tout.append((threshold-y1)*(x2-x1)/(y2-y1) + x1)
        return tout[0], tout[1]

    # _____________________________________________________________ #
    def tof(self, method='corr', *args):
        """
        Computes the time of flight relative to another signal. Three different methods for
        computing the time of flight are currently supported.

        Parameters
        ----------
        method: string, optional
            The method to be used for computing the signal time of flight. The following methods
            are currently supported:

                * *corr* : Use a correlation peak to compute the time of flight relative to another
                  signal. Another signal should be provided as input for performing the correlation.

                * *max* : The maximum value of the signal is used to compute the time of flight.
                  This time of flight is relative to the signal's time 0.

                * *thresh* : Compute the time the signal first crosses a given threshold value. The
                  threshold should  be given as argument in dB units. Note that the signal is
                  normalized by it's maximum for the purpose of finding the threshold crossing,
                  thus the maximum dB value is 0. If no threshold is given, the default is -12 dB.

        Returns
        -------
            : float
                The computed time of flight, with the same units as the Signal index.
        """
        if method.lower() == 'corr':
            try:
                other = args[0]
            except IndexError:
                raise ValueError('Another signal should be specified to compute the tof using the '
                                 'correlation method.')
            c = fftconvolve(self('n'), other('n')[::-1], mode='full')
            ind = self.size - np.argmax(c)
            return self.ts * ind
        elif method.lower() == 'max':
            return self.normalize('max').idxmax()
        elif method.lower() == 'thresh':
            try:
                thresh = args[0]
            except IndexError:
                thresh = -12
            return self.limits(thresh)[0]
        else:
            raise ValueError('method not supported. See documentation for supported methods.')

    # _____________________________________________________________ #
    def spectrogram(self, width, overlap=0, **kwargs):
        """
        Computes the spectrogram (short-time Fourier transform) of the signal. This method uses
        the function :func:`scipy.signal.spectrogram`.

        Parameters
        ----------
        width : float
            Substitute for the argument *nperseg* in the function :func:`scipy.signal.spectrogram`.
            Here, *width* has same units as *index*.

        overlap : float, optional
            Substitute for the argument *noverlap* in the function :func:`scipy.signal.spectrogram`.
            Here, *overlap was same units as *index*.

        Returns
        -------
        : Signal2D
            A Signal2D class representing the

        Note
        ----
        For other supported keyword arguments, see the documentation of
        :func:`scipy.signal.spectrogram`. However, the two arguments *nperseg* and *noverlap*
        should not be used.
        """
        nperseg = int(width * self.fs)
        nol = int(overlap * self.fs)
        f, t, S = spectrogram(self.values, fs=self.fs, nperseg=nperseg, noverlap=nol,
                              mode='magnitude', **kwargs)
        return Signal2D(S, index=f, columns=t)

    # _____________________________________________________________ #
    def psd(self, width, overlap=0, **kwargs):
        """
        Computes the periodogram of the signal. This method uses the function
        :func:`scipy.signal.welch`.

        Parameters
        ----------
        width : float
            Substitute for the argument *nperseg* in the function :func:`scipy.signal.spectrogram`.
            Here, *width* has same units as *index*.

        overlap : float, optional
            Substitute for the argument *noverlap* in the function :func:`scipy.signal.spectrogram`.
            Here, *overlap was same units as *index*.

        Returns
        -------
        : Signal2D
            A Signal2D class representing the

        Note
        ----
        For other supported keyword arguments, see the documentation of
        :func:`scipy.signal.welch`. However, the two arguments *nperseg* and *noverlap*
        should not be used.
        """
        nperseg = int(width * self.fs)
        nol = int(overlap * self.fs)
        f, pxx = welch(self.values, fs=self.fs, nperseg=nperseg, noverlap=nol, **kwargs)
        return Signal(pxx, index=f)

    # _____________________________________________________________ #
    def coherence(self, other, width, overlap=0, **kwargs):
        """
        Compute the short-time correlation coefficient of the signal. Uses the function
        :func:`scipy.signal.coherence` for this computation.

        Parameters
        ----------
        other : Signal
            The other Signal that will be used to perform the short time cross correlation.

        width : float
            Window size (in Signal index units) that will be used in computing the short-time
            correlation coefficient.

        overlap : float, optional
            Units (index units) of overlap between consecutive computations.

        Returns
        -------
        : array_like
            The computed short tiem cross-correlation function.
        """
        other = other(self.index)
        nperseg = int(width * self.fs)
        nol = int(overlap * self.fs)
        f, cxy = coherence(self.values, other.values, self.fs, nperseg=nperseg, noverlap=nol,
                           **kwargs)
        return Signal(cxy, index=f)

    # _____________________________________________________________ #
    def filter_freq(self, cutoff, option='lp', win_fcn='boxcar'):
        """
        Applies a frequency domain filter to the signal.

        Parameters
        ----------
        cutoff : float or (2,) tuple
            The cutoff frequency (Hz) of the filter. This is a scalar value if type
            is ``'lp'`` or ``'hp'``. When type is ``'bp'``, cutoff  should be a 2 element list,
            where the first element specifies the lower cutoff frequency, and the second element
            specifies the upper cutoff frequency.

        option : string, optional
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

        win_fcn : string, optional
            Apply a specific window in the frequency domain. See the function
            :func:`scipy.signal.get_window` for a complete list of available windows, and how to
            pass extra parameters for a specific window function.

        Returns
        -------
        : Signal
            The filtered Signal.
        """
        fdomain = self.fft()
        index1 = 0
        index2 = self.fs / 2.0
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
    def center_frequency(self, threshold=-6, nfft=None):
        """
        Computes the center frequency of the signal, which is the mean of the bandwidth limits.

        Parameters
        ----------
        threshold : float, optional
            Threshold value in dB, indicating the noise floor level. This value should be
            negative, as the specturm is normalized by its maximum values, and thus the maximum
            amplitude is 0 dB. Default is -6 dB.

        nfft : bool, optional
            Since this computation is based on the Fourier transform, indicate the number of
            points to be used on computing the FFT.

        Returns
        -------
        : float
            The value of the center frequency in Hz.
        """
        if threshold > 0:
            raise ValueError("threshold should be dB value <= 0.")
        fdomain = self.fft(ssb=True, nfft=nfft)
        yn = fdomain.operate('nd')
        frequencies = yn[yn >= threshold].index
        return (frequencies.max() + frequencies.min())/2

    # _____________________________________________________________ #
    def peak_frequency(self, threshold=-6, nfft=None):
        """
        Computes the peak frequency of the signal.

        Parameters
        ----------
        threshold : float, optional
            Threshold value in dB, indicating the noise floor level. This value should be
            negative, as the spectrum is normalized by its maximum values, and thus the maximum
            amplitude is 0 dB. Default is -6 dB.

        nfft : bool, optional
            Since this computation is based on the Fourier transform, indicate the number of
            points to be used on computing the FFT.

        Returns
        -------
        : float
            The value of the center frequency in Hz.
        """
        if threshold > 0:
            raise ValueError("threshold should be dB value <= 0.")
        return self.fft(ssb=True, nfft=nfft).idxmax()

    # _____________________________________________________________ #
    def bandwidth(self, threshold=-6, nfft=None):
        """
        Computes the bandwidth of the signal by finding the range of frequencies
        where the signal is above a given threshold.

        Parameters
        ----------
        threshold : float, optional
            Units is decibel (dB) <= 0. The spectrum is normalized by its maximum value,
            and thus the maximum amplitude is 0 dB. Default is -6 dB.

        nfft : bool, optional
            Since this computation is based on the Fourier transform, indicate the number of
            points to be used on computing the FFT.

        Returns
        -------
        : float
            The total signal bandwidth.
        """
        if threshold > 0:
            raise ValueError("threshold should be dB value <= 0.")

        fdomain = self.fft(ssb=True, nfft=nfft)
        lims = fdomain.abs().limits(threshold)
        print(lims)
        return lims[1] - lims[0]

    # _____________________________________________________________ #
    def maxof(self, option='peak'):
        """
        Computes the maximum of the signal according to a given method.

        Parameters
        ----------
        option : str, optional
            The method to be used to compute the maximum. Supported options are:

            ==================    ======================================
            *option*               Meaning
            ==================    ======================================
            abs                   Max of the signal absolute value
            env                   Max of the signal envelop
            peak *(Default)*      Max of the raw signal
            fft                   Max of the signal FFT magnitude
            ==================    ======================================

        Returns
        -------
        : float
            The maximum value of the specified signal form.
        """
        if option == 'peak':
            y = self
        elif option == 'abs':
            y = self.abs()
        elif option == 'env':
            y = self('e')
        elif option == 'fft':
            yf = self.fft(ssb=True)
            y = yf.abs()/yf.size
        else:
            raise ValueError("The value for option is unknown. Should be: 'abs',"
                             "'env', 'peak', or 'fft'.")
        return np.max(y.values)

    # _____________________________________________________________ #
    def energy(self, option='abs'):
        """
        Computes the energy of the given waveform in the specified domain. Simpson's integration
        is used for computing the total energy.

        Parameters
        ----------
        option : str, optional
            The method to be used to compute the energy. Supported options are:

            =================    ======================================
            *option*               Meaning
            =================    ======================================
            abs                  Use the absolute value of the signal
            env                  Use the envelop of the signal
            =================    ======================================

        Returns
        -------
        : float
            The computes energy in the signal.
        """
        if option == 'abs':
            return simps(self.values**2, x=self.index)
        elif option == 'env':
            return simps(self('e'), x=self.index)
        else:
            raise ValueError("The value for option is unknown. Should be either 'abs' or 'env'.")

    # _____________________________________________________________ #
    def remove_mean(self):
        """
        Subtracts the mean of the signal.

        Returns
        -------
        : Signal
            The :class:`Signal` with its mean subtracted.
        """
        return self - self.mean()

    # _____________________________________________________________ #
    @property
    def ts(self):
        """ Get the signal sampling period. """
        return np.mean(np.diff(self.index[self.index >= 0]))

    # _____________________________________________________________ #
    @property
    def extent(self):
        """ Get the signal index extent. """
        return self.index.max() - self.index.min()

    # _____________________________________________________________ #
    @property
    def fs(self):
        """ Get the signal sampling frequency. """
        return 1.0/self.ts

