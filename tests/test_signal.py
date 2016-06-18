import unittest
from utkit import Signal
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt


class TestSignal(unittest.TestCase):
    fc = 1e6
    fs = 100e6
    n = 1000
    t = np.arange(n)/fs
    y = np.sin(2*np.pi*fc*t)
    f = fftfreq(len(y), 1/fs)
    Y = fft(y)

    def test_indexing(self):
        """
        Test if inputing only the sampling interval results in the same signal as inputing the
        whole time vector.
        """
        s1 = Signal(self.y, index=1/self.fs)
        s2 = Signal(self.y, index=self.t)
        pdt.assert_series_equal(s1, s2)

    def test_index_type(self):
        """
        Confirm that Signal will only accept monotonically increasing indices. Thus, FFT signals
        are only allowed if shifted.
        """
        self.assertRaises(ValueError, Signal, self.Y, index=self.f)

    def test_window(self):
        """

        """
        u = Signal(self.y, index=self.t)
        #uw = u.window(index1=5e-6, index2=6e-6, win_fcn='hann')
        #Y = fft(uw)
        #f = fftfreq(uw.size, 1/uw.fs)
        #uf = Signal(fftshift(Y), index=fftshift(f))
        #uf.window(index1=1e6, index2=2e6, fftbins=True).abs().plot()
        #plt.show()

    def test_call(self):
        """

        """
        u = Signal(self.y, index=self.t).fft(nfft=2**17, ssb=True)
        a = u(np.arange(1, 5e6, 10))
        plt.plot(a.abs())
        plt.plot(u.abs())
        plt.xlim([0, 5e6])
        plt.show()

if __name__ == '__main__':
    unittest.main()
