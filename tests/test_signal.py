import unittest
from utkit import Signal
import numpy as np
import pandas.util.testing as pdt
import numpy.testing as npt

class TestSignal(unittest.TestCase):
    fc = 1e6
    fs = 100e6
    n = 1000

    t = np.arange(n)/fs
    y = np.sin(2*np.pi*fc*t)

    s = Signal(y, index=t)

    def test_constructor_samplingintervalindex(self):
        """
        Test index input using sampling interval results in the same signal as inputing the
        whole time vector.
        """
        s1 = Signal(self.y, ts=1/self.fs)
        s2 = Signal(self.y, index=self.t)
        pdt.assert_series_equal(s1, s2)

    def test_contrusctor_monotonicindex(self):
        """
        Test that error is raised if index is not monotonic.
        """
        self.assertRaises(Exception, Signal, [0, 1, 2], index=[-1, 2, 0])

    def test_call_ok(self):
        """

        """
        s = Signal([0, 1, 2, 3], index=[1, 2, 3, 4])
        npt.assert_allclose(s(2), 1)

if __name__ == '__main__':
    unittest.main()
