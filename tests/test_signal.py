import unittest
from utkit import Signal
import numpy as np
import pandas as pd


class TestSignal(unittest.TestCase):

    def test_indexing(self):
        fc = 1e6
        Fs = 100e6
        t  = np.arange(1000)/Fs
        s = Signal(np.sin(2*np.pi*fc*t), index=t)
        self.assertIsInstance(s, pd.Series)

if __name__ == '__main__':
    unittest.main()
