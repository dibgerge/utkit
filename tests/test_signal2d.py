import unittest
from utkit import Signal2D
import numpy as np
import pandas.util.testing as pdt
import numpy.testing as npt


class TestSignal2D(unittest.TestCase):

    def test_constructor_scalarindex(self):
        s = Signal2D(np.arange(9).reshape(3, 3), index=2e-6, columns=[0., 1., 2.])
        s2 = Signal2D(np.arange(9).reshape(3, 3), index=[0, 2e-6, 4e-6], columns=[0., 1., 2.])
        pdt.assert_frame_equal(s, s2)

    def test_constructor_scalarindexfromdict(self):
        s = Signal2D({1: [0, 1, 3]}, index=2e-6)
        s2 = Signal2D([[0], [1], [3]], index=[0, 2e-6, 4e-6], columns=[1])
        pdt.assert_frame_equal(s, s2)

    def test_constructor_scalarcolumns(self):
        s = Signal2D(np.arange(9).reshape(3, 3), index=[0, 2e-6, 4e-6], columns=1.0)
        s2 = Signal2D(np.arange(9).reshape(3, 3), index=[0, 2e-6, 4e-6], columns=[0., 1., 2.])
        pdt.assert_frame_equal(s, s2)

    def test_constructor_scalarbothaxes(self):
        s = Signal2D(np.arange(9).reshape(3, 3), index=2e-6, columns=1.0)
        s2 = Signal2D(np.arange(9).reshape(3, 3), index=[0, 2e-6, 4e-6], columns=[0., 1., 2.])
        pdt.assert_frame_equal(s, s2)

    def test_constructor_scalarbothaxesFromdict(self):
        s = Signal2D({1.0: [0, 1, 3], 2.0: [4, 5, 6]}, index=2e-6)
        s2 = Signal2D([[0, 4], [1, 5], [3, 6]], index=[0, 2e-6, 4e-6], columns=[1.0, 2.0])
        pdt.assert_frame_equal(s, s2)
