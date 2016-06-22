import unittest
from utkit import Signal, Signal2D
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

    def test_slicing(self):
        s = Signal2D(np.arange(9).reshape(3, 3), index=np.arange(3.), columns=np.arange(3.)*0.1)
        self.assertTrue(type(s[0.1]) == Signal)

    # def test_slicing2(self):
    #     s = Signal2D(np.arange(9).reshape(3, 3), index=np.arange(3.), columns=np.arange(3.)*0.1)
    #     self.assertTrue(type(s.loc[:, 0.1]) == Signal)
    #     self.assertTrue(type(s.loc[1., :]) == Signal)
    #
    # def test_slicing3(self):
    #     s = Signal2D(np.arange(9).reshape(3, 3), index=np.arange(3.), columns=np.arange(3.)*0.1)
    #     print(type(s.iloc[1, :]))
    #     self.assertTrue(type(s.iloc[:, 1]) == Signal)
    #     self.assertTrue(type(s.iloc[1, :]) == Signal)

    def test_shift_axis_axisnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[-1, 0, 1, 2], columns=[-1, 0, 1])
        pdt.assert_frame_equal(s.shift_axis(1), s2)

    def test_shift_axis_axisnone2(self):
        s = Signal2D(np.ones(12).reshape(4, 3))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[-1, 0, 1, 2], columns=[-2, -1, 0])
        pdt.assert_frame_equal(s.shift_axis([1, 2]), s2)

    def test_shift_axis_axisindex(self):
        s = Signal2D(np.ones(12).reshape(4, 3))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[-1, 0, 1, 2])
        pdt.assert_frame_equal(s.shift_axis(1, axis='y'), s2)

    def test_shift_axis_axisindex2(self):
        s = Signal2D(np.ones(12).reshape(4, 3))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[-1, 0, 1, 2])
        pdt.assert_frame_equal(s.shift_axis(1, axis=0), s2)

    def test_shift_axis_columns(self):
        s = Signal2D(np.ones(12).reshape(4, 3))
        s2 = Signal2D(np.ones(12).reshape(4, 3), columns=[-1, 0, 1])
        pdt.assert_frame_equal(s.shift_axis(1, axis='x'), s2)

    def test_shift_axis_columns2(self):
        s = Signal2D(np.ones(12).reshape(4, 3))
        s2 = Signal2D(np.ones(12).reshape(4, 3), columns=[-1, 0, 1])
        pdt.assert_frame_equal(s.shift_axis(1, axis=1), s2)

    def test_scale_axis_axisnone_startstopnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 0.5, 1., 1.5], columns=[0., 0.5, 1.0])
        pdt.assert_frame_equal(s.scale_axis(0.5), s2)

    def test_scale_axis_axisnone_startstopnone2(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 0.5, 1., 1.5], columns=[0., 0.25, 0.5])
        pdt.assert_frame_equal(s.scale_axis([0.5, 0.25]), s2)

    def test_scale_axis_axisnone_startnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 0.5, 1., 3.], columns=[0., 0.25, 0.5])
        pdt.assert_frame_equal(s.scale_axis([0.5, 0.25], stop=2.), s2)

    def test_scale_axis_axisnone_startnone2(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 0.5, 1., 3.], columns=[0., 0.25, 2.])
        pdt.assert_frame_equal(s.scale_axis([0.5, 0.25], stop=[2., 1.]), s2)

    def test_scale_axis_axisnone_stopnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 0.5, 1., 1.5], columns=[0., 0.25, 0.5])
        pdt.assert_frame_equal(s.scale_axis([0.5, 0.25], start=1.), s2)

    def test_scale_axis_axisnone_stopnone2(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 1., 1., 1.5], columns=[0., 0.25, 0.5])
        pdt.assert_frame_equal(s.scale_axis([0.5, 0.25], start=[2., 1.]), s2)

    def test_scale_axis_axis0_startstopnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 0.5, 1., 1.5], columns=[0., 1., 2.0])
        pdt.assert_frame_equal(s.scale_axis(0.5, axis=0), s2)

    def test_scale_axis_axis0_startnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 0.5, 1., 3.], columns=[0., 1., 2.0])
        pdt.assert_frame_equal(s.scale_axis(0.5, stop=2, axis=0), s2)

    def test_scale_axis_axis0_stopnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 1., 4., 6.], columns=[0., 1., 2.0])
        pdt.assert_frame_equal(s.scale_axis(2., start=2, axis=0), s2)

    def test_scale_axis_axis1(self):
        s = Signal2D(np.ones(20).reshape(4, 5), index=np.arange(4.), columns=np.arange(5.))
        s2 = Signal2D(np.ones(20).reshape(4, 5), index=[0., 1., 2., 3.],
                      columns=[0., 1., 4., 6., 8.])
        pdt.assert_frame_equal(s.scale_axis(2., start=2., stop=4., axis=1), s2)

    def test_scale_axis_axis1_startstopnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 1., 2., 3.], columns=[0., 0.5, 1.])
        pdt.assert_frame_equal(s.scale_axis(0.5, axis=1), s2)

    def test_scale_axis_axis1_startnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 1., 2., 3.], columns=[0., 0.5, 2.0])
        pdt.assert_frame_equal(s.scale_axis(0.5, stop=1., axis=1), s2)

    def test_scale_axis_axis1_stopnone(self):
        s = Signal2D(np.ones(16).reshape(4, 4), index=np.arange(4.), columns=np.arange(4.))
        s2 = Signal2D(np.ones(16).reshape(4, 4), index=[0., 1., 2., 3.], columns=[0., 1., 4., 6.])
        pdt.assert_frame_equal(s.scale_axis(2., start=2, axis=1), s2)
