import pandas as pd
import numpy as np
from . import Scan2D
from pandas.tools.util import cartesian_product

# _______________________________________________________________________#
class Scan3D(pd.Panel):
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
        return Scan3D

    _constructor_sliced = Scan2D

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
            Scan2D: A copy of Scan2D with shiftes axes.
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
            Scan2D :
                A copy of Scan2D with scaled axes.
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
            Scan2D :
                A Scan2D object representing the B-scan.
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
            Scan2D :
                A Scan2D object representing the D-scan.
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
            Scan2D :
                A Scan2D object representing the C-scan.
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
                A 4-element tuple where each element is a flattened array of the Scan3D,
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

            obj = Signal(values[tuple(indexer)], index=slice_axis, name=pts)
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
        if isinstance(results[0], Signal):
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
