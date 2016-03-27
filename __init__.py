"""
UTkit is used for performing signal processing on time series signals generally,
which for example are encountered in Ultrasonics.
The library extends the popular library :mod:`pandas` to support functions commonly
used in UT analysis. The module :mod:`utkit` implements three classes:

* :class:`utkit.uSeries`: representation of a time series.
* :class:`utkit.uFrame`: representation of a 1-D collection of time series (single line scan).
* :class:`utkit.RasterScan`: representation of a 2-D collection of time series (raster scan).

.. autoclass:: utkit.uSeries
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __call__
    :member-order: groupwise

.. autoclass:: utkit.uFrame
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __call__
    :member-order: groupwise

.. autoclass:: utkit.RasterScan
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __call__
    :member-order: groupwise
"""
import utkit.utkit
import utkit.io
from .utkit import *
