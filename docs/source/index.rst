UTkit Documentation
===================

:mod:`utkit` extends the data structures in the popular library :mod:`pandas` in order to
support short time signals generally encountered in wave propagation measurmenets. The main
purpose of this library is to implement signal processing procedures for *signal shaping*,
*transforms*, and *feature extraction* in a simple manner. All the while, keeping track and managing
signal values and their corresponding indices (time/space values).

At the core of the library, there are three data structures:

.. automodule:: utkit

IO Methods
----------

The library also provides methods to read datafiles generated from common equipment used in
ultrasound inspections (such as data acquisition hardware and oscilloscopes). In addition,
methods that read data from modeling software outputs are available.

.. automodule:: utkit.io
