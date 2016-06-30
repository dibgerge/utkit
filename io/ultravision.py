"""

"""
import pandas as pd
import numpy as np
import utkit


def read_ultravision(fname):
    """
    Reads ultrasound scans saved in UltraVision (ZETEC, Inc. software) text file format.

    Parameters
    ----------
    fname : string
        The full path or relative path to the file.

    Returns
    -------
    : utkit.Signal3D
        :class:`utkit.Signal3D` object representing a scan.
    """
    header = pd.read_table(fname, sep=' =\t', skipinitialspace=True, header=None, nrows=19,
                           engine='python', index_col=0, squeeze=True)

    t = np.arange(int(header['USoundQty (sample)']))*float(header['USoundResol [True Depth] (mm)'])*1e-3

    # convert x/y indices to meters (from millimeters) in the uv file
    Xv = np.arange(int(header['ScanQty (sample)'])) * float(header['ScanResol (mm)'])*1e-3
    Yv = np.arange(int(header['IndexQty (sample)'])) * float(header['IndexResol (mm)'])*1e-3
    X, Y = np.meshgrid(Xv, Yv)
    u = pd.read_table(fname,
                      skiprows=19,
                      sep='\t',
                      skipinitialspace=True,
                      header=None,
                      engine='c',
                      index_col=False)
    # the last column is giving NaN values. Just remove it.
    u = u.iloc[:, :-1]
    u.index = [X.ravel(), Y.ravel()]
    u.columns = t
    p = u.to_panel()
    return utkit.Signal3D(p.transpose(2, 0, 1).values,
                          items=Yv,
                          major_axis=t,
                          minor_axis=Xv)
