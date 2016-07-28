import numpy as np
import pandas as pd
import utkit
import re


def read_civa_cscan(fname):
    """
    Reads a C-scan file saved from a CIVA simulation. The X-Y axis coordinates are returned in
    units of meters.

    Parameters
    ----------
    fname : string
        Full path name of the CIVA C-scan file.

    Returns
    -------
    cscan : utkit.Signal2D
        The simulation C-scan.
    """
    scan = pd.read_table(fname, sep=';', usecols=[0, 1, 4], encoding='iso8859_15',
                         index_col=[0, 1], squeeze=True).unstack()
    # convert axis from millimeter to meters
    scan.index *= 1e-3
    scan.columns *= 1e-3
    return utkit.Signal2D(scan)


def read_civa_tcscan(fname):
    """
    Reads a True C-scan file saved from a CIVA simulation.

    Parameters
    ----------
    fname : string
        Full path name of the CIVA C-scan file

    Returns
    -------
    cscan : utkit.Signal2D
        The simulation True C-scan.
    """
    with open(fname) as fid:
        for i, line in enumerate(fid):
            parts = line.split(';')
            if i == 0:
                xstart = float(parts[1])
                ystart = float(parts[3])
            if i == 2:
                xstep = float(parts[1])
            if i == 3:
                ystep = float(parts[1])
                break
    d = np.genfromtxt(fname,  delimiter=';', skip_header=5, usecols=(0, 1, 5))
    # convert steps from millimeter to meters by multiplying a factor of 1e-3
    X = (np.arange(min(d[:, 0]), 1+max(d[:, 0]))*xstep + xstart)*1e-3
    Y = (np.arange(min(d[:, 1]), max(d[:, 1]))*ystep + ystart)*1e-3

    ind = np.nonzero(np.diff(d[:, 1]))[0]
    C = np.zeros((len(ind), len(X)))
    ind = np.append(0, ind)

    for i in range(len(ind)-1):
        leftover = len(X) - ind[i+1] + ind[i]
        C[i, :] = np.concatenate((d[ind[i]:ind[i+1], 2], np.zeros(leftover)))
    return utkit.Signal2D(C, index=Y, columns=X)


def read_civa_bscan(fname):
    """
    Reads a B-scan txt file saved in CIVA-UT modeling software.

    Parameters
    ----------
    fname : str
        Name of the file, including the full path if not in the current directory.

    Returns
    -------
    bscan : utkit.Signal2D
        A Scan2D object containing the B-scan.
    """
    # this is the default start of the header in a civa b-scan txt file
    skip_lines = 12
    # read the header
    with open(fname) as fid:
        for i, line in enumerate(fid):
            if i == skip_lines-1:
                coords = re.findall(r'\d*\.?\d+', line)
                coords = np.array([float(val) for val in coords])
                cols = line.split(';')
                ind = np.array([j for j, c in enumerate(cols) if 'val' in c])
                break

    d = np.genfromtxt(fname,  delimiter=';', skip_header=skip_lines)
    # convert from microseconds in CIVA b-scan file to seconds
    Y = d[:, 0]*1e-6
    # convert from millimeters to meters
    X = coords[ind-1]*1e-3
    b = d[:, ind]
    return utkit.Signal2D(b, index=Y, columns=X)
