import numpy as np
import utkit
import re


def civa_bscan(fname):
    """
    Reads a B-scan txt file saved in CIVA-UT modeling software.

    Parameters:
        fname:
            Name of the file, including the fullpath if not in the current directory

    Returns:
        Bscan (:class:`utkit.Signal2D`):
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
