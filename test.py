import series
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_fft(u):
    fdomain = u.fft(NFFT=10000, option='single')
    plt.plot(abs(fdomain))
    plt.xlim([0, 10e6])
    plt.show()


def test_filter():
    Fs = 50e6
    t = np.arange(100)/Fs
    y = np.sin(2*np.pi*1e6*t) + np.sin(2*np.pi*4e6*t)
    u = series.Series(y, index=t)

    uf1 = u.filter(cutoff=2e6, option='lp', win_fcn='hann')
    uf2 = u.filter(cutoff=2e6, option='hp', win_fcn='hann')
    uf3 = u.filter(cutoff=[2e6, 3e6], option='bp', win_fcn='hann')

    _, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(uf1)
    ax[1].plot(uf2)
    ax[2].plot(uf3)
    plt.show()


def test_center_frequency():
    Fs = 50e6
    t = np.arange(2000)/Fs
    y = np.sin(2*np.pi*1e6*t)
    u = series.Series(y, index=t)
    fc = []
    for i in range(1, 10):
        uw = u.window(0, i*1e-6)
        fc.append(uw.center_frequency(option='max'))
    print(fc)


def test_limits():
    Fs = 50e6
    t = np.arange(2000)/Fs
    y = np.sin(2*np.pi*1e6*t)
    u = series.Series(y, index=t)
    u = u.window(0, 8e-6)
    print(u.limits())


def test_bandwidth():
    Fs = 50e6
    t = np.arange(2000)/Fs
    y = np.sin(2*np.pi*1e6*t)
    u = series.Series(y, index=t)
    for i in range(1, 10):
        u2 = u.window(0, i*1e-6)
        print("bandwidth = ", u2.bandwidth()*1e-6, " MHZ")


def test_addition():
    Fs1 = 50e6
    Fs2 = 49e6
    t1 = np.arange(2000)/Fs1
    y1 = np.sin(2*np.pi*1e6*t1)
    u1 = series.Series(y1, index=t1)
    u1 = u1.window(1e-6, 5e-6)
    t2 = np.arange(2000)/Fs2
    y2 = np.sin(2*np.pi*1e6*t2)
    u2 = series.Series(y2, index=t2)
    u2 = u2.window(3.5e-6, 7.5e-6)
    ut = u1+u2
    plt.plot(ut)
    plt.show()

if __name__ == '__main__':
    # test_filter()
    # test_center_frequency()
    # test_bandwidth()
    test_addition()
