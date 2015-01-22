#! /usr/bin/env python
#-*- encoding: utf-8 -*-

import numpy as np
from numpy.random import randn
"""

"""


def slide3():
    print "##############"
    ints = np.ones(10, dtype=np.uint16)
    floats = np.ones(10, dtype=np.float32)
    print np.issubdtype(ints.dtype, np.integer)
    print np.issubdtype(floats.dtype, np.floating)

    print "######Super Class########"
    print np.float64.mro()


def slide4():
    arr = np.arange(8)
    print arr
    print "#####reshape1#########"
    print arr.reshape((4, 2))
    print "#####reshape2#########"
    print arr.reshape((4, 2)).reshape((2, 4))

    print "##############"
    arr = np.arange(15)
    other_arr = np.ones((3, 5))
    print other_arr.shape
    print arr.reshape(other_arr.shape)

    print "######revel, flatten########"
    arr = np.arange(15).reshape((5, 3))
    print arr
    print arr.ravel()


def slide5():
    arr = np.arange(12).reshape((3, 4))
    print arr

    print "#####C LANG#########"
    print arr.ravel()
    print "#####FORTRAN#########"
    print arr.ravel('F')


def slide6():
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[7, 8, 9], [10, 11, 12]])
    print "#####axis=0#########"
    print np.concatenate([arr1, arr2], axis=0)
    print "#####axis=1#########"
    print np.concatenate([arr1, arr2], axis=1)

    print "#####vstack#########"
    print np.vstack((arr1, arr2))
    print "#####hstack#########"
    print np.hstack((arr1, arr2))

    print "####split [0:1][1:3][3:]##########"
    arr = randn(5, 2)
    print arr
    first, second, third = np.split(arr, [1, 3])
    print "#####first#########"
    print first
    print "#####second#########"
    print second
    print "#####third#########"
    print third

    arr = np.arange(6)
    arr1 = arr.reshape((3, 2))
    arr2 = randn(3, 2)
    print "######r_########"
    print np.r_[arr1, arr2]
    print "######c_########"
    print np.c_[np.r_[arr1, arr2], arr]


def slide7():
    arr = np.arange(3)
    print arr.repeat(3)
    print arr.repeat([2, 3, 4])

    print "#####matrix#############"
    arr = randn(2, 2)
    print arr.repeat(2, axis=0)
    print arr.repeat([2, 3], axis=0)
    print arr.repeat([2, 3], axis=1)

    print "#####tile#############"
    print np.tile(arr, 2)
    print np.tile(arr, (2, 1))
    print np.tile(arr, (3, 2))


def slide8():
    arr = np.arange(10) * 100
    inds = [7, 1, 2, 6]
    print arr[inds]
    print "######take##############"
    print arr.take(inds)
    print "######put###############"
    arr.put(inds, 42)
    print arr
    arr.put(inds, [40, 41, 42, 43])
    print arr

    print "######take along other axis####"
    inds = [2, 0, 2, 1]
    arr = randn(2, 4)
    print arr.take(inds, axis=1)


def slide9():
    arr = np.arange(5)
    print arr
    print arr * 4

    print "####axis=0###########"
    arr = randn(4, 3)
    print arr.mean(0)
    demeaned = arr - arr.mean(0)
    print demeaned
    print demeaned.mean(0)

    print "####axis=1###########"
    row_means = arr.mean(1)
    print row_means.reshape((4, 1))
    demeaned = arr - row_means.reshape((4, 1))
    print demeaned.mean(1)


def slide10():
    print "######3D##########"
    arr = np.zeros((4, 4))
    arr_3d = arr[:, np.newaxis, :]
    print arr_3d
    print arr_3d.shape

    print "######1D##########"
    arr_1d = np.random.normal(size=3)
    print arr_1d
    print arr_1d[:, np.newaxis]
    print arr_1d[np.newaxis, :]

    print "######tensor##########"
    arr = randn(3, 4, 5)
    print arr
    depth_means = arr.mean(2)
    print depth_means
    demeaned = arr - depth_means[:, :, np.newaxis]
    print demeaned

    def demean_axis(arr, axis=0):
        means = arr.mean(axis)
        indexer = [slice(None)] * arr.ndim
        indexer[axis] = np.newaxis
        return arr - means[indexer]

    print "##############"
    arr = np.zeros((4, 3))
    print arr
    arr[:] = 5
    print arr
    col = np.array([1.28, -0.42, 0.44, 1.6])
    arr[:] = col[:, np.newaxis]
    print arr
    arr[:2] = [[-1.37], [0.509]]
    print arr


def slide11():
    print "#####reduce1D####"
    arr = np.arange(10)
    print np.add.reduce(arr)
    print arr.sum()

    print "#####reduce2D####"
    arr = randn(5, 5)
    arr[::2].sort(1)
    print arr[:, :-1] < arr[:, 1:]

    print np.logical_and.reduce(arr[:, :-1] < arr[:, 1:], axis=1)

    print "#####accumulate####"
    arr = np.arange(15).reshape((3, 5))
    print np.add.accumulate(arr, axis=1)

    print "#####multiply outer#########"
    arr = np.arange(3).repeat([1, 2, 2])
    print arr
    print np.multiply.outer(arr, np.arange(5))

    print "#####subtract outer#########"
    result = np.subtract.outer(randn(3, 4), randn(5))
    print result.shape

    print "#####reduceat 1D###########"
    arr = np.arange(10)
    print np.add.reduceat(arr, [0, 5, 8])

    print "#####reduceat 2D###########"
    arr = np.multiply.outer(np.arange(4), np.arange(5))
    print arr
    print np.add.reduceat(arr, [0, 2, 4], axis=1)


def slide12():
    def add_elements(x, y):
        return x + y

    add_them = np.frompyfunc(add_elements, 2, 1)
    print add_them(np.arange(8), np.arange(8))

    add_them = np.vectorize(add_elements, otypes=[np.float64])
    print add_them(np.arange(8), np.arange(8))


def slide13():
    dtype = [('x', np.float64), ('y', np.int32)]
    sarr = np.array([(1.5, 6), (np.pi, -2)], dtype=dtype)
    print sarr
    print sarr[0]
    print sarr[0]['y']
    print sarr['x']

    print "#####nested########"
    dtype = [('x', np.int64, 3), ('y', np.int32)]
    arr = np.zeros(4, dtype=dtype)
    print arr
    print arr[0]['x']
    print arr['x']

    print "##################"
    dtype = [('x', [('a', 'f8'), ('b', 'f4')]), ('y', np.int32)]
    data = np.array([((1, 2), 5), ((3, 4), 6)], dtype=dtype)
    print data
    print data['x']
    print data['y']
    print data['x']['a']


def slide14():
    arr = randn(6)
    arr.sort()
    print arr

    print "###numpy.ndarray.sort#####"
    arr = randn(3, 5)
    print arr

    arr[:, 0].sort()
    print arr

    print "###numpy.sort#####"
    arr = randn(5)
    print arr
    print np.sort(arr)
    print arr
    print "####matrix sort######"
    arr = randn(3, 5)
    print arr
    arr.sort(axis=1)
    print arr
    print "####descending########"
    print arr[:, ::-1]


def slide15():
    values = np.array([5, 0, 1, 3, 2])
    indexer = values.argsort()
    print indexer
    print values[indexer]

    print "####2D#####"
    arr = randn(3, 5)
    arr[0] = values
    arr[:, arr[0].argsort()]

    first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
    last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])
    sorter = np.lexsort((first_name, last_name))
    print zip(last_name[sorter], first_name[sorter])


def slide16():
    values = np.array(['2:first',
                       '2:second',
                       '1:first',
                       '1:second',
                       '1:third'])

    key = np.array([2, 2, 1, 1, 1])
    indexer = key.argsort(kind='mergesort')
    print indexer
    print values.take(indexer)


def slide17():
    arr = np.array([0, 1, 7, 12, 15])
    print arr.searchsorted(9)
    print arr.searchsorted([0, 8, 11, 16])
    arr = np.array([0, 0, 0, 1, 1, 1, 1])
    print arr.searchsorted([0, 1])
    print arr.searchsorted([0, 1], side='right')

    print "######bucket edges###############"
    data = np.floor(np.random.uniform(0, 10000, size=50))
    bins = np.array([0, 100, 1000, 5000, 10000])
    print data

    labels = bins.searchsorted(data)
    print labels
    from pandas import Series
    print Series(data).groupby(labels).mean()

    print np.digitize(data, bins)


def slide18():
    X = np.array([[8.82768214, 3.82222409, -1.14276475, 2.04411587],
                 [3.82222409, 6.75272284, 0.83909108, 2.08293758],
                 [-1.14276475, 0.83909108, 5.01690521, 0.79573241],
                 [ 2.04411587, 2.08293758, 0.79573241, 6.24095859]])

    print X[:, 0]
    y = X[:, :1]
    print np.dot(y.T, np.dot(X, y))

    Xm = np.matrix(X)
    ym = Xm[:, 0]

    print Xm
    print Xm.I * X


def slide19():
    mmap = np.memmap('mymmap', dtype='float64', mode='w+', shape=(10000, 10000))
    print mmap
    section = mmap[:5]
    section[:] = randn(5, 10000)
    mmap.flush()
    print mmap
    del mmap


if __name__ == '__main__':
    slide3()
    # slide4()
    # slide5()
    # slide6()
    # slide7()
    # slide8()
    # slide9()
    # slide10()
    # slide11()
    # slide12()
    # slide13()
    # slide14()
    # slide15()
    # slide16()
    # slide17()
    # slide18()
    # slide19()

# End of Line.
