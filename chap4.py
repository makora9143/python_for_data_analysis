#! /usr/bin/env python
#-*- encoding: utf-8 -*-

"""

"""

import numpy as np


# The NumPy ndarray: A Multidimensional Array Object
# Slide No.5
# data = np.array([[0.9526, -0.246, -0.8856],
#                  [0.5639, 0.2379, 0.9104]])

# print data
# print data * 10
# print data + data

# print data.shape
# print data.dtype

# # Creating ndarrays
# # Slide No.6

# data1 = [6, 7.5, 8, 0, 1]
# arr1 = np.array(data1)
# print arr1

# data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
# arr2 = np.array(data2)
# print arr2
# print arr2.ndim
# print arr2.shape

# print arr1.dtype
# print arr2.dtype

# print np.zeros(10)
# print np.zeros((3, 6))
# print np.empty((2, 3, 2))

# # Data Types
# # Slide No.8

# arr1 = np.array([1, 2, 3], dtype=np.float64)
# arr2 = np.array([1, 2, 3], dtype=np.int32)

# print arr1.dtype
# print arr2.dtype

# arr = np.array([1, 2, 3, 4, 5])
# print arr.dtype

# # cast
# float_arr = arr.astype(np.float64)
# print float_arr.dtype

# arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
# print arr
# print arr.astype(np.int32)


# # Slide No.9

# arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# print arr
# print arr * arr
# print arr - arr
# print 1 / arr
# print arr ** 0.5

# # Slide No.10

# arr = np.arange(10)

# print arr
# print arr[5]
# print arr[5:8]

# arr[5:8] = 12
# print arr

# arr_slice = arr[5:8]
# arr_slice[1] = 12345
# print arr
# arr_slice[:] = 64
# print arr

# arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print arr2d[2]
# print arr2d[0][2]
# print arr2d[0, 2]

# arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print arr3d
# print arr3d[0]

# old_values = arr3d[0].copy()
# arr3d[0] = 42
# print arr3d

# arr3d[0] = old_values
# print arr3d

# print arr2d
# print arr2d[:2]
# print arr2d[:2, 1:]

# # Slide No.11

# names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# print names
# data = np.random.randn(7, 4)
# print data

# print names == 'Bob'
# print data[names == 'Bob']

# print names != 'Bob'
# print data[-(names == 'Bob')]

# mask = (names == 'Bob') | (names == 'Will')
# print mask
# print data[mask]

# data[data < 0] = 0
# print data

# data[names != 'Joe'] = 7
# print data

# # Slide No.12

# arr = np.empty((8, 4))

# for i in range(8):
#     arr[i] = i
# print arr
# print arr[[4, 3, 0, 6]]

# arr = np.arange(32).reshape((8, 4))
# print arr

# print arr[[1, 5, 7, 2], [0, 3, 1, 2]]
# print arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]

# # Slide No.13

# arr = np.arange(15).reshape((3, 5))
# print arr
# print arr.T

# arr = np.random.randn(6, 3)

# print np.dot(arr.T, arr)

# arr = np.arange(16).reshape((2, 2, 4))

# print arr
# print "(0, 1, 2)"
# print arr.transpose((0, 1, 2))
# print "(0, 2, 1)"
# print arr.transpose((0, 2, 1))
# print "(1, 0, 2)"
# print arr.transpose((1, 0, 2))
# print "(1, 2, 0)"
# print arr.transpose((1, 2, 0))
# print "(2, 0, 1)"
# print arr.transpose((2, 0, 1))
# print "(2, 1, 0)"
# print arr.transpose((2, 1, 0))

# # Slide No.14
# arr = np.arange(10)

# print np.sqrt(arr)
# print np.exp(arr)

# x = np.random.randn(8)
# y = np.random.randn(8)
# print x
# print y

# print np.maximum(x, y)

# arr = np.random.randn(7)
# print arr
# print arr * 5
# print np.modf(arr)

# # Slide No.17
# points = np.arange(-5, 5, 0.01)
# print points
# xs, ys = np.meshgrid(points, points)
# print ys
# print xs
# import matplotlib.pyplot as plt
# z = np.sqrt(xs ** 2 + ys ** 2)
# print z

# plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
# plt.show()

# # Slide No.18
# xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
# yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
# cond = np.array([True, False, True, True, False])

# result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
# print result

# result = np.where(cond, xarr, yarr)
# print result

# arr = np.random.randn(4, 4)
# print arr
# print np.where(arr > 0, 2, -2)
# print np.where(arr > 0, 2, arr)

# # Slide No.19
# arr = np.random.randn(5, 4)

# print arr.mean()
# print np.mean(arr)
# print arr.sum()

# print arr.mean(axis=1)
# print arr.sum(0)

# # Slide No.20
# arr = np.randn(100)
# print (arr > 0).sum()

# # Slide No.21
# arr = np.random.randn(8)
# print arr
# arr.sort()
# print arr

# arr = np.random.randn(5, 3)
# print arr
# print np.sort(arr)
# print arr

# # Slide No.22
# names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# print np.unique(names)

# ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
# print np.unique(ints)

# # Slide No.24

# x = np.array([[1., 2., 3.], [4., 5., 6.]])
# y = np.array([[6., 23.], [-1, 7], [8, 9]])

# print x.dot(y)
# print y.dot(x)

# from np.linalg import inv, qr
# X = np.random.randn(5, 5)
# mat = X.T.dot(X)
# print inv(mat)
# print mat.dot(inv(mat))

# q, r = qr(mat)
# print r

# Slide No.28
import random
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

import matplotlib.pyplot as plt
plt.plot(walk)
plt.show()

nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
plt.plot(walk)
plt.show()

nwalks = 5000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
plt.plot(walks)
plt.show()



# End of Line.
