#! /usr/bin/env python
#-*- encoding: utf-8 -*-

"""

"""

import numpy as np


# The NumPy ndarray: A Multidimensional Array Object
# Slide No.5
data = np.array([[0.9526, -0.246, -0.8856],
                      [0.5639, 0.2379, 0.9104]])

print data
print data * 10
print data + data

print data.shape
print data.dtype

# Creating ndarrays
# Slide No.6

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print arr1

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
print arr2
print arr2.ndim
print arr2.shape

print arr1.dtype
print arr2.dtype

print np.zeros(10)
print np.zeros((3, 6))
print np.empty((2, 3, 2))

# Data Types
# Slide No.8

arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)

print arr1.dtype
print arr2.dtype

arr = np.array([1, 2, 3, 4, 5])
print arr.dtype

# cast
float_arr = arr.astype(np.float64)
print float_arr.dtype

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
print arr
print arr.astype(np.int32)
# End of Line.
