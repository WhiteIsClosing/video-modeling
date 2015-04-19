import sys
sys.path.insert(0, '/deep/u/kuanfang/optical-flow-pred/')
import numpy
from utils.load import shuffle

A = numpy.array([[1, 1, 1, 1], \
                [2, 1, 1, 1], \
                [3, 1, 1, 1], \
                [4, 1, 1, 1], \
                [5, 1, 1, 1], \
                [6, 1, 1, 1], \
                [7, 1, 1, 1], \
                [8, 1, 1, 1], \
                [9, 1, 1, 1]])

B = numpy.array([[10, 1, 1, 1], \
                [20, 1, 1, 1], \
                [30, 1, 1, 1], \
                [40, 1, 1, 1], \
                [50, 1, 1, 1], \
                [60, 1, 1, 1], \
                [70, 1, 1, 1], \
                [80, 1, 1, 1], \
                [90, 1, 1, 1]])

# numpy.random.permutation(A)
print A
print B

for i in range(5):
  A, B = shuffle(A, B, 3, 43)
  print A
  print B

# numpy.random.seed(1)
# for i in range(10):
#   arr = numpy.arange(9).reshape((3, 3))
#   numpy.random.permutation(arr)
#   print arr
