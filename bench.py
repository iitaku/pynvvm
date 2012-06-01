import timeit

s = '''
import test
import numpy as np
n = 2048
x = np.random.randn(n*n).astype(np.float32)
y = np.random.randn(n*n).astype(np.float32)
a = np.float32(2.71828183)
'''

c = timeit.Timer('test.cpu(x, y, a, n)', s)
g = timeit.Timer('test.gpu(x, y, a, n)', s)

print(c.timeit(number=100))
print(g.timeit(number=100))
