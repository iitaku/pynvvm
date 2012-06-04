import timeit

s = '''
import test
import numpy as np
n = 4096
x = np.random.randn(n*n).astype(np.float32)
y = np.random.randn(n*n).astype(np.float32)
a = np.float32(2.71828183)
'''

c = timeit.Timer('test.cpu(x, y, a, n)', s)
g = timeit.Timer('test.gpu(x, y, a, n)', s)
p = timeit.Timer('test.py(x, y, a, n)', s)

print(' numpy : ' + str(c.timeit(number=10)/10.0))
print('pynvvm : ' + str(g.timeit(number=10)/10.0))
print('python : ' + str(p.timeit(number=2)/2.0))

