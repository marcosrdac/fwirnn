import re
from distutils.version import StrictVersion
import matplotlib.pyplot as plt
from os import listdir, system, popen
from os.path import join
import numpy as np


def discarray(filename, mode='r', dtype=float, shape=None, order='C'):
    file_mode = f'{mode[0]}b{mode[1:]}'
    if not isinstance(shape, tuple):
        shape = (shape, )
    with open(filename, file_mode) as io:
        if 'w' in mode:
            ndims_shape = np.array((len(shape), *shape), dtype=np.int64)
            ndims_shape.tofile(io)
        if 'r' in mode:
            ndims = np.fromfile(io, dtype=np.int64, count=1)[0]
            shape = tuple(np.fromfile(io, dtype=np.int64, count=ndims))
        offset = io.tell()
        arr = np.memmap(io,
                        mode=mode,
                        dtype=dtype,
                        shape=shape,
                        offset=offset,
                        order=order)
    return arr


def todiscarray(filename, arr, order='C'):
    disk_arr = discarray(filename,
                         'w+',
                         dtype=arr.dtype,
                         shape=arr.shape,
                         order=order)
    disk_arr[...] = arr[...]


folder = 'data_v'
# ls = [join(folder, f) for f in listdir(folder)]
ls = [join(folder, f[:-1]) for f in popen('ls data_v | sort -V')]

mean_vs = []
for i, fn in enumerate(ls):

    v = discarray(fn, dtype=np.float32)
    # plt.imshow(v)
    # plt.show()
    if i == 0:
        # abs_mean_delta = 0
        max_abs_delta = 0
    else:
        v_diff = v - v_old
        # v_diff = v_diff[v_diff >= 5]
        # abs_mean_delta = np.mean(np.abs(v_diff/v))

        max_abs_delta = np.mean(np.abs(v_diff))
        # max_abs_delta = np.sum(np.abs(v_diff)/v)

    v_old = v
    # if i % 31 != 0:
        # continue

    # mean_vs.append(abs_mean_delta)
    mean_vs.append(max_abs_delta)

plt.plot(mean_vs)
plt.grid()
plt.show()
