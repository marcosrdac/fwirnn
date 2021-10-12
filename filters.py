import numpy as np
import tensorflow as tf
from scipy.ndimage import uniform_filter


def zero_shallow(arr, depth):
    filt = arr.copy()
    filt[:depth, :] = 0
    return filt


def depth_lowpass(arr, ws, min_depth=0):
    filt = uniform_filter(arr, size=ws)
    filt[:min_depth, :] = arr[:min_depth, :]
    return filt


def np_surface_to_depth(arr, min_depth=0):
    filt = np.asarray(arr).copy()
    filt[min_depth + 1:, :] = arr[None, min_depth, :]
    return filt


def tf_surface_to_depth(arr, min_depth=0):
    filt = tf.Variable(arr)
    new_block = tf.tile(arr[None, min_depth, :],
                        [arr.shape[0] - 1 - min_depth, 1])
    filt[min_depth + 1:, :].assign(new_block)
    return filt


def surface_to_depth(arr, *args, **kwargs):
    if isinstance(arr, tf.Tensor):
        return tf_surface_to_depth(arr, *args, **kwargs)
    else:
        return np_surface_to_depth(arr, *args, **kwargs)


def clip(arr, keep=1.0):
    max_val = keep * np.max(np.abs(arr))
    arr[arr < -max_val] = -max_val
    arr[arr > max_val] = max_val
    return arr


def seis_gain(d, dt=0.015, a=2., b=0.):
    '''Simple amplitude gain function.'''
    d = np.asarray(d)
    nt, nx = d.shape
    t = np.arange(nt) * dt
    tgain = np.power(t, a) * np.exp(b * t)
    return tgain[:, None] * d


if __name__ == '__main__':
    shape = nz, nx = 5, 5
    v = np.empty(shape, dtype=np.float32)
    v[None:nz // 3, :] = 2000.
    v[nz // 3:nz // 2, :] = 3500.
    v[nz // 2:None, :] = 6000.
    v = tf.Variable(v)
    print(v)
    print(surface_to_depth(v))
    print(tf_surface_to_depth(v, 1))
    print(v)
