from utils.discarrays import discarray
from typing import Sequence, Callable
import numpy as np
from os.path import join

MODEL_DIR = join('/home/marcosrdac/tmp/awm_data', 'model')


class VelocityModel:
    def __init__(self,
                 name: str,
                 load: Callable,
                 shape: Sequence[int] = None,
                 delta: Sequence[int] = None,
                 *load_args,
                 **load_kwargs):
        self.name = name
        self.shape = shape
        self.delta = delta
        self.nz, self.nx = shape or (None, None)
        self.dz, self.dx = delta or (None, None)
        self.load_args = (shape, delta, *load_args)
        self.load_kwargs = {'shape': shape, 'delta': delta, **load_kwargs}
        self.load = lambda: load(*self.load_args, **self.load_kwargs)


def load_marmousi(*args, **kwargs):
    path = kwargs.get('path') or args[0]
    da = discarray(path, mode='r', order='F', dtype=float)
    return da.astype(np.float32)


def load_multi_layered(*args, **kwargs):
    shape = kwargs.get('shape') or args[0]
    interfaces = kwargs.get('interfaces') or args[1]
    velocities = kwargs.get('velocities') or args[2]

    model = np.empty(shape, dtype=np.float32)
    d_old = None
    for d, v in zip(interfaces, velocities):
        model[d_old:d] = v
        d_old = d
    model[d_old:] = velocities[-1]
    return model


marmousi_model = VelocityModel(name='marmousi',
                               load=load_marmousi,
                               shape=(350, 350),
                               delta=(25, 8),
                               path=join(MODEL_DIR, 'marmousi',
                                         'marmousi_f.bin'))

multi_layered_model = VelocityModel(name='multi_layered',
                                    shape=(80, 121),
                                    delta=(10, 10),
                                    interfaces=(20, 40),
                                    velocities=(2000, 3500, 6000),
                                    load=load_multi_layered)
