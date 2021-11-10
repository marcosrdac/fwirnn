#!/usr/bin/env python3

from os.path import join
from typing import Sequence, Callable
from utils.saving import discarray
from os.path import join, expanduser, isdir
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

# Directory settings
# - Base data directory
DIR_CONFIG = {'data': '/home/marcosrdac/tmp/awm_data'}
# - User managed directories
# -- Input data
DIR_CONFIG['model'] = join(DIR_CONFIG['data'], 'model')
DIR_CONFIG['source'] = join(DIR_CONFIG['data'], 'source')
# - Automatically managed directories
# -- Training outputs
DIR_CONFIG['result'] = join(DIR_CONFIG['data'], 'result')
DIR_CONFIG['Y_old'] = join(DIR_CONFIG['data'], 'Y.pkl')

# Train config
TRAIN_CONFIG = {}
# - Independent labels to also keep track of metrics
TRAIN_CONFIG['batch_size'] = 20  # int or None
TRAIN_CONFIG['max_epochs'] = 1000
TRAIN_CONFIG['test_size'] = 1 / 4
# TRAIN_CONFIG['learning_rates'] = 10 ** np.linspace(-1, 2, 8)
# TRAIN_CONFIG['learning_rates'] = 1e-3, 1e-2, 1e-1,
TRAIN_CONFIG['learning_rates'] = 2e-2,
TRAIN_CONFIG['early_stopping'] = {
    'enable': True,
    'metric': 'accuracy',
    'greater_is_better': True,
    'patience': 20,
    'delta': 0,
}

# Model definitions
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
                               shape=(375, 369),
                               delta=(8, 25),
                               path=join(DIR_CONFIG['model'], 'marmousi',
                                         'marmousi_f.bin'))

multi_layered_model = VelocityModel(name='multi_layered',
                                    shape=(80, 121),
                                    delta=(10, 10),
                                    interfaces=(20, 40),
                                    velocities=(2000, 3500, 6000),
                                    load=load_multi_layered)

# Other automatic configs
# WARNING: do not mess here unless you know what you are doing
# - Result directories' generator
DIR_CONFIG['result_dirs'] = lambda *d: {  # i.e. model_name, train_id
    'root': join(DIR_CONFIG['result'], *d),
    'v_data': join(DIR_CONFIG['result'], *d, 'data_v'),
    'v_plot': join(DIR_CONFIG['result'], *d, 'plot_v'),
    'seis_data': join(DIR_CONFIG['result'], *d, 'data_seis'),
    'seis_plot': join(DIR_CONFIG['result'], *d, 'plot_seis'),
    'metric_data': join(DIR_CONFIG['result'], *d, 'data_metric'),
    'metric_plot': join(DIR_CONFIG['result'], *d, 'plot_metric'),
    'dataset': join(DIR_CONFIG['result'], *d, 'dataset'),
}

if __name__ == '__main__':
    from os import listdir
    from utils.pretty import pprint

    def nprint(*args, **kwargs):
        return print(*['\n' + str(args[0]), *args[1:]], **kwargs)

    print('Directory definitions:')
    for dir_nick, dir_path in DIR_CONFIG.items():
        if dir_nick == 'result_dirs':
            continue
        dir_exists = 'X' if isdir(dir_path) else ' '
        num_files = len(listdir(dir_path)) if isdir(dir_path) else 0
        print(f'- [{dir_exists}]',
              f'{dir_nick} -> {dir_path!r}',
              f'(with {num_files} files)',
              sep=' ')

    from models import Marmousi
    model = Marmousi()
    v = model.load()

#    nprint('Patch settings:')
#    pprint(PATCH_CONFIG)
#
#    nprint('Train settings:')
#    pprint(TRAIN_CONFIG, dontprint=('rngkeys'))
#
#    nprint('U-net settings:')
#    pprint(UNET_CONFIG, dontprint='rngkeys')
