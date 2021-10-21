#!/usr/bin/env python3

from os.path import join, expanduser, isdir
import numpy as np

# Directory settings
# - Base data directory
DIR_CONFIG = {'data': '/home/marcosrdac/tmp/awm_data'}
# - User managed directories
# -- Input data
DIR_CONFIG['model'] = join(DIR_CONFIG['data'], 'input')
# - Automatically managed directories
# -- Training outputs
DIR_CONFIG['result'] = join(DIR_CONFIG['data'], 'result')

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

# Model settings
MODELS = {}
# - U-net architectures
MODELS['marmousi'] = {
    'dx':
    3,
    'a':
    dict(
        rescale=(-2, -2, 0, 2, 2),
        nfeat=(
            (8, ),
            (16, ),
            (32, ),
            (16, ),
            (8, ),
        ),
        norm=True,
        # drop=(),
        droplast=.3),
}

# Other automatic configs
# WARNING: do not mess here unless you know what you are doing
# - Result directories' generator
DIR_CONFIG['result_dirs'] = lambda *d: {  # i.e. model_name, train_id
    'checkpoint_data': join(DIR_CONFIG['result'], *d, 'data',  'checkpoint'),
    'checkpoint_plot': join(DIR_CONFIG['result'], *d, 'plot', 'checkpoint'),
    'metrics_data': join(DIR_CONFIG['result'], *d, 'data', 'metrics'),
    'metrics_plot': join(DIR_CONFIG['result'], *d, 'plot', 'history'),
    'X': join(DIR_CONFIG['result'], *d, 'data', 'x'),
    'Y': join(DIR_CONFIG['result'], *d, 'data', 'y'),
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

#    nprint('Patch settings:')
#    pprint(PATCH_CONFIG)
#
#    nprint('Train settings:')
#    pprint(TRAIN_CONFIG, dontprint=('rngkeys'))
#
#    nprint('U-net settings:')
#    pprint(UNET_CONFIG, dontprint='rngkeys')
