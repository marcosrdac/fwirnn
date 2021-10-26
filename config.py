#!/usr/bin/env python3

from os.path import join, expanduser, isdir
import numpy as np

# Directory settings
# - Base data directory
DIR_CONFIG = {'data': '/home/marcosrdac/tmp/awm_data'}
# - User managed directories
# -- Input data
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

# Other automatic configs
# WARNING: do not mess here unless you know what you are doing
# - Result directories' generator
DIR_CONFIG['result_dirs'] = lambda *d: {  # i.e. model_name, train_id
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
