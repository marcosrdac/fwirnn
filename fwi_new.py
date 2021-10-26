from random import shuffle
from os import makedirs
from os.path import join, dirname, isfile, isdir, dirname, basename
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Any, Sequence, Dict, Callable
import itertools
from utils.filters import depth_lowpass, zero_shallow, clip, seis_gain, surface_to_depth
from awm import make_awm
from utils.plotutils import plot_seismograms, plot_velocities
from functools import partial
from utils.wavelets import rickerwave
from seismicarrays import make_default_arrays, parse_geometry, make_array
from pickle import dumps, loads
import tensorflow.keras.optimizers as optimizers
from utils.discarrays import discarray, todiscarray
from utils.structures import AccumulatingDict


def mse(ŷ, y=None):
    e = ŷ - y if y is not None else ŷ
    return 1 / 2 * tf.reduce_mean(e**2)


def mae(ŷ, y=None):  # mean absolute deviation
    e = ŷ - y if y is not None else ŷ
    return tf.reduce_mean(tf.abs(e))


def make_loss_fun(lamb=1.):
    def loss_fun(ŷ, y, v=None):
        if v is None or lamb == 0:
            return mse(ŷ - y)
        else:
            return mse(ŷ - y) + lamb * mae(v)

    return loss_fun


def make_ground_truth(v, X, seis):
    Y = []
    for xi in X:
        srcsgns, srccrds, reccrds = xi
        yi = seis(v=v, srcsgns=srcsgns, srccrds=srccrds, reccrds=reccrds)
        Y.append(yi)
    return Y


def make_combinations(d: Dict[Any, Sequence]):
    return (dict(zip(d.keys(), c)) for c in itertools.product(*d.values()))


def make_seis_wo_dw(seis, sporder):
    def seis_wo_dw(v, *args, **kwargs):
        v_s = surface_to_depth(v, sporder // 2)
        s = seis(v, *args, **kwargs)
        dw = seis(v_s, *args, **kwargs)
        return s - dw

    return seis_wo_dw


def train_epoch(optimizer,
                name,
                e,
                v_e,
                i_xiyi,
                loss_fun,
                seis,
                result_dirs,
                v=None,
                v_0=None,
                zero_before_depth=0,
                accumulate_gradients=False,
                show=False):

    stop = False

    if accumulate_gradients:
        loss_grad = np.zeros_like(v_e)

    for t, (i, (xi, yi)) in enumerate(i_xiyi, start=1):
        print(f'  e={e} t={t} i={i}')

        with tf.GradientTape() as tape:
            tape.watch(v_e)

            # performing forward modeling
            srcsgns, srccrds, reccrds = xi
            ŷi = seis(v_e, srcsgns=srcsgns, srccrds=srccrds, reccrds=reccrds)

            # use only valid time samples
            yi = yi[:ŷi.shape[0]]

            # normalizing amplitudes
            norm_coef = np.sum(yi * ŷi) / np.sum(ŷi * ŷi)
            ŷi = norm_coef * ŷi

            loss = loss_fun(ŷi, yi)

        loss_grad_i = tape.gradient(loss, v_e).numpy()

        # loss_grad_i = clip(loss_grad_i, 0.02)
        loss_grad_i = zero_shallow(loss_grad_i, zero_before_depth)

        if accumulate_gradients:
            loss_grad += loss_grad_i / len(i_xiyi)

        fig_filename = f'{name}_e={e}_t={t}_i={i}_loss={loss:.4g}.png'
        fig_path = join(result_dirs['v_plot'], fig_filename)
        fig_title = f'Epoch {e}, Iteration {t} (Shot {i}), Loss={loss:.4g}'
        plot_velocities(
            v,
            v_0,
            v_e,
            loss_grad if accumulate_gradients else loss_grad_i,
            title=fig_title,
            figname=fig_path,
            show=show,
        )

        fig_path = join(result_dirs['seis_plot'], fig_filename)
        plot_seismograms(
            ŷi,
            yi,
            title=fig_title,
            figname=fig_path,
            show=show,
        )

        if not np.isfinite(loss):
            stop = True
            break

        if not accumulate_gradients:
            optimizer.apply_gradients(((loss_grad_i, v_e), ))

    if accumulate_gradients:
        optimizer.apply_gradients(((loss_grad, v_e), ))

    # save model data
    v_path = join(result_dirs['v_data'], f'{name}_e={e}_t={t}_i={i}.bin')
    todiscarray(v_path, v_e.numpy())

    if not np.all(np.isfinite(v_e)):
        stop = True

    # parallelization tip
    # grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    return optimizer, v_e, stop


if __name__ == '__main__':
    from datetime import datetime
    from config import DIR_CONFIG
    from models import marmousi_model, multi_layered_model
    from scipy.signal import convolve
    NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def calc_dt(v, dz, dx):
        '''Maximum dt used for a model, given grid spacing.'''
        return np.sqrt((3 / 4) / ((v / dx)**2 + (v / dz)**2))

    model = marmousi_model
    # model = multi_layered_model
    v = model.load()

    shape = nz, nx = model.shape
    delta = dz, dx = model.delta
    model_name = model.name

    dt_dat = 0.004
    dt_min = calc_dt(np.max(v), dz, dx)
    # dt_div = int(np.ceil(dt_dat / dt_min))
    # dt = dt_dat / dt_div
    dt = dt_min

    # nt = int(3 / dt)
    nt = 1000

    # seismic source
    # nu = 2  # Hz
    nu_max = np.min(v) / (10 * dx)
    nu = nu_max
    srcsgn = rickerwave(nu, dt)

    if model_name == 'multi_layered':
        array_desc = dict(geometry='40-6-0-6-40',
                          rr=1,
                          ss=3,
                          ns=None,
                          nx=nx,
                          dx=1,
                          all_recs=True)
    elif model_name == 'marmousi':
        downscale = 4
        v = v[::downscale, ::downscale]
        shape = nz, nx = v.shape
        dz, dx = downscale * dz, downscale * dx

        array_desc = dict(geometry=f'1300-{3*dx}-0',
                          rr=dx,
                          ss=(nx // 20) * dx,
                          dx=dx,
                          nx=nx,
                          ns=None,
                          all_recs=True)

    make_srcsgns, srccrds, reccrds, true_srccrds, true_reccrds = make_array(
        **array_desc, )

    srcsgns = make_srcsgns(srcsgn)

    print(f'Number of shots: {len(srccrds)}')

    # modeling parameters
    sporder = 8
    awm = make_awm(v.shape,
                   dz,
                   dx,
                   dt,
                   tsolver='fd',
                   spsolver='fd',
                   sporder=sporder,
                   vmax=np.max(v))
    seis = partial(awm, nt=nt, out='seis')

    seis_wo_dw = make_seis_wo_dw(seis, sporder)

    # START MAKING DATASET
    X = (*zip(srcsgns, srccrds, reccrds), )

    if not isfile(DIR_CONFIG['Y_old']):
        Y = make_ground_truth(v, X, seis=seis_wo_dw)
        Y_text = dumps(Y)
        with open(DIR_CONFIG['Y_old'], 'wb') as f:
            f.write(Y_text)
    else:
        with open(DIR_CONFIG['Y_old'], 'rb') as f:
            Y_text = f.read()
        Y = loads(Y_text)
    # END MAKING DATASET

    freqs = 2, 4, 8, 17
    multi_scale_sources = {freq: rickerwave(freq, dt) for freq in freqs}

    # making directories
    result_dirs = DIR_CONFIG['result_dirs'](f'{NOW}_{model_name}')
    DIR_CONFIG = {**DIR_CONFIG, **result_dirs}
    for d in result_dirs.values():
        makedirs(d, exist_ok=True)

    loss_fun = make_loss_fun(lamb=0)

    # initial model for fwi
    maintain_before_depth = sporder // 2
    # maintain_before_depth = 0
    v_0 = depth_lowpass(v, ws=30, min_depth=maintain_before_depth)

    # training
    show = False
    accumulate_gradients = False
    shuffle_shots = True
    zero_before_depth = maintain_before_depth
    epochs = 100
    optimizer_param_spaces = {
        'adam': {
            # 'learning_rate': (1 * 10**i for i in range(1, 2 + 1)), #2 is good
            'learning_rate': (1 * 10**i for i in range(1, 1 + 1)),
            'beta_1': (.9, ),  # .7,
            'beta_2': (.9, ),  # .7,
        },
        'sgd': {
            'learning_rate': (1 * 10**i for i in range(8, 9 + 1)),
        },
        # 'adam': {
        # 'learning_rate': (1 * 10**i for i in range(1, 4 + 1)),
        # 'beta_1': (.5, .7, .9),
        # 'beta_2': (.7, .9, .999),
        # },
        # 'sgd': {
        # 'learning_rate': (1 * 10**i for i in range(6, 9 + 1)),
        # },
    }

    # Setting up history
    histories = {}

    for optimizer_name, param_space in optimizer_param_spaces.items():

        if optimizer_name == 'adam':
            optimizer_generator = optimizers.Adam
        elif optimizer_name == 'sgd':
            optimizer_generator = optimizers.SGD
        else:
            print('Unknown optimizer; skipping.')
            continue

        for optimizer_params in make_combinations(param_space):
            optimizer = optimizer_generator(**optimizer_params)

            param_strs = [f'{p}={v}' for p, v in optimizer_params.items()]

            name = '_'.join(
                [NOW, model_name, f'opt={optimizer_name}', *param_strs])

            histories[' '.join(param_strs)] = {
                'train': AccumulatingDict(),
                'test': AccumulatingDict(),
            }

            # for the pickling part
            # histories = {k: dict(v) for k, v in histories.items()}

            v_e = tf.Variable(v_0, trainable=True)

            v_path = join(result_dirs['v_data'], f'{name}_e=0_t=0_i=0.bin')
            todiscarray(v_path, v_e.numpy())

            for freq, srcsgn in multi_scale_sources.items():
                srcsgns = make_srcsgns(srcsgn)
                X_freq = [*zip(srcsgns, srccrds, reccrds)]
                Y_freq = [convolve(y, srcsgn[:, None], mode='same') for y in Y]

                # Shots
                i_xiyi = [*enumerate(zip(X_freq, Y_freq), start=1)]

                for e in range(1, epochs + 1):

                    if accumulate_gradients:
                        loss_grad = np.zeros_like(v_e)

                    if shuffle_shots:  # TODO: goes to train epoch
                        shuffle(i_xiyi)

                    optimizer, ve, stop = train_epoch(
                        optimizer=optimizer,
                        name=name,
                        e=e,
                        v_e=v_e,
                        i_xiyi=i_xiyi,
                        loss_fun=loss_fun,
                        seis=seis_wo_dw,
                        result_dirs=result_dirs,
                        v=v,
                        v_0=v_0,
                        zero_before_depth=zero_before_depth,
                        accumulate_gradients=False,
                        show=show)
                    if stop:
                        break

                if stop:
                    break
