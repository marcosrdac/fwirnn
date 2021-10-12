from random import shuffle
from os import makedirs
from os.path import join, dirname, isfile, isdir, dirname, basename
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Any, Sequence, Dict, Callable
import itertools
from filters import depth_lowpass, zero_shallow, clip, seis_gain, surface_to_depth
from awm import make_awm
from plotutils import plot_seismograms, plot_velocities
from functools import partial
from sources import rickerwave
from seismicarrays import make_default_arrays, parse_geometry, make_array
from pickle import dumps, loads
import tensorflow.keras.optimizers as optimizers
from discarrays import discarray, todiscarray
from utils.abcutils import AccumulatingDict


def mse(ŷ, y=None):
    e = ŷ - y if y is not None else ŷ
    return 1 / 2 * tf.reduce_mean(e**2)


def mae(ŷ, y=None):  # mean absolute deviation
    e = ŷ - y if y is not None else ŷ
    return tf.reduce_mean(tf.abs(e))


def make_loss_fun(lamb):
    assert lamb >= 0
    if lamb == 0:
        return mse
    else:

        def loss_fun(ŷ, y):
            e = ŷ - y
            return mse(e) + lamb * mae(e)

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


if __name__ == '__main__':
    from datetime import datetime
    from params import DIR_CONFIG
    start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


    result_dirs = DIR_CONFIG['result_dirs'](start)
    img_dir = join(dirname(IMG_DIR), f'{start}_{basename(IMG_DIR)}')
    seis_dir = join(dirname(SEIS_DIR), f'{start}_{basename(SEIS_DIR)}')
    v_dir = join(dirname(V_DIR), f'{start}_{basename(V_DIR)}')

    for folder in (dirname(SHOTS_FILE), img_dir, seis_dir, v_dir):
        makedirs(folder, exist_ok=True)

    model = 'marmousi'
    if model == '3_layers':
        # velocity field
        shape = nz, nx = 70, 120
        v = np.empty(shape, dtype=np.float32)
        v[None:nz // 3, :] = 2000.
        v[nz // 3:nz // 2, :] = 3500.
        v[nz // 2:None, :] = 6000.
        nt = 1200
        dz, dx, dt = 100., 100., 0.0075

        # seismic source
        nu = 2  # Hz
        signal = rickerwave(nu, dt)

        # header
        srcsgns, srccrds, reccrds, true_srccrds, true_reccrds = make_array(
            geometry='40-6-0-6-40',
            rr=1,
            ss=3,
            ns=None,
            srcsgn=signal,
            nx=nx,
            dx=1,
            all_recs=True,
        )
    if model == 'marmousi':
        v = discarray(
            '/home/marcosrdac/cld/Dropbox/home/pro/0/awm/awm2d/models/marmousi.bin',
            mode='r',
            order='F',
            dtype=float).astype(np.float32)
        dz, dx, dt = 260., 260., 0.008
        shape = nz, nx = v.shape
        nt = 3000

        factor = 4
        v = v[::factor, ::factor]
        shape = nz, nx = v.shape
        dz, dx = factor * dz, factor * dx

        # seismic source
        nu = 2  # Hz
        signal = rickerwave(nu, dt)

        # header
        srcsgns, srccrds, reccrds, true_srccrds, true_reccrds = make_array(
            srcsgn=signal,
            geometry=f'1300-{3*dx}-0',
            rr=dx,
            ss=(nx//20)*dx,
            dx=dx,
            nx=nx,
            ns=None,
            all_recs=True,
        )

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

    # dataset
    X = (*zip(srcsgns, srccrds, reccrds), )

    if not isfile(SHOTS_FILE):
        Y = make_ground_truth(v, X, seis=seis_wo_dw)
        Y_text = dumps(Y)
        with open(SHOTS_FILE, 'wb') as f:
            f.write(Y_text)
    else:
        with open(SHOTS_FILE, 'rb') as f:
            Y_text = f.read()
        Y = loads(Y_text)

    # shots
    i_xiyi = [*enumerate(zip(X, Y), start=1)]

    loss_fun = make_loss_fun(lamb=.3)

    # initial model for fwi
    maintain_before_depth = sporder // 2
    # maintain_before_depth = 0
    v_0 = depth_lowpass(v, ws=30, min_depth=maintain_before_depth)

    # training
    model_name = '3layed'
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

            name = '_'.join([model_name, start, optimizer_name, *param_strs])

            histories[' '.join(param_strs)] = {
                'train': AccumulatingDict(),
                'test': AccumulatingDict(),
            }

            # for the pickling part
            # histories = {k: dict(v) for k, v in histories.items()}

            v_e = tf.Variable(v_0, trainable=True)
            for e in range(1, epochs + 1):

                if accumulate_gradients:
                    loss_grad = np.zeros_like(v_e)

                if shuffle_shots:
                    shuffle(i_xiyi)

                for t, (i, (xi, yi)) in enumerate(i_xiyi, start=1):
                    print(f'  e={e} t={t} i={i}')

                    v_filename = f'{name}_e={e}_t={t}_i={i}.bin'
                    v_path = join(v_dir, v_filename)
                    todiscarray(v_path, v_e.numpy())

                    with tf.GradientTape() as tape:
                        tape.watch(v_e)
                        srcsgns, srccrds, reccrds = xi
                        ŷi = seis_wo_dw(v_e,
                                        srcsgns=srcsgns,
                                        srccrds=srccrds,
                                        reccrds=reccrds)

                        amp_norm = np.sum(yi * ŷi) / np.sum(ŷi * ŷi)
                        ŷi = amp_norm * ŷi

                        # plt.imshow(ŷi, aspect='auto')
                        # plt.show()

                        # multiscale stuff
                        # y_diff = ŷi - yi
                        # y_diff_freq = tf.signal.rfft(y_diff)
                        # # f = 1.  # Hz
                        # # kidx = int(.9 * y_diff.shape[0])
                        # kidx = y_diff.shape[0] * 2 // 3
                        # filt = np.ones(y_diff_freq.shape)
                        # filt[kidx:, :] = 0
                        # y_diff_freq *= filt
                        # y_diff = tf.signal.irfft(y_diff_freq)
                        # loss = mse(y_diff, 0)

                        loss = loss_fun(ŷi, yi)

                    loss_grad_i = tape.gradient(loss, v_e).numpy()

                    # loss_grad_i = clip(loss_grad_i, 0.02)
                    loss_grad_i = zero_shallow(loss_grad_i, zero_before_depth)

                    if accumulate_gradients:
                        loss_grad += loss_grad_i / len(X)

                    fig_filename = f'{name}_e={e}_t={t}_i={i}_loss={loss:.4g}.png'
                    fig_path = join(img_dir, fig_filename)
                    fig_title = f'Epoch {e}, Iteration {t} (Shot {i}), Loss={loss:.4g}'
                    plot_velocities(
                        v,
                        v_0,
                        v_e,
                        loss_grad if accumulate_gradients else loss_grad_i,
                        title=fig_title,
                        figname=fig_path,
                        show=show,
                        # show = e % 2 == 0 and i == len(X)-1,
                        # show = e == epochs,
                    )

                    fig_path = join(seis_dir, fig_filename)
                    fig_title = f'Epoch {e}, Iteration {t} (Shot {i}), Loss={loss:.4g}'
                    plot_seismograms(
                        ŷi,
                        yi,
                        title=fig_title,
                        figname=fig_path,
                        show=show,
                    )

                    if not np.isfinite(loss):
                        break

                    if not accumulate_gradients:
                        optimizer.apply_gradients(((loss_grad_i, v_e), ))

                if not np.isfinite(loss):
                    break

                if accumulate_gradients:
                    optimizer.apply_gradients(((loss_grad, v_e), ))

                if not np.all(np.isfinite(v_e)):
                    break

                # parallelization tip
                # grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
