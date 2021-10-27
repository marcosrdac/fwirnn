from os import makedirs
from os.path import join, dirname, isfile, isdir, dirname, basename
import numpy as np
import tensorflow as tf
from tensorflow import math as tfm, image as tfi
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
import tensorflow.keras.optimizers as optimizers
from utils.discarrays import discarray, todiscarray
from utils.structures import AccumulatingDict
from stop_conditions import StopCondition
import pickle


def dump_to_file(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def mse(ŷ, y=None):
    e = ŷ - y if y is not None else ŷ
    return tf.reduce_mean(e**2)


def mae(ŷ, y=None):  # mean absolute deviation
    e = ŷ - y if y is not None else ŷ
    return tf.reduce_mean(tf.abs(e))


def rmse(ŷ, y=None):
    return tf.math.sqrt(mse(ŷ, y))


def mse_loss(ŷ, y=None):
    return 1 / 2 * mse(ŷ, y)


def mae_loss(ŷ, y=None):  # mean absolute deviation
    return mae(ŷ, y)


def make_loss_fun(lamb=0, a=0., dz=1, dx=1):
    if lamb == 0:

        def loss_fun(ŷ, y, *args, **kwargs):
            return mse_loss(ŷ, y)
    else:
        kern_fwd = tf.constant(np.asarray([0, -1, 1], dtype=np.float32))
        kern_bwd = tf.constant(np.asarray([-1, 1, 0], dtype=np.float32))

        kernz_fwd = kern_fwd[:, None, None, None] / dz
        kernz_bwd = kern_bwd[:, None, None, None] / dz
        kernx_fwd = kern_fwd[None, :, None, None] / dx
        kernx_bwd = kern_bwd[None, :, None, None] / dx

        def loss_fun(ŷ, y, v):
            v = v[None, :, :, None]

            vz_fwd = tf.nn.convolution(v, kernz_fwd, padding="SAME")[0, ..., 0]
            vx_fwd = tf.nn.convolution(v, kernx_fwd, padding="SAME")[0, ..., 0]
            grad_v = tfm.sqrt(tfm.square(vz_fwd) + tfm.square(vx_fwd))

            vz_bwd = tf.nn.convolution(v, kernz_bwd, padding="SAME")[0, ..., 0]
            vx_bwd = tf.nn.convolution(v, kernx_bwd, padding="SAME")[0, ..., 0]
            vzz = (vz_fwd - vz_bwd) / dz
            vxx = (vx_fwd - vx_bwd) / dx
            lap_v = vzz + vxx

            pred_loss = mse_loss(ŷ - y)
            reg_loss = mae_loss((1 - a) * grad_v + a * lap_v)
            return pred_loss + lamb * reg_loss

    return loss_fun


def make_ground_truth(seis_fun, v, X):
    Y = []
    for xi in X:
        srcsgns, srccrds, reccrds = xi
        yi = seis(v=v, srcsgns=srcsgns, srccrds=srccrds, reccrds=reccrds)
        Y.append(yi)
    return Y


def make_combinations(d: Dict[Any, Sequence]):
    return (dict(zip(d.keys(), c)) for c in itertools.product(*d.values()))


def make_seis_wo_direct_fun(seis, sporder):
    def seis_wo_direct_fun(v, *args, **kwargs):
        v_s = surface_to_depth(v, sporder // 2)
        s = seis(v, *args, **kwargs)
        dw = seis(v_s, *args, **kwargs)
        return s - dw

    return seis_wo_direct_fun


def make_val_loss_grad_fun(seis_fun, loss_fun):
    def val_loss_grad_fun(v, x, y):
        srcsgns, srccrds, reccrds = x

        with tf.GradientTape() as tape:
            tape.watch(v)

            ŷ = seis_fun(v, srcsgns=srcsgns, srccrds=srccrds, reccrds=reccrds)
            y = y[:ŷ.shape[0]]

            # normalizing amplitudes
            norm_coef = tf.reduce_sum(y * ŷ) / tf.reduce_sum(ŷ * ŷ)
            ŷ *= norm_coef

            loss = loss_fun(ŷ, y)
        loss_grad = tape.gradient(loss, v).numpy()

        return ŷ, loss, loss_grad

    return val_loss_grad_fun


def train_epoch(epoch,
                optimizer,
                v_e,
                X,
                Y,
                loss_grad_fun,
                seis,
                result_dirs,
                preffixes=(),
                v_true=None,
                v_0=None,
                zero_before_depth=0,
                batch_size=1,
                accum_grads=False,
                show=False):
    diverged = False

    n_batches = X.shape[0] // batch_size
    batches = np.random.permutation(X.shape[0])
    batches = batches[:n_batches * batch_size]
    batches = batches.reshape((n_batches, batch_size))

    epoch_mean_loss = 0
    epoch_mean_loss_grad = np.zeros_like(v_e) if accum_grads else None
    for batch_num, batch in enumerate(batches):
        iter_desc = [f'e={1+epoch}', f'b={1+batch_num}']

        # TODO
        # cluster parallelization can be applied here, i.e.:
        # `tf.distribute.get_replica_context().all_reduce('sum', grads)`
        batch_mean_loss = 0
        batch_mean_loss_grad = np.zeros_like(v_e)
        for shot_num, shot_idx in batch:
            shot_desc = [
                *iter_desc,
                f's={1+shot_num}'
                f'i={1+shot_idx}',
            ]

            print('  ', *shot_desc)
            x, y = X[shot_idx], Y[shot_idx]
            ŷ, shot_loss, shot_loss_grad = val_loss_grad_fun(v_e, x, y).numpy()

            # processing gradients
            # shot_loss_grad = clip(shot_loss_grad, 0.02)
            shot_loss_grad = zero_shallow(shot_loss_grad, zero_before_depth)

            batch_mean_loss += shot_loss / batch_size
            batch_mean_loss_grad += shot_loss_grad / batch_size

        epoch_mean_loss += batch_mean_loss / n_batches
        if accum_grads:
            epoch_mean_loss_grad += batch_mean_loss_grad / n_batches

        base_desc = [*preffixes, *iter_desc]
        full_desc = [*base_desc, f'loss={batch_mean_loss:.4g}']
        full_title = [
            f'Epoch {1+epoch}', f'Batch {1+batch_num}',
            f'(Last shot={1+shot_idx})', '-', f'Loss={batch_mean_loss:.4g}'
        ]

        # plotting data
        fig_filename = '_'.join(full_desc) + '.png'
        fig_title = ' '.join(full_title)

        if result_dirs.get('v_plot'):
            fig_path = join(result_dirs['v_plot'], fig_filename)
            plot_velocities(
                v_true,
                v_0,
                v_e,
                epoch_mean_loss_grad if accum_grads else batch_mean_loss_grad,
                title=fig_title,
                figname=fig_path,
                show=show)

        if result_dirs.get('seis_plot'):
            fig_path = join(result_dirs['seis_plot'], fig_filename)
            plot_seismograms(ŷ,
                             y,
                             title=fig_title,
                             figname=fig_path,
                             show=show)

        if not np.isfinite(batch_mean_loss):
            diverged = True
            break

        # apply gradients
        if not accum_grads:
            optimizer.apply_gradients(((batch_mean_loss_grad, v_e), ))

    if accum_grads:
        optimizer.apply_gradients(((epoch_mean_loss_grad, v_e), ))

    if not np.all(np.isfinite(v_e)):
        diverged = True

    if result_dirs.get('v_data'):
        epoch_desc = [
            *preffixes,
            f'e={1+epoch}',
            f'loss={epoch_mean_loss:.4g}',
            f'err={v_rmse:.4g}',
        ]

        # save model variables
        v_file = '_'.join(epoch_desc) + '.bin'
        v_path = join(result_dirs['v_data'], v_file)
        todiscarray(v_path, v_e.numpy())

    metrics = {'loss': epoch_mean_loss}

    return optimizer, v_e, metrics, diverged


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
    v_true = model.load()

    shape = nz, nx = model.shape
    delta = dz, dx = model.delta
    model_name = model.name

    dt_dat = 0.004
    dt_min = calc_dt(np.max(v_true), dz, dx)
    # dt_div = int(np.ceil(dt_dat / dt_min))
    # dt = dt_dat / dt_div
    dt = dt_min

    # nt = int(3 / dt)
    nt = 1000

    # seismic source
    # nu = 2  # Hz
    nu_max = np.min(v_true) / (10 * dx)
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
        v_true = v_true[::downscale, ::downscale]
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
    awm = make_awm(v_true.shape,
                   dz,
                   dx,
                   dt,
                   tsolver='fd',
                   spsolver='fd',
                   sporder=sporder,
                   vmax=np.max(v_true))
    seis_fun = partial(awm, nt=nt, out='seis')
    seis_wo_direct_fun = make_seis_wo_direct_fun(seis, sporder)

    # START MAKING DATASET
    X = [*zip(srcsgns, srccrds, reccrds)]

    if not isfile(DIR_CONFIG['Y_old']):
        Y = make_ground_truth(seis_wo_direct_fun, v_true, X)
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
    val_loss_grad_fun = make_val_loss_grad_fun(seis_wo_direct_fun, loss_fun)

    # initial model for fwi
    maintain_before_depth = sporder // 2
    # maintain_before_depth = 0
    v_0 = depth_lowpass(v_true, ws=30, min_depth=maintain_before_depth)

    # training
    show = False
    accum_grads = False
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

            preffixes = [NOW, model_name, f'o={optimizer_name}', *param_strs]

            name = '_'.join(preffixes)

            histories = {
                'train': AccumulatingDict(),
                'test': AccumulatingDict(),
            }

            v_e = tf.Variable(v_0, trainable=True)

            v_filename = '_'.join([*preffixes, 'e=0']) + '.bin'
            v_path = join(result_dirs['v_data'], v_filename)
            todiscarray(v_path, v_e.numpy())

            for freq, srcsgn in multi_scale_sources.items():
                # train_idx, test_idx
                srcsgns = make_srcsgns(srcsgn)
                X_freq = [*zip(srcsgns, srccrds, reccrds)]
                Y_freq = [convolve(y, srcsgn[:, None], mode='same') for y in Y]

                stop_condition = StopCondition(growing_is_good=False,
                                               patience=15)

                for epoch in range(1, epochs + 1):

                    optimizer, v_e, train_metrics, diverged = train_epoch(
                        epoch=epoch,
                        optimizer=optimizer,
                        v_e=v_e,
                        X=X,
                        Y=X,
                        val_loss_grad_fun=val_loss_grad_fun,
                        result_dirs=result_dirs,
                        v_0=v_0,
                        v_true=v_true,
                        zero_before_depth=zero_before_depth,
                        accum_grads=False,
                        preffixes=preffixes,
                        show=show)

                    histories['train'] += train_metrics

                    v_rmse = rmse(v_e, v_true)

                    stop_condition.update(train_metrics['loss'])
                    if stop_condition.stop or diverged:
                        break

                if diverged:
                    break

            # saving histories
            histories_path = join(
                result_dirs['metric_data'],
                '_'.join(preffixes) + '.pkl',
            )
            histories = {k: dict(v) for k, v in histories.items()}
            dump_to_file(histories_path, histories)
