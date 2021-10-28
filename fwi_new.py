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
from utils.stopping import StopCondition
from sklearn.model_selection import train_test_split
import pickle


def dump_to_file(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def mse(ŷ, y=None, reduce=tf.reduce_mean):
    e = ŷ - y if y is not None else ŷ
    return reduce(e**2)


def mae(ŷ, y=None, reduce=tf.reduce_mean):
    e = ŷ - y if y is not None else ŷ
    return reduce(tf.abs(e))


def rmse(ŷ, y=None):
    return tf.math.sqrt(mse(ŷ, y, reduce=tf.reduce_mean))


def mse_loss(ŷ, y=None, **kwargs):
    return 1 / 2 * mse(ŷ, y, **kwargs)


def mae_loss(ŷ, y=None, **kwargs):  # mean absolute deviation
    return mae(ŷ, y, **kwargs)


def make_loss_fun(lamb=0, a=0., dz=1, dx=1, reduce=tf.reduce_mean):
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

            pred_loss = mse_loss(ŷ - y, reduce=reduce)
            reg_loss = mae_loss((1 - a) * grad_v + a * lap_v, reduce=reduce)
            return pred_loss + lamb * reg_loss

    return loss_fun


def make_ground_truth(seis_fun, v, X):
    Y = []
    print(f'Modeling {len(X)} seismograms')
    for i, xi in enumerate(X, start=1):
        print(f'- seismogram #{i}')
        srcsgns, srccrds, reccrds = xi
        yi = seis_fun(v=v, srcsgns=srcsgns, srccrds=srccrds, reccrds=reccrds)
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


def make_eval_epoch(seis_fun, loss_fun):
    def eval_epoch(v, X, Y, idx):
        print('Running test shots:')
        mean_loss = 0
        for shot_num, shot_idx in enumerate(idx):
            print(f'  s={1+shot_num}', f'i={1+shot_idx}', end=' ')
            x, y = X[shot_idx], Y[shot_idx]
            srcsgns, srccrds, reccrds = x
            ŷ = seis_fun(v=v,
                         srcsgns=srcsgns,
                         srccrds=srccrds,
                         reccrds=reccrds)
            loss = loss_fun(ŷ, y)
            print(f'loss={loss:.4g}')
            mean_loss += loss
        mean_loss /= len(idx)
        print(f'  mean_loss={mean_loss:.4g}')
        metrics = {'loss': mean_loss}
        return metrics

    return eval_epoch


def make_train_epoch(val_loss_grad_fun,
                     dz,
                     dx,
                     dt,
                     accum_grads=False,
                     batch_size=1,
                     result_dirs={}):
    def train_epoch(epoch,
                    optimizer,
                    v_e,
                    X,
                    Y,
                    preffixes=(),
                    v_0=None,
                    v_true=None,
                    zero_before_depth=0,
                    idx_train=None,
                    show=False):
        diverged = False

        idx_train = np.arange(len(X)) if idx_train is None else idx_train
        n_samples = len(idx_train)
        n_batches = n_samples // batch_size
        batches = np.random.permutation(idx_train)
        batches = batches[:n_batches * batch_size]
        batches = batches.reshape((n_batches, batch_size))

        epoch_mean_loss = 0
        epoch_mean_loss_grad = np.zeros_like(v_e) if accum_grads else None
        for batch_num, batch in enumerate(batches):
            iter_desc = [f'e={1+epoch}', f'b={1+batch_num}']

            # TODO
            # cluster parallelization can be exploited here, i.e.:
            # `tf.distribute.get_replica_context().all_reduce('sum', grads)`
            batch_mean_loss = 0
            batch_mean_loss_grad = np.zeros_like(v_e)
            for shot_num, shot_idx in enumerate(batch):
                shot_desc = [
                    *iter_desc,
                    f's={1+shot_num}',
                    f'i={1+shot_idx}',
                ]

                print(' ', *shot_desc, end=' ')
                x, y = X[shot_idx], Y[shot_idx]
                ŷ, shot_loss, shot_loss_grad = val_loss_grad_fun(v_e, x, y)
                print(f'loss={shot_loss:.4g}')

                # processing gradients
                # shot_loss_grad = clip(shot_loss_grad, 0.02)
                shot_loss_grad = zero_shallow(shot_loss_grad,
                                              zero_before_depth)

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

            if accum_grads:
                loss_grad = epoch_mean_loss_grad
            else:
                loss_grad = batch_mean_loss_grad
            if result_dirs.get('v_plot'):
                fig_path = join(result_dirs['v_plot'], fig_filename)
                # TODO plot_velocities should not depend on v true
                plot_velocities(v_e=v_e,
                                loss_grad=loss_grad,
                                v_0=v_0,
                                v_true=v_true,
                                dz=dz,
                                dx=dx,
                                xlabel='x (m)',
                                ylabel='z (m)',
                                cbar_label='m/s',
                                title=fig_title,
                                figname=fig_path,
                                show=show)

            if result_dirs.get('seis_plot'):
                fig_path = join(result_dirs['seis_plot'], fig_filename)
                plot_seismograms(
                    ŷ,
                    y,
                    dt=dt,
                    # dx=dx,
                    # xlabel='x (km)',
                    dx=1,
                    xlabel='Trace index',
                    ylabel='t (s)',
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
            ]

            # save model variables
            v_file = '_'.join(epoch_desc) + '.bin'
            v_path = join(result_dirs['v_data'], v_file)
            todiscarray(v_path, v_e.numpy())

        metrics = {'loss': epoch_mean_loss}

        return optimizer, v_e, metrics, diverged

    return train_epoch


if __name__ == '__main__':
    from datetime import datetime
    from config import DIR_CONFIG, marmousi_model, multi_layered_model
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

    print(nt, dt, nu_max)

    if model_name == 'multi_layered':
        array_desc = dict(geometry='40-6-0-6-40',
                          rr=1,
                          ss=3,
                          ns=None,
                          nx=nx,
                          dx=1,
                          all_recs=True)
    elif model_name == 'marmousi':
        downscale = 1
        v_true = v_true[::downscale, ::downscale]
        shape = nz, nx = v_true.shape
        dz, dx = downscale * dz, downscale * dx

        array_desc = dict(
            geometry=f'1300-{1*dx}-0-{1*dx}-1300',
            rr=dx,
            # ss=(nx // 20) * dx,
            ss=(nx // 5) * dx,
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
    seis_wo_direct_fun = make_seis_wo_direct_fun(seis_fun, sporder)

    # START MAKING DATASET
    X = [*zip(srcsgns, srccrds, reccrds)]
    if not isfile(DIR_CONFIG['Y_old']):
        Y = make_ground_truth(seis_wo_direct_fun, v_true, X)
        with open(DIR_CONFIG['Y_old'], 'wb') as f:
            pickle.dump(Y, f)
    else:
        with open(DIR_CONFIG['Y_old'], 'rb') as f:
            Y = pickle.load(f)
    # END MAKING DATASET

    test_split = 1 / 5
    freqs = 2, 4, 8, 17
    multi_scale_sources = {freq: rickerwave(freq, dt) for freq in freqs}

    # making directories
    result_dirs = DIR_CONFIG['result_dirs'](f'{NOW}_{model_name}')
    DIR_CONFIG = {**DIR_CONFIG, **result_dirs}
    for d in result_dirs.values():
        makedirs(d, exist_ok=True)

    loss_fun = make_loss_fun(lamb=0, reduce=tf.reduce_sum)
    val_loss_grad_fun = make_val_loss_grad_fun(seis_wo_direct_fun, loss_fun)
    train_epoch = make_train_epoch(val_loss_grad_fun,
                                   accum_grads=False,
                                   result_dirs=result_dirs,
                                   dz=dz,
                                   dx=dx,
                                   dt=dt)
    eval_epoch = make_eval_epoch(seis_wo_direct_fun, loss_fun)

    # hold-out validation
    idx = np.arange(len(X))
    idx_test = np.random.choice(idx, size=int(test_split * len(X)))
    idx_train = np.delete(idx, idx_test)

    # initial model for fwi
    maintain_before_depth = sporder // 2
    # maintain_before_depth = 0
    v_0 = depth_lowpass(v_true, ws=30, min_depth=maintain_before_depth)

    # training
    show = False
    accum_grads = False
    zero_before_depth = maintain_before_depth
    max_epochs = 100
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
                'param': AccumulatingDict(),
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
                                               patience=10)

                for epoch in range(max_epochs):

                    optimizer, v_e, train_metrics, diverged = train_epoch(
                        epoch=epoch,
                        optimizer=optimizer,
                        v_e=v_e,
                        X=X,
                        Y=Y,
                        idx_train=idx_train,
                        v_0=v_0,
                        v_true=v_true,
                        zero_before_depth=zero_before_depth,
                        preffixes=preffixes,
                        show=show)

                    test_metrics = eval_epoch(v_e, X, Y, idx_test)

                    histories['train'] += train_metrics
                    histories['test'] += test_metrics

                    if v_true is not None:
                        v_rmse = rmse(v_e, v_true)
                        histories['param'] += {'rmse': v_rmse}

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
