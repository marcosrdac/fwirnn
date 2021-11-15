from os import makedirs
from os.path import join, isfile, isdir, dirname, basename
import numpy as np
import tensorflow as tf
from tensorflow import math as tfm, image as tfi
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Any, Sequence, Dict, Callable
import itertools
from utils.filters import depth_lowpass, zero_shallow, clip, seis_gain, surface_to_depth
from awm import make_awm, calc_dt_min, calc_dt_mod_and_samp_rate, calc_freq_max
from utils.plotutils import plot_seismograms, plot_velocities
from functools import partial
from utils.wavelets import rickerwave
from seismicarrays import make_default_arrays, parse_geometry, make_array
import tensorflow.keras.optimizers as optimizers
from utils.saving import discarray, discarray_to
from utils.saving import pickle_to, unpickle_from
from utils.saving import yaml_to, unyaml_from
from utils.structures import StateInfo, AccumulatingDict
from utils.stopping import StopCondition
from sklearn.model_selection import train_test_split


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


def make_ground_truth(seis_fun, v, X, verbose=True):
    Y = []
    if verbose:
        print(f'Modeling {len(X)} seismograms')
    for i, xi in enumerate(X, start=1):
        if verbose:
            print(f'- Seismogram #{i}')
        srcsgns, srccrds, reccrds = xi
        yi = seis_fun(v=v, srcsgns=srcsgns, srccrds=srccrds, reccrds=reccrds)
        if not np.all(np.isfinite(yi)):
            print('NaNs encountered. Breaking.')
            break
        Y.append(yi.numpy())
    if verbose:
        print()
    return Y


def make_combinations(d: Dict[Any, Sequence]):
    return (dict(zip(d.keys(), c)) for c in itertools.product(*d.values()))


def make_seis_wo_direct_fun(seis, sp_order):
    def seis_wo_direct_fun(v, *args, **kwargs):
        #v_s = surface_to_depth(v, sp_order // 2)
        v_s = surface_to_depth(v, sp_order)
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


def make_eval_epoch(seis_fun, loss_fun, verbose=True):
    def eval_epoch(v, X, Y, idx, info=None):
        if info is None:
            epoch_info = StateInfo()
        else:
            epoch_info = info.copy()

        mean_loss = 0
        if len(idx) > 0:
            if verbose:
                print('Testing')

            for shot_num, shot_idx in enumerate(idx):
                shot_info = epoch_info.copy()
                shot_info['shot'] = 1 + shot_num
                shot_info['shot_idx'] = 1 + shot_idx, 'i'
                if verbose:
                    print('- ',
                          shot_info.print('optimizer', 'frequency', 'epoch', 'shot',
                                          'shot_idx'),
                          sep='',
                          end=' ')
                x, y = X[shot_idx], Y[shot_idx]
                srcsgns, srccrds, reccrds = x
                ŷ = seis_fun(v=v,
                             srcsgns=srcsgns,
                             srccrds=srccrds,
                             reccrds=reccrds)
                loss = loss_fun(ŷ, y)
                mean_loss += loss
                shot_info['loss'] = loss
                if verbose:
                    print(shot_info.print('loss'))

            mean_loss /= len(idx)
        else:
            if verbose:
                print('Testing: no test shots configured')

        epoch_info['mean_loss'] = mean_loss
        print('  > ',
              epoch_info.print('frequncy', 'epoch', 'mean_loss'),
              sep='')
        metrics = {'loss': mean_loss}
        return metrics

    return eval_epoch


def make_train_epoch(val_loss_grad_fun,
                     dz,
                     dx,
                     dt,
                     accum_grads=False,
                     batch_size=1,
                     result_dirs={},
                     verbose=True):
    def train_epoch(optimizer,
                    v_e,
                    X,
                    Y,
                    info=StateInfo(),
                    v_0=None,
                    v_true=None,
                    zero_before_depth=0,
                    idx_train=None,
                    show=False):

        if verbose:
            print('Training')

        idx_train = np.arange(len(X)) if idx_train is None else idx_train
        n_samples = len(idx_train)
        n_batches = n_samples // batch_size
        batches = np.random.permutation(idx_train)
        batches = batches[:n_batches * batch_size]
        batches = batches.reshape((n_batches, batch_size))

        epoch_mean_loss = 0
        epoch_mean_loss_grad = tf.zeros_like(v_e) if accum_grads else None
        for batch_num, batch in enumerate(batches):
            batch_info = info.copy()
            batch_info['batch'] = 1 + batch_num

            # TODO
            # cluster parallelization can be exploited here, i.e.:
            # `tf.distribute.get_replica_context().all_reduce('sum', grads)`
            batch_mean_loss = 0
            # batch_mean_loss_grad = tf.zeros_like(v_e)
            batch_loss_grad = tf.zeros_like(v_e)
            for shot_num, shot_idx in enumerate(batch):
                shot_info = batch_info.copy()
                shot_info['shot'] = 1 + shot_num
                shot_info['shot_idx'] = 1 + shot_idx, 'i'
                if verbose:
                    print('- ',
                          shot_info.print('optimizer', 'frequency', 'epoch', 'batch',
                                          'shot', 'shot_idx'),
                          sep='',
                          end=' ')
                x, y = X[shot_idx], Y[shot_idx]
                ŷ, shot_loss, shot_loss_grad = val_loss_grad_fun(v_e, x, y)
                shot_info['loss'] = shot_loss
                if verbose:
                    print(shot_info.print('loss'))

                # processing gradients
                # shot_loss_grad = clip(shot_loss_grad, 0.02)
                shot_loss_grad = zero_shallow(shot_loss_grad,
                                              zero_before_depth)

                batch_mean_loss += shot_loss / batch_size
                # batch_mean_loss_grad += shot_loss_grad / batch_size
                batch_loss_grad += shot_loss_grad

            # update epoch with batch results
            epoch_mean_loss += batch_mean_loss / n_batches
            if accum_grads:
                # epoch_mean_loss_grad += batch_mean_loss_grad / n_batches
                epoch_mean_loss_grad += batch_loss_grad / n_batches
            batch_info['loss'] = batch_mean_loss

            # save model variables
            if result_dirs.get('v_data'):
                v_file = batch_info.filename() + '.bin'
                v_path = join(result_dirs['v_data'], v_file)
                discarray_to(v_path, v_e.numpy())

            # plotting data
            fig_filename = batch_info.filename() + '.png'
            fig_title = batch_info.pprint('frequency', 'epoch', 'batch',
                                          'shot', 'shot_idx', 'loss')

            if accum_grads:
                loss_grad = epoch_mean_loss_grad
            else:
                # loss_grad = batch_mean_loss_grad
                loss_grad = batch_loss_grad
            if result_dirs.get('v_plot'):
                fig_path = join(result_dirs['v_plot'], fig_filename)
                # TODO plot_velocities should not depend on v true
                plot_velocities(
                    v_e=v_e,
                    loss_grad=loss_grad,
                    v_0=v_0,
                    v_true=v_true,
                    dz=dz,
                    dx=dx,
                    xlabel='x (m)',  # TODO: not only m
                    ylabel='z (m)',  # TODO: not only m
                    cbar_label='m/s',  # TODO: not only m/s
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
                    ylabel='t (s)',  # TODO: not only t
                    title=fig_title,
                    figname=fig_path,
                    show=show)

            diverged = not np.isfinite(batch_mean_loss)
            if diverged:
                break

            # apply gradients
            if not accum_grads:
                # optimizer.apply_gradients([(batch_mean_loss_grad, v_e)])
                optimizer.apply_gradients([(batch_loss_grad, v_e)])

        if accum_grads:
            optimizer.apply_gradients([(epoch_mean_loss_grad, v_e)])

        # save model variables
        if result_dirs.get('v_data'):
            epoch_info = info.copy()
            epoch_info['loss'] = epoch_mean_loss
            v_file = epoch_info.filename() + '.bin'
            v_path = join(result_dirs['v_data'], v_file)
            discarray_to(v_path, v_e.numpy())

        metrics = {'loss': epoch_mean_loss}

        return optimizer, v_e, metrics

    return train_epoch


if __name__ == '__main__':
    from datetime import datetime
    from config import DIR_CONFIG, marmousi_model, multi_layered_model
    from scipy.signal import convolve
    NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    model = marmousi_model
    #model = multi_layered_model
    v_true = model.load()

    shape = nz, nx = model.shape
    delta = dz, dx = model.delta
    model_name = model.name
    dt = 0.004
    print(v_true.shape)

    if model_name == 'multi_layered':
        array_desc = dict(
            geometry='8-6-0-6-8',
            rr=1,
            # ss=5,
            ss=3,
            # ss=40,
            ns=None,
            nx=nx,
            dx=1,
            all_recs=True)
        t_max = .7  # s
    elif model_name == 'marmousi':
        # downscale = 4
        downscale = 2
        v_true = v_true[::downscale, ::downscale]
        shape = nz, nx = v_true.shape
        dz, dx = downscale * dz, downscale * dx

        array_desc = dict(
            # geometry=f'{20*dx}-{1*dx}-0-{1*dx}-{20*dx}',
            geometry=f'{0*dx}-{10*dx}-0-{10*dx}-{0*dx}',
            rr=dx,
            # ss=(nx // 20) * dx,
            # ss=(nx // 20) * dx,
            # ss=12 * dx,
            # ss=3 * dx,
            ss=6 * dx,
            dx=dx,
            nx=nx,
            ns=None,
            all_recs=True)

        t_max = 4  # s

    nt = int(t_max / dt)

    # modeling parameters
    #   seismic source
    sp_order = 8
    freq_max = calc_freq_max(v_true.min(), dz, dx, sp_order=8)
    freq = int(freq_max)
    srcsgn = rickerwave(freq, dt)  # TODO interpolate before while...
    awm, samp_rate = make_awm(v_true.shape,
                              dz,
                              dx,
                              dt,
                              v_max=np.max(v_true),
                              tsolver='fd',
                              spsolver='fd',
                              sp_order=sp_order,
                              return_samp_rate=True)

    dt_min = calc_dt_min(np.max(v_true), dz, dx, sp_order=sp_order)
    dt_mod = dt / samp_rate
    print(f'nt={nt}', f'nt_mod={samp_rate*nt}', f'dt={dt}', f'dt_mod={dt_mod}',
          f'dt_min={dt_min}', f'freq={freq}', f'freq_max={freq_max}')

    make_srcsgns, srccrds, reccrds, true_srccrds, true_reccrds = make_array(
        **array_desc, )
    print(f'Number of shots: {len(srccrds)}')
    print(dz, dx)
    srcsgns = make_srcsgns(srcsgn)

    seis_fun = partial(awm, nt=nt, out='seis')
    seis_wo_direct_fun = make_seis_wo_direct_fun(seis_fun, sp_order)

    # START MAKING DATASET
    X = [*zip(srcsgns, srccrds, reccrds)]
    if isfile(DIR_CONFIG['Y_old']):
        Y = unpickle_from(DIR_CONFIG['Y_old'])
    else:
        Y = make_ground_truth(seis_wo_direct_fun, v_true, X)
        pickle_to(DIR_CONFIG['Y_old'], Y)
    # END MAKING DATASET

    test_split = 1 / 3  # 1/4
    # freqs = 7, 12, 15
    # freqs = 2, 7, 14
    freqs = freq/3, 2*freq/3, freq
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
    idx_test = np.random.choice(idx, size=int(test_split * len(X)), replace=False)
    idx_train = np.delete(idx, idx_test)
    idx_path = join(result_dirs['root'], 'validation_setup.yaml')
    yaml_to(idx_path, {'idx_train': [int(i) for i in idx_train],
                       'idx_test': [int(i) for i in idx_test]})

    # initial model for fwi
    # maintain_before_depth = sp_order // 2
    maintain_before_depth = sp_order
    # maintain_before_depth = 0
    v_0 = depth_lowpass(v_true, ws=30, min_depth=maintain_before_depth)

    # training
    show = False
    accum_grads = False
    zero_before_depth = maintain_before_depth
    max_epochs = 30

    optimizer_param_spaces = {
        'adam': {
            # 'learning_rate': (1 * 10**i for i in range(0, 2 + 1)),
            # 'beta_1': (0.5, 0.7, 0.9,),  # .7,
            # 'beta_2': (0.7, .9, 0.999 ),  # .7,
            # 'learning_rate': (1 * 10**i for i in range(0, 1))[::-1],
            'learning_rate': [1 * 10**i for i in range(0, 2+1)][::-1],
            'beta_1': [0.9][::-1],  # .7,
            'beta_2': [0.999][::-1],  # .7,
        },

        #'sgd': {
        #  'learning_rate': [1 * 10**i for i in range(5, 7 + 1)][::-1],
        #},

        # 'momentum': {
        #   'learning_rate': [1 * 10**i for i in range(5, 7 + 1)][::-1],
        #   'momentum': (0.5, 0.9,),  # .7,
        # },
    }

    for optimizer_name, param_space in optimizer_param_spaces.items():

        if optimizer_name == 'adam':
            optimizer_generator = optimizers.Adam
        elif optimizer_name == 'momentum':  # = SGD in tf2
            optimizer_generator = optimizers.SGD
        elif optimizer_name == 'sgd':
            optimizer_generator = optimizers.SGD
        else:
            print('Unknown optimizer; skipping.')
            continue

        for optimizer_params in make_combinations(param_space):

            # fwi function
            fwi_info = StateInfo()
            fwi_info['optimizer'] = optimizer_name
            for p, v in optimizer_params.items():
                fwi_info[p] = v, p
            fwi_info['frequency'] = 0

            histories = {
                'param': AccumulatingDict(),
                'train': AccumulatingDict(),
                'test': AccumulatingDict(),
            }

            v_e = tf.Variable(v_0, trainable=True)

            v_filename = fwi_info.filename() + '.bin'
            v_path = join(result_dirs['v_data'], v_filename)
            discarray_to(v_path, v_e.numpy())

            for freq, srcsgn in multi_scale_sources.items():
                optimizer = optimizer_generator(**optimizer_params)

                # fwi_fun
                fwi_info['frequency'] = freq

                # train_idx, test_idx
                srcsgns = make_srcsgns(srcsgn)
                X_freq = [*zip(srcsgns, srccrds, reccrds)]
                Y_freq = [convolve(y, srcsgn[:, None], mode='same') for y in Y]

                test_stop_cond = StopCondition(growing_is_good=False,
                                               patience=5)

                for epoch in range(max_epochs):
                    fwi_info['epoch'] = 1 + epoch

                    v_old = v_e.numpy()

                    # divergence should be availed in the loop
                    optimizer, v_e, train_metrics = train_epoch(
                        optimizer=optimizer,
                        v_e=v_e,
                        X=X_freq,
                        Y=Y_freq,
                        idx_train=idx_train,
                        v_0=v_0,
                        v_true=v_true,
                        zero_before_depth=zero_before_depth,
                        info=fwi_info,
                        show=show)

                    test_metrics = eval_epoch(v_e, X_freq, Y_freq, idx_test)

                    mean_abs_delta_v_e = np.mean(np.abs(v_e.numpy() - v_old))

                    histories['train'] += train_metrics
                    histories['test'] += test_metrics
                    histories['param'] += {
                        'mean_abs_delta_v_e': mean_abs_delta_v_e
                    }

                    if v_true is not None:
                        v_rmse = rmse(v_e, v_true)
                        histories['param'] += {'rmse': v_rmse}

                    test_stop_cond.update(test_metrics['loss'], v_e.numpy())
                    diverged = not np.all(np.isfinite(v_e))
                    if test_stop_cond.stop or diverged:
                        v_e = tf.Variable(test_stop_cond.best_checkpoint)
                        break

                diverged = not np.all(np.isfinite(v_e))
                if diverged:
                    break

            # saving histories
            histories_path = join(
                result_dirs['metric_data'],
                fwi_info.filename() + '.pkl',
            )
            histories = {k: dict(v) for k, v in histories.items()}
            pickle_to(histories_path, histories)
