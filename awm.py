import itertools
from typing import Any, Sequence, Dict, Callable
import numpy as np
import tensorflow as tf
from functools import partial
from sources import rickerwave
# from laplacians import make_fd_laplacian
# from wavesolvers import make_fd_wavesolver
from seismicarrays import make_default_array, make_default_arrays
from filters import depth_lowpass, zero_shallow, clip, seisgain, tf_surface_to_depth
from scipy.special import jv
from pickle import loads, dumps
from os import makedirs
from os.path import join, dirname, isfile, isdir, dirname, basename
import tensorflow.keras.optimizers as optimizers
from discarrays import discarray, todiscarray
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use("TkAgg")


def central_difference_coefs(degree: int, order: int):
    '''
    Generates central difference coeficients.
    '''
    assert order % 2 == 0
    p = (order + 1) // 2
    # defining P matrix
    P = np.empty((2 * p + 1, 2 * p + 1))
    P[0, :] = 1
    P[1, :] = np.arange(-p, p + 1)
    for i in np.arange(1, P.shape[0]):
        P[i, :] = P[1, :]**(i)
    # defining d matrix
    d = np.zeros(P.shape[0])
    d[degree] = np.math.factorial(degree)
    # solving P c = d
    c = np.linalg.solve(P, d)
    return c


def make_fd_laplacian(dz, dx, order=2):
    stencil = central_difference_coefs(2, order)
    stencil = tf.constant(stencil, dtype=tf.float32)
    zkern = stencil[::-1, None, None, None] / dz**2
    xkern = stencil[None, ::-1, None, None] / dx**2

    def laplacian(P):
        P = P[None, :, :, None]
        Pzz = tf.nn.convolution(P, zkern, padding="SAME")[0, ..., 0]
        Pxx = tf.nn.convolution(P, xkern, padding="SAME")[0, ..., 0]
        return Pzz + Pxx

    return laplacian


def make_s_laplacian(shape, dz: float = 1., dx: float = 1.):
    '''
    Spectral laplacian operator.
    '''
    def gen_filter(shape, dz, dx):
        nz, nx = shape
        kz_ax = 2 * np.pi * np.fft.fftfreq(nz, dz)
        kx_ax = 2 * np.pi * np.fft.fftfreq(nx, dx)
        kz, kx = np.meshgrid(kz_ax, kx_ax, indexing='ij')
        phi = -(kz**2 + kx**2)
        return phi

    phi = gen_filter(shape, dz, dx)

    def laplacian(A):
        A = tf.cast(A, tf.complex64)
        return tf.math.real(tf.signal.ifft2d(phi * tf.signal.fft2d(A)))

    return laplacian


def make_fd_wavesolver(dt, laplacian):
    '''
    Finite difference 2D acoustic wave equation time solver.
    '''
    def wavesolver(v, P_old, P_cur):
        lap_P_cur = laplacian(P_cur)
        P_new = 2 * P_cur - P_old + (v * dt)**2 * lap_P_cur
        return P_new

    return wavesolver


def make_re_wavesolver(dz, dx, dt, vmax, laplacian):
    '''
    Rapid expansion 2D wave equation time solver using Chebyshev polynomials
    (Pestana & Stoffa, 2010; Araujo & Pestana, 2019).
    '''
    dl = np.sqrt(1 / dz**2 + 1 / dx**2)
    R = np.pi * vmax * dl
    R_dt = R * dt
    # m > R dt  (=R dt + 1 generates low freq noise)
    m = int(np.ceil(R_dt)) + 2
    k = np.arange(m)
    c = np.where(k == 0, 1, 2)
    cJ = np.asarray([ci * jv(2 * ki, R_dt) for ci, ki in zip(c, k)])

    def wavesolver(v, P_old, P_cur):
        cos_LdtP = tf.zeros_like(P_cur)
        for k in range(m):
            if k == 0:
                P_new = P_cur  # maybe a copy is needed
            else:
                lap_P_cur = laplacian(P_cur)
                if k == 1:
                    P_new = P_cur + 2 * (v / R)**2 * lap_P_cur
                else:
                    P_new = 2 * P_cur + 4 * (v / R)**2 * lap_P_cur - P_new
            cos_LdtP += cJ[k] * P_new
            P_cur, P_new = P_new, P_cur
        P_new = 2 * cos_LdtP - P_old
        return P_new

    return wavesolver


def padcoords(crds, value, until=2):
    crds = np.asarray(crds).copy()
    crds[:, :until] += value
    return crds


def make_record(out='snaps', shape=None, reccrds=None, taper=0):
    if out == 'snaps':
        nz, nx = shape
        recorded_shape = nz - 2 * taper, nx - 2 * taper

        def record(P):
            return P[taper:nz - taper, taper:nx - taper]
    elif out == 'seis':
        if reccrds is None:
            nz, nx = shape
            recorded_shape = nx - 2 * taper,

            def record(P):
                return P[taper, taper:nx - taper]
        else:
            reccrds = padcoords(reccrds, taper, until=2)
            recorded_shape = len(reccrds),

            def record(P):
                return tf.gather_nd(P, reccrds)
    else:
        raise NotImplementedError("Only 'seis' or 'snaps' options allowed"
                                  "as 'out' argument.")
    return record, recorded_shape


def make_addsources(srcsgns, srccrds, taper=0):
    if srccrds is None:

        def addsources(P, t):
            return P

    else:
        padded_srccrds = padcoords(srccrds, taper, until=2)

        def addsource(P, t, sgn, crd):
            return tf.tensor_scatter_nd_add(
                P,
                crd[None, :2],
                sgn[None, t - crd[-1]],
            )

        def addsources(P, t):
            for srccrd, srcsgn in zip(padded_srccrds, srcsgns):
                sz, sx, st = srccrd
                P = tf.cond(
                    (st <= t) & (t < st + srcsgn.size),
                    lambda: addsource(P, t, srcsgn, srccrd),
                    lambda: P,
                )
            return P

    return addsources


def make_attenuate(shape: tuple,
                   taper: int = 60,
                   attenuation: float = 0.0035,
                   offset: int = 0):
    def factor(depth):
        return np.exp(-(attenuation * depth)**2)

    def get_factors(shape):
        nz, nx = shape
        factors = np.ones((nz + 2 * taper, nx + 2 * taper))
        nz, nx = padded_shape = factors.shape
        for i in range(offset, taper):
            depth = taper - i + 1
            factors[i, i:nx - i] = \
                factors[nz - i - 1, i:nx - i] = \
                factors[i:nz - i, i] = \
                factors[i:nz - i, nx - i - 1] = \
                factor(depth)
        return factors, padded_shape

    if taper:
        factors, padded_shape = get_factors(shape)
        factors = tf.constant(factors, dtype=tf.float32)

        def attenuate(P):
            return factors * P
    else:
        padded_shape = shape

        def attenuate(P):
            return P

    return attenuate, padded_shape


def make_awm_cell(
    wavesolver: Callable,
    attenuate: Callable = lambda x: x,
    addsources: Callable = lambda x: x,
    record: Callable = lambda x: x,
):
    '''
    2D acoustic wave equation time solver as a Jordan recurrent cell. This
    module is meant to be used with scan.
    '''
    def awm_cell(carry, t, v):
        out, (P_old, P_cur) = carry
        P_old = attenuate(P_old)
        P_cur = addsources(P_cur, t)
        P_new = wavesolver(v, P_old, P_cur)
        P_new = attenuate(P_new)
        out = record(P_new)
        return out, (P_cur, P_new)

    return awm_cell


def make_awm(shape,
             dz,
             dx,
             dt,
             spsolver='fd',
             tsolver='fd',
             sporder=2,
             vmax=None):
    attenuate, padded_shape = make_attenuate(shape)
    taper = (padded_shape[0] - shape[0]) // 2

    if spsolver == 'fd':
        laplacian = make_fd_laplacian(dz, dx, sporder)
    elif spsolver == 's':
        laplacian = make_s_laplacian(padded_shape, dz, dx)

    if tsolver == 'fd':
        wavesolver = make_fd_wavesolver(dt, laplacian)
    elif tsolver == 're':
        assert vmax
        wavesolver = make_re_wavesolver(dz, dx, dt, vmax, laplacian)

    def awm(v,
            nt,
            out='snaps',
            srcsgns=None,
            srccrds=None,
            reccrds=None,
            P_old=None,
            P_cur=None):
        addsources = make_addsources(srcsgns, srccrds, taper)
        record, recorded_shape = make_record(out, padded_shape, reccrds, taper)
        awm_cell = make_awm_cell(wavesolver, attenuate, addsources, record)

        v = tf_pad_equal(v, taper)

        P_old = P_old if P_old is not None else tf.zeros_like(v)
        P_cur = P_cur if P_cur is not None else P_old
        out = tf.zeros(recorded_shape)
        carry = out, (P_old, P_cur)
        const_v_awm_cell = partial(awm_cell, v=v)
        outs, carry = tf.scan(const_v_awm_cell, tf.range(nt), carry)
        return outs

    return awm


def tf_pad_equal(A, padding):
    _A = A
    for i in range(padding):
        _A = tf.pad(_A, [[1, 1], [1, 1]], "SYMMETRIC")
    return _A


def mse(ŷ, y):
    return 1 / 2 * tf.reduce_mean((ŷ - y)**2)


def make_mse_loss(model, y):
    def loss(v):
        ŷ = model(v)
        return mse(ŷ, y)

    return loss


def make_ground_truth(v, X, seis):
    Y = []
    for xi in X:
        srcsgns, srccrds, reccrds = xi
        yi = seis(v=v, srcsgns=srcsgns, srccrds=srccrds, reccrds=reccrds)
        Y.append(yi)
    return Y


def plot_velocities(v,
                    v_0,
                    v_e,
                    loss_grad,
                    title=None,
                    figname=None,
                    figsize=(8, 5),
                    v_unit='m/s',
                    show=None,
                    dpi=300):
    fig, axes = plt.subplots(2,
                             2,
                             sharex=True,
                             sharey=True,
                             figsize=figsize,
                             dpi=dpi)

    if title:
        fig.suptitle(title)

    vmin, vmax = np.min(v_e), np.max(v_e)
    avmax = np.max(np.abs((vmin, vmax)))
    axes.flat[0].set_title('$v_e$')
    im = axes.flat[0].imshow(v_e, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.flat[0])
    cbar.set_label(f'{v_unit}')

    axes.flat[1].set_title('$v$')
    im = axes.flat[1].imshow(v, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.flat[1])
    cbar.set_label(f'{v_unit}')

    vmin, vmax = np.min(loss_grad), np.max(loss_grad)
    avmax = np.max(np.abs((vmin, vmax)))
    axes.flat[3].set_title(r'$\nabla$ loss($v_e$)')
    im = axes.flat[3].imshow(loss_grad,
                             vmin=-avmax,
                             vmax=avmax,
                             cmap='seismic')
    cbar = fig.colorbar(im, ax=axes.flat[3])
    cbar.set_label(f'{v_unit}')

    delta = (v_e - v_0)
    vmin, vmax = np.min(delta), np.max(delta)
    avmax = np.max(np.abs((vmin, vmax)))
    axes.flat[2].set_title('$v_e - v_0$')
    im = axes.flat[2].imshow(delta, vmin=-avmax, vmax=avmax, cmap='seismic')
    cbar = fig.colorbar(im, ax=axes.flat[2])
    cbar.set_label(f'{v_unit}')

    if figname:
        fig.savefig(figname)
    else:
        show = True if show is None else show

    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


def plot_seismograms(seis_i,
                     seis_t,
                     dt=0.015,
                     title=None,
                     figname=None,
                     figsize=(8, 5),
                     seis_unit='Ampliude',
                     show=None,
                     dpi=300):
    fig, axes = plt.subplots(1,
                             3,
                             sharex=True,
                             sharey=True,
                             figsize=figsize,
                             dpi=dpi)

    if title:
        fig.suptitle(title)

    vmin = vmax = None

    axes.flat[0].set_title('Predicted')
    # seis_i_gain = seisgain(seis_i, dt=dt, a=.8, b=0.001)
    im = axes.flat[0].imshow(seis_i,
                             vmin=vmin,
                             vmax=vmax,
                             cmap='gray',
                             aspect='auto')

    axes.flat[1].set_title('Ground-truth')
    #seis_t_gain = seisgain(seis_t, dt=dt, a=.8, b=0.001)
    im = axes.flat[1].imshow(seis_t,
                             vmin=vmin,
                             vmax=vmax,
                             cmap='gray',
                             aspect='auto')

    axes.flat[2].set_title('Difference')
    delta_gain = seisgain(seis_i - seis_t, dt=dt, a=.8, b=0.001)
    im = axes.flat[2].imshow(delta_gain, cmap='gray', aspect='auto')

    if figname:
        fig.savefig(figname)
    else:
        show = True if show is None else show

    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


def make_combinations(d: Dict[Any, Sequence]):
    return (dict(zip(d.keys(), c)) for c in itertools.product(*d.values()))


if __name__ == '__main__':
    from datetime import datetime
    start = datetime.now().strftime("%Y%m%d%H%M%S")

    from params import SHOTS_FILE, IMG_DIR, SEIS_DIR, V_DIR

    # TODO create DIRS variable in params
    img_dir = join(dirname(IMG_DIR), f'{start}_{basename(IMG_DIR)}')
    seis_dir = join(dirname(SEIS_DIR), f'{start}_{basename(SEIS_DIR)}')
    v_dir = join(dirname(V_DIR), f'{start}_{basename(V_DIR)}')

    for folder in (dirname(SHOTS_FILE), img_dir, seis_dir, v_dir):
        makedirs(folder, exist_ok=True)

    # velocity field
    shape = nz, nx = 70, 120
    nt = 800

    v = np.empty(shape, dtype=np.float32)
    v[None:nz // 3, :] = 2000.
    v[nz // 3:nz // 2, :] = 3500.
    v[nz // 2:None, :] = 6000.

    # modeling parameters
    dz, dx, dt = 100., 100., 0.0075
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

    def seis_wo_dw(v, *args, **kwargs):
        v_s = tf_surface_to_depth(v)
        s = seis(v, *args, **kwargs)
        dw = seis(v_s, *args, **kwargs)
        fig, axes = plt.subplots(1, 3)
        m = np.max(np.abs(s))
        axes[0].imshow(s, aspect='auto', vmin=-m, vmax=m)
        axes[1].imshow(dw, aspect='auto', vmin=-m, vmax=m)
        axes[2].imshow(s - dw, aspect='auto', vmin=-m, vmax=m)
        plt.show()
        return s - dw


    # seismic source
    nu = 2
    signal = rickerwave(nu, dt)

    # dataset
    srcsgns, srccrds, reccrds = make_default_arrays(shape, signal)
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

    # initial model for fwi
    v_0 = depth_lowpass(v, ws=50, min_depth=0)

    # training
    model_name = '3layed'
    show = False
    accumulate_gradients = False
    zero_before_depth = 10
    epochs = 30
    optimizer_param_spaces = {
        'adam': {
            'learning_rate': (1 * 10**i for i in range(1, 3 + 1)),
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

            name = '_'.join([
                model_name,
                start,
                optimizer_name,
                *[f'{p}={v}' for p, v in optimizer_params.items()],
            ])

            print(name)

            v_e = tf.Variable(v_0, trainable=True)
            for e in range(1, epochs + 1):

                if accumulate_gradients:
                    loss_grad = np.zeros_like(v_e)

                for i, (xi, yi) in enumerate(zip(X, Y), start=1):
                    print(f'  e={e} i={i}')

                    v_filename = f'{name}_e={e}_i={i}.bin'
                    v_path = join(v_dir, v_filename)
                    todiscarray(v_path, v_e.numpy())

                    with tf.GradientTape() as tape:
                        tape.watch(v_e)
                        srcsgns, srccrds, reccrds = xi
                        ŷi = seis_wo_dw(v_e,
                                        srcsgns=srcsgns,
                                        srccrds=srccrds,
                                        reccrds=reccrds)
                        plt.imshow(ŷi, aspect='auto')
                        plt.show()

                        loss = mse(ŷi, yi)

                    loss_grad_i = tape.gradient(loss, v_e).numpy()

                    # loss_grad_i = clip(loss_grad_i, 0.02)
                    # loss_grad_i = zero_shallow(loss_grad_i, zero_before_depth)

                    if accumulate_gradients:
                        loss_grad += loss_grad_i / len(X)

                    fig_filename = f'{name}_e={e}_i={i}_loss={loss:.4g}.png'
                    fig_path = join(img_dir, fig_filename)
                    fig_title = f'Epoch {e}, Shot {i}, Loss={loss:.4g}'
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
                    fig_title = f'Epoch {e}, Shot {i}, Loss={loss:.4g}'
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

                # parallelization tip
                # grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
