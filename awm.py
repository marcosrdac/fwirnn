from typing import Any, Sequence, Dict, Callable
import numpy as np
import tensorflow as tf
from functools import partial
from scipy.special import jv
from os import makedirs
from os.path import join, dirname, isfile, isdir, dirname, basename
from utils.saving import discarray, discarray_to
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


def make_fd_laplacian(dz: int, dx: int, order: int = 2) -> Callable:
    stencil = central_difference_coefs(2, order)
    stencil = tf.constant(stencil, dtype=tf.float32)
    # will not reverse stencil as it is symmetric
    zkern = stencil[:, None, None, None] / dz**2
    xkern = stencil[None, :, None, None] / dx**2

    def laplacian(P: tf.Tensor) -> tf.Tensor:
        P = P[None, :, :, None]
        Pzz = tf.nn.convolution(P, zkern, padding="SAME")[0, ..., 0]
        Pxx = tf.nn.convolution(P, xkern, padding="SAME")[0, ..., 0]
        return Pzz + Pxx

    return laplacian


def make_s_laplacian(shape, dz: float = 1., dx: float = 1.) -> Callable:
    '''
    Spectral laplacian operator.
    '''
    def gen_filter(shape: tuple, dz: int, dx: int) -> np.array:
        nz, nx = shape
        kz_ax = 2 * np.pi * np.fft.fftfreq(nz, dz)
        kx_ax = 2 * np.pi * np.fft.fftfreq(nx, dx)
        kz, kx = np.meshgrid(kz_ax, kx_ax, indexing='ij')
        phi = -(kz**2 + kx**2)
        return phi

    phi = gen_filter(shape, dz, dx)

    def laplacian(A: tf.Tensor) -> tf.Tensor:
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
            # print(t, crd[-1], t-crd[-1])
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


def calc_dt_max(v_max, dz, dx, p=.95):
    '''
    Maximum dt used for a model of maximum velocity v_max, given grid 
    spacings dz and dx.
    '''
    return p*np.sqrt((3 / 4) / ((v_max / dx)**2 + (v_max / dz)**2))


def calc_dt_mod_and_samp_rate(dt, dt_max):
    rate = int(np.ceil(dt / dt_max))
    dt_mod = dt / rate
    return dt_mod, rate


def calc_freq_max(v_min, *dl, p=.95):
    dl_max = np.max(dl)
    freq_max = v_min / (10 * dl_max)
    return p * freq_max


def make_awm(shape, dz, dx, dt, v_max, spsolver='fd', tsolver='fd', sporder=2):

    attenuate, padded_shape = make_attenuate(shape)
    taper = (padded_shape[0] - shape[0]) // 2

    if spsolver == 'fd':
        laplacian = make_fd_laplacian(dz, dx, sporder)
    elif spsolver == 's':
        laplacian = make_s_laplacian(padded_shape, dz, dx)

    samp_rate = 1
    if tsolver == 'fd':
        dt_max = calc_dt_max(v_max, dz, dx)
        dt_mod, samp_rate = calc_dt_mod_and_samp_rate(dt, dt_max)
        wavesolver = make_fd_wavesolver(dt_mod, laplacian)
    elif tsolver == 're':
        assert v_max
        wavesolver = make_re_wavesolver(dz, dx, dt, v_max, laplacian)

    def awm(v,
            nt,
            out='snaps',
            srcsgns=None,
            srccrds=None,
            reccrds=None,
            P_old=None,
            P_cur=None):

        nt_mod = nt * samp_rate
        addsources = make_addsources(srcsgns, srccrds, taper)
        record, recorded_shape = make_record(out, padded_shape, reccrds, taper)
        awm_cell = make_awm_cell(wavesolver, attenuate, addsources, record)

        v = tf_pad_equal(v, taper)

        P_old = P_old if P_old is not None else tf.zeros_like(v)
        P_cur = P_cur if P_cur is not None else P_old
        out = tf.zeros(recorded_shape)
        carry = out, (P_old, P_cur)
        const_v_awm_cell = partial(awm_cell, v=v)
        outs, carry = tf.scan(const_v_awm_cell, tf.range(nt_mod), carry)
        return outs[::samp_rate, ...]

    return awm


def tf_pad_equal(A, padding):
    _A = A
    for i in range(padding):
        _A = tf.pad(_A, [[1, 1], [1, 1]], "SYMMETRIC")
    return _A


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


if __name__ == '__main__':
    from utils.wavelets import rickerwave
    from datetime import datetime
    from seismicarrays import make_default_arrays, make_array, parse_geometry
    from utils.filters import depth_lowpass, surface_to_depth
    start = datetime.now().strftime("%Y%m%d%H%M%S")

    # from config import SHOTS_FILE, IMG_DIR, SEIS_DIR, V_DIR

    # TODO create DIRS variable in params
    # img_dir = join(dirname(IMG_DIR), f'{start}_{basename(IMG_DIR)}')
    # seis_dir = join(dirname(SEIS_DIR), f'{start}_{basename(SEIS_DIR)}')
    # v_dir = join(dirname(V_DIR), f'{start}_{basename(V_DIR)}')

    # for folder in (dirname(SHOTS_FILE), img_dir, seis_dir, v_dir):
    # makedirs(folder, exist_ok=True)

    # velocity field

    # shape = nz, nx = 70, 120
    # v = np.empty(shape, dtype=np.float32)
    # v[None:nz // 3, :] = 2000.
    # v[nz // 3:nz // 2, :] = 3500.
    # v[nz // 2:None, :] = 6000.

    v = discarray(
        '/home/marcosrdac/cld/Dropbox/home/pro/0/awm/awm2d/models/marmousi.bin',
        mode='r',
        dtype=float).astype(np.float32)
    shape = nz, nx = v.shape

    # dz, dx, dt = 100., 100., 0.0075
    dz, dx, dt = 100., 100., 0.004

    dt_max = calc_dt_max(v.max(), dz, dx)
    dt_mod, samp_rate = calc_dt_mod_and_samp_rate(dt, dt_max)

    nt = int(1 / dt)  # 1 s
    print(nt, nt * samp_rate, dt, dt_max, dt_mod, samp_rate)

    # seismic source
    nu = 2
    signal = rickerwave(nu, dt)

    # dataset
    srcsgns, srccrds, reccrds, true_srccrds, true_reccrds = make_array(
        srcsgn=signal,
        geometry='5200-130-0',
        rr=130,
        ss=5200,
        dx=260,
        nx=nx,
        ns=None,
        all_recs=True,
    )

    print('Number of shots =', len(srcsgns))

    # modeling parameters
    sporder = 8

    awm = make_awm(v.shape,
                   dz,
                   dx,
                   dt,
                   tsolver='fd',
                   spsolver='fd',
                   sporder=sporder,
                   v_max=np.max(v))
    seis = partial(awm, nt=nt, out='seis')

    def seis_wo_dw(v, *args, **kwargs):
        v = tf.Variable(v)
        v_s = surface_to_depth(v, sporder // 2)
        s = seis(v, *args, **kwargs)
        dw = seis(v_s, *args, **kwargs)
        m = np.max(np.abs(s))
        cut = 10**(-2)
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(s,
                       aspect='auto',
                       vmin=-m,
                       vmax=m,
                       norm=mpl.colors.SymLogNorm(cut))
        axes[1].imshow(dw,
                       aspect='auto',
                       vmin=-m,
                       vmax=m,
                       norm=mpl.colors.SymLogNorm(cut))
        axes[2].imshow(s - dw,
                       aspect='auto',
                       vmin=-m,
                       vmax=m,
                       norm=mpl.colors.SymLogNorm(cut))
        plt.show()
        return s - dw

    # low passed model
    # v = depth_lowpass(v, ws=20, min_depth=0)

    s = 9
    seis_wo_dw(v, srcsgns=srcsgns[s], srccrds=srccrds[s], reccrds=reccrds[s])
