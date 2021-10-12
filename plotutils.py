import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from filters import depth_lowpass, zero_shallow, clip, seis_gain, tf_surface_to_depth


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
    if not np.isfinite(avmax):
        avmax = 0
    axes.flat[3].set_title(r'$\nabla$ loss($v_e$)')
    im = axes.flat[3].imshow(loss_grad,
                             vmin=-avmax,
                             vmax=avmax,
                             norm=mpl.colors.SymLogNorm(1e-8),
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
    # seis_t_gain = seisgain(seis_t, dt=dt, a=.8, b=0.001)
    im = axes.flat[1].imshow(seis_t,
                             vmin=vmin,
                             vmax=vmax,
                             cmap='gray',
                             aspect='auto')

    axes.flat[2].set_title('Difference')
    delta_gain = seis_gain(seis_i - seis_t, dt=dt, a=.8, b=0.001)
    im = axes.flat[2].imshow(delta_gain, cmap='gray', aspect='auto')

    if figname:
        fig.savefig(figname)
    else:
        show = True if show is None else show

    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)
